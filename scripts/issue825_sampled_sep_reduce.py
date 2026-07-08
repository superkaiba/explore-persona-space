"""Issue #825 `sampled-separator-control` G3b: pooled-draw store reduction.

Consumes the arm-C POOLED store (one extracted row per (article, draw) at the
FIXED prefix-final anchor — window_id ``wiki:NNNNN:cd<k>``) and emits, in the
``issue931_extract_store`` shard format ``issue931_fit_cells.load_regime_store``
consumes verbatim:

  C-avg    one row per KEPT article: Y = fp32 mean over VALID draws (>=
           ``--k-valid-floor`` of them, else the article drops — reported);
           X arrays = the lowest valid draw's rows ("single X": the anchor sits
           in the shared prefix, identical across draws up to bf16 batch
           nondeterminism)
  C-single the draw-0 row allowlist (the round-5 allowlist mechanism)

X-identity gate (plan v22 section 7, HALT-class): per article, cosine between
each draw's ``x_sep`` @ L19 and the reference draw's — calibrated between the
legit bf16 nondeterminism band (round-7 ``armC_equivalence`` early-cos >=
0.99996) and a structural wrong-offset failure (<< 0.9). On a breach the
script STILL WRITES both output stores + the distribution JSON, then exits
rc=7 (UPLOAD-THEN-HALT: the dispatcher records GATE_FAIL and halts after p5
uploads — persist-by-default beats fail-fast). ``--smoke`` records the gate
non-binding; ``--self-test`` (smoke) plants a perturbed row and asserts the
gate mechanics catch it.

CLI:
  uv run python scripts/issue825_sampled_sep_reduce.py --model base \
      --pooled-data-dir data/issue_825/sampled_sep/base/armC \
      --avg-data-dir data/issue_825/sampled_sep/base/armC_avg \
      --single-data-dir data/issue_825/sampled_sep/base/armC_single \
      --out-dir eval_results/issue_825/sampled-separator-control/base
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps bind before numpy/torch import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402
import issue931_fit_cells as fit931  # noqa: E402

SCRIPT = "scripts/issue825_sampled_sep_reduce.py"
FOLLOWUP_LABEL = "sampled-separator-control"
K_VALID_FLOOR = 6  # of 10 draws (plan v22 section 11: majority-of-draws floor)
# HALT threshold (plan v22 section 7): same-surface reference = round-7
# armC_equivalence early-cos >= 0.99996 (legit bf16 band); structural
# wrong-offset failure reads << 0.9 — >=10x margin each way.
X_IDENTITY_COS_MIN = 0.999
GATE_RC = 7
ROW_ID_RE = re.compile(r"^(?P<article>wiki:\d+):cd(?P<draw>\d+):a(?P<anchor>\d+)$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=("base", "instruct"))
    ap.add_argument("--pooled-data-dir", type=Path, required=True, help="arm-C pooled data dir")
    ap.add_argument("--avg-data-dir", type=Path, required=True)
    ap.add_argument("--single-data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True, help="reduce_summary.json dest")
    ap.add_argument("--k-valid-floor", type=int, default=K_VALID_FLOOR)
    ap.add_argument("--halt-cos", type=float, default=X_IDENTITY_COS_MIN)
    ap.add_argument("--smoke", action="store_true", help="gate recorded, not binding")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="SMOKE ONLY: plant a perturbed x_sep row and assert the gate catches it",
    )
    return ap.parse_args()


def parse_rows(row_ids: np.ndarray) -> tuple[list[str], list[int], list[int]]:
    """row_id 'wiki:NNNNN:cd<k>:a<t>' -> (article, draw, anchor) per row."""
    articles, draws, anchors = [], [], []
    for rid in row_ids:
        m = ROW_ID_RE.match(str(rid))
        assert m, f"unexpected pooled row_id shape: {rid}"
        articles.append(m.group("article"))
        draws.append(int(m.group("draw")))
        anchors.append(int(m.group("anchor")))
    return articles, draws, anchors


def x_identity_stats(
    x_sep: np.ndarray, rows_of: dict[str, list[int]], draws: list[int], hl: int
) -> dict:
    """Per-article min cosine of each draw's x_sep @ hl vs the reference
    (lowest) draw — vectorized per article (one matmul per article group)."""
    per_article: dict[str, float] = {}
    all_cos: list[float] = []
    for article, idxs in rows_of.items():
        order = sorted(idxs, key=lambda i: draws[i])
        ref = x_sep[order[0], hl, :].astype(np.float64)
        rest = x_sep[order, hl, :].astype(np.float64)
        num = rest @ ref
        den = np.linalg.norm(rest, axis=1) * np.linalg.norm(ref)
        cos = num / np.maximum(den, 1e-30)
        per_article[article] = float(cos.min())
        all_cos.extend(float(v) for v in cos[1:])  # exclude the self-cos
    arr = np.asarray(all_cos, dtype=np.float64)
    return {
        "per_article_min": per_article,
        "min": float(arr.min()) if arr.size else 1.0,
        "mean": float(arr.mean()) if arr.size else 1.0,
        "p01": float(np.quantile(arr, 0.01)) if arr.size else 1.0,
        "n_comparisons": int(arr.size),
    }


def write_store(dest_dir: Path, regime: str, rows: list[dict]) -> None:
    """One shard in the issue931_extract_store format (fp32 tensors)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0] if isinstance(rows[0][k], np.ndarray)]
    payload = {
        "row_ids": [r["row_id"] for r in rows],
        "group_ids": [r["group_id"] for r in rows],
        "char_ids": [r["char_id"] for r in rows],
        "arrays": {
            k: torch.from_numpy(np.stack([r[k] for r in rows]).astype(np.float32)) for k in keys
        },
    }
    n_layers, hidden = rows[0][keys[0]].shape
    for k, v in payload["arrays"].items():
        assert v.shape == (len(rows), n_layers, hidden), (k, v.shape)
    pt_path = dest_dir / f"{regime}_shard000.pt"
    tmp = pt_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(pt_path)
    sidecar = {
        "regime": regime,
        "shard_index": 0,
        "n_rows": len(rows),
        "row_ids": payload["row_ids"],
        "group_ids": payload["group_ids"],
        "keys": keys,
        "shape_per_row": [int(n_layers), int(hidden)],
        "metadata": common.metadata(SCRIPT, common.BUILD_SEED, len(rows)),
    }
    (dest_dir / f"{regime}_shard000.json").write_text(json.dumps(sidecar, indent=2))
    print(f"[i825-ss-reduce] wrote {pt_path} ({len(rows)} rows)")


def main() -> int:
    args = parse_args()
    if args.self_test:
        assert args.smoke, "--self-test is smoke-only"
    store = fit931.load_regime_store(args.pooled_data_dir / "store" / "armC", "armC")
    row_ids = store["row_ids"]
    groups = store["group_ids"]
    arrays = store["arrays"]
    articles, draws, anchors = parse_rows(row_ids)
    for a, g in zip(articles, groups, strict=True):
        assert a == str(g), ("group_id != parsed article", a, g)
    rows_of: dict[str, list[int]] = {}
    for i, a in enumerate(articles):
        rows_of.setdefault(a, []).append(i)

    n_layers = arrays["x_sep"].shape[1]
    hl = common.HEADLINE_LAYER if n_layers > common.HEADLINE_LAYER else n_layers - 1

    # --- X-identity gate (per article, vs the lowest-draw reference row) ---
    gate = x_identity_stats(arrays["x_sep"], rows_of, draws, hl)
    gate_pass = gate["min"] >= args.halt_cos
    if args.self_test:
        planted = arrays["x_sep"].copy()
        first_article = next(iter(rows_of))
        victim = sorted(rows_of[first_article], key=lambda i: draws[i])[-1]
        planted[victim, hl, :] = np.roll(planted[victim, hl, :], 7)  # structural corruption
        planted_gate = x_identity_stats(planted, rows_of, draws, hl)
        assert planted_gate["min"] < args.halt_cos, (
            "planted x_sep corruption NOT caught — gate mechanics broken",
            planted_gate["min"],
        )
        print(f"[i825-ss-reduce] gate self-test PASS (planted min {planted_gate['min']:.4f})")

    # --- C-avg: mean-Y (fp32) over valid draws; X arrays from the lowest
    # valid draw; article kept iff K_valid >= floor -----------------------
    array_keys = sorted(arrays.keys())
    avg_rows: list[dict] = []
    single_rows: list[dict] = []
    k_valid: dict[str, int] = {}
    for article in sorted(rows_of, key=lambda a: int(a.split(":")[1])):
        idxs = sorted(rows_of[article], key=lambda i: draws[i])
        k_valid[article] = len(idxs)
        anchor = anchors[idxs[0]]
        assert all(anchors[i] == anchor for i in idxs), (article, "anchor drift across draws")
        d0 = [i for i in idxs if draws[i] == 0]
        if d0:
            i0 = d0[0]
            single_rows.append(
                {
                    "row_id": str(row_ids[i0]),
                    "group_id": article,
                    "char_id": "sep",
                    **{k: arrays[k][i0].astype(np.float32) for k in array_keys},
                }
            )
        if len(idxs) < args.k_valid_floor:
            continue
        rep = idxs[0]  # lowest valid draw = the "single X" representative
        row = {
            "row_id": f"{article}:cdavg:a{anchor}",
            "group_id": article,
            "char_id": "sep",
        }
        for k in array_keys:
            if k == "y":
                row[k] = arrays[k][idxs].astype(np.float64).mean(axis=0).astype(np.float32)
            else:
                row[k] = arrays[k][rep].astype(np.float32)
        avg_rows.append(row)

    assert avg_rows, "C-avg store is empty (every article below the K_valid floor)"
    assert single_rows, "C-single store is empty (no draw-0 rows)"
    write_store(args.avg_data_dir / "store" / "armC", "armC", avg_rows)
    write_store(args.single_data_dir / "store" / "armC", "armC", single_rows)

    kv = np.asarray(list(k_valid.values()), dtype=np.int64)
    md = common.metadata(SCRIPT, common.BUILD_SEED, len(avg_rows))
    md["issue"] = 825
    summary = {
        "metadata": md,
        "followup_label": FOLLOWUP_LABEL,
        "model": args.model,
        "n_pooled_rows": len(row_ids),
        "n_articles": len(rows_of),
        "k_valid_floor": int(args.k_valid_floor),
        "k_valid_distribution": {
            str(k): int((kv == k).sum()) for k in range(int(kv.max()) + 1 if kv.size else 1)
        },
        "n_avg_rows": len(avg_rows),
        "n_single_rows": len(single_rows),
        "n_articles_below_floor": int((kv < args.k_valid_floor).sum()),
        "x_identity_gate": {
            "headline_layer": int(hl),
            "halt_cos": args.halt_cos,
            "pass": bool(gate_pass),
            "binding": not args.smoke,
            "min": gate["min"],
            "mean": gate["mean"],
            "p01": gate["p01"],
            "n_comparisons": gate["n_comparisons"],
            "per_article_min": gate["per_article_min"],
            "calibration": (
                "same-surface reference: round-7 armC_equivalence early-cos >= 0.99996 "
                "(legit bf16 batch nondeterminism); structural wrong-offset failure "
                "<< 0.9 (plan v22 section 7)"
            ),
        },
        "convention": (
            "C-avg: fp32 mean-Y over valid draws, X arrays from the lowest valid draw "
            "(single X — the anchor sits in the shared prefix); C-single: draw-0 "
            "allowlist (round-5 mechanism)"
        ),
    }
    common.write_json(args.out_dir / "reduce_summary.json", summary)
    print(
        f"[i825-ss-reduce] {args.model}: pooled={len(row_ids)} articles={len(rows_of)} "
        f"avg={len(avg_rows)} single={len(single_rows)} "
        f"x_identity_min={gate['min']:.6f} pass={gate_pass} binding={not args.smoke}"
    )
    if not gate_pass and not args.smoke:
        print(
            f"[i825-ss-reduce] X-IDENTITY GATE FAIL (min {gate['min']:.6f} < {args.halt_cos}) — "
            "stores + summary written; exiting rc=7 (upload-then-halt)",
            file=sys.stderr,
        )
        return GATE_RC
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
