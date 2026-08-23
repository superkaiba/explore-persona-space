"""Recompute the registered arm's per-pair cosine margins from persisted tensors.

Issue #2215, `discrimination-battery-expansion` fold, revision round 3. The
pod-side C' analysis wrote ``perpair/dv3_dbe_pairs.jsonl`` next to the eval
JSONs but the file was never committed, so the per-pair margin figure
(``issue2215_dbe_figures.py::fig_perpair_margins``) could not be re-rendered
locally. The inputs it derives from ARE persisted (plan v6 §C'):
``issue2215_dbe/analysis_tensors/predictions/predictions_L19.pt`` on the HF
data repo carries the per-arm prediction matrices, per-pooling targets, and
the validity mask, and ``.../vc_bank_dbe/bank_dbe.json`` carries the pair
table + judge verdicts. This driver rebuilds exactly the rows the plotter
consumes — the registered arm (``779ce``) at the registered config (layer 19,
tail pooling, cosine) — through the SAME shared helpers the pod-side analysis
used (``issue2215_analysis``: ``PairTable.from_bank`` / ``build_cell_views`` /
``sim_blocks`` / ``observed_2afc``), validates the result against the
pod-side render's sidecar points and the committed qualitative-example
margins, and writes the jsonl.

Deliberately OMITTED fields vs the pod-side file: ``correct_cos_*`` and
``margin_euc_*``. The persisted tensors are fp16-quantized, so a near-zero
margin can flip sign relative to the pod-side fp64 pipeline; the committed
``dv3_dbe_map_discrimination.json`` accuracies stay the sole authority for
correctness counts — this file exists to make the per-pair FIGURE locally
re-renderable, and the plotter reads only the margin values.

Validation (both fail loud):
- per-panel point counts match the pod-side sidecar exactly;
- every recomputed (margin_a, margin_b) point sits within ``--tol`` of a
  sidecar point of the same panel (greedy nearest match), bounding the fp16
  round-trip error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2215_analysis as ANA  # noqa: E402

ARM = "779ce"


def recompute_rows(bank: dict, pred: dict) -> list[dict]:
    """Registered-arm per-pair cosine margins from the persisted predictions."""
    pt = ANA.PairTable.from_bank(bank, None)
    views = ANA.build_cell_views(bank, pt)
    assert pt.ids == list(pred["ids"]), "row-order drift vs predictions_L19.pt"
    layer = int(pred["layer"])
    assert layer == ANA.PRIMARY_LAYER, (layer, ANA.PRIMARY_LAYER)
    valid = pred["valid"].numpy().astype(bool)
    p_np = pred["fitted"][ARM].to(torch.float64).numpy()
    t_np = pred["targets"][ANA.POOL_PRIMARY].to(torch.float64).numpy()
    assert p_np.shape == t_np.shape == (len(pt.ids), p_np.shape[1]), (p_np.shape, t_np.shape)
    judge_valid = np.array([p["judge_valid"] for p in bank["pairs"]], dtype=bool)
    included_pair = judge_valid & valid[pt.a_row] & valid[pt.b_row]
    rows: list[dict] = []
    for cell in pt.cells:
        cv = views[cell]
        loc = cv.ctx_rows
        s = ANA.sim_blocks(p_np[loc], t_np[loc])["cosine"]
        m_a, m_b = ANA.observed_2afc(s, cv.a_loc, cv.b_loc)
        vp_valid = included_pair[cv.pair_idx] & valid[loc][cv.a_loc] & valid[loc][cv.b_loc]
        for j_local, k in enumerate(cv.pair_idx):
            if not vp_valid[j_local]:
                continue
            row = {
                "pair_id": pt.pair_ids[int(k)],
                "cell": pt.pair_cell[int(k)],
                "carrier": pt.pair_carrier[int(k)],
                "value_pair": pt.pair_vp[int(k)],
                "arm": ARM,
                "layer": layer,
                "pooling": ANA.POOL_PRIMARY,
                "margin_cos_a": float(m_a[j_local]),
                "margin_cos_b": float(m_b[j_local]),
            }
            rows.append(row)
    assert rows, "empty recompute — selection predicates emptied the pair set"
    return rows


def validate_vs_sidecar(rows: list[dict], sidecar: dict, tol: float) -> float:
    """Per-panel count match + nearest-point deviation bound vs the pod render."""
    pts = sidecar["points"]
    by_group: dict[int, list[tuple[float, float]]] = {}
    for p in pts.values() if isinstance(pts, dict) else pts:
        # savefig_paper keys point coords by the panel's axis labels when set
        # (bottom row / left column only in this grid), else bare x/y.
        x = p.get("margin, direction a", p.get("x"))
        y = p.get("margin, direction b", p.get("y"))
        by_group.setdefault(int(p["_group"]), []).append((float(x), float(y)))
    by_cell: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        by_cell.setdefault(r["cell"], []).append((r["margin_cos_a"], r["margin_cos_b"]))
    assert sum(len(v) for v in by_group.values()) == len(rows), (
        sum(len(v) for v in by_group.values()),
        len(rows),
    )
    # Panels were rendered in the figure's cell order (accuracy-ascending);
    # match groups to cells by nearest-point assignment rather than order.
    worst = 0.0
    unmatched_groups = set(by_group)
    for cell, mine in by_cell.items():
        arr_mine = np.array(mine)
        best_g, best_d = None, np.inf
        for g in unmatched_groups:
            arr_g = np.array(by_group[g])
            if len(arr_g) != len(arr_mine):
                continue
            d = np.abs(arr_mine[:, None, :] - arr_g[None, :, :]).sum(axis=2)
            dev = float(d.min(axis=1).max())
            if dev < best_d:
                best_g, best_d = g, dev
        assert best_g is not None, (cell, "no size-matched sidecar panel")
        assert best_d <= tol, (cell, best_g, best_d, tol)
        unmatched_groups.discard(best_g)
        worst = max(worst, best_d)
        print(f"[validate] {cell}: n={len(mine)} matched group {best_g} max|Δ|={best_d:.2e}")
    assert not unmatched_groups, unmatched_groups
    return worst


def validate_vs_qualitative(rows: list[dict], qual: dict, tol: float) -> float:
    """Drawn-example margins from qualitative_examples.json match the recompute."""
    by_pair = {r["pair_id"]: r for r in rows}
    worst = 0.0
    n = 0
    for cell, rec in qual["per_type"].items():
        for ex in rec.get("examples", []) + ([rec["worst_miss"]] if rec.get("worst_miss") else []):
            r = by_pair.get(ex["pair_id"])
            assert r is not None, (cell, ex["pair_id"], "drawn pair missing from recompute")
            d = max(
                abs(r["margin_cos_a"] - ex["margin_cos_a"]),
                abs(r["margin_cos_b"] - ex["margin_cos_b"]),
            )
            assert d <= tol, (ex["pair_id"], d, tol)
            worst = max(worst, d)
            n += 1
    print(f"[validate] qualitative examples: {n} drawn pairs, max|Δ|={worst:.2e}")
    return worst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", type=Path, required=True, help="staged bank_dbe.json")
    ap.add_argument("--pred", type=Path, required=True, help="staged predictions_L19.pt")
    ap.add_argument("--sidecar", type=Path, default=None, help="pod-side figure .meta.json")
    ap.add_argument("--qual", type=Path, default=None, help="committed qualitative_examples.json")
    ap.add_argument("--out", type=Path, required=True, help="perpair/dv3_dbe_pairs.jsonl dest")
    ap.add_argument("--tol", type=float, default=2e-3, help="max abs fp16 round-trip deviation")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] OK", flush=True)
        return 0
    bank = json.loads(args.bank.read_text())
    pred = torch.load(args.pred, map_location="cpu", weights_only=False)
    rows = recompute_rows(bank, pred)
    print(f"[recompute] {len(rows)} per-pair rows, arm={ARM} L{pred['layer']} tail/cosine")
    if args.sidecar:
        validate_vs_sidecar(rows, json.loads(args.sidecar.read_text()), args.tol)
    if args.qual:
        validate_vs_qualitative(rows, json.loads(args.qual.read_text()), args.tol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.jsonl")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(args.out)
    print(f"[write] {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
