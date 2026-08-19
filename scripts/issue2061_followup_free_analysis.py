"""Free-analysis follow-up for task #2061 (Step 9a-ter, 0 GPU-h).

Two descriptive reads over EXISTING artifacts (no new training / eval / data):

P1 — Activation counts + engine-symmetric classification of per-cell winners.
  From `eval_results/issue_2061/null/*.jsonl` (56 cells), take every cell that
  exceeds its OWN per-cell p97.5 null quantile under EITHER read — the
  P2-derived true statistic (`true_max_delta_r2`, 18 cells) or the
  engine-symmetric identity statistic (`engine_identity_max_delta`, 16 cells;
  same refit engine as the null draws) — and join, for each read's argmax
  feature, the feature's nonzero-activation ROW COUNT in the cell's two
  stages from the P1 TopK-sparse encoded payloads
  (`data/issue_2061/hf_dl/issue2061_sae_predictability/sae_encoded/`).
  Question: are the local exceedances rare-feature noise (like the global
  winner, active in ~1 row) or populated features?

P2 — Cross-corpus rank agreement (plan §7 secondary read (ii), DESCRIPTIVE
  COLOUR only). For each of the 4 adjacent transitions, context arm only:
  rank features by ΔR²_j per (render, corpus) cell from the per-feature R²
  JSONLs, take the top-100 improved features per cell, and report pairwise
  cross-corpus overlap (|intersection|/100) + Spearman rank agreement over
  the union. Prefix arms SKIPPED: the 24 chat-render prefix cells are
  pre-registered render-degenerate (plan §Design "Pre-registered prefix-arm
  degeneracy") and 7/28 prefix cells are regularization-limited per the run
  record — a prefix rank list would rank fit noise.

Loading is vectorized: each ~250 MB per-feature JSONL is scanned ONCE with a
single C-level regex pass (no per-row json.loads loop); payload counts use
numpy array ops over the fixed-width (n, k=32) sparse codes.

Usage:
  uv run python scripts/issue2061_followup_free_analysis.py \
      [--null-dir eval_results/issue_2061/null] \
      [--encoded-dir data/issue_2061/hf_dl/issue2061_sae_predictability/sae_encoded] \
      [--r2-dir data/issue_2061/hf_dl/issue2061_sae_predictability/analysis_tensors/per_feature_r2] \
      [--output-dir eval_results/issue_2061/followup_free_analysis] \
      [--figure-dir figures/issue_2061] [--skip-p1] [--skip-p2]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy import

import numpy as np  # noqa: E402

# Sibling-script import (bare module name via the script-dir sys.path insert —
# same pattern as issue2061_null.py / issue2061_figures.py).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue2061_turnstore as ts  # noqa: E402

LAYER = 29
TRANSITIONS = (("base", "sft"), ("sft", "dpo"), ("dpo", "rlvr"), ("rlvr", "longer-rlvr"))
CELLS: tuple[tuple[str, str], ...] = (
    ("chat", "gsm8k_train_full"),
    ("chat", "if11k"),
    ("chat", "lmsys23k"),
    ("chat", "math7500"),
    ("chat", "sft11k"),
    ("chat", "uf11k"),
    ("naturalistic", "lmsys23k"),
)
TOP_N = 100
RARE_BELOW = 10  # rare: max per-stage nonzero-row count < 10
POPULATED_AT_LEAST = 100  # populated: min per-stage nonzero-row count >= 100

# One row per line, keys in writer order (scripts/issue2061_fit_per_feature.py:684:
# json.dumps default separators, "feature_id" then "R2"; NaN serialized as null).
_ROW_RE = re.compile(rb'\{"feature_id": (\d+), "R2": (null|-?[0-9][0-9.eE+-]*), ')


def _git_commit_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_SCRIPT_DIR,
    )
    return out.stdout.strip() or "unavailable"


def _meta(inputs: list[str]) -> dict:
    return {
        "generator": "scripts/issue2061_followup_free_analysis.py",
        "git_commit": _git_commit_sha(),
        "numpy_version": np.__version__,
        "created_unix": time.time(),
        "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "layer": LAYER,
        "inputs": inputs,
    }


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------


def load_null_cells(null_dir: Path) -> list[dict]:
    """All 56 per-cell null rows (one JSON row per *_L29.jsonl file)."""
    rows = []
    for path in sorted(null_dir.glob(f"*_L{LAYER}.jsonl")):
        with path.open() as f:
            rows.append(json.loads(f.readline()))
    if len(rows) != 56:
        raise ValueError(f"{null_dir}: expected 56 per-cell null files, found {len(rows)}")
    return rows


def load_r2_vector(path: Path) -> np.ndarray:
    """(d_sae,) float64 R² from one per-feature JSONL — ONE C-level regex pass.

    Fail-loud parity with `issue2061_figures._load_per_feature_r2`: every line
    must match, and feature_id must equal the row index (never mis-attribute
    ΔR² to the wrong feature) — both asserted vectorized here.
    """
    data = path.read_bytes()
    matches = _ROW_RE.findall(data)
    n_lines = data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1)
    if len(matches) != n_lines:
        raise ValueError(
            f"{path.name}: regex matched {len(matches)} rows but file has {n_lines} lines — "
            "writer format drift; refusing a partial parse."
        )
    ids = np.array([m[0] for m in matches]).astype(np.int64)
    if not (ids == np.arange(len(ids))).all():
        raise ValueError(f"{path.name}: rows not in feature-id order; refusing to mis-index.")
    vals = np.array([m[1] for m in matches])
    vals[vals == b"null"] = b"nan"
    return vals.astype(np.float64)


# ---------------------------------------------------------------------------
# P1 — winner activation counts
# ---------------------------------------------------------------------------


def _payload_feature_counts(
    encoded_dir: Path, stage: str, render: str, corpus: str, feature_ids: list[int]
) -> tuple[dict[int, int], int]:
    """{feature_id: nonzero-activation row count} + n_rows for one payload.

    A row counts as active when the feature appears in the fixed-width TopK
    code with val != 0 (padding is (idx=0, val=0.0) — the val guard keeps
    feature 0 correct; `issue2061_turnstore.to_fixed_width_sparse`).
    """
    name = ts.encoded_target_name(stage, render, corpus, "answer", LAYER)
    payload = ts.load_encoded_target(encoded_dir / name)
    idx = payload["idx"].numpy()
    val = payload["val"].numpy()
    counts = {int(f): int((((idx == int(f)) & (val != 0)).any(axis=1)).sum()) for f in feature_ids}
    return counts, int(payload["n_rows"])


def _count_class(count_before: int, count_after: int) -> str:
    if max(count_before, count_after) < RARE_BELOW:
        return "rare"
    if min(count_before, count_after) >= POPULATED_AT_LEAST:
        return "populated"
    return "intermediate"


def run_p1(null_dir: Path, encoded_dir: Path, output_dir: Path) -> dict:
    rows = load_null_cells(null_dir)

    def exceeds_true(r: dict) -> bool:
        return bool(r["true_max_delta_r2"] > r["null_quantiles_per_cell"]["p97.5"])

    def exceeds_engine(r: dict) -> bool:
        return bool(r["engine_identity_max_delta"] > r["null_quantiles_per_cell"]["p97.5"])

    selected = [r for r in rows if exceeds_true(r) or exceeds_engine(r)]
    # Reference row: the GLOBAL true winner (max true ΔR² across all 56 cells)
    # — the analyzer's prior read found it active in ~1 row; carried here so
    # the local winners are classified against the same yardstick.
    global_winner = max(
        rows,
        key=lambda r: r["true_max_delta_r2"] if np.isfinite(r["true_max_delta_r2"]) else -np.inf,
    )
    if not any(r is global_winner for r in selected):
        selected = selected + [global_winner]

    out_cells = []
    for r in selected:
        stage_before, stage_after = r["pair"].split("_", 1)
        fids = sorted({int(r["true_argmax_feature_id"]), int(r["engine_identity_argmax"])})
        counts_before, n_rows_before = _payload_feature_counts(
            encoded_dir, stage_before, r["render"], r["corpus"], fids
        )
        counts_after, n_rows_after = _payload_feature_counts(
            encoded_dir, stage_after, r["render"], r["corpus"], fids
        )
        if n_rows_before != int(r["n_rows_before"]) or n_rows_after != int(r["n_rows_after"]):
            raise ValueError(
                f"{r['pair']}/{r['render']}/{r['corpus']}: payload rows "
                f"({n_rows_before}, {n_rows_after}) != null-cell meta "
                f"({r['n_rows_before']}, {r['n_rows_after']}) — stale payload?"
            )

        def winner(fid: int, delta: float) -> dict:
            cb, ca = counts_before[fid], counts_after[fid]
            return {
                "feature_id": fid,
                "max_delta_r2": float(delta),
                "nonzero_rows_before": cb,
                "nonzero_rows_after": ca,
                "n_rows_before": n_rows_before,
                "n_rows_after": n_rows_after,
                "class": _count_class(cb, ca),
            }

        out_cells.append(
            {
                "cell": f"{r['pair']}_{r['render']}_{r['corpus']}_{r['arm']}",
                "pair": r["pair"],
                "render": r["render"],
                "corpus": r["corpus"],
                "arm": r["arm"],
                "stage_before": stage_before,
                "stage_after": stage_after,
                "per_cell_p97_5": r["null_quantiles_per_cell"]["p97.5"],
                "exceeds_own_bar_p2_derived": exceeds_true(r),
                "exceeds_own_bar_engine_symmetric": exceeds_engine(r),
                "is_global_true_winner": bool(r is global_winner),
                "winner_p2_derived": winner(
                    int(r["true_argmax_feature_id"]), r["true_max_delta_r2"]
                ),
                "winner_engine_symmetric": winner(
                    int(r["engine_identity_argmax"]), r["engine_identity_max_delta"]
                ),
            }
        )

    def class_hist(cells: list[dict], key: str, flag: str) -> dict[str, int]:
        hist: dict[str, int] = {"rare": 0, "intermediate": 0, "populated": 0}
        for c in cells:
            if c[flag]:
                hist[c[key]["class"]] += 1
        return hist

    result = {
        "meta": {
            **_meta([str(null_dir), str(encoded_dir)]),
            "selection": "cells exceeding OWN per-cell p97.5 under either read, "
            "plus the global true winner as a reference row",
            "count_convention": "rows where the feature appears in the fixed-width TopK "
            "(k=32) sparse code with val != 0, over ALL payload rows of the stage",
            "class_rule": f"rare: max(before, after) < {RARE_BELOW}; populated: "
            f"min(before, after) >= {POPULATED_AT_LEAST}; else intermediate",
            "n_cells_total": len(rows),
            "n_exceed_p2_derived": sum(1 for r in rows if exceeds_true(r)),
            "n_exceed_engine_symmetric": sum(1 for r in rows if exceeds_engine(r)),
            "n_union": sum(1 for r in rows if exceeds_true(r) or exceeds_engine(r)),
        },
        "class_histogram_p2_derived_winners": class_hist(
            out_cells, "winner_p2_derived", "exceeds_own_bar_p2_derived"
        ),
        "class_histogram_engine_symmetric_winners": class_hist(
            out_cells, "winner_engine_symmetric", "exceeds_own_bar_engine_symmetric"
        ),
        "cells": out_cells,
    }
    out_path = output_dir / "p1_winner_activation_counts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[p1] wrote {out_path} ({len(out_cells)} cells)")
    return result


# ---------------------------------------------------------------------------
# P2 — cross-corpus rank agreement (context arm only)
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via rank + Pearson (average ranks on ties)."""
    from scipy.stats import rankdata

    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def run_p2(r2_dir: Path, output_dir: Path, figure_dir: Path | None) -> dict:
    stages = sorted({s for pair in TRANSITIONS for s in pair})
    r2: dict[tuple[str, str, str], np.ndarray] = {}
    for stage in stages:
        for render, corpus in CELLS:
            name = f"{stage}_{render}_{corpus}_context_L{LAYER}.jsonl"
            t0 = time.time()
            r2[(stage, render, corpus)] = load_r2_vector(r2_dir / name)
            print(f"[p2] loaded {name} ({time.time() - t0:.1f}s)", flush=True)

    transitions_out = []
    for stage_before, stage_after in TRANSITIONS:
        pair_str = f"{stage_before}_{stage_after}"
        top: dict[tuple[str, str], np.ndarray] = {}
        delta: dict[tuple[str, str], np.ndarray] = {}
        cells_out = []
        for render, corpus in CELLS:
            d = r2[(stage_after, render, corpus)] - r2[(stage_before, render, corpus)]
            delta[(render, corpus)] = d
            finite = np.isfinite(d)
            improved = finite & (d > 0)
            n_take = min(TOP_N, int(improved.sum()))
            work = np.where(improved, d, -np.inf)
            top_ids = np.argpartition(-work, n_take - 1)[:n_take]
            top_ids = top_ids[np.argsort(-work[top_ids])]
            top[(render, corpus)] = top_ids
            cells_out.append(
                {
                    "render": render,
                    "corpus": corpus,
                    "n_finite": int(finite.sum()),
                    "n_improved": int(improved.sum()),
                    "n_top": n_take,
                    "top_feature_ids": [int(v) for v in top_ids],
                    "top_delta_r2": [float(d[v]) for v in top_ids],
                }
            )

        pairs_out = []
        for (ra, ca), (rb, cb) in combinations(CELLS, 2):
            ta, tb = top[(ra, ca)], top[(rb, cb)]
            inter = np.intersect1d(ta, tb)
            union = np.union1d(ta, tb)
            da, db = delta[(ra, ca)][union], delta[(rb, cb)][union]
            used = np.isfinite(da) & np.isfinite(db)
            rho = _spearman(da[used], db[used]) if int(used.sum()) >= 3 else float("nan")
            pairs_out.append(
                {
                    "cell_a": f"{ra}_{ca}",
                    "cell_b": f"{rb}_{cb}",
                    "n_intersection": int(inter.size),
                    "overlap_frac": float(inter.size / min(ta.size, tb.size)),
                    "n_union": int(union.size),
                    "n_used_for_spearman": int(used.sum()),
                    "spearman_rho_union": rho,
                }
            )

        chat_pairs = [
            p
            for p in pairs_out
            if p["cell_a"].startswith("chat") and p["cell_b"].startswith("chat")
        ]
        transitions_out.append(
            {
                "pair": pair_str,
                "cells": cells_out,
                "pairs": pairs_out,
                "mean_overlap_all_pairs": float(np.mean([p["overlap_frac"] for p in pairs_out])),
                "mean_overlap_chat_only": float(np.mean([p["overlap_frac"] for p in chat_pairs])),
                "mean_spearman_all_pairs": float(
                    np.nanmean([p["spearman_rho_union"] for p in pairs_out])
                ),
                "mean_spearman_chat_only": float(
                    np.nanmean([p["spearman_rho_union"] for p in chat_pairs])
                ),
            }
        )

    d_sae = len(next(iter(r2.values())))
    result = {
        "meta": {
            **_meta([str(r2_dir)]),
            "read": "plan §7 secondary read (ii) — DESCRIPTIVE COLOUR only (v13): weak "
            "discriminator between stage-general feature strengthening and stage-general "
            "format/templating shift; never adjudicates a winner's mechanism",
            "arm": "context only",
            "prefix_skip_reason": "24/28 chat-render prefix delta cells pre-registered "
            "render-degenerate (plan §Design 'Pre-registered prefix-arm degeneracy'); "
            "7/28 prefix cells regularization-limited per the run record — a prefix rank "
            "list would rank fit noise",
            "top_n": TOP_N,
            "d_sae": d_sae,
            "chance_overlap_frac": TOP_N / d_sae,
            "improved_def": "finite ΔR²_j = R²_after − R²_before > 0 (both stages non-null)",
            "spearman_def": "rho over the UNION of the two cells' top sets, on ΔR² values, "
            "pairs with a non-finite ΔR² in either cell dropped",
        },
        "transitions": transitions_out,
    }
    out_path = output_dir / "p2_cross_corpus_rank_agreement.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[p2] wrote {out_path}")

    if figure_dir is not None:
        _p2_figure(result, figure_dir, inputs=[str(r2_dir)])
    return result


def _p2_figure(result: dict, figure_dir: Path, inputs: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r[:4]}/{c.replace('_train_full', '')}" for r, c in CELLS]
    n = len(CELLS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    vmax = max(p["overlap_frac"] for t in result["transitions"] for p in t["pairs"])
    vmax = max(vmax, 0.05)
    im = None
    for ax, t in zip(axes.flat, result["transitions"], strict=True):
        mat = np.full((n, n), np.nan)
        index = {f"{r}_{c}": i for i, (r, c) in enumerate(CELLS)}
        for p in t["pairs"]:
            i, j = index[p["cell_a"]], index[p["cell_b"]]
            mat[i, j] = mat[j, i] = p["overlap_frac"]
        im = ax.imshow(mat, vmin=0.0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), labels, fontsize=8)
        ax.set_title(t["pair"], fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="top-100 overlap fraction")
    fig.suptitle(
        f"#2061 P2 cross-corpus top-{TOP_N} ΔR² overlap per transition "
        f"(context arm, L{LAYER}; chance ≈ {TOP_N}/262144)",
        fontsize=11,
    )
    path = figure_dir / "i2061_p2_cross_corpus_overlap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    meta = {
        "caption": "Pairwise cross-corpus overlap of the top-100 improved SAE features "
        "per adjacent transition (context arm). Diagonal blank.",
        "commit_sha": _git_commit_sha(),
        "inputs": inputs,
        "generator": "scripts/issue2061_followup_free_analysis.py",
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)
    print(f"[p2] wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"), add_help=True)
    ap.add_argument("--null-dir", type=Path, default=Path("eval_results/issue_2061/null"))
    ap.add_argument(
        "--encoded-dir",
        type=Path,
        default=Path("data/issue_2061/hf_dl/issue2061_sae_predictability/sae_encoded"),
    )
    ap.add_argument(
        "--r2-dir",
        type=Path,
        default=Path(
            "data/issue_2061/hf_dl/issue2061_sae_predictability/analysis_tensors/per_feature_r2"
        ),
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("eval_results/issue_2061/followup_free_analysis")
    )
    ap.add_argument("--figure-dir", type=Path, default=Path("figures/issue_2061"))
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--skip-p1", action="store_true")
    ap.add_argument("--skip-p2", action="store_true")
    args = ap.parse_args()

    if not args.skip_p1:
        run_p1(args.null_dir, args.encoded_dir, args.output_dir)
    if not args.skip_p2:
        run_p2(args.r2_dir, args.output_dir, None if args.no_figure else args.figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
