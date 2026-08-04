"""P3 selection-symmetric null battery for task #2061.

Implements plan §Design "Selection-symmetric null" — the MF2 fix from the
v4 union revision:

  1. ONE SHARED `draw_seed_schedule = [42 + d for d in range(1000)]`
     across ALL 64 (stage-pair, corpus, arm) delta cells. Draw index `d`
     uses the SAME seed on every cell.
  2. Per-cell, per-draw: shuffle STAGE label within-corpus (which rows
     come from stage_after vs stage_before), refit ridge, record
     `max_j_d = max_j ΔR²_j` for that (cell, draw) pair.
  3. Per-draw GLOBAL max reduction (PRIMARY headline null):
     `global_max_d = max_cell max_j_d` — the null distribution of the
     joint (feature × cell) selection statistic. Report p50/p95/p97.5/p99
     of this GLOBAL null distribution.
  4. Per-cell p97.5 retained as SECONDARY diagnostic only.
  5. Persist per-draw `max_j_d` values (1000 scalars per cell × 64
     cells = 64,000 scalars total, KB-scale) so the GLOBAL max
     reduction is recoverable post-hoc.

Reads B-2 per-feature R² JSONL for both stages of each pair. Emits:
- `eval_results/issue_2061/null/<pair>_<corpus>_<arm>_L29.jsonl` — per-cell
  with `null_max_j_per_draw` (shape (1000,) float32) + p50/p95/p97.5/p99
  quantiles + primary/secondary diagnostic fields.
- `eval_results/issue_2061/null/GLOBAL_L29.json` — p50/p95/p97.5/p99 of
  `{global_max_d}` — the PRIMARY headline null quantile.

**Verdict**: `max_{j, cell} ΔR²_j` from the true assignment vs the p97.5
quantile of the GLOBAL null distribution. Any true (j, cell) pair
exceeding the global p97.5 is a candidate improvement; the headline is
whether ANY (feature, cell) pair clears the global bar.

**Batched inner-loop**: this scaffold writes the SYNCHRONIZATION contract
correctly. The inner per-draw ridge refit is currently a placeholder —
production-scale (1000 draws × 64 cells) requires the batched masked-GEMM
recipe per `.claude/rules/vectorize-many-cell-fits.md`. A follow-up round
vectorizes the refit; the orchestration + reduction contract here is
unchanged.

Usage:
  # Compute GLOBAL null over all cells whose per-cell R² JSONL exists:
  uv run python scripts/issue2061_null.py --all-cells

  # Single cell (debug):
  uv run python scripts/issue2061_null.py --pair base_sft --corpus lmsys23k --arm context
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

STAGES = ["base", "sft", "dpo", "rlvr", "longer-rlvr"]
STAGE_PAIRS = [
    ("base", "sft"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("rlvr", "longer-rlvr"),
]
LAYER = 29
N_DRAWS = 1000
DRAW_SEED_BASE = 42  # plan §Design synchronization contract


def draw_seed_schedule(n_draws: int = N_DRAWS, base: int = DRAW_SEED_BASE) -> list[int]:
    """The ONE shared permutation seed schedule across all 64 cells.

    Load-bearing for the per-draw GLOBAL max reduction: draw index `d`
    uses the SAME seed on every cell, so `max_cell max_j_d` is a coherent
    joint (feature × cell) selection statistic.
    """
    return [base + d for d in range(n_draws)]


def load_per_feature_r2(jsonl_path: Path) -> np.ndarray:
    """Read B-2's per-feature R² JSONL into a (d_sae,) float array.

    Nulls (features with ss_tot == 0) become NaN.
    """
    r2 = []
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            r2.append(row["R2"] if row["R2"] is not None else np.nan)
    return np.asarray(r2, dtype=np.float64)


def compute_true_delta_max(
    r2_before: np.ndarray,
    r2_after: np.ndarray,
) -> tuple[float, int]:
    """True ΔR²_j = R²_after - R²_before per feature; return (max, argmax).

    NaN features are excluded from the max (both sides must be non-NaN).
    """
    delta = r2_after - r2_before
    mask = ~np.isnan(delta)
    if not mask.any():
        return float("nan"), -1
    valid = np.where(mask)[0]
    idx_local = int(np.nanargmax(delta[mask]))
    idx = int(valid[idx_local])
    return float(delta[idx]), idx


def per_cell_null_placeholder(
    r2_before: np.ndarray,
    r2_after: np.ndarray,
    seeds: list[int],
) -> np.ndarray:
    """Per-cell null draws: (n_draws,) array of max_j |ΔR²_j| under
    stage-label shuffling.

    **PLACEHOLDER**: This scaffold implements the DRAW-SEED-SYNCHRONIZATION
    contract correctly (uses the shared `seeds` list, one seed per draw
    across all cells), so the per-draw GLOBAL max reduction downstream is
    coherent. The inner per-draw statistic here is a stand-in — a
    bootstrap over the FEATURE-WISE ΔR² distribution — that captures the
    scale of the null but NOT the ridge-refit dynamics.

    Full recipe (deferred to a follow-up round; plan §Design
    "Selection-symmetric null"): for each draw d, shuffle the stage label
    within-corpus using `np.random.default_rng(seeds[d])`, refit the
    ridge on the shuffled partition, recompute R²_j per fold, and take
    max_j (R²_shuffled_after - R²_shuffled_before). Batched per
    `.claude/rules/vectorize-many-cell-fits.md`: all 1000 draws as one
    masked GEMM against the precomputed pool reduction.
    """
    delta = r2_after - r2_before
    valid_delta = delta[~np.isnan(delta)]
    if valid_delta.size == 0:
        return np.zeros(len(seeds), dtype=np.float32)

    # Bootstrap placeholder: for each draw, resample |ΔR²_j| with replacement
    # using the shared seed, take the max. This preserves the scale + shape
    # of the empirical distribution while the true refit recipe is pending.
    out = np.empty(len(seeds), dtype=np.float32)
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        # Symmetrize around zero (stage-label shuffling under H0 gives
        # sign-symmetric deltas): resample with random sign flips.
        signs = rng.choice([-1.0, 1.0], size=valid_delta.size)
        boot = signs * valid_delta
        out[i] = float(np.max(boot))
    return out


def write_cell_jsonl(
    output_path: Path,
    pair: tuple[str, str],
    corpus: str,
    arm: str,
    true_max: float,
    true_argmax: int,
    null_max_j_per_draw: np.ndarray,
) -> None:
    """Emit per-cell record with per-draw max_j_d + local quantiles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantiles = {f"p{q}": float(np.percentile(null_max_j_per_draw, q)) for q in [50, 95, 97.5, 99]}
    row = {
        "pair": f"{pair[0]}_{pair[1]}",
        "corpus": corpus,
        "arm": arm,
        "layer": LAYER,
        "true_max_delta_r2": true_max,
        "true_argmax_feature_id": true_argmax,
        "null_quantiles_per_cell": quantiles,
        "null_max_j_per_draw": null_max_j_per_draw.astype(np.float32).tolist(),
        "n_draws": len(null_max_j_per_draw),
        "draw_seed_base": DRAW_SEED_BASE,
    }
    with output_path.open("w") as f:
        f.write(json.dumps(row) + "\n")


def compute_global_null(
    per_cell_max_j: dict[tuple[str, str, str, str], np.ndarray],
) -> dict:
    """Per-draw GLOBAL max reduction (PRIMARY headline null quantile).

    For each draw d, form `global_max_d = max_cell max_j_d`. Report
    p50/p95/p97.5/p99 of this GLOBAL null distribution.
    """
    if not per_cell_max_j:
        raise ValueError("No per-cell null draws available.")
    stacked = np.stack(list(per_cell_max_j.values()), axis=0)  # (n_cells, n_draws)
    global_max_per_draw = stacked.max(axis=0)  # (n_draws,)
    return {
        "global_null_quantiles": {
            f"p{q}": float(np.percentile(global_max_per_draw, q)) for q in [50, 95, 97.5, 99]
        },
        "global_max_per_draw": global_max_per_draw.astype(np.float32).tolist(),
        "n_cells": len(per_cell_max_j),
        "n_draws": stacked.shape[1],
        "draw_seed_base": DRAW_SEED_BASE,
        "cells": [
            {"pair": p, "corpus": c, "arm": a, "layer": L} for (p, c, a, L) in per_cell_max_j.keys()
        ],
    }


def _load_r2_file(
    r2_dir: Path,
    stage: str,
    corpus: str,
    arm: str,
    render: str = "chat",
) -> np.ndarray | None:
    """Try both naming conventions from B-2."""
    candidates = [
        r2_dir / f"{stage}_{render}_{corpus}_{arm}_L{LAYER}.jsonl",
        r2_dir / f"{stage}_{corpus}_{arm}_L{LAYER}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return load_per_feature_r2(path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", type=str, default=None, help="e.g. 'base_sft'; omit for all 4 pairs"
    )
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--render", type=str, default="chat")
    parser.add_argument("--arm", choices=["prefix", "context"], default=None)
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument(
        "--r2-dir", type=Path, default=Path("eval_results/issue_2061/per_feature_r2")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/issue_2061/null"))
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = draw_seed_schedule(args.n_draws)
    print(f"[setup] Shared draw_seed_schedule: {len(seeds)} seeds, base={DRAW_SEED_BASE}")

    # Enumerate target (pair, corpus, arm) cells.
    pairs = [tuple(args.pair.split("_", 1))] if args.pair else STAGE_PAIRS
    arms = [args.arm] if args.arm else ["prefix", "context"]

    # Auto-detect corpora from the r2_dir.
    if args.corpus:
        corpora = [args.corpus]
    elif args.all_cells:
        seen = set()
        for path in args.r2_dir.glob(f"*_L{LAYER}.jsonl"):
            # <stage>_<render>_<corpus>_<arm> or <stage>_<corpus>_<arm>
            parts = path.stem.rsplit("_", 3)
            if len(parts) >= 3:
                seen.add(parts[-3])  # corpus is 3rd-from-last
        corpora = sorted(seen)
    else:
        corpora = []

    if not corpora:
        print("[error] No corpora found (use --corpus or --all-cells)")
        return 1

    per_cell_max_j: dict[tuple[str, str, str, str], np.ndarray] = {}
    for pair in pairs:
        for corpus in corpora:
            for arm in arms:
                cell_key = (f"{pair[0]}_{pair[1]}", corpus, arm, f"L{LAYER}")
                output_path = args.output_dir / f"{cell_key[0]}_{corpus}_{arm}_L{LAYER}.jsonl"
                if output_path.exists():
                    print(f"[skip] Exists: {output_path}")
                    # Reload the per-draw values for global aggregation.
                    with output_path.open() as f:
                        row = json.loads(f.readline())
                    per_cell_max_j[cell_key] = np.asarray(
                        row["null_max_j_per_draw"], dtype=np.float32
                    )
                    continue

                r2_before = _load_r2_file(args.r2_dir, pair[0], corpus, arm, args.render)
                r2_after = _load_r2_file(args.r2_dir, pair[1], corpus, arm, args.render)
                if r2_before is None or r2_after is None:
                    print(f"[skip] Missing R² for {pair}/{corpus}/{arm}")
                    continue

                t0 = time.time()
                true_max, true_argmax = compute_true_delta_max(r2_before, r2_after)
                null_max_j = per_cell_null_placeholder(r2_before, r2_after, seeds)
                elapsed = time.time() - t0

                write_cell_jsonl(
                    output_path,
                    pair=pair,
                    corpus=corpus,
                    arm=arm,
                    true_max=true_max,
                    true_argmax=true_argmax,
                    null_max_j_per_draw=null_max_j,
                )
                per_cell_max_j[cell_key] = null_max_j
                print(
                    f"[cell] {cell_key[0]}/{corpus}/{arm} true_max={true_max:.4f} "
                    f"argmax={true_argmax} local_p97.5={np.percentile(null_max_j, 97.5):.4f} "
                    f"({elapsed:.1f}s)"
                )

    if not per_cell_max_j:
        print("[error] No per-cell nulls computed — cannot form GLOBAL null")
        return 1

    global_null = compute_global_null(per_cell_max_j)
    global_path = args.output_dir / f"GLOBAL_L{LAYER}.json"
    with global_path.open("w") as f:
        json.dump(global_null, f, indent=2)

    print(f"\n[global] GLOBAL null (p97.5): {global_null['global_null_quantiles']['p97.5']:.4f}")
    print(f"[global] Wrote {global_path}")
    print(f"[global] n_cells={global_null['n_cells']} n_draws={global_null['n_draws']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
