# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ≥, ×) in scientific docstrings + log messages.
"""Issue #715 — aggregate LoRA checkpoints into the P1 Pareto frontier + D* select.

Joins per-(condition, seed, checkpoint) narrow-task acquisition (x) and OOD
EM-rate (y) into ``pareto_em_vs_narrow.json`` + the hero figure
``figures/issue_715/pareto_em_vs_narrow.png``. Computes:

- The SFT-LoRA and DFT-LoRA Pareto frontiers (per checkpoint × seed).
- Paired-by-seed bootstrap CIs on the (SFT−DFT) EM difference at matched
  narrow-acquisition x-coordinates (linearly interpolated per seed to a common
  x-grid over the overlapping x-range).
- The D* selection (the narrow-task acquisition value with the largest SFT-vs-DFT
  EM gap, in the 60-90% band — the P4 operating point).

Metadata recorded (Stats(1) + Codex Statistics critic): ``interpolation_method``,
``overlap_x_range``, ``bootstrap_resampling`` (n_boot, paired-by-seed scheme).

Carries per-cell raw judge labels by REFERENCE (the em_rate/*.json + narrow_task/
*.json paths) so the analyzer can recompute either the conservative per-seed rule
OR the paired-difference CI (plan §6.5 P1 note).

Usage (off-pod on the VM, or pod-side):
    uv run python scripts/issue715_aggregate_pareto.py \
        --eval-dir eval_results/issue_715 [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("issue715_pareto")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LORA_CONDITIONS = ("sft_lora", "dft_lora")
N_BOOT = 2000
BOOT_SEED = 42


def _load_cells(eval_dir: Path) -> dict:
    """Join em_rate + narrow_task JSONs into per-(condition, seed) point lists.

    Returns {condition: {seed: [{step, x (narrow_rate), x_cont (mean_bad_medical),
    y (em_rate), coherence-ish, em_raw, narrow_raw}, ...]}}.
    """
    em_files = {
        f.stem: json.loads(f.read_text())
        for f in (eval_dir / "em_rate").glob("*.json")
        if not f.name.startswith("raw_")
    }
    narrow_files = {
        f.stem: json.loads(f.read_text())
        for f in (eval_dir / "narrow_task").glob("*.json")
        if not f.name.startswith("raw_")
    }

    cells: dict = {c: {} for c in LORA_CONDITIONS}
    for tag, em in em_files.items():
        cond = em.get("condition")
        if cond not in LORA_CONDITIONS:
            continue
        narrow = narrow_files.get(tag)
        if narrow is None:
            logger.warning("EM cell %s has no matching narrow_task file; skipping", tag)
            continue
        seed = em["seed"]
        cells[cond].setdefault(seed, []).append(
            {
                "step": em["checkpoint_step"],
                "x": narrow["narrow_rate"],
                "x_cont": narrow.get("mean_bad_medical"),
                "y": em["em_rate"],
                "n_em": em.get("n_total"),
                "em_raw": str(eval_dir / "em_rate" / f"raw_{tag}.json"),
                "narrow_raw": str(eval_dir / "narrow_task" / f"raw_{tag}.json"),
            }
        )
    for cond in cells:
        for seed in cells[cond]:
            cells[cond][seed].sort(key=lambda p: p["x"])
    return cells


def _interp_y_at(points: list[dict], xq: float) -> float | None:
    """Linear interpolation of y (EM) at narrow-acquisition x for one seed's curve.

    Returns None if xq is outside this seed's measured x-range (no extrapolation).
    """
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    if not xs or xq < min(xs) or xq > max(xs):
        return None
    for i in range(1, len(xs)):
        if xs[i - 1] <= xq <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (xq - x0) / (x1 - x0)
    return ys[-1]


def _overlap_x_range(cells: dict) -> tuple[float, float] | None:
    """The x-range covered by BOTH conditions across ALL their seeds."""
    lo, hi = 0.0, 1.0
    for cond in LORA_CONDITIONS:
        if not cells[cond]:
            return None
        cond_lo = max(min(p["x"] for p in pts) for pts in cells[cond].values())
        cond_hi = min(max(p["x"] for p in pts) for pts in cells[cond].values())
        lo = max(lo, cond_lo)
        hi = min(hi, cond_hi)
    return (lo, hi) if lo < hi else None


def _paired_bootstrap_diff(cells: dict, x_grid: list[float], n_boot: int) -> dict:
    """Paired-by-seed bootstrap CI on (SFT−DFT) EM at each matched-x grid point.

    For each x in x_grid, interpolate each seed's SFT and DFT EM, form the
    per-seed paired difference d_s = y_sft(x) − y_dft(x), and bootstrap-resample
    SEEDS (paired) to get a 95% CI on the mean difference. Records whether the CI
    excludes 0 (the P1 supported-iff condition at that operating point).
    """
    import random

    rng = random.Random(BOOT_SEED)
    # Common seed set present in both conditions.
    seeds = sorted(set(cells["sft_lora"]) & set(cells["dft_lora"]))
    out = []
    for xq in x_grid:
        per_seed_diff = []
        for s in seeds:
            y_sft = _interp_y_at(cells["sft_lora"][s], xq)
            y_dft = _interp_y_at(cells["dft_lora"][s], xq)
            if y_sft is not None and y_dft is not None:
                per_seed_diff.append(y_sft - y_dft)
        if len(per_seed_diff) < 2:
            out.append({"x": xq, "n_seeds": len(per_seed_diff), "mean_diff": None, "ci": None})
            continue
        mean_diff = sum(per_seed_diff) / len(per_seed_diff)
        boot_means = []
        for _ in range(n_boot):
            sample = [per_seed_diff[rng.randrange(len(per_seed_diff))] for _ in per_seed_diff]
            boot_means.append(sum(sample) / len(sample))
        boot_means.sort()
        lo = boot_means[int(0.025 * n_boot)]
        hi = boot_means[int(0.975 * n_boot)]
        out.append(
            {
                "x": xq,
                "n_seeds": len(per_seed_diff),
                "mean_diff_sft_minus_dft": mean_diff,
                "ci95": [lo, hi],
                "ci_excludes_zero": (lo > 0) or (hi < 0),
                "dft_below_sft": mean_diff > 0,  # SFT−DFT > 0 => DFT lower EM
            }
        )
    return {"per_x": out, "n_seeds_paired": len(seeds)}


def _select_dstar(cells: dict, x_grid: list[float], boot: dict) -> dict:
    """D* = the matched-x with the largest (SFT−DFT) EM gap in the 60-90% band.

    "60-90% band" interpreted over the narrow-acquisition x (the dose where EM
    appears, not floor/saturated). If no in-band point has a positive gap, falls
    back to the highest joint-sample-size point in band.
    """
    band = [r for r in boot["per_x"] if r.get("mean_diff_sft_minus_dft") is not None]
    in_band = [r for r in band if 0.60 <= r["x"] <= 0.90]
    candidates = in_band or band
    if not candidates:
        return {"dstar_x": None, "reason": "no matched-x points available"}
    # Largest gap; tie-break to more paired seeds.
    best = max(candidates, key=lambda r: (r["mean_diff_sft_minus_dft"], r["n_seeds"]))
    return {
        "dstar_x": best["x"],
        "dstar_gap_sft_minus_dft": best["mean_diff_sft_minus_dft"],
        "dstar_ci95": best.get("ci95"),
        "dstar_in_60_90_band": bool(in_band),
        "reason": "largest SFT-DFT EM gap in the 0.60-0.90 narrow-acquisition band",
    }


def _plot(cells: dict, dstar: dict, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"sft_lora": "#1f77b4", "dft_lora": "#d62728"}
    labels = {"sft_lora": "Standard SFT-LoRA", "dft_lora": "DFT-LoRA"}
    for cond in LORA_CONDITIONS:
        for seed, pts in sorted(cells[cond].items()):
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            ax.plot(
                xs,
                ys,
                marker="o",
                color=colors[cond],
                alpha=0.6,
                label=labels[cond] if seed == min(cells[cond]) else None,
            )
    if dstar.get("dstar_x") is not None:
        ax.axvline(
            dstar["dstar_x"], color="gray", ls="--", alpha=0.7, label=f"D* = {dstar['dstar_x']:.3f}"
        )
    ax.set_xlabel("Narrow-task acquisition (held-out bad-medical rate)")
    ax.set_ylabel("OOD EM-rate (Betley main-8)")
    ax.set_title("Issue #715 — EM vs narrow-task acquisition (LoRA Pareto)")
    ax.legend()
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("[phase=pareto] wrote figure %s", fig_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #715 Pareto aggregation + D* selection")
    parser.add_argument("--eval-dir", default=str(PROJECT_ROOT / "eval_results" / "issue_715"))
    parser.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures" / "issue_715"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from issue715_common import reproducibility_metadata

    eval_dir = Path(args.eval_dir)
    cells = _load_cells(eval_dir)
    n_cells = sum(len(pts) for cond in cells for pts in cells[cond].values())
    logger.info("Loaded %d Pareto cells across %s", n_cells, LORA_CONDITIONS)

    overlap = _overlap_x_range(cells)
    n_boot = 50 if args.smoke else args.n_boot
    if overlap is None:
        x_grid: list[float] = []
        boot = {"per_x": [], "n_seeds_paired": 0}
        dstar = {"dstar_x": None, "reason": "no overlapping x-range between conditions"}
    else:
        lo, hi = overlap
        x_grid = [lo + (hi - lo) * i / (args.n_grid - 1) for i in range(args.n_grid)]
        boot = _paired_bootstrap_diff(cells, x_grid, n_boot)
        dstar = _select_dstar(cells, x_grid, boot)

    result = {
        "cells": cells,
        "overlap_x_range": list(overlap) if overlap else None,
        "matched_x_grid": x_grid,
        "paired_bootstrap": boot,
        "dstar_selection": dstar,
        "stats_metadata": {
            "interpolation_method": "per-seed linear interpolation of EM-rate to a "
            "common narrow-acquisition x-grid over the overlapping x-range; "
            "no extrapolation outside each seed's measured range",
            "bootstrap_resampling": f"paired-by-seed: per-x per-seed difference "
            f"d_s = EM_sft(x) - EM_dft(x), resample SEEDS with replacement, "
            f"n_boot={n_boot}, 95% percentile CI, boot_seed={BOOT_SEED}",
            "overlap_x_range_note": "x-range covered by BOTH conditions across all their seeds",
            "p1_supported_iff": "at >=1 matched-x operating point the SFT-DFT EM gap "
            ">0 with a 95% CI excluding 0 in >=2 of 3 seeds AND DFT EM >=30% below SFT",
        },
        "metadata": reproducibility_metadata({"script": "issue715_aggregate_pareto"}),
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "pareto_em_vs_narrow.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("[phase=pareto] wrote %s (D*=%s)", out_path, dstar.get("dstar_x"))

    if n_cells > 0:
        _plot(cells, dstar, Path(args.fig_dir) / "pareto_em_vs_narrow.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
