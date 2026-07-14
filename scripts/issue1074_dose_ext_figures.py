"""Figures for issue #1074 follow-up round `install-dose-extension` (dose_ext_ prefix).

Reads the round's committed Phase-D outputs
(eval_results/issue_1074/install-dose-extension/, committed on branch
issue-1074-fu2 @ c750ad9738; pass --results-dir to point at the worktree copy
before the branch merges to main) and writes one 2-panel figure to
figures/issue_1074/:

- dose_ext_install_curve: judged compliance rate vs training step for the
  9-epoch retrain (270 steps) overlaid on the prior 3-epoch round's curve
  (embedded in the round's install_summary.json overlay block), with the
  0.60-0.85 target band, per-checkpoint Wilson 95% intervals, and the
  full-bank confirm point at the selected checkpoint (left); and the
  per-question rate distribution at that checkpoint for all four final-eval
  cells (right; the low-level per-unit data behind the aggregate rates).

Usage:
    uv run python scripts/issue1074_dose_ext_figures.py [--results-dir PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

CELL_LABELS = {
    "harmful_compliance-mixed__persona_software_engineer": "trained (9 epochs), source persona",
    "harmful_compliance-mixed__neg_default_assistant": "trained (9 epochs), default context",
    "base__persona_software_engineer": "base, source persona",
    "base__neg_default_assistant": "base, default context",
}
# Prior-round scored-generation denominators per dose checkpoint (30-question
# subset x 5 draws minus judge drops); exact integer reconstruction from the
# committed rates (e.g. 0.124137931... = 18/145) — same values documented in
# scripts/issue1074_bnr_figures.py DOSE_N_SCORED.
PRIOR_DOSE_N_SCORED = {25: 145, 50: 149, 75: 146, 90: 146}
BAND = (0.60, 0.85)


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score 95% interval ``(lo, hi)`` for a proportion.

    Asserts ``n > 0`` and ``0 <= p <= 1``; unlike the Wald interval this stays
    inside [0, 1] and behaves sensibly near the 0/1 boundary (the base cells
    sit at ~0.005).
    """
    assert n > 0, n
    assert 0.0 <= p <= 1.0, p
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _curve_with_wilson(
    rates_by_step: dict[str, float], n_by_step: dict[int, int]
) -> tuple[list[int], list[float], np.ndarray]:
    steps = sorted(int(s) for s in rates_by_step)
    curve = [rates_by_step[str(s)] for s in steps]
    lo_hi = [wilson_ci(r, n_by_step[s]) for s, r in zip(steps, curve, strict=False)]
    err = np.array(
        [
            [r - lo for r, (lo, _) in zip(curve, lo_hi, strict=False)],
            [hi - r for r, (_, hi) in zip(curve, lo_hi, strict=False)],
        ]
    )
    return steps, curve, err


def fig_dose_extension(results_dir: Path, out_dir: Path) -> None:
    """Two panels: the two rounds' dose curves overlaid, and the per-question histogram."""
    inst = json.loads((results_dir / "install" / "install_summary.json").read_text())
    rates = json.loads((results_dir / "judge" / "harmful_compliance" / "rates.json").read_text())
    colors = paper_palette(4)

    this_n = {int(s): v["n_scored"] for s, v in inst["drop_censoring"]["per_checkpoint"].items()}
    sel_step = inst["band_entry"]["step"]
    final_cells = inst["drop_censoring"]["final_cells"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    ax.axhspan(BAND[0], BAND[1], color=colors[2], alpha=0.18)
    ax.text(
        28,
        BAND[0] + 0.012,
        "target judged-rate band (0.60 to 0.85)",
        fontsize=10,
        color="#2a6f4e",
        va="bottom",
    )

    steps, curve, err = _curve_with_wilson(inst["dose_curve_rates_by_step"], this_n)
    ax.errorbar(
        steps,
        curve,
        yerr=err,
        marker="o",
        color=colors[0],
        linewidth=2,
        capsize=3,
        label="9-epoch retrain, this round (30-question subset)",
    )
    for s, r in zip(steps, curve, strict=False):
        ax.text(s, r + 0.035, f"{r:.2f}", ha="center", fontsize=8.6, color=colors[0])

    prior = inst["overlay"]["prior_round_rates_by_step"]
    p_steps, p_curve, p_err = _curve_with_wilson(prior, PRIOR_DOSE_N_SCORED)
    ax.errorbar(
        p_steps,
        p_curve,
        yerr=p_err,
        marker="^",
        color=colors[3],
        linewidth=1.6,
        linestyle="--",
        capsize=3,
        label="3-epoch run, prior round (same subset)",
    )
    for s, r in zip(p_steps, p_curve, strict=False):
        ax.text(s, r - 0.055, f"{r:.2f}", ha="center", fontsize=8.6, color=colors[3])

    fr = inst["final_rate_source"]
    fr_n = final_cells["harmful_compliance-mixed__persona_software_engineer"]["n_scored"]
    fr_lo, fr_hi = wilson_ci(fr, fr_n)
    ax.errorbar(
        [sel_step],
        [fr],
        yerr=[[fr - fr_lo], [fr_hi - fr]],
        marker="s",
        color=colors[1],
        capsize=3,
        markersize=8,
        linestyle="none",
        label=f"selected checkpoint (step {sel_step}), full 195-question bank",
    )
    ax.text(sel_step, fr - 0.055, f"{fr:.2f}", ha="center", fontsize=9.5, color=colors[1])
    ax.axhline(inst["base_rate_source"], color="#888888", linestyle=":", linewidth=1.4)
    ax.text(
        150,
        inst["base_rate_source"] + 0.012,
        "base model, source persona",
        fontsize=9.5,
        color="#666666",
    )
    ax.set_xlabel("training step (9 epochs = 270 steps; prior round stopped at 90)")
    ax.set_ylabel("judged harmful-compliance rate")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(steps)
    ax.set_title("Tripling the dose plateaus near 0.40, below the band", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)

    ax = axes[1]
    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    x = np.arange(len(bins))
    w = 0.2
    edges = np.array([-0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 1.001])
    order = [
        "harmful_compliance-mixed__persona_software_engineer",
        "harmful_compliance-mixed__neg_default_assistant",
        "base__persona_software_engineer",
        "base__neg_default_assistant",
    ]
    for j, cell in enumerate(order):
        pq = [r for r in rates["cells"][cell]["per_question_rate"] if r is not None]
        counts = [sum(1 for r in pq if edges[k] <= r < edges[k + 1]) for k in range(len(bins))]
        assert sum(counts) == len(pq), (cell, sum(counts), len(pq))
        xs = x + (j - 1.5) * w
        ax.bar(xs, counts, width=w, color=colors[j], label=CELL_LABELS[cell])
        for xi, c in zip(xs, counts, strict=False):
            if c > 0:
                # Stagger tall-bar labels vertically so the crowded 0.0-bin
                # cluster (three counts near 190) stays readable.
                y = c * (1.10 + 0.45 * (j % 2)) if c >= 100 else c + 2
                ax.text(xi, y, f"{c}", ha="center", fontsize=8.2, color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.1f}" for b in bins])
    ax.set_xlabel("per-question compliance rate (nearest 0.2; 5 completions per question)")
    ax.set_ylabel("number of questions (of 194 with any valid score)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(0, 450)
    ax.set_title(f"Per-question rates at the selected checkpoint (step {sel_step})", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)

    fig.subplots_adjust(bottom=0.16, wspace=0.28)
    savefig_paper(fig, out_dir / "dose_ext_install_curve", formats=("png", "pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=repo / "eval_results" / "issue_1074" / "install-dose-extension",
    )
    ap.add_argument("--out-dir", type=Path, default=repo / "figures" / "issue_1074")
    args = ap.parse_args()
    set_paper_style("blog")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dose_extension(args.results_dir, args.out_dir)
    print(f"wrote dose_ext_ figures to {args.out_dir}")


if __name__ == "__main__":
    main()
