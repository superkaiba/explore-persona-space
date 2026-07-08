"""Figures for issue #1074 follow-up round `base-negatives-regen` (bnr_ prefix).

Reads the round's committed Phase-D outputs
(eval_results/issue_1074/base-negatives-regen/, committed on branch
issue-1074-fu1 @ eafec28462; pass --results-dir to point at the worktree copy
before the branch merges to main) and writes two 2-panel figures to
figures/issue_1074/:

- bnr_negative_yield_quota: per-member kept contrastive negatives vs the
  24-of-35 pairing quota, base-Qwen round vs the parent abliterated round
  (left), and the per-member drop composition for both rounds (right).
- bnr_install_dose_curve: judged compliance rate vs training step with the
  0.60-0.85 target band (left), and the per-question rate distribution at the
  selected checkpoint for all four final-eval cells (right).

Usage:
    uv run python scripts/issue1074_bnr_figures.py [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import json
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
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

MEMBER_LABELS = {
    "neg_sp_police": "police officer",
    "neg_sp_ph4": "second persona",
    "neg_reph_curious": "curious rephrase",
    "neg_wc_short": "WildChat-short",
    "neg_default_assistant": "default assistant",
}
CELL_LABELS = {
    "harmful_compliance-mixed__persona_software_engineer": "trained, source persona",
    "harmful_compliance-mixed__neg_default_assistant": "trained, default context",
    "base__persona_software_engineer": "base, source persona",
    "base__neg_default_assistant": "base, default context",
}
# Scored-generation denominators per dose checkpoint (30-question subset x 5
# draws minus judge drops); exact integer reconstruction from the committed
# rates (e.g. 0.124137931... = 18/145).
DOSE_N_SCORED = {25: 145, 50: 149, 75: 146, 90: 146}


def fig_negative_yield(results_dir: Path, out_dir: Path) -> None:
    d = json.loads((results_dir / "negative_yield.json").read_text())
    members = list(MEMBER_LABELS)
    quota, budget = d["quota"], d["budget"]
    rounds = [
        ("mixed", "base-Qwen negatives (this round)"),
        ("parent_ablit", "abliterated negatives (parent round)"),
    ]
    colors = paper_palette(4)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    x = np.arange(len(members))
    w = 0.38
    for j, (key, label) in enumerate(rounds):
        kept = np.array([d[key][m]["kept"] for m in members], dtype=float)
        lo = np.array([d[key][m]["kept_rate_ci95"][0] * budget for m in members])
        hi = np.array([d[key][m]["kept_rate_ci95"][1] * budget for m in members])
        xs = x + (j - 0.5) * w
        ax.bar(xs, kept, width=w, color=colors[j], label=label)
        ax.errorbar(
            xs,
            kept,
            yerr=[kept - lo, hi - kept],
            fmt="none",
            ecolor="#444444",
            elinewidth=1.2,
            capsize=3,
        )
        for xi, k in zip(xs, kept, strict=False):
            ax.text(
                xi,
                1.0,
                f"{int(k)}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
    ax.axhline(quota, color="#333333", linestyle="--", linewidth=1.4)
    ax.text(
        len(members) - 0.55,
        quota + 0.5,
        f"pairing quota ({quota} of {budget})",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MEMBER_LABELS[m] for m in members], rotation=20, ha="right")
    ax.set_ylabel(f"kept contrastive negatives (of {budget})")
    ax.set_ylim(0, budget)
    ax.set_title("Kept judge-clean negatives per panel member", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, frameon=False)

    ax = axes[1]
    drop_kinds = [("judged harmful-compliant", colors[3]), ("no valid judge score", "#9a9a9a")]
    for j, (key, _label) in enumerate(rounds):
        none_d = np.array([d[key][m]["judge_none_drops"] for m in members], dtype=float)
        kept = np.array([d[key][m]["kept"] for m in members], dtype=float)
        compl = budget - kept - none_d
        xs = x + (j - 0.5) * w
        ax.bar(
            xs, compl, width=w, color=drop_kinds[0][1], label=drop_kinds[0][0] if j == 0 else None
        )
        ax.bar(
            xs,
            none_d,
            width=w,
            bottom=compl,
            color=drop_kinds[1][1],
            label=drop_kinds[1][0] if j == 0 else None,
        )
        for xi, c, n in zip(xs, compl, none_d, strict=False):
            if c > 0:
                ax.text(xi, c / 2, f"{int(c)}", ha="center", va="center", fontsize=9)
            ax.text(
                xi, c + n + 0.3, f"{int(n)}", ha="center", va="bottom", fontsize=9, color="#666666"
            )
    ax.set_xticks(x)
    ax.set_xticklabels([MEMBER_LABELS[m] for m in members], rotation=20, ha="right")
    ax.set_ylabel(f"dropped negatives (of {budget} generated)")
    ax.set_title("Why negatives dropped (left bar: this round; right: parent)", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, frameon=False)

    fig.subplots_adjust(bottom=0.34, wspace=0.28)
    savefig_paper(fig, out_dir / "bnr_negative_yield_quota", formats=("png", "pdf"))
    plt.close(fig)


def fig_install_dose(results_dir: Path, out_dir: Path) -> None:
    inst = json.loads((results_dir / "install" / "install_summary.json").read_text())
    rates = json.loads((results_dir / "judge" / "harmful_compliance" / "rates.json").read_text())
    colors = paper_palette(4)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    band = (0.60, 0.85)
    ax.axhspan(band[0], band[1], color=colors[2], alpha=0.18)
    ax.text(
        26,
        band[0] + 0.012,
        "target judged-rate band (0.60 to 0.85)",
        fontsize=10,
        color="#2a6f4e",
        va="bottom",
    )
    steps = sorted(int(s) for s in inst["dose_curve_rates_by_step"])
    curve = [inst["dose_curve_rates_by_step"][str(s)] for s in steps]
    err = np.array(
        [
            [curve[i] - proportion_ci(curve[i], DOSE_N_SCORED[s])[0] for i, s in enumerate(steps)],
            [proportion_ci(curve[i], DOSE_N_SCORED[s])[1] - curve[i] for i, s in enumerate(steps)],
        ]
    )
    ax.errorbar(
        steps,
        curve,
        yerr=err,
        marker="o",
        color=colors[0],
        linewidth=2,
        capsize=3,
        label="checkpoint read (30-question subset)",
    )
    for s, r in zip(steps, curve, strict=False):
        ax.text(s, r + 0.03, f"{r:.2f}", ha="center", fontsize=10, color=colors[0])
    fr = inst["final_rate_source"]
    fr_lo, fr_hi = proportion_ci(fr, 948)
    ax.errorbar(
        [90],
        [fr],
        yerr=[[fr - fr_lo], [fr_hi - fr]],
        marker="s",
        color=colors[1],
        capsize=3,
        markersize=8,
        linestyle="none",
        label="selected checkpoint, full 195-question bank",
    )
    ax.text(90, fr - 0.055, f"{fr:.2f}", ha="center", fontsize=10, color=colors[1])
    ax.axhline(inst["base_rate_source"], color="#888888", linestyle=":", linewidth=1.4)
    ax.text(
        26,
        inst["base_rate_source"] + 0.012,
        "base model, source persona",
        fontsize=9.5,
        color="#666666",
    )
    ax.set_xlabel("training step (3 epochs = 90 steps)")
    ax.set_ylabel("judged harmful-compliance rate")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(steps)
    ax.set_title("Install dose curve never reaches the band", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    ax = axes[1]
    cells = list(CELL_LABELS)
    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    x = np.arange(len(bins))
    w = 0.2
    edges = np.array([-0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 1.001])
    for j, cell in enumerate(cells):
        pq = [r for r in rates["cells"][cell]["per_question_rate"] if r is not None]
        counts = [sum(1 for r in pq if edges[k] <= r < edges[k + 1]) for k in range(len(bins))]
        assert sum(counts) == len(pq), (cell, sum(counts), len(pq))
        xs = x + (j - 1.5) * w
        ax.bar(xs, counts, width=w, color=colors[j], label=CELL_LABELS[cell])
        for xi, c in zip(xs, counts, strict=False):
            if c > 0:
                ax.text(xi, c + 2, f"{c}", ha="center", fontsize=8.2, color=colors[j])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:.1f}" for b in bins])
    ax.set_xlabel("per-question compliance rate (binned to nearest 0.2; up to 5 draws each)")
    ax.set_ylabel("number of questions (of 194 with any valid score)")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Per-question rates at the selected checkpoint", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)

    fig.subplots_adjust(bottom=0.16, wspace=0.28)
    savefig_paper(fig, out_dir / "bnr_install_dose_curve", formats=("png", "pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=repo / "eval_results" / "issue_1074" / "base-negatives-regen",
    )
    ap.add_argument("--out-dir", type=Path, default=repo / "figures" / "issue_1074")
    args = ap.parse_args()
    set_paper_style("blog")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_negative_yield(args.results_dir, args.out_dir)
    fig_install_dose(args.results_dir, args.out_dir)
    print(f"wrote bnr_ figures to {args.out_dir}")


if __name__ == "__main__":
    main()
