# Qwen marker token, em-dash, multiplication sign all intentional
"""Task #472 follow-up re-analysis figures (analyzer, placement-null-full-trajectory).

Two figures, both reading from the committed follow-up JSONs (no GPU):

  placement_full_trajectory_arms — mean held-out leakage by placement arm (near /
    spread / far) across all 6 checkpoints, with Holm-adjusted Friedman verdicts
    per checkpoint annotated. Documents the placement-null DOWNGRADE: the arms
    separate at >=1 checkpoint, so the earliest-checkpoint null was an
    artifact of reading too early.

  count_matched_step_levels — held-out leakage by count level (negex 100/200/400
    and negp 2/4/8) at five matched absolute training-step targets, with the
    per-axis interpolation-error floor band drawn for scale. Documents that the
    'more negatives = more leakage' finding survives the step-matched read.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472" / "placement-null-full-trajectory"
FIG = WT / "figures" / "issue_472"

PLACEMENT_JSON = SLAB / "reanalysis_placement_full_trajectory.json"
COUNT_JSON = SLAB / "reanalysis_count_matched_step.json"


def _placement_arms_figure() -> None:
    """Mean bystander delta-G by placement arm across all 6 checkpoints."""
    data = json.loads(PLACEMENT_JSON.read_text())
    checkpoints = data["checkpoints"]
    holm = data["holm_friedman_across_checkpoints"]
    verdicts = data["verdicts"]["per_checkpoint"]

    steps = [c["step"] for c in checkpoints]
    arms_order = ["near", "spread", "far"]
    arm_labels = {
        "near": "Near negatives",
        "spread": "Spread negatives",
        "far": "Far negatives",
    }
    arm_colors = {
        "near": paper_palette_role("control"),
        "spread": paper_palette_role("baseline"),
        "far": paper_palette_role("primary"),
    }

    means = {a: [] for a in arms_order}
    ci_lo = {a: [] for a in arms_order}
    ci_hi = {a: [] for a in arms_order}
    for ck in checkpoints:
        for a in arms_order:
            cell = ck["per_arm"][a]
            means[a].append(cell["pooled_mean_delta_g"])
            ci_lo[a].append(cell["boot_ci95"][0])
            ci_hi[a].append(cell["boot_ci95"][1])

    # Build step → holm-verdict via the frac label in the holm dict keys
    del verdicts  # rely on Holm keys directly
    frac_to_step = {ck["frac"]: ck["step"] for ck in checkpoints}
    step_to_holm = {}
    for k, h in holm.items():
        # k like "ck3_frac0.33" — extract the frac value after "frac"
        frac_str = k.split("frac")[1]
        frac = float(frac_str)
        step_to_holm[frac_to_step[frac]] = h

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for a in arms_order:
        yerr_lo = np.array(means[a]) - np.array(ci_lo[a])
        yerr_hi = np.array(ci_hi[a]) - np.array(means[a])
        ax.errorbar(
            steps,
            means[a],
            yerr=[yerr_lo, yerr_hi],
            label=arm_labels[a],
            color=arm_colors[a],
            marker="o",
            capsize=3,
            linewidth=1.8,
        )

    # Annotate per-checkpoint verdicts: bold "S" for separated, "·" for indistinguishable
    ymax = max(max(ci_hi[a]) for a in arms_order)
    for step in steps:
        h = step_to_holm[step]
        sym = "S" if h["reject_null"] else "ns"
        ax.text(
            step,
            ymax + 0.18,
            sym,
            ha="center",
            va="bottom",
            fontsize=9,
            color=(paper_palette_role("primary") if h["reject_null"] else "#777"),
            fontweight=("bold" if h["reject_null"] else "normal"),
        )

    ax.set_xlabel("Training step (matched across placement arms)")
    ax.set_ylabel("Mean held-out marker log-prob shift (nats)")
    ax.set_xticks(steps)
    ymin = min(min(ci_lo[a]) for a in arms_order)
    ax.set_ylim(ymin - 0.15, ymax + 0.55)
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.25)

    n_sep = sum(1 for h in step_to_holm.values() if h["reject_null"])
    subtitle = (
        f"S = Holm-adjusted Friedman p rejects equal arms ({n_sep}/6 checkpoints, "
        "incl. terminal); ns = indistinguishable. Error bars: 95% bootstrap CI "
        "over 47 held-out probes."
    )
    # Use ax.set_title + ax.text for a blog-style left-aligned title + subtitle
    ax.set_title(
        "Placement arms separate at 4/6 checkpoints (incl. the earliest) under the paired test",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        pad=18,
    )
    ax.text(
        0.0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555",
        ha="left",
        va="bottom",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    savefig_paper(fig, "issue_472/placement_full_trajectory_arms", dir=str(WT / "figures"))
    plt.close(fig)


def _count_matched_step_figure() -> None:
    """Held-out leakage by count level at matched absolute training-step targets."""
    data = json.loads(COUNT_JSON.read_text())
    targets = [10, 13, 19, 29, 38]  # matched_step_targets

    negex_levels = ["100", "200", "400"]
    negp_levels = ["2", "4", "8"]

    floor_check = data["interpolation_error_nats"]["resolution_floor_check"]
    negex_floor = floor_check["negex"]["max_per_probe_interp_error_nats"]
    negp_floor = floor_check["negp"]["max_per_probe_interp_error_nats"]

    # Pull level means + CIs at each target
    def _pull(axis_key: str, levels: list[str]):
        per_target = data["matched_step_comparisons"][axis_key]["per_target"]
        means = {lv: [] for lv in levels}
        ci_lo = {lv: [] for lv in levels}
        ci_hi = {lv: [] for lv in levels}
        for t in targets:
            cell = per_target[f"step_{t}"]
            for lv in levels:
                means[lv].append(cell["level_means"][lv])
                lo, hi = cell["level_boot_ci95"][lv]
                ci_lo[lv].append(lo)
                ci_hi[lv].append(hi)
        return means, ci_lo, ci_hi

    negex_means, negex_lo, negex_hi = _pull("negex", negex_levels)
    negp_means, negp_lo, negp_hi = _pull("negp", negp_levels)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)

    level_palette = {
        "low": paper_palette_role("control"),
        "mid": paper_palette_role("baseline"),
        "high": paper_palette_role("primary"),
    }

    def _draw(ax, levels, means, lo, hi, level_label_fn, axis_label, floor_nats):
        slots = {levels[0]: "low", levels[1]: "mid", levels[2]: "high"}
        bar_w = 1.6
        group_w = bar_w * 3 + 0.8
        x_centers = np.arange(len(targets)) * group_w
        for j, lv in enumerate(levels):
            xs = x_centers + (j - 1) * bar_w
            ys = np.array(means[lv])
            yerr_lo = ys - np.array(lo[lv])
            yerr_hi = np.array(hi[lv]) - ys
            ax.bar(
                xs,
                ys,
                width=bar_w * 0.9,
                color=level_palette[slots[lv]],
                label=level_label_fn(lv),
                yerr=[yerr_lo, yerr_hi],
                capsize=2.5,
                linewidth=0,
            )
        # Per-probe interpolation-error floor — drawn as a band at the bottom
        ax.axhspan(0, floor_nats, color="#bbbbbb", alpha=0.35, zorder=0)
        ax.text(
            x_centers[-1] + group_w * 0.45,
            floor_nats * 0.95,
            f"per-probe interp-error floor ({floor_nats:.2f} nats)",
            fontsize=8,
            color="#555",
            ha="right",
            va="top",
        )
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"step {t}" for t in targets])
        ax.set_xlabel(axis_label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper left", frameon=False, fontsize=9)

    _draw(
        axes[0],
        negex_levels,
        negex_means,
        negex_lo,
        negex_hi,
        lambda lv: f"{lv} examples / persona",
        "Negative examples per persona (axis A)",
        negex_floor,
    )
    _draw(
        axes[1],
        negp_levels,
        negp_means,
        negp_lo,
        negp_hi,
        lambda lv: f"{lv} negative personas",
        "Number of negative personas (axis B)",
        negp_floor,
    )

    axes[0].set_ylabel("Mean held-out marker log-prob shift (nats)")
    ymax = max(
        max(max(negex_hi[lv]) for lv in negex_levels),
        max(max(negp_hi[lv]) for lv in negp_levels),
    )
    axes[0].set_ylim(0, ymax + 1.2)

    fig.suptitle(
        '"More negatives = more leakage" survives matched training step',
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.945,
        "Held-out leakage at five matched absolute training-step targets. Levels stay ~10 "
        "nats apart at every step — 8-9x the per-probe interpolation-error floor; not a "
        "training-budget artifact.",
        ha="left",
        fontsize=8.5,
        color="#555",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    savefig_paper(fig, "issue_472/count_matched_step_levels", dir=str(WT / "figures"))
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    FIG.mkdir(parents=True, exist_ok=True)
    _placement_arms_figure()
    _count_matched_step_figure()


if __name__ == "__main__":
    main()
