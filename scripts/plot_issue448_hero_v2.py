"""Regenerate the issue #448 hero 4-knob sweep figure with a subtitle that
doesn't assert a single wrong bystander denominator.

Round-1 v1 subtitle: "Mean across 23 held-out bystanders. ..."

That subtitle was wrong for two cells: the +pos-personas=2 cell uses 22
bystanders (comedian excluded), and the +pos-personas=4 cell uses 20
bystanders (comedian + assistant + software_engineer excluded). Critic
round-2 caught it. The body caption already discloses the variable
denominator; the figure subtitle should not assert a single count it
doesn't satisfy.

Inputs:  eval_results/issue_448/analyze_summary.json
Outputs: figures/issue_448/hero_4knob_sweep.{png,pdf,meta.json}
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

KNOB_PANEL_TITLES = {
    "pos_ex_per_persona": "Examples per positive persona",
    "pos_personas": "Number of positive personas",
    "neg_ex_per_persona": "Examples per contrastive negative persona",
    "neg_personas": "Number of contrastive negative personas",
}

KNOB_X_LABELS = {
    "pos_ex_per_persona": "Marker examples / positive persona",
    "pos_personas": "Positive personas",
    "neg_ex_per_persona": "Non-marker examples / negative persona",
    "neg_personas": "Negative personas",
}

KNOB_ORDER = [
    "pos_ex_per_persona",
    "pos_personas",
    "neg_ex_per_persona",
    "neg_personas",
]

ANCHOR_LEVELS = {
    "pos_ex_per_persona": 200,
    "pos_personas": 1,
    "neg_ex_per_persona": 200,
    "neg_personas": 2,
}


def main() -> None:
    summary_path = Path("eval_results/issue_448/analyze_summary.json")
    with summary_path.open() as f:
        summary = json.load(f)

    set_paper_style("blog")

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.6), sharey=True)

    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")

    y_min = np.inf
    y_max = -np.inf

    for ax, knob in zip(axes, KNOB_ORDER):
        info = summary["per_knob"][knob]
        cells = info["axis_cells"]
        levels = info["axis_levels"]
        deltas = info["mean_deltas"]

        # Pull per-cell CI half-widths to draw error bars.
        ci_halfwidths = []
        for cell in cells:
            cell_data = summary["per_cell"][cell]
            half = cell_data.get("ci_halfwidth_23")
            if half is None:
                lo, hi = cell_data["ci_bystander_delta_23"]
                mean = cell_data["mean_bystander_delta_23"]
                half = max(hi - mean, mean - lo)
            ci_halfwidths.append(half)

        ax.errorbar(
            levels,
            deltas,
            yerr=ci_halfwidths,
            color=primary,
            marker="o",
            markersize=6,
            markerfacecolor=primary,
            linewidth=1.6,
            capsize=3,
            zorder=2,
        )

        # Anchor cell marker (open circle).
        anchor_level = ANCHOR_LEVELS[knob]
        anchor_idx = levels.index(anchor_level)
        ax.scatter(
            [anchor_level],
            [deltas[anchor_idx]],
            facecolors="white",
            edgecolors=primary,
            s=80,
            linewidths=1.8,
            zorder=3,
        )

        ax.set_title(KNOB_PANEL_TITLES[knob], fontsize=11, loc="left", pad=8)
        ax.set_xlabel(KNOB_X_LABELS[knob], fontsize=9)

        # Log-x for the example-count knobs; linear-x for persona-count knobs.
        if knob in ("pos_ex_per_persona", "neg_ex_per_persona"):
            ax.set_xscale("log")
            ax.set_xticks(levels)
            ax.set_xticklabels([str(v) for v in levels])
        else:
            ax.set_xticks(levels)
            ax.set_xticklabels([str(v) for v in levels])

        # Range annotation (bottom-right of each panel).
        direction = "down" if info["monotone_down"] else "up"
        rng = info["delta_range_nats"]
        ax.text(
            0.97,
            0.04,
            f"{direction}, range {rng:.2f} nats",
            transform=ax.transAxes,
            fontsize=9,
            color=neutral,
            ha="right",
            va="bottom",
            style="italic",
        )

        y_min = min(y_min, min(d - h for d, h in zip(deltas, ci_halfwidths)))
        y_max = max(y_max, max(d + h for d, h in zip(deltas, ci_halfwidths)))

    # Shared y-axis label on the leftmost panel.
    axes[0].set_ylabel("Mean bystander leakage Δ\nlog p( ※) nats above base", fontsize=10)

    # Add a tiny pad to the y-limits so error bars don't kiss the frame.
    pad = (y_max - y_min) * 0.04
    for ax in axes:
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.6)

    # Anthropic-blog title block — manual placement, since the four-panel grid
    # doesn't get along with set_title_subtitle (which targets a single ax).
    fig.suptitle(
        "Contrastive-negative knobs reduce bystander marker leakage; "
        "positive-side knobs nudge it up",
        x=0.005,
        y=0.99,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.005,
        0.935,
        "Mean across held-out bystanders (23 in 9 cells; 22 in +pos-personas=2; 20 in +pos-personas=4). "
        "Error bars = 95% bootstrap CI half-width. Open circle = anchor cell "
        "(1 positive × 200 examples + 2 negatives × 200 examples). "
        "N = 1 seed (42). Qwen-2.5-7B-Instruct, contrastive LoRA SFT.",
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.text(
        0.005,
        0.01,
        "Source: task #448, eval at commit c1cc2b6e",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#888888",
        style="italic",
    )

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.88])
    savefig_paper(fig, "issue_448/hero_4knob_sweep", dir="figures/")
    plt.close(fig)
    print("Wrote figures/issue_448/hero_4knob_sweep.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
