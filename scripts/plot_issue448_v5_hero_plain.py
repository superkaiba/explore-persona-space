"""Hero figure for issue #448 on-policy re-run (v5).

Two-row, two-column panel:
- Top row: the four recipe knobs at the marker-implant scale (0-25 nats),
  showing that the recipe knob is invisible against the magnitude of the
  trained-vs-base shift.
- Bottom row: the SAME points, zoomed to the actual cross-knob spread
  (~0.003 nats), showing that even under the magnifying glass no monotone
  trend survives.

Plain-English knob names everywhere reader-facing; cell slugs only in the
data dictionary.
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

# Plain-English label for each knob axis (top of each column).
KNOB_TITLE = {
    "pos_ex_per_persona": "Positive examples per persona",
    "pos_personas": "Positive personas (extra)",
    "neg_ex_per_persona": "Negative examples per persona",
    "neg_personas": "Negative personas",
}

# Plain-English x-axis label for each knob.
KNOB_XLABEL = {
    "pos_ex_per_persona": "examples / source persona",
    "pos_personas": "extra positive personas",
    "neg_ex_per_persona": "examples / negative persona",
    "neg_personas": "negative personas",
}

# The hypothesized direction (down = "should reduce bystander leakage").
KNOB_HYPOTH = {
    "pos_ex_per_persona": "up",  # more positives → leakage up (predicted)
    "pos_personas": "up",
    "neg_ex_per_persona": "down",  # more negatives → leakage down (predicted)
    "neg_personas": "down",
}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    summary_path = repo / "eval_results/issue_448_v5/analyze_summary.json"
    figures_dir = repo / "figures/issue_448_v5"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(summary_path.read_text())
    per_cell = summary["per_cell"]
    per_knob = summary["per_knob_h1b"]  # held-out is the headline DV

    set_paper_style("blog")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex="col")

    color = paper_palette_role("primary")

    for col, knob in enumerate(
        ["pos_ex_per_persona", "pos_personas", "neg_ex_per_persona", "neg_personas"]
    ):
        axis_cells = per_knob[knob]["axis_cells"]
        levels = per_knob[knob]["axis_levels"]

        means = [per_cell[s]["mean_bystander_delta_held_out"] for s in axis_cells]
        ci_lo = [per_cell[s]["ci_bystander_delta_held_out"][0] for s in axis_cells]
        ci_hi = [per_cell[s]["ci_bystander_delta_held_out"][1] for s in axis_cells]

        means = np.asarray(means)
        ci_lo = np.asarray(ci_lo)
        ci_hi = np.asarray(ci_hi)
        err = np.vstack([means - ci_lo, ci_hi - means])

        # Top: full implant scale 0-25 nats
        ax_top = axes[0, col]
        ax_top.errorbar(
            levels, means, yerr=err, fmt="o-", color=color, capsize=4, lw=1.4, markersize=6
        )
        ax_top.axhline(0, color="grey", lw=0.5)
        ax_top.set_title(KNOB_TITLE[knob], fontsize=11)
        ax_top.set_ylim(0, 26)
        if knob.endswith("ex_per_persona"):
            ax_top.set_xscale("log")

        # Bottom: zoomed to the cross-knob spread (~0.003 nats). Error bars
        # are omitted here on purpose — the per-cell within-eval CI is ~±0.6
        # nats, two orders of magnitude wider than the zoomed window. Plotting
        # them would obscure the cross-knob curve and over-state precision;
        # the top panel's bars already carry the uncertainty story.
        ax_bot = axes[1, col]
        ax_bot.plot(levels, means, "o-", color=color, lw=1.4, markersize=6)
        cell_min = means.min()
        cell_max = means.max()
        cell_pad = max((cell_max - cell_min) * 1.5, 0.005)
        center = (cell_max + cell_min) / 2
        ax_bot.set_ylim(center - cell_pad, center + cell_pad)
        ax_bot.set_xlabel(KNOB_XLABEL[knob], fontsize=10)
        if knob.endswith("ex_per_persona"):
            ax_bot.set_xscale("log")

        # Range annotation in bottom panel
        rng = cell_max - cell_min
        ax_bot.text(
            0.97,
            0.05,
            f"range: {rng:.4f} nats",
            transform=ax_bot.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="grey",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
        )

    axes[0, 0].set_ylabel("Trained − base log P(※)\n(nats, held-out bystanders)")
    axes[1, 0].set_ylabel("Same, zoomed in\n(window ≈ 0.01 nats)")

    fig.suptitle(
        "None of the four recipe knobs moves on-policy bystander marker leakage",
        fontsize=13,
        x=0.02,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.93,
        "Held-out bystander subset (n=15 personas, 20 generic eval questions, 1 seed). "
        "Threshold for declaring a knob-effect was ≥1.0 nat with a permutation p < 0.10; "
        "observed cross-knob range is ~0.003 nats, three orders of magnitude smaller.",
        fontsize=9,
        color="#555",
        ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out = savefig_paper(fig, "issue_448_v5/hero_4knob_null_plain", dir=str(repo / "figures/"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
