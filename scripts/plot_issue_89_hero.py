"""Plot the compact hero figure for the issue #81 trait-variation follow-up.

Produces `figures/leakage_i81/trait_ranking/fig_hero_compact.{png,pdf}` — a
single wide figure with two side-by-side panels: (L) mean cos(source,
bystander) vs level; (R) mean marker-emission rate vs level. 5 lines per
panel (one per Big-5 axis). Error bars = 1 SE across 25 (source × bystander-
noun) cells per point.

Data source: `eval_results/leakage_i81/trait_ranking/per_cell_ranking.csv`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)


def main() -> None:
    set_paper_style("generic")

    root = Path("eval_results/leakage_i81/trait_ranking")
    df = pd.read_csv(root / "per_cell_ranking.csv")

    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    levels = ["L1", "L2", "L3", "L4", "L5"]

    # Compute per-(trait, level) mean + SE from 25 cells
    stats = {}
    for t in traits:
        rows_cos = []
        rows_rate = []
        for lv in levels:
            cell = df[(df["trait"] == t) & (df["level"] == lv)]
            rows_cos.append(
                (cell["cos_a1"].mean(), cell["cos_a1"].std(ddof=1) / np.sqrt(len(cell)))
            )
            rows_rate.append(
                (cell["rate_a1"].mean(), cell["rate_a1"].std(ddof=1) / np.sqrt(len(cell)))
            )
        stats[t] = {"cos": rows_cos, "rate": rows_rate}

    colors = paper_palette(len(traits))
    fig, (ax_cos, ax_rate) = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)

    x = np.arange(len(levels))
    for i, t in enumerate(traits):
        cos_mean = [m for m, _ in stats[t]["cos"]]
        cos_se = [s for _, s in stats[t]["cos"]]
        rate_mean = [m * 100 for m, _ in stats[t]["rate"]]
        rate_se = [s * 100 for _, s in stats[t]["rate"]]
        ax_cos.errorbar(
            x,
            cos_mean,
            yerr=cos_se,
            marker="o",
            capsize=3,
            color=colors[i],
            label=t,
            linewidth=1.6,
            markersize=4,
        )
        ax_rate.errorbar(
            x,
            rate_mean,
            yerr=rate_se,
            marker="o",
            capsize=3,
            color=colors[i],
            label=t,
            linewidth=1.6,
            markersize=4,
        )

    ax_cos.set_title("Representation distance from source", fontsize=10)
    ax_cos.set_ylabel("mean cos(source, bystander)  [layer 20]")
    ax_cos.set_xticks(x)
    ax_cos.set_xticklabels(levels)
    ax_cos.set_xlabel("Bystander trait-gradation level")
    ax_cos.grid(True, alpha=0.3)

    ax_rate.set_title("Marker leakage", fontsize=10)
    ax_rate.set_ylabel("mean marker emission rate (%)")
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels(levels)
    ax_rate.set_xlabel("Bystander trait-gradation level")
    ax_rate.grid(True, alpha=0.3)

    # One shared legend below the panels
    handles, labels = ax_cos.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Per-trait gradation trajectories — 5 Big-5 axes (N=25 source×noun cells per point, err=1 SE)",
        fontsize=10,
    )

    savefig_paper(fig, "leakage_i81/trait_ranking/fig_hero_compact", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
