#!/usr/bin/env python3
"""Issue #238 hero figure — full-param vs LoRA persona-geometry collapse.

Plots full-param SFT M1 deltas alongside the #205 LoRA baselines at the
flagship layer L20 (Method A) and the deepest layer L27 (Method A).

Two-panel grouped bar chart:
- Each panel shows EM (left bars) vs benign (right bars).
- Each panel shows three methods:
    LoRA (#205, lr=1e-4),
    Full-param @ lr=2e-5 (matched LoRA scale),
    Full-param @ lr=1e-4 (LR control).

Usage:
    uv run python scripts/plot_issue238_hero.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
RUN_RESULT = REPO / "eval_results" / "issue_238" / "run_result.json"
FIG_DIR = REPO / "figures"


def main() -> None:
    set_paper_style("blog")

    with open(RUN_RESULT) as f:
        data = json.load(f)

    ratios = data["delta_ratios"]
    lora = data["lora_deltas_from_205"]
    results = data["results"]

    # We plot two panels: L20 (flagship), L27 (deepest).
    panel_layers = [20, 27]
    methods = ["A"]  # Method A is the canonical from #237
    method = "A"

    # Per panel: 6 bars (EM-LoRA, EM-Full-2e5, EM-Full-1e4, Benign-LoRA, Benign-Full-2e5, Benign-Full-1e4)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=False)
    palette = paper_palette(3)
    color_lora = palette[0]
    color_full_2e5 = palette[1]
    color_full_1e4 = palette[2]

    bar_labels = [
        "LoRA\n(#205, lr=1e-4)",
        "Full-param\n(lr=2e-5)",
        "Full-param\n(lr=1e-4)",
    ]
    colors_em = [color_lora, color_full_2e5, color_full_1e4]
    colors_benign = [color_lora, color_full_2e5, color_full_1e4]

    for ax, layer in zip(axes, panel_layers):
        # Pull values
        em_lora = lora[f"M{method}_L{layer}_em"]
        benign_lora = lora[f"M{method}_L{layer}_benign"]
        em_full_2e5 = results[f"M1_{method}_L{layer}_full_em_lr2e5"]["delta_mean_offdiag"]
        em_full_1e4 = results[f"M1_{method}_L{layer}_full_em_lr1e4"]["delta_mean_offdiag"]
        benign_full_2e5 = results[f"M1_{method}_L{layer}_full_benign_lr2e5"]["delta_mean_offdiag"]
        benign_full_1e4 = results[f"M1_{method}_L{layer}_full_benign_lr1e4"]["delta_mean_offdiag"]

        em_vals = [em_lora, em_full_2e5, em_full_1e4]
        benign_vals = [benign_lora, benign_full_2e5, benign_full_1e4]

        n_bars = 3
        group_gap = 0.35
        bar_width = 0.7
        x_em = np.arange(n_bars)
        x_benign = np.arange(n_bars) + n_bars + group_gap

        b1 = ax.bar(
            x_em, em_vals, color=colors_em, width=bar_width, edgecolor="black", linewidth=0.4
        )
        b2 = ax.bar(
            x_benign,
            benign_vals,
            color=colors_benign,
            width=bar_width,
            edgecolor="black",
            linewidth=0.4,
            hatch="///",
        )

        # Value labels on bars
        for rect, v in zip(list(b1) + list(b2), em_vals + benign_vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.003,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        # Group labels under x-axis
        ax.set_xticks([(n_bars - 1) / 2, n_bars + group_gap + (n_bars - 1) / 2])
        ax.set_xticklabels(["EM\n(bad_legal_advice)", "Benign\n(Tulu-3-SFT)"], fontsize=8)

        ax.set_ylabel(r"$\Delta$ mean off-diagonal cos-sim (post − base)")
        ax.set_title(f"Layer {layer} (Method A)", fontsize=10)
        ax.set_ylim(0, max(em_vals + benign_vals) * 1.30)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    # Single shared legend across the figure (above)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_lora, edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, color=color_full_2e5, edgecolor="black", linewidth=0.4),
        plt.Rectangle((0, 0), 1, 1, color=color_full_1e4, edgecolor="black", linewidth=0.4),
    ]
    fig.legend(
        legend_handles,
        bar_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig_paper(fig, "issue_238/hero_fullparam_vs_lora", dir=str(FIG_DIR) + "/")
    plt.close(fig)
    print(f"Saved hero figure to {FIG_DIR}/issue_238/hero_fullparam_vs_lora.{{png,pdf}}")


if __name__ == "__main__":
    main()
