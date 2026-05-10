#!/usr/bin/env python3
"""Issue #238 supporting figures.

(1) Delta-ratio (full-param / LoRA) across all 5 layers, two panels (Method A
    and Method B), bars colored by data_type (EM vs benign) and grouped by LR.
(2) Weight-delta global L2 norms vs M1 delta at L20 — tests whether geometric
    collapse scales with parameter-update magnitude. Two scatter points per
    condition (Method A and B).

Usage:
    uv run python scripts/plot_issue238_supporting.py
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


def plot_ratio_panels(data: dict) -> None:
    """Two panels (Method A, Method B); x = layer; bars per (lr × data_type)."""
    set_paper_style("blog")
    layers = [7, 14, 20, 21, 27]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True)
    palette = paper_palette(4)
    # Bars: EM-2e5, EM-1e4, Benign-2e5, Benign-1e4
    bar_labels = [
        "EM lr=2e-5",
        "EM lr=1e-4",
        "Benign lr=2e-5",
        "Benign lr=1e-4",
    ]

    for ax, method in zip(axes, ["A", "B"]):
        x = np.arange(len(layers), dtype=float)
        bar_width = 0.18

        for i, (key_suffix, color) in enumerate(
            [
                ("em_2e5", palette[0]),
                ("em_1e4", palette[1]),
                ("benign_2e5", palette[2]),
                ("benign_1e4", palette[3]),
            ]
        ):
            ratios = []
            for layer in layers:
                key = f"ratio_{method}_L{layer}_{key_suffix}"
                ratios.append(data["delta_ratios"][key]["ratio"])
            offset = (i - 1.5) * bar_width
            ax.bar(
                x + offset,
                ratios,
                bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.3,
                label=bar_labels[i] if method == "A" else None,
            )

        # Reference lines
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.axhline(1.5, color="grey", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.text(
            0.02,
            0.5,
            "H1 boundary (0.5×)",
            transform=ax.transAxes,
            fontsize=6,
            color="grey",
            verticalalignment="bottom",
        )
        ax.text(
            0.02,
            0.94,
            "H3 boundary (1.5×)",
            transform=ax.transAxes,
            fontsize=6,
            color="grey",
            verticalalignment="top",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{layer}" for layer in layers])
        ax.set_xlabel("Extraction layer")
        ax.set_title(f"Method {method}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel(r"$\Delta_\mathrm{full} / \Delta_\mathrm{LoRA}$")
    fig.legend(
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=7,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    savefig_paper(fig, "issue_238/ratio_by_layer", dir=str(FIG_DIR) + "/")
    plt.close(fig)


def plot_weight_delta_vs_collapse(data: dict) -> None:
    """Scatter: global ||theta_full - theta_base||_2 vs M1 delta at L20."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=False)
    palette = paper_palette(4)

    # 4 conditions × 2 methods × 1 layer (L20)
    conditions = [
        ("full_em_lr2e5", "EM lr=2e-5", palette[0], "o"),
        ("full_em_lr1e4", "EM lr=1e-4", palette[1], "s"),
        ("full_benign_lr2e5", "Benign lr=2e-5", palette[2], "^"),
        ("full_benign_lr1e4", "Benign lr=1e-4", palette[3], "D"),
    ]

    for ax, method in zip(axes, ["A", "B"]):
        for cond, label, color, marker in conditions:
            wd_global = data["weight_delta_norms"][cond]["global_l2"]
            m1_delta = data["results"][f"M1_{method}_L20_{cond}"]["delta_mean_offdiag"]
            ax.scatter(
                wd_global,
                m1_delta,
                color=color,
                marker=marker,
                s=60,
                edgecolor="black",
                linewidth=0.4,
                label=label,
                zorder=3,
            )

        ax.set_xlabel(r"Global $\|\theta_\mathrm{full} - \theta_\mathrm{base}\|_2$")
        ax.set_ylabel(r"$\Delta$ mean off-diag cos-sim, L20")
        ax.set_title(f"Method {method}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_xlim(0, 110)

    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    savefig_paper(fig, "issue_238/weight_delta_vs_collapse", dir=str(FIG_DIR) + "/")
    plt.close(fig)


def main() -> None:
    with open(RUN_RESULT) as f:
        data = json.load(f)

    plot_ratio_panels(data)
    plot_weight_delta_vs_collapse(data)
    print("Saved issue_238 supporting figures.")


if __name__ == "__main__":
    main()
