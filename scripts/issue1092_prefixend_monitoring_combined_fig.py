"""Combined figure: prefix-end vs averaged-context trait monitoring (#1092).

Merges the two round figures (`prefixend_monitoring.png` — supervised parity +
averaging curves — and `prefixend_monitoring_constructions.png` — three readout
constructions) into ONE 2x2 figure with a single color encoding (color = input
state) and one shared legend. The supervised bars appear once (top row) instead
of twice across the two source figures.

Re-plot of banked values only — no recomputation. Inputs:
  eval_results/issue_1092/inline_prefixend_monitoring/results.json
  eval_results/issue_1092/inline_prefixend_monitoring/readout_constructions.json
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval_results" / "issue_1092" / "inline_prefixend_monitoring"
LAYER = "14"
CELL = "cell_inst_own"

TRAITS = ["sycophancy", "hallucination"]
TRAIT_TITLES = {"sycophancy": "Sycophancy", "hallucination": "Hallucination"}
ARMS = ["averaged_context", "prefix_end"]
ARM_LABELS = {
    "averaged_context": "query-averaged context vectors (mean of N question-contexts; N=48 in top row)",
    "prefix_end": "pre-query prefix-end state (one vector per prefix, zero questions)",
}
CONSTRUCTIONS = ["supervised", "raw_rb_projection", "map_mediated"]
CONSTRUCTION_LABELS = {
    "supervised": "supervised probe\n(trained on judge scores)",
    "raw_rb_projection": "raw projection onto\npersona vector $r_B$",
    "map_mediated": "map-mediated: transport\nmap, then $r_B$",
}


def _yerr(read: dict) -> np.ndarray:
    lo, hi = read["ci95"]
    r = read["r"]
    return np.array([[max(r - lo, 0.0)], [max(hi - r, 0.0)]])


def main() -> None:
    results = json.loads((SRC / "results.json").read_text())
    constructions = json.loads((SRC / "readout_constructions.json").read_text())

    per_trait = {row["trait"]: row for row in results["cells"][CELL][LAYER]}
    con = constructions["constructions"]

    # reads[trait][construction][arm] -> {"r":, "ci95":}
    reads: dict[str, dict[str, dict[str, dict]]] = {}
    for trait in TRAITS:
        supervised = per_trait[trait]["reads"]
        reads[trait] = {
            "supervised": {arm: supervised[arm] for arm in ARMS},
            "raw_rb_projection": {arm: con["raw_rb_projection"][CELL][trait][arm] for arm in ARMS},
            "map_mediated": {arm: con["map_mediated"][CELL][trait][arm] for arm in ARMS},
        }

    set_paper_style("blog")
    arm_colors = {
        "averaged_context": paper_palette_role("baseline"),
        "prefix_end": paper_palette_role("primary"),
    }
    ceiling_color = paper_palette_role("neutral")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))

    # ---- top row: readout r by construction and input state ----
    for col, trait in enumerate(TRAITS):
        ax = axes[0, col]
        xs = np.arange(len(CONSTRUCTIONS), dtype=float)
        for j, arm in enumerate(ARMS):
            offs = -0.19 if j == 0 else 0.19
            for i, cname in enumerate(CONSTRUCTIONS):
                read = reads[trait][cname][arm]
                ax.bar(
                    xs[i] + offs,
                    read["r"],
                    width=0.34,
                    color=arm_colors[arm],
                    yerr=_yerr(read),
                    error_kw={"ecolor": "#333333", "elinewidth": 1.1, "capsize": 0},
                )
        ceiling = per_trait[trait]["monitoring_r_ceiling_from_reliability"]
        ax.axhline(ceiling, ls="--", lw=1.2, color=ceiling_color, zorder=1)
        ax.axhline(0.0, lw=0.8, color="#999999", zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([CONSTRUCTION_LABELS[c] for c in CONSTRUCTIONS])
        ax.set_ylim(-0.85, 1.0)
        if col == 0:
            ax.set_ylabel("monitoring correlation r\n(readout vs judge score)")
        ax.set_title(
            f"{TRAIT_TITLES[trait]}: three readouts of the same two states",
            loc="left",
        )

    # ---- bottom row: averaging curve vs the single prefix-end read ----
    for col, trait in enumerate(TRAITS):
        ax = axes[1, col]
        curve = per_trait[trait]["averaging_curve_context"]
        ns = np.array([p["N"] for p in curve], dtype=float)
        mean = np.array([p["r_mean"] for p in curve])
        sd = np.array([p["r_sd"] for p in curve])
        ax.plot(
            ns,
            mean,
            marker="o",
            ms=4.5,
            color=arm_colors["averaged_context"],
            label=ARM_LABELS["averaged_context"],
        )
        ax.fill_between(
            ns,
            mean - 1.96 * sd,
            mean + 1.96 * sd,
            color=arm_colors["averaged_context"],
            alpha=0.18,
            lw=0,
        )
        pe = per_trait[trait]["reads"]["prefix_end"]
        ax.axhline(pe["r"], ls="--", lw=1.6, color=arm_colors["prefix_end"])
        ax.axhspan(pe["ci95"][0], pe["ci95"][1], color=arm_colors["prefix_end"], alpha=0.12, lw=0)
        ceiling = per_trait[trait]["monitoring_r_ceiling_from_reliability"]
        ax.axhline(ceiling, ls="--", lw=1.2, color=ceiling_color, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(int(n)) for n in ns])
        ax.minorticks_off()
        ax.set_ylim(0.35, 0.97)
        ax.set_xlabel("questions averaged per prefix (N)")
        if col == 0:
            ax.set_ylabel("monitoring correlation r\n(supervised probe vs judge score)")
        ax.set_title(
            f"{TRAIT_TITLES[trait]}: context-averaging curve vs the single prefix-end read",
            loc="left",
        )

    # ---- one shared legend for the whole figure ----
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=arm_colors["prefix_end"]),
        plt.Rectangle((0, 0), 1, 1, color=arm_colors["averaged_context"]),
        plt.Line2D([], [], ls="--", lw=1.2, color=ceiling_color),
    ]
    labels = [
        ARM_LABELS["prefix_end"],
        ARM_LABELS["averaged_context"],
        "judge-score reliability ceiling",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        "Trait monitoring from one pre-query vector vs question-averaged context reads\n"
        "(#1092 dense core: 149 real conversation prefixes × 48 queries, instruct model, layer 14)",
        x=0.01,
        ha="left",
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.925))
    savefig_paper(fig, "issue_1092/prefixend_monitoring_combined", dir=str(ROOT / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    main()
