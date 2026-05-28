"""Plot issue #382 clean-result figures.

Hero figure: per-cell marker fire-rate after Phase 1 (install) and Phase 2
(benign SFT), showing the Phase 1 → Phase 2 erasure across the assistant+
trigger headline cell.

Reads: eval_results/issue382/summary.json
Writes: figures/issue_382/hero.png + .pdf + .meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parent.parent
SUMMARY = REPO / "eval_results" / "issue382" / "summary.json"

# Plain-English cell labels for the figure (no project-internal A/B/C/D/D'/F/F' codes).
CELL_ORDER = [
    "A_phase1_assistant_trigger",
    "C_phase1_assistant_no_trigger",
    "D_phase1_villain_trigger",
    "Dprime_phase1_kt_trigger",
    "B_phase2_assistant_trigger",
    "F_phase2_assistant_no_trigger",
    "Fprime_phase2_villain_trigger",
]
CELL_LABELS = {
    "A_phase1_assistant_trigger": "Assistant\n+ trigger",
    "C_phase1_assistant_no_trigger": "Assistant\nno trigger",
    "D_phase1_villain_trigger": "Villain\n+ trigger",
    "Dprime_phase1_kt_trigger": "K-teacher\n+ trigger",
    "B_phase2_assistant_trigger": "Assistant\n+ trigger",
    "F_phase2_assistant_no_trigger": "Assistant\nno trigger",
    "Fprime_phase2_villain_trigger": "Villain\n+ trigger",
}

SEEDS = ["42", "137", "256"]


def load_data():
    with open(SUMMARY) as f:
        summary = json.load(f)
    # mean per cell across seeds, plus pooled Wilson CI on the union (3 seeds × 600 = 1800).
    out = {}
    for cell in CELL_ORDER:
        per_seed_rates = [summary[s]["marker"][cell]["fire_rate"] for s in SEEDS]
        total_fire = sum(summary[s]["marker"][cell]["n_fire"] for s in SEEDS)
        total_n = sum(summary[s]["marker"][cell]["n_total"] for s in SEEDS)
        pooled_rate = total_fire / total_n
        lo, hi = proportion_ci(pooled_rate, total_n)
        out[cell] = {
            "per_seed": per_seed_rates,
            "pooled_rate": pooled_rate,
            "pooled_n": total_n,
            "lo": lo,
            "hi": hi,
            "n_fire": total_fire,
        }
    return out


def plot_hero(data, out_path: str):
    set_paper_style("blog", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    n_cells = len(CELL_ORDER)
    x = np.arange(n_cells)
    width = 0.65

    # Color per phase + headline-cell highlight.
    p1_color = paper_palette_role("primary")
    p2_color = paper_palette_role("control")
    headline_color = paper_palette_role("baseline")  # for the headline B cell

    rates = [data[c]["pooled_rate"] for c in CELL_ORDER]
    los = [data[c]["lo"] for c in CELL_ORDER]
    his = [data[c]["hi"] for c in CELL_ORDER]
    err_lo = [r - lo for r, lo in zip(rates, los)]
    err_hi = [hi - r for r, hi in zip(rates, his)]

    colors = []
    for cell in CELL_ORDER:
        if (
            cell == "A_phase1_assistant_trigger"
            or cell.startswith("A_")
            or cell.startswith("C_")
            or cell.startswith("D")
        ):
            colors.append(p1_color)
        elif cell == "B_phase2_assistant_trigger":
            colors.append(headline_color)
        else:
            colors.append(p2_color)

    ax.bar(
        x,
        rates,
        color=colors,
        width=width,
        yerr=[err_lo, err_hi],
        error_kw={"elinewidth": 0.9, "ecolor": "#1A1A1A", "capsize": 3},
        edgecolor="white",
        linewidth=0.5,
    )

    # Overlay per-seed dots so the reader sees per-seed agreement (3 dots per bar).
    for i, cell in enumerate(CELL_ORDER):
        for s_rate in data[cell]["per_seed"]:
            ax.plot(
                x[i],
                s_rate,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgecolor="#1A1A1A",
                markeredgewidth=0.9,
                linestyle="",
                zorder=5,
            )

    # Phase separator (above x-axis line, between cells 3 and 4).
    ax.axvline(3.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)

    # Phase labels — positioned just under the title block, well above bar tops.
    # Use y = 1.02 in data coords (since ylim ends at 1.10) so they sit ABOVE
    # the 1.0 line but BELOW the title.
    ax.text(
        1.5,
        1.04,
        "Phase 1: after marker install",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#333333",
        fontweight="semibold",
    )
    ax.text(
        5.0,
        1.04,
        "Phase 2: after benign medical SFT",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#333333",
        fontweight="semibold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CELL_LABELS[c] for c in CELL_ORDER], fontsize=9.5)
    ax.set_ylabel("Marker [ZLT] fire rate")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    set_title_subtitle(
        ax,
        "Benign Phase 2 SFT completely erases the conditional marker",
        subtitle=(
            "Fire rate per cell, pooled across 3 seeds (n=1800 each, 95% Wilson CI). "
            "Per-seed values overlaid as white dots."
        ),
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.16)
    savefig_paper(fig, out_path, dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    print("Per-cell pooled fire rates (n=1800 across 3 seeds):")
    for cell in CELL_ORDER:
        d = data[cell]
        print(
            f"  {cell:42s} rate={d['pooled_rate']:.4f}  CI=[{d['lo']:.4f}, {d['hi']:.4f}]  "
            f"per_seed={d['per_seed']}"
        )
    plot_hero(data, "issue_382/hero")
    print("\nWrote figures/issue_382/hero.{png,pdf,meta.json}")
