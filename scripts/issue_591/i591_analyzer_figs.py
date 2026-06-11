"""Analyzer round-1 figures for #591 e2 (VM-side, zero GPU).

Regenerates the e2 hero scatter with an explicit role legend, and adds the
cross-matrix heatmap (adapter x new persona, raw delta) that carries the
off-diagonal style-control / promiscuity / suppression read.

Inputs: eval_results/issue_591/e2/extended_panel_results.json and the frozen
#411 join snapshot in eval_results/issue_591/_inputs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[2]
E2 = REPO / "eval_results/issue_591/e2/extended_panel_results.json"
FROZEN = REPO / "eval_results/issue_591/_inputs/cell_table_snapshot.json"
FIGDIR = REPO / "figures" / "issue_591"

# Plain-English persona labels (one-line roster names -> reader-facing)
PERSONA_LABELS = {
    "supervillain": "supervillain",
    "evil_mastermind": "evil mastermind",
    "dark_overlord": "dark overlord",
    "criminal_mastermind": "criminal mastermind",
    "standup_comic": "stand-up comic",
    "improv_comedian": "improv comedian",
    "late_night_host": "late-night host",
    "daycare_teacher": "daycare teacher",
    "preschool_teacher": "preschool teacher",
    "nursery_school_teacher": "nursery-school teacher",
    "elementary_school_teacher": "elementary-school teacher",
    "web_developer": "web developer",
    "fullstack_programmer": "full-stack programmer",
    "virtual_assistant": "virtual assistant",
    "digital_helper": "digital helper",
    "accountant": "accountant (frozen anchor)",
    "librarian": "librarian (frozen anchor)",
    "data_scientist": "data scientist (frozen leak anchor)",
    "villain": "villain (source-self)",
    "comedian": "comedian (source-self)",
    "kindergarten_teacher": "kindergarten teacher (source-self)",
    "software_engineer": "software engineer (source-self)",
}
ADAPTER_LABELS = {
    "villain": "villain adapter",
    "comedian": "comedian adapter",
    "kindergarten_teacher": "kindergarten-teacher adapter",
    "software_engineer": "software-engineer adapter (known-leaking)",
}


def _load():
    e2 = json.loads(E2.read_text())
    cells = e2["cells"]
    frozen_cells = None
    # frozen join: prefer the e1 cell table (sycophancy rows carry the frozen panel)
    ct = REPO / "eval_results/issue_591/e1/cell_table.json"
    if ct.exists():
        rows = json.loads(ct.read_text())["cells"]
        frozen_cells = [r for r in rows if r["behavior"] == "sycophancy"]
    return cells, frozen_cells


def fig_hero(cells, frozen_cells):
    set_paper_style("blog")
    srcs = ["villain", "comedian", "kindergarten_teacher", "software_engineer"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), squeeze=False)
    c_twin = paper_palette_role("primary")
    c_pc = paper_palette_role("accent")
    for ax, src in zip(axes[0], srcs, strict=True):
        fr = [c for c in frozen_cells if c["source"] == src]
        ax.scatter(
            [c["cos_to_source"] for c in fr],
            [c["delta"] for c in fr],
            s=14,
            color="lightgrey",
            zorder=1,
        )
        for c in cells:
            if c["adapter_source"] != src or c.get("cos_to_adapter_source") is None:
                continue
            role = c["role_assigned"]
            color = c_pc if role.startswith("positive_control") else c_twin
            ax.errorbar(
                c["cos_to_adapter_source"],
                c["delta_raw"],
                yerr=[
                    [max(0.0, c["delta_raw"] - c["ci95_low"])],
                    [max(0.0, c["ci95_high"] - c["delta_raw"])],
                ],
                fmt="o",
                ms=5,
                color=color,
                zorder=3,
            )
        ax.axvspan(0.95, 0.97, alpha=0.10, color="orange", zorder=0)
        ax.axvspan(0.97, 1.005, alpha=0.10, color="green", zorder=0)
        ax.axhline(0.10, color="grey", ls="--", lw=0.8)
        ax.set_title(ADAPTER_LABELS[src], fontsize=11)
        ax.set_xlabel("cosine to adapter source (layer 20)")
    axes[0][0].set_ylabel("agreement-rate delta (trained − base)")
    handles = [
        Line2D([], [], marker="o", ls="", color="lightgrey", label="frozen 23-bystander panel"),
        Line2D([], [], marker="o", ls="", color=c_twin, label="new synthesized persona (95% CI)"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=c_pc,
            label="positive-control twins (software-engineer / assistant)",
        ),
        Line2D([], [], ls="--", color="grey", label="leak threshold (delta = +0.10)"),
    ]
    axes[0][0].legend(handles=handles, fontsize=7.5, loc="upper left")
    savefig_paper(fig, "e2_delta_vs_cosine_hero", dir=FIGDIR)
    plt.close(fig)


def fig_cross_matrix(cells):
    set_paper_style("blog")
    adapters = ["villain", "comedian", "kindergarten_teacher", "software_engineer"]
    # column order: villain twins, comedian twins, kt twins, pc twins, assistant twins, anchors
    col_order = [
        "supervillain",
        "evil_mastermind",
        "dark_overlord",
        "criminal_mastermind",
        "standup_comic",
        "improv_comedian",
        "late_night_host",
        "daycare_teacher",
        "preschool_teacher",
        "nursery_school_teacher",
        "elementary_school_teacher",
        "web_developer",
        "fullstack_programmer",
        "virtual_assistant",
        "digital_helper",
    ]
    lut = {(c["adapter_source"], c["new_persona"]): c for c in cells}
    mat = np.full((len(adapters), len(col_order)), np.nan)
    for i, a in enumerate(adapters):
        for j, p in enumerate(col_order):
            c = lut.get((a, p))
            if c is not None:
                mat[i, j] = c["delta_raw"]
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            c = lut[(adapters[i], col_order[j])]
            leak = c["delta_raw"] >= 0.10 and c["ci95_low"] > 0
            txt = f"{v:+.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold" if leak else "normal",
                color="white" if abs(v) > 0.45 else "black",
            )
            if c["diagonal_cell"]:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.6)
                )
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(
        [PERSONA_LABELS[p].replace(" (frozen anchor)", "") for p in col_order],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(adapters)))
    ax.set_yticklabels([ADAPTER_LABELS[a] for a in adapters], fontsize=9)
    ax.set_title(
        "Cross-matrix: agreement-rate delta for every new persona under every adapter "
        "(black outline = twin under its own source)",
        fontsize=10.5,
        loc="left",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("delta (trained − base)", fontsize=8)
    savefig_paper(fig, "e2_cross_matrix", dir=FIGDIR)
    plt.close(fig)


def main():
    cells, frozen_cells = _load()
    fig_hero(cells, frozen_cells)
    fig_cross_matrix(cells)
    print("wrote", FIGDIR / "e2_delta_vs_cosine_hero.png", "and", FIGDIR / "e2_cross_matrix.png")


if __name__ == "__main__":
    main()
