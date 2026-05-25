"""Per-cell heatmap of source-implantation rate vs mean bystander leakage for #383.

Three panels (one per source persona). Each panel is a 24-row matrix with two
columns: source rate (left) and mean bystander leakage (right). Rows are
ordered by source rate descending. Row labels spell out the recipe in
plain English so you can scan and see which factor combinations produce
high source / low leakage.

Run from repo root:

    uv run python scripts/plot_issue383_per_cell_heatmap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    savefig_paper,
    set_paper_style,
)

EVAL_ROOT = Path(".claude/worktrees/issue-383/eval_results/issue_383")
SOURCES = ["librarian", "programmer", "surgeon"]

# Compact plain-English label per (axis, bit). Axis order in cell key = A B C D E.
FACTOR_LEGEND = {
    "A": {"1": "long-sys", "0": "short-sys"},
    "B": {"1": "long-ans", "0": "short-ans"},
    "C": {"1": "neutral", "0": "persona"},
    "D": {"1": "Claude", "0": "base-Qwen"},
    "E": {"1": "whole-loss", "0": "marker-loss"},
}
AXES = ["A", "B", "C", "D", "E"]


def recipe_label(cell_key: str) -> str:
    parts = [FACTOR_LEGEND[axis][bit] for axis, bit in zip(AXES, cell_key, strict=True)]
    return " · ".join(parts)


def load_cells_for_source(source: str) -> list[dict]:
    rows = []
    for cell_dir in sorted(EVAL_ROOT.glob("cell_*")):
        cell_key = cell_dir.name.removeprefix("cell_")
        if len(cell_key) != 5 or not all(c in "01" for c in cell_key):
            continue
        mfile = cell_dir / f"source_{source}" / "seed_42" / "metrics.json"
        if not mfile.exists():
            continue
        m = json.loads(mfile.read_text())
        if m.get("failed"):
            continue
        rows.append(
            {
                "cell": cell_key,
                "label": recipe_label(cell_key),
                "source_rate": float(m["source_substring_rate"]),
                "leakage_rate": float(m["leakage_rate_full"]),
            }
        )
    rows.sort(key=lambda r: r["source_rate"], reverse=True)
    return rows


def main() -> None:
    set_paper_style("blog")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 22.0),
        gridspec_kw={
            "hspace": 0.55,
            "top": 0.965,
            "bottom": 0.035,
            "left": 0.32,
            "right": 0.92,
        },
        constrained_layout=False,
    )

    cmap_src = plt.get_cmap("Blues")
    cmap_lk = plt.get_cmap("Reds")

    for ax, source in zip(axes, SOURCES, strict=True):
        rows = load_cells_for_source(source)
        n = len(rows)

        mat_src = np.array([[r["source_rate"]] for r in rows])  # n x 1
        mat_lk = np.array([[r["leakage_rate"]] for r in rows])  # n x 1
        combined = np.hstack([mat_src, mat_lk])  # n x 2 — just for axes layout

        # Plot each column with its own colormap by using imshow twice.
        ax.imshow(
            mat_src,
            extent=(-0.5, 0.5, n - 0.5, -0.5),
            aspect="auto",
            cmap=cmap_src,
            vmin=0,
            vmax=1.0,
            interpolation="nearest",
        )
        im_lk = ax.imshow(
            mat_lk,
            extent=(0.5, 1.5, n - 0.5, -0.5),
            aspect="auto",
            cmap=cmap_lk,
            vmin=0,
            vmax=1.0,
            interpolation="nearest",
        )

        # Annotate each cell with its rate value.
        for i in range(n):
            sv = rows[i]["source_rate"]
            lv = rows[i]["leakage_rate"]
            ax.text(
                0,
                i,
                f"{sv:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if sv > 0.55 else "#222",
                fontweight="bold" if sv > 0.9 else "normal",
            )
            ax.text(
                1,
                i,
                f"{lv:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if lv > 0.55 else "#222",
                fontweight="bold" if lv > 0.9 else "normal",
            )

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["source\nrate", "bystander\nleakage"], fontsize=9)
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

        ax.set_yticks(np.arange(n))
        ax.set_yticklabels([r["label"] for r in rows], fontsize=8.5, family="monospace")

        # Light gridlines between cells.
        ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)

        # Bold separator between the two columns.
        ax.axvline(0.5, color="#444", lw=1.5)

        ax.set_title(
            f"source = {source}\n(24 cells, ordered by source rate ↓)",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )

    fig.suptitle(
        "Per-cell source vs bystander-leakage, by recipe (one panel per source persona)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    out_dir = Path("figures")
    savefig_paper(fig, "issue_383/per_cell_heatmap", dir=str(out_dir))
    plt.close(fig)
    print("saved figures/issue_383/per_cell_heatmap.png + .pdf + .meta.json")


if __name__ == "__main__":
    main()
