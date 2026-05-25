"""Heatmap of [ZLT] marker firing across the 24-persona eval panel for #383.

Three side-by-side panels, one per source persona. Each cell is one of 24
trained recipes; each column is one of the 24 panel personas (source listed
first, then 23 bystanders alphabetical). Color is the marker substring rate
[0, 1].

Reads the 72 per-cell metrics.json files from the issue-383 worktree.

Run from repo root:

    uv run python scripts/plot_issue383_heatmap.py
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


def load_cells_for_source(source: str) -> list[dict]:
    """Return [{cell, source_rate, panel: {persona: rate}}] for one source."""
    out = []
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
        panel_scores = m.get("persona_panel_scores", {})
        panel = {p: float(panel_scores[p]["substring_rate"]) for p in panel_scores}
        out.append(
            {
                "cell": cell_key,
                "source_rate": float(m["source_substring_rate"]),
                "panel": panel,
            }
        )
    out.sort(key=lambda r: r["source_rate"], reverse=True)
    return out


def main() -> None:
    set_paper_style("blog")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18.0, 10.0),
        gridspec_kw={"wspace": 0.55, "top": 0.88, "bottom": 0.22, "left": 0.05, "right": 0.94},
        constrained_layout=False,
    )

    cmap = plt.get_cmap("YlOrRd")
    cmap.set_under("#f5f5f5")

    for ax, source in zip(axes, SOURCES, strict=True):
        rows = load_cells_for_source(source)
        if not rows:
            ax.set_title(f"{source}: no data")
            continue

        # Column order: source persona first, then 23 bystanders sorted alphabetically.
        all_personas = sorted(rows[0]["panel"].keys())
        bystanders = [p for p in all_personas if p != source]
        columns = [source] + bystanders

        # Matrix: cells (rows) x personas (cols), values in [0, 1].
        mat = np.zeros((len(rows), len(columns)), dtype=float)
        for i, row in enumerate(rows):
            for j, p in enumerate(columns):
                mat[i, j] = row["panel"].get(p, 0.0)

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=1.0,
            interpolation="nearest",
        )

        # Column ticks: only label every 2nd, but always label the source (col 0).
        ax.set_xticks(np.arange(len(columns)))
        labels = []
        for j, p in enumerate(columns):
            if j == 0 or j % 2 == 0:
                labels.append(p.replace("_", " "))
            else:
                labels.append("")
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
        # Highlight source persona xtick.
        ax.get_xticklabels()[0].set_fontweight("bold")
        ax.get_xticklabels()[0].set_color("#0033a0")

        # Row ticks: cell keys (5-bit) — only show some so they're readable.
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(
            [f"{r['cell']}  ({r['source_rate']:.2f})" for r in rows],
            fontsize=8,
        )
        ax.set_ylabel("cell key (source rate)", fontsize=9)

        # Vertical line separating source column from bystanders + box around source col.
        ax.axvline(0.5, color="#0033a0", lw=1.5)
        ax.axvline(-0.5, color="#0033a0", lw=1.5)

        ax.set_title(
            f"source = {source}  (n=24 cells, ordered by source rate ↓)",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )

    # One shared colorbar on the right.
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.012,
        pad=0.02,
        shrink=0.7,
    )
    cbar.set_label("[ZLT] substring rate", fontsize=9)

    fig.suptitle(
        "[ZLT] firing across the 24-persona eval panel for each trained cell, by source persona",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.005,
        0.005,
        "source: eval_results/issue_383/cell_*/source_*/seed_42/metrics.json (persona_panel_scores) — "
        "first column of each panel is the trained source persona (highlighted in blue); remaining 23 are bystanders.",
        fontsize=7.5,
        color="#666666",
    )
    out_dir = Path("figures")
    savefig_paper(fig, "issue_383/panel_heatmap", dir=str(out_dir))
    plt.close(fig)
    print("saved figures/issue_383/panel_heatmap.png + .pdf + .meta.json")


if __name__ == "__main__":
    main()
