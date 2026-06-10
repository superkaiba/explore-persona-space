# Greek ΔG, marker, long lines OK in plot script
"""Overlay figure for the #477 follow-up (negative-panel-self-leakage).

Inputs:
  - eval_results/issue_477/reval_grid/grid.json (35 cells, repo root, on main):
      held_out_delta_g_mean = BYSTANDER mean (panel excluded every trained negative)
      source_self_delta_g_mean = SOURCE persona ΔG on the same adapter
  - eval_results/issue_477/negpanel_eval/negpanel_grid.json (this worktree):
      panel_delta_g_mean = TRAINED-NEGATIVE mean (each cell scored on its own
        trained-negative panel)

Outputs (figures/issue_477/):
  - negpanel_overlay.{png,pdf,meta.json} — three curves (source, bystander,
    trained negatives) vs negative-persona count, faceted by LoRA rank, for
    the low-rank phase at LR=2e-6.

We deliberately omit the calib (LR-lever) cells from the overlay: they
saturate emission at the trained negatives (0.23-1.00) and the four floats
the marker-leakage contract reports are mostly at the saturation ceiling
there, so a "vs count" line drawn through them is rank-shuffling among
near-equal values. The calib cells get a separate fenced summary in the
body's finding prose instead.

All figures use the "blog" paper-plots style.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BYSTANDER_GRID = REPO_ROOT / "eval_results" / "issue_477" / "reval_grid" / "grid.json"
NEGPANEL_GRID = REPO_ROOT / "eval_results" / "issue_477" / "negpanel_eval" / "negpanel_grid.json"


def load_grids() -> tuple[dict, dict]:
    """Return (bystander_rows_by_adapter, negpanel_rows_by_adapter)."""
    with open(BYSTANDER_GRID) as f:
        b = json.load(f)
    with open(NEGPANEL_GRID) as f:
        n = json.load(f)
    by_b = {r["adapter_dirname"]: r for r in b["rows"]}
    by_n = {r["adapter_dirname"]: r for r in n["rows"]}
    return by_b, by_n


def fig_negpanel_overlay() -> None:
    by_b, by_n = load_grids()
    ranks = [2, 4, 8]
    counts = [2, 4, 8, 16]

    # Phase = calA, LR = 2e-6, the 12 low-rank cells.
    rows_b = [r for r in by_b.values() if r["phase"] == "calA" and r["lr"] == 2e-6]
    rows_n = [r for r in by_n.values() if r["phase"] == "calA" and r["lr"] == 2e-6]

    set_paper_style("blog")
    # constrained_layout (default from set_paper_style) collides with
    # subplots_adjust + manual fig.suptitle in a subplot grid (memory note).
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.6), sharey=True)

    palette = paper_palette_blog(3)
    color_source = palette[0]  # warm primary
    color_bystander = palette[1]  # second
    color_negative = palette[2]  # third

    for i, rank in enumerate(ranks):
        ax = axes[i]
        src_ys = []
        bys_ys = []
        neg_ys = []
        for c in counts:
            cell_b = next(r for r in rows_b if r["rank"] == rank and r["count"] == c)
            cell_n = next(r for r in rows_n if r["rank"] == rank and r["count"] == c)
            src_ys.append(cell_b["source_self_delta_g_mean"])
            bys_ys.append(cell_b["held_out_delta_g_mean"])
            neg_ys.append(cell_n["panel_delta_g_mean"])

        ax.plot(
            counts,
            src_ys,
            marker="o",
            color=color_source,
            label="Source (the trained persona)",
            linewidth=1.8,
            markersize=6,
        )
        ax.plot(
            counts,
            bys_ys,
            marker="s",
            color=color_bystander,
            label="Bystanders (held-out, not in any cell's negatives)",
            linewidth=1.8,
            markersize=6,
        )
        ax.plot(
            counts,
            neg_ys,
            marker="^",
            color=color_negative,
            label="Trained negatives (this cell's own negative panel)",
            linewidth=1.8,
            markersize=6,
            linestyle="--",
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks(counts)
        ax.set_xticklabels([str(c) for c in counts])
        ax.set_xlabel("Number of contrastive negative personas")
        if i == 0:
            ax.set_ylabel("Mean ΔlogP marker, trained − base (nats)")
        ax.set_ylim(-1.5, 25)
        ax.axhline(0, color="#1A1A1A", linewidth=0.5, linestyle=":", alpha=0.4)
        ax.set_title(
            f"LoRA rank {rank}",
            loc="left",
            fontweight="semibold",
            pad=6,
        )

    # Title via fig.text (NOT suptitle) so it sits flush with the left edge
    # and reserves vertical space without clashing with constrained_layout.
    fig.text(
        0.04,
        0.95,
        "Trained negatives are NOT point-suppressed:",
        ha="left",
        fontweight="semibold",
        fontsize=12,
    )
    fig.text(
        0.04,
        0.905,
        "they track bystanders to within ~0–3 nats across the count bundle.",
        ha="left",
        fontweight="regular",
        fontsize=10,
        alpha=0.85,
    )

    # One legend below the figure (room reserved by bottom margin).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.22, wspace=0.10)
    savefig_paper(fig, "issue_477/negpanel_overlay", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig_negpanel_overlay()
    print("Wrote figures/issue_477/negpanel_overlay.{png,pdf,meta.json}")
