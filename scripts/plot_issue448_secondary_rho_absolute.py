# ruff: noqa: RUF001, RUF002
"""Per-cell Spearman ρ on the ABSOLUTE post-training log-prob DV.

Shows ρ_abs (point + 95% bootstrap CI) for the 6 non-degenerate cells.
Side panel shows the Δ-based ρ for direct comparison — the change in
sign tells the reader the original "negative ρ" was a Δ + ceiling
artifact, not a geometric finding.

Inputs:  eval_results/issue_448/secondary_absolute_summary.json
Outputs: figures/issue_448/secondary_rho_per_cell.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

CELL_LABELS = {
    "c1_anchor": "Anchor (villain, 1×200 pos, 2×200 neg)",
    "c2_pos_ex_100": "+pos-ex/persona = 100",
    "c3_pos_ex_400": "+pos-ex/persona = 400",
    "c4_pos_ex_800": "+pos-ex/persona = 800",
    "c5_pos_personas_2": "+pos personas = 2",
    "c6_pos_personas_4": "+pos personas = 4",
    "c7_neg_ex_100": "+neg-ex/persona = 100",
    "c8_neg_ex_400": "+neg-ex/persona = 400",
    "c9_neg_ex_800": "+neg-ex/persona = 800",
    "c10_neg_personas_4": "+neg personas = 4",
    "c11_neg_personas_8": "+neg personas = 8",
}


def main() -> None:
    summary = json.loads(Path("eval_results/issue_448/secondary_absolute_summary.json").read_text())

    set_paper_style("blog")

    cell_order = [
        "c1_anchor",
        "c2_pos_ex_100",
        "c3_pos_ex_400",
        "c4_pos_ex_800",
        "c5_pos_personas_2",
        "c6_pos_personas_4",
        "c7_neg_ex_100",
        "c8_neg_ex_400",
        "c9_neg_ex_800",
        "c10_neg_personas_4",
        "c11_neg_personas_8",
    ]

    # Disable constrained_layout BEFORE subplots() so subplots_adjust below works
    # (memory: set_title_subtitle_breaks_subplot_grids).
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(13.0, 6.4), gridspec_kw={"wspace": 0.04})

    n = len(cell_order)
    y_positions = np.arange(n)[::-1]

    primary = paper_palette_role("primary")
    accent = paper_palette_role("accent")
    neutral = paper_palette_role("neutral")
    baseline = paper_palette_role("baseline")

    def plot_panel(ax, key: str, predicted_text: str, xlim_left: float = -1.0) -> None:
        for i, cell in enumerate(cell_order):
            y = y_positions[i]
            sec = summary["per_cell"][cell].get(key, {})
            rho = sec.get("rho_point")
            lo = sec.get("rho_ci_low")
            hi = sec.get("rho_ci_high")
            if rho is None:
                ax.scatter([0.0], [y], marker="x", s=70, color=neutral, zorder=3)
                ax.text(
                    -0.95,
                    y,
                    "degenerate (cosine spread too small to estimate)",
                    fontsize=8.5,
                    color=neutral,
                    va="center",
                    ha="left",
                )
                continue
            straddles_zero = lo < 0 < hi
            color = accent if straddles_zero else primary
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, alpha=0.9, zorder=2)
            cap_height = 0.18
            ax.plot([lo, lo], [y - cap_height, y + cap_height], color=color, linewidth=1.2)
            ax.plot([hi, hi], [y - cap_height, y + cap_height], color=color, linewidth=1.2)
            ax.scatter([rho], [y], s=46, color=color, zorder=3)
        ax.axvline(0, color="#9a9a9a", linestyle="--", linewidth=0.9, zorder=1)
        ax.set_xlim(xlim_left, 1.0)
        ax.grid(True, axis="x", alpha=0.18, linewidth=0.6)

    plot_panel(ax_abs, "spearman_absolute", "predicted ρ > 0", xlim_left=-1.0)
    ax_abs.set_yticks(y_positions)
    ax_abs.set_yticklabels([CELL_LABELS[c] for c in cell_order], fontsize=9)
    ax_abs.set_xlabel(
        "Spearman ρ on absolute post log p(marker)\n"
        "(positive = farther from negative → higher emission, as originally predicted)"
    )

    plot_panel(ax_delta, "spearman_delta", "previously reported", xlim_left=-1.0)
    ax_delta.set_yticks(y_positions)
    ax_delta.set_yticklabels([])
    ax_delta.set_xlabel(
        "Spearman ρ on Δ (= post − base) — previously reported\n"
        "(contaminated by ceiling: ρ(base, Δ) ≈ −0.93 at anchor)"
    )

    # Fig-level title / subtitle. set_title_subtitle on a single subplot
    # in a multi-panel layout squashes the grid (memory:
    # set_title_subtitle_breaks_subplot_grids).
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.27, right=0.98)
    fig.text(
        0.27,
        0.94,
        "Switching the DV from Δ to absolute post log p(marker) collapses the secondary finding",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.27,
        0.85,
        "Per-cell Spearman ρ ± 95% bootstrap CI. Left: absolute post log p(marker), the "
        "operational leakage measure.\n"
        "Right: Δ = post − base, the previously reported metric "
        "(ρ(base, Δ) ≈ −0.93 at anchor — Δ is anti-correlated with base prior\n"
        "by ceiling saturation). Orange = CI crosses zero. Three cells "
        "(+pos personas = 2, +pos personas = 4, +neg personas = 8) are degenerate by\n"
        "the §4.2.5 cosine-spread guard and skipped.",
        fontsize=9.5,
        color="#555",
        ha="left",
    )

    savefig_paper(fig, "issue_448/secondary_rho_per_cell", dir="figures/")
    plt.close(fig)
    print("Wrote figures/issue_448/secondary_rho_per_cell.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
