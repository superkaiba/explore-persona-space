"""Regenerate the issue #448 secondary-rho per-cell figure with an honest
title + caption that surfaces the 3-of-8 cells whose 95% bootstrap CIs
straddle zero.

Round-1 v1 figure title: "Bystanders FARTHER from contrastive negatives
leak LESS, not more — opposite of the prediction".

That title is technically true at the point-estimate level (8/8
non-degenerate ρ < 0) but overstates the certainty: the 3 cells with the
widest negative-side cover (c8 neg-ex=400, c9 neg-ex=800, c10
neg-personas=4) have upper CIs > 0. Critic round-1 caught this.

This regen states the truthful version: 5 of 8 cells reliably below zero;
the 3 cells with the widest negative-side cover have CIs that straddle
zero. The c10 collapse is highlighted as the cleanest mechanistic test.

Inputs:  eval_results/issue_448/analyze_summary.json
Outputs: figures/issue_448/secondary_rho_per_cell.{png,pdf,meta.json}
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
    set_title_subtitle,
)

# Plain-English label for each cell (panel-side labels).
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
    summary_path = Path("eval_results/issue_448/analyze_summary.json")
    with summary_path.open() as f:
        summary = json.load(f)

    set_paper_style("blog")

    # Build per-cell rows in canonical c1 -> c11 order.
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

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    n = len(cell_order)
    y_positions = np.arange(n)[::-1]  # top -> bottom rendering matches list order

    primary = paper_palette_role("primary")
    accent = paper_palette_role("accent")
    neutral = paper_palette_role("neutral")

    for i, cell in enumerate(cell_order):
        y = y_positions[i]
        sec = summary["per_cell"][cell].get("secondary_spearman", {})
        rho = sec.get("rho_point")
        lo = sec.get("rho_ci_low")
        hi = sec.get("rho_ci_high")
        if rho is None:
            ax.scatter([0.0], [y], marker="x", s=70, color=neutral, zorder=3)
            ax.text(
                0.05,
                y,
                "degenerate (cosine spread too small to estimate)",
                fontsize=9,
                color=neutral,
                va="center",
                ha="left",
            )
            continue

        straddles_zero = hi > 0
        color = accent if straddles_zero else primary
        # Bar (CI).
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, alpha=0.9, zorder=2)
        # Whisker caps.
        cap_height = 0.18
        ax.plot([lo, lo], [y - cap_height, y + cap_height], color=color, linewidth=1.2)
        ax.plot([hi, hi], [y - cap_height, y + cap_height], color=color, linewidth=1.2)
        # Point.
        ax.scatter([rho], [y], s=46, color=color, zorder=3)

    ax.axvline(0, color="#9a9a9a", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(
        0.02,
        -0.6,
        "predicted direction (ρ > 0)",
        fontsize=8.5,
        color=neutral,
        ha="left",
        va="center",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([CELL_LABELS[c] for c in cell_order], fontsize=9)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Spearman ρ (per-bystander leakage, distance to nearest contrastive negative)")
    ax.grid(True, axis="x", alpha=0.18, linewidth=0.6)

    # Anthropic-blog title block — short, honest. Long-form caption lives in the body.
    set_title_subtitle(
        ax,
        "Point estimates all negative, but CIs straddle zero in the 3 widest-negative-cover cells",
        "Per-cell Spearman ρ ± 95% bootstrap CI. Predicted direction ρ > 0. "
        "Orange = CI crosses zero (c8, c9, c10).",
        source="task #448, analyze_summary.json",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_448/secondary_rho_per_cell", dir="figures/")
    plt.close(fig)
    print("Wrote figures/issue_448/secondary_rho_per_cell.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
