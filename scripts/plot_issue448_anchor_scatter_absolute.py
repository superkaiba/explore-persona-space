# ruff: noqa: RUF001, RUF002
"""Anchor-cell raw scatter on the ABSOLUTE post-training log-prob DV.

x-axis: cosine distance to the nearest trained contrastive negative
        (medical_doctor or police_officer) at layer 20.
y-axis: per-bystander mean log p( ※) at end_of_canonical_response,
        AFTER anchor recipe training (absolute, not Δ).

Annotates comedian + french_person — the two HIGHEST-base-prior
bystanders that on Δ were the two lowest-leakage outliers driving
ρ_Δ ≈ −0.55. On absolute post they should sit mid-pack (rank 15/23
and 19/23 from the high-emission side), no longer bottom outliers.

Inputs:  eval_results/issue_448/secondary_absolute_summary.json
Outputs: figures/issue_448/secondary_anchor_scatter_raw.{png,pdf,meta.json}
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


def main() -> None:
    summary = json.loads(Path("eval_results/issue_448/secondary_absolute_summary.json").read_text())
    anchor = summary["per_cell"]["c1_anchor"]

    bystanders = anchor["bystanders"]
    x = np.array(anchor["x_nearest_neg_distance"])
    y = np.array(anchor["y_post_abs_logp"])
    rho_abs = anchor["spearman_absolute"]["rho_point"]
    ci_lo = anchor["spearman_absolute"]["rho_ci_low"]
    ci_hi = anchor["spearman_absolute"]["rho_ci_high"]

    set_paper_style("blog")

    primary = paper_palette_role("primary")
    accent = paper_palette_role("accent")
    neutral = paper_palette_role("neutral")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Plot all bystanders.
    ax.scatter(x, y, s=58, color=primary, edgecolor="white", linewidth=0.6, zorder=3)

    # Annotate the load-bearing personas: comedian, french_person (the two
    # highest-base-prior outliers); lawyer (the highest-base-prior LOW-prior
    # bystander, ranks low on post too); philosopher (the highest-post-abs).
    annotated = {
        "comedian": (10, -10),
        "french_person": (-12, 14),
        "lawyer": (12, -10),
        "child": (10, 10),
        "philosopher": (-10, -10),
    }
    for label, (dx, dy) in annotated.items():
        if label in bystanders:
            i = bystanders.index(label)
            ax.annotate(
                label,
                (x[i], y[i]),
                xytext=(x[i] + dx * 0.005, y[i] + dy * 0.04),
                fontsize=9,
                color="#222",
                ha="left",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#888",
                    "lw": 0.7,
                    "shrinkA": 0,
                    "shrinkB": 2,
                },
            )

    ax.set_xlabel(
        "Cosine distance to nearest trained contrastive negative\n"
        "(medical_doctor or police_officer; layer-20 centroids)"
    )
    ax.set_ylabel(
        "Absolute post-training mean log p(marker) at end of canonical response (nats)\n"
        "(higher y = more emission; ceiling at 0; anchor mean = −1.18)"
    )
    ax.grid(True, alpha=0.18, linewidth=0.6)

    # Annotate the ρ_abs in a corner.
    ax.text(
        0.97,
        0.05,
        f"Spearman ρ = {rho_abs:+.2f}\n95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]\nN = {len(bystanders)} bystanders",
        transform=ax.transAxes,
        fontsize=10,
        color="#333",
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.4",
            "edgecolor": "#bbb",
            "facecolor": "white",
            "alpha": 0.92,
        },
    )

    set_title_subtitle(
        ax,
        "On absolute post log p, comedian + french_person sit mid-pack, not bottom-outlier",
        "Anchor cell, 23 held-out bystanders. The two highest-base-prior bystanders (comedian "
        "and french_person) ranked 22/23 and 23/23 on Δ — driving the −0.55 ρ_Δ. On absolute "
        "post they rank 15/23 and 19/23 from the high-emission side: ordinary, not outliers.",
        source="task #448, anchor cell, secondary_absolute_summary.json",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_448/secondary_anchor_scatter_raw", dir="figures/")
    plt.close(fig)
    print("Wrote figures/issue_448/secondary_anchor_scatter_raw.{png,pdf,meta.json}")

    # Print the ranking tables for confirmation.
    print()
    print("Ranking on absolute post log p (low emission → high):")
    for i, idx in enumerate(np.argsort(y)):
        marker = " <<< " if bystanders[idx] in ("comedian", "french_person") else ""
        print(f"  rank {i + 1:2d}: {bystanders[idx]:25s} post={y[idx]:+.3f}{marker}")


if __name__ == "__main__":
    main()
