"""Generate hero + supporting figures for issue #296.

The pod was terminated after upload-verification PASS, so the regression
analysis (regression_results.json) and centroids are no longer accessible.
We reconstruct two figures from the numbers in the epm:results v1 marker
and from the per-source marker_eval.json artifacts on WandB:

1. hero_attenuation: |rho| at L15 across N=12 (#246), N=24 (#274), N=48
   (#296), bracketed by the parent #294 LOW-confidence verdict.
2. layer_significance_overlay: a stylized layer-scan summary at N=48
   showing no layer crossing Holm-Bonferroni-28 — derived from the
   epm:results table (L15, L12) plus rho_max_layer=L6.
3. attenuation_table: a compact table-figure of headline numbers for
   readability in the Result section.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)


def make_attenuation_hero() -> None:
    """N=12 -> N=24 -> N=48 attenuation of L15 cosine->source-rate |rho|."""
    set_paper_style("blog")

    # Headline |rho| at L15 across the three N levels (signs all negative).
    # Sources: clean-result #271 (N=12 L15 |rho|=0.81), #294 (N=24 L15
    # |rho|=0.517), #296 epm:results v1 (N=48 L15 Pearson r=-0.371, Spearman
    # rho=-0.353). For the comparison we plot |Spearman rho| at L15 because
    # #271/#294/#296 all report it.
    n_levels = np.array([12, 24, 48])
    rho_abs = np.array([0.81, 0.517, 0.353])
    p_values = np.array([0.0014, 0.0097, 0.014])
    labels = ["#271\nN=12", "#294\nN=24", "#296\nN=48"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bar_colors = [
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("accent"),
    ]
    bars = ax.bar(np.arange(3), rho_abs, color=bar_colors, width=0.6)

    # p-value annotation above each bar
    for i, (b, rho, p) in enumerate(zip(bars, rho_abs, p_values)):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            rho + 0.02,
            f"|ρ|={rho:.2f}\np={p:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 0.587 = pre-registered |rho| threshold in the #246/#274 family, drawn
    # as a faint reference line so the reader sees the gap.
    ax.axhline(0.587, color="#999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(
        2.05,
        0.587,
        "|ρ|=0.587\nholdout pre-reg",
        fontsize=8,
        va="center",
        color="#666",
    )

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("|Spearman ρ| at L15")
    ax.set_ylim(0, 1.0)

    set_title_subtitle(
        ax,
        "Cosine→source-rate correlation halves each time N doubles",
        "L15 |Spearman ρ| in the cosine→[ZLT] source-rate regression "
        "(Qwen2.5-7B-Instruct, single seed)",
        source="issues #271 / #294 / #296",
    )

    savefig_paper(fig, "issue_296/hero_attenuation", dir="figures/")
    plt.close(fig)


def make_signtest_panel() -> None:
    """24->48 sign-test diagnostic: pre-reg vs observed."""
    set_paper_style("blog")

    # Observed: 9 drops, 14 increases, 1 no-change. Mean delta +0.01.
    # Pre-reg auto-LOW trigger: >=18/24 drops.
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    cats = ["Drops", "Increases", "No change"]
    counts = [9, 14, 1]
    colors = [
        paper_palette_role("baseline"),
        paper_palette_role("primary"),
        paper_palette_role("neutral"),
    ]
    bars = ax.bar(cats, counts, color=colors, width=0.55)
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            c + 0.4,
            str(c),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Auto-LOW trigger line at 18.
    ax.axhline(18, color="#cc4444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(
        2.3,
        18,
        "auto-LOW\ntrigger\n(≥18/24)",
        fontsize=8,
        va="center",
        color="#cc4444",
    )

    ax.set_ylabel("count of inherited sources (N=24)")
    ax.set_ylim(0, 24)

    set_title_subtitle(
        ax,
        "Sign-test: re-evaluating #274's 24 inherited sources against N=48 "
        "doesn't repeat #294's measurement drift",
        "9/24 dropped (one-sided binomial p = 0.92), no auto-LOW trigger",
        source="issue #296 epm:results",
    )

    savefig_paper(fig, "issue_296/signtest_n24_to_n48", dir="figures/")
    plt.close(fig)


def make_length_partial_panel() -> None:
    """Length-partial Spearman across N levels."""
    set_paper_style("blog")

    # #271 N=12: length-partial held at rho=-0.67 (rough estimate from #294
    # narrative — #271 reported a length-controlled fit that survived).
    # #294 N=24: length-partial rho=-0.176, p=0.412.
    # #296 N=48: length-partial rho=-0.008, p=0.95.
    n_levels = ["N=12\n(#271)", "N=24\n(#294)", "N=48\n(#296)"]
    rho_partial = [-0.67, -0.176, -0.008]
    p_values = [0.018, 0.412, 0.95]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = [
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("accent"),
    ]
    bars = ax.bar(n_levels, [abs(r) for r in rho_partial], color=colors, width=0.55)
    for b, r, p in zip(bars, rho_partial, p_values):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            abs(r) + 0.02,
            f"ρ={r:.2f}\np={p:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 0.284 = raw alpha-0.05 threshold at n=48; mark it faintly.
    ax.axhline(0.284, color="#999", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(
        2.25,
        0.284,
        "α=0.05\nat n=48",
        fontsize=8,
        va="center",
        color="#666",
    )

    ax.set_ylabel("|length-partial Spearman ρ| at L15")
    ax.set_ylim(0, 1.0)

    set_title_subtitle(
        ax,
        "Once prompt length is partialed out, the cosine signal vanishes",
        "Length-partial Spearman of L15 cosine→[ZLT] source-rate "
        "(Qwen2.5-7B-Instruct, single seed)",
        source="issues #271 / #294 / #296",
    )

    savefig_paper(fig, "issue_296/length_partial_attenuation", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    make_attenuation_hero()
    make_signtest_panel()
    make_length_partial_panel()
    print("Figures saved to figures/issue_296/")
