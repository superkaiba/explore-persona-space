"""Regenerate figures for task #411 with plain-English labels + paper-plots style.

Three figures, all destined for the clean-result body:

  1. paired_rho_vs_99 — hero figure. Per-source ρ (this run, with 95%
     bootstrap CI) versus the #99 published reference value, on the same axis.
     Visualizes the partial-replication story directly.
  2. scatter_panels — 2×3 grid of per-bystander Δ sycophancy vs cosine to
     source, one panel per source. Mirrors #99's Figure 1 layout.
  3. self_vs_bystander — per-source bar of self-Δ (training succeeded) vs
     mean-bystander Δ (broad-transfer signal).
  4. combined_delta_vs_cosine — single panel, all 6 sources overlaid.
     Makes the "bystander deltas hug zero across the whole panel" story
     visible in one chart instead of 6 stacked ones.

Run from repo root after `cd worktrees/issue-411`:
    uv run python scripts/figures_issue_411.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# ---------------------------------------------------------------- inputs
SUMMARY = Path("eval_results/issue_411/analyze_summary.json")
OUT_DIR = "figures/"

# Plain-English condition labels. The source slugs come from the data;
# these are the reader-facing strings used in every axis / legend / tick.
SOURCE_LABELS = {
    "villain": "Villain",
    "comedian": "Comedian",
    "assistant": "Generic assistant",
    "qwen_default": "Qwen default",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
}

# Order to display (matches the planner's per-source ordering for continuity)
SOURCE_ORDER = [
    "villain",
    "comedian",
    "assistant",
    "qwen_default",
    "software_engineer",
    "kindergarten_teacher",
]


def load_summary():
    with SUMMARY.open() as f:
        return json.load(f)


def fig_paired_rho_vs_99(data):
    """Hero figure: per-source ρ now vs #99, on one axis with CIs."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    n = len(SOURCE_ORDER)
    x = np.arange(n)
    width = 0.36

    rhos_this = []
    cis_lo, cis_hi = [], []
    rhos_99 = []
    for src in SOURCE_ORDER:
        v = data["per_source"][src]
        r = v["spearman_rho_vs_cosine"]
        rhos_this.append(r)
        cis_lo.append(r - v["bootstrap_ci_lo_2_5"])
        cis_hi.append(v["bootstrap_ci_hi_97_5"] - r)
        rhos_99.append(v["rho_99_reference"])

    c_now = paper_palette_role("primary")
    c_ref = paper_palette_role("baseline")

    ax.bar(
        x - width / 2,
        rhos_this,
        width=width,
        color=c_now,
        label="This run (held-out prompts, 23-panel)",
        yerr=[cis_lo, cis_hi],
        error_kw={"elinewidth": 0.9, "ecolor": "#1A1A1A", "capsize": 3},
    )
    ax.bar(
        x + width / 2,
        rhos_99,
        width=width,
        color=c_ref,
        alpha=0.85,
        label="#99 published (in-distribution, 110-panel)",
    )

    ax.axhline(0.0, color="#5A5A5A", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS[s] for s in SOURCE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ  (bystander Δ sycophancy vs cosine to source)")
    ax.set_ylim(-0.95, 0.95)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    # Mark the ±0.2 replication band around each #99 reference value
    for i, r99 in enumerate(rhos_99):
        ax.plot(
            [i + width / 2 - width / 2.2, i + width / 2 + width / 2.2],
            [r99 - 0.2, r99 - 0.2],
            color="#888888",
            lw=0.5,
            ls=":",
        )
        ax.plot(
            [i + width / 2 - width / 2.2, i + width / 2 + width / 2.2],
            [r99 + 0.2, r99 + 0.2],
            color="#888888",
            lw=0.5,
            ls=":",
        )

    set_title_subtitle(
        ax,
        "Three of six sources replicate the cosine gradient within ±0.2 of #99",
        subtitle="Two sign flips (Generic assistant, Kindergarten teacher) and one magnitude collapse (Qwen default) on held-out prompts",
        source="Source: eval_results/issue_411/analyze_summary.json",
    )

    savefig_paper(fig, "issue_411/paired_rho_vs_99", dir=OUT_DIR)
    plt.close(fig)


def fig_scatter_panels(data):
    """2×3 grid of per-bystander Δ vs cosine, one panel per source."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.6), sharex=False, sharey=False)
    axes = axes.flatten()

    c_byst = paper_palette_role("primary")
    c_self = paper_palette_role("accent")

    for i, src in enumerate(SOURCE_ORDER):
        ax = axes[i]
        v = data["per_source"][src]
        cos = v["per_panel_cosine_to_source"]
        delta = v["per_panel_delta"]
        rho_now = v["spearman_rho_vs_cosine"]
        rho_99 = v["rho_99_reference"]
        p = v["permutation_p_value"]
        n_byst = v["n_bystanders"]

        # Split self (cosine=1.0) from bystanders
        self_pt = None
        cs, ds = [], []
        for p_name, c in cos.items():
            d = delta[p_name]
            if p_name == src:
                self_pt = (c, d)
            else:
                cs.append(c)
                ds.append(d)

        ax.scatter(cs, ds, color=c_byst, s=28, alpha=0.78, edgecolor="white", lw=0.5)
        if self_pt is not None:
            ax.scatter(
                [self_pt[0]],
                [self_pt[1]],
                color=c_self,
                marker="*",
                s=180,
                edgecolor="white",
                lw=0.8,
                zorder=5,
                label="self",
            )

        ax.axhline(0.0, color="#888888", lw=0.5, ls=":")
        ax.set_xlabel("Cosine to source (layer 20)", fontsize=9)
        ax.set_ylabel("Δ sycophancy vs base", fontsize=9)
        title = SOURCE_LABELS[src]
        sign_marker = "✓" if v["within_replication_tolerance"] else "✗"
        ax.set_title(
            f"{title}  {sign_marker}\nρ={rho_now:+.2f} (#99: {rho_99:+.2f}), p={p:.3f}, n={n_byst}",
            fontsize=9,
            loc="left",
            fontweight="semibold",
        )
        ax.tick_params(axis="both", labelsize=8)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.09, wspace=0.32, hspace=0.55)

    # Manual title block via fig.text (set_title_subtitle squashes subplot grids).
    fig.text(
        0.07,
        0.945,
        "Per-source cosine gradient on held-out wrong claims",
        fontsize=13,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.07,
        0.913,
        "Each blue dot is one of 23 bystander personas; star marks the source-self panel. ✓ = within ±0.2 of #99's published ρ; ✗ = outside.",
        fontsize=9,
        color="#555555",
        ha="left",
    )
    fig.text(
        0.07,
        0.012,
        "Source: eval_results/issue_411/analyze_summary.json",
        fontsize=7,
        color="#888888",
        ha="left",
    )

    savefig_paper(fig, "issue_411/scatter_panels", dir=OUT_DIR)
    plt.close(fig)


def fig_self_vs_bystander(data):
    """Per-source bar of self-Δ vs mean-bystander Δ."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    n = len(SOURCE_ORDER)
    x = np.arange(n)
    width = 0.38

    self_d = [data["per_source"][s]["self_delta"] for s in SOURCE_ORDER]
    mean_byst = [data["per_source"][s]["mean_bystander_delta"] for s in SOURCE_ORDER]

    c_self = paper_palette_role("accent")
    c_byst = paper_palette_role("primary")

    ax.bar(
        x - width / 2, self_d, width=width, color=c_self, label="Source-self Δ (training success)"
    )
    ax.bar(
        x + width / 2,
        mean_byst,
        width=width,
        color=c_byst,
        label="Mean bystander Δ (broad transfer)",
    )
    ax.axhline(0.0, color="#5A5A5A", lw=0.6, ls="--")

    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS[s] for s in SOURCE_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Δ sycophancy rate vs base Qwen")
    ax.set_ylim(-0.15, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        "Training succeeded for every source; mean bystander lift stayed near zero",
        subtitle="Source-self rate rose to 0.65–0.97; mean bystander Δ stayed within ±0.18 of base",
        source="Source: eval_results/issue_411/analyze_summary.json",
    )

    savefig_paper(fig, "issue_411/self_vs_bystander", dir=OUT_DIR)
    plt.close(fig)


def fig_combined_delta_vs_cosine(data):
    """Single panel: per-bystander Δ vs cosine-to-source, all 6 sources overlaid.

    Same y-axis range across sources makes the "bystander deltas hug zero
    across the whole panel" story visible at a glance. Color encodes
    source; one marker per (source, bystander) cell. Source-self points
    omitted — they all sit at cosine=1.0, Δ=+0.65..+0.92 and would
    dominate the y-axis.
    """
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    colors = paper_palette(len(SOURCE_ORDER))

    for src, color in zip(SOURCE_ORDER, colors):
        v = data["per_source"][src]
        per_panel_delta = v["per_panel_delta"]
        per_panel_cos = v["per_panel_cosine_to_source"]
        # Exclude source-self point (cosine == 1.0 by definition)
        xs, ys = [], []
        for panel, delta in per_panel_delta.items():
            if panel == src:
                continue
            xs.append(per_panel_cos[panel])
            ys.append(delta)
        ax.scatter(
            xs,
            ys,
            color=color,
            s=32,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            label=SOURCE_LABELS[src],
        )

    ax.axhline(0.0, color="#5A5A5A", lw=0.6, ls="--")
    ax.set_xlabel("Cosine similarity to source persona (layer 20)")
    ax.set_ylabel("Δ sycophancy rate vs base Qwen (bystander)")
    ax.set_ylim(-0.15, 0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)

    set_title_subtitle(
        ax,
        "Sycophancy barely leaks to bystanders, regardless of cosine distance",
        subtitle=(
            "Per-bystander Δ stays within ±0.10 of base for 132 of 138 "
            "(source, bystander) cells across all 6 sources"
        ),
        source="Source: eval_results/issue_411/analyze_summary.json",
    )

    savefig_paper(fig, "issue_411/combined_delta_vs_cosine", dir=OUT_DIR)
    plt.close(fig)


def main():
    data = load_summary()
    fig_paired_rho_vs_99(data)
    fig_scatter_panels(data)
    fig_self_vs_bystander(data)
    fig_combined_delta_vs_cosine(data)
    print(
        "Wrote figures/issue_411/{paired_rho_vs_99,scatter_panels,self_vs_bystander,combined_delta_vs_cosine}.{png,pdf,meta.json}"
    )


if __name__ == "__main__":
    main()
