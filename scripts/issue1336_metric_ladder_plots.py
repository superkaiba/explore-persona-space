"""Generate figures for the metric-ladder round of #1336.

Reads the aggregated pair data (56 metric_ladder pair files, layer 30) and
renders the hero figure (sufficient-tier map per pair × corpus) plus the
underlying per-unit low-level view.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from explore_persona_space.analysis.paper_plots import (
    set_paper_style,
    savefig_paper,
    paper_palette,
    paper_palette_role,
    set_title_subtitle,
)

set_paper_style("blog")

AGG = json.load(open("/tmp/issue-1336-v2/aggregate.json"))
ROWS = AGG["rows"]
BAND = 0.020709261538715756  # elicit_band_v2 from v2_bars

# Ordering
PAIR_ORDER = [
    ("base__sft", "base → SFT"),
    ("base__dpo", "base → DPO"),
    ("base__rlvr", "base → RLVR"),
    ("base__rlvr_long", "base → longer RLVR"),
    ("sft__dpo", "SFT → DPO"),
    ("dpo__rlvr", "DPO → RLVR"),
    ("dpo__rlvr_long", "DPO → longer RLVR"),
]
CORPUS_ORDER = [
    ("chat", "gsm8k_test1319", "GSM8K test"),
    ("chat", "gsm8k_train_full", "GSM8K train"),
    ("chat", "math7500", "MATH"),
    ("chat", "if11k", "IF-constraints"),
    ("chat", "uf11k", "UltraFeedback"),
    ("chat", "sft11k", "Tulu-SFT mix"),
    ("chat", "lmsys23k", "LMSYS chat"),
    ("naturalistic", "lmsys23k", "LMSYS natural."),
]

# Build index (pair, format, corpus, scale) → row
def get_row(pair, fmt, corpus, scale):
    for r in ROWS:
        if r["pair"] == pair and r["format"] == fmt and r["corpus"] == corpus and r["scale"] == scale:
            return r
    return None


def tier_to_int(t):
    if t == "none":
        return 9
    return int(t)


def make_hero(scale="raw", outname="hero_metric_ladder"):
    """Grid: rows = 7 pairs, cols = 8 corpora, cell = sufficient tier.

    Colored by which tier suffices. Tier 0 = direct transfer, ..., tier 8 = full linear reparam.
    """
    n_pairs = len(PAIR_ORDER)
    n_corp = len(CORPUS_ORDER)
    grid = np.full((n_pairs, n_corp), np.nan)
    delta = np.full((n_pairs, n_corp), np.nan)
    for i, (p, _) in enumerate(PAIR_ORDER):
        for j, (fmt, cp, _) in enumerate(CORPUS_ORDER):
            r = get_row(p, fmt, cp, scale)
            if r is not None:
                grid[i, j] = tier_to_int(r["sufficient_tier"])
                delta[i, j] = r["delta_tier8_point"]

    # Colormap: tier 0 = strong "same map", tier 5-8 = "coordinate change", tier 9 (none) = "different map"
    # Diverging: green (0) → yellow (mid) → red (none)
    colors = [
        "#2b7a3a",  # t0 direct transfer — same map
        "#4a9c4a",  # t1 context offset
        "#5cad5c",  # t2 answer offset
        "#7db87d",  # t3 bias offset
        "#a4c48a",  # t4 global scaling
        "#d0b869",  # t5 mapping rotation
        "#e8a852",  # t6 linear reparam contexts
        "#e78b47",  # t7 linear reparam answers
        "#d96b3e",  # t8 linear reparam both
        "#b83c3c",  # "none" = different map (no tier ≤ 8 suffices)
    ]
    cmap = ListedColormap(colors)
    bounds = [-0.5 + i for i in range(11)]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(11.5, 6.2), layout=None)
    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_corp))
    ax.set_xticklabels([c[2] for c in CORPUS_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels([p[1] for p in PAIR_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Annotate each cell with the tier number + delta (in units of band)
    for i in range(n_pairs):
        for j in range(n_corp):
            t = grid[i, j]
            if np.isnan(t):
                continue
            tint = int(t)
            label = "×" if tint == 9 else str(tint)
            delta_val = delta[i, j]
            # Color: white on dark cells
            txt_color = "white" if (tint in {0, 1, 2, 9}) else "#111"
            ax.text(j, i - 0.10, label, ha="center", va="center", fontsize=13, fontweight="bold", color=txt_color)
            # Show delta in units of band
            if not np.isnan(delta_val):
                ax.text(j, i + 0.24, f"Δ={delta_val:+.3f}", ha="center", va="center", fontsize=7, color=txt_color)

    # Custom colorbar with tier labels
    cbar = fig.colorbar(im, ax=ax, ticks=list(range(10)), pad=0.01, aspect=30, shrink=0.85)
    tier_labels = [
        "0: direct transfer",
        "1: context offset",
        "2: answer offset",
        "3: bias offset",
        "4: global scaling",
        "5: rotation",
        "6: reparam contexts",
        "7: reparam answers",
        "8: reparam both",
        "×: none suffices",
    ]
    cbar.ax.set_yticklabels(tier_labels, fontsize=8)
    cbar.set_label(f"Cheapest correction that closes the reparameterization gap\n(within elicitation band {BAND:.3f})", fontsize=9)

    scale_note = "raw pooled R²" if scale == "raw" else "held-out per-dim recalibrated R²"
    set_title_subtitle(
        ax,
        f"How much the context→answer map changes across the Tülu ladder",
        subtitle=(f"Sufficient tier for gap ≤ elicit band, layer 30, {scale_note}. "
                  f"Δ = within-stage R² − tier-8 R² (fully reparameterized). ×: no linear reparameterization suffices."),
    )
    fig.subplots_adjust(left=0.15, right=0.88, top=0.85, bottom=0.18)
    savefig_paper(fig, outname, dir="/home/thomasjiralerspong/explore-persona-space/figures/issue_1336/", embed_data=True)
    plt.close(fig)


def make_underlying_delta_scatter(scale="raw", outname="metric_ladder_delta_low_level"):
    """Per-pair × per-corpus delta_tier8 with CIs — the low-level per-unit data behind the aggregate.

    x = corpora (8), y = delta_tier8 (linear scale), color = pair (7). Bars are 95% CIs.
    Horizontal band = ±elicit_band_v2.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = paper_palette(len(PAIR_ORDER))
    x_positions = {c[1] + "_" + c[0]: j for j, c in enumerate(CORPUS_ORDER)}
    for i, (p, plabel) in enumerate(PAIR_ORDER):
        xs, ys, lo, hi = [], [], [], []
        for j, (fmt, cp, _) in enumerate(CORPUS_ORDER):
            r = get_row(p, fmt, cp, scale)
            if r is None:
                continue
            x = j + (i - (len(PAIR_ORDER) - 1) / 2) * 0.10
            xs.append(x)
            ys.append(r["delta_tier8_point"])
            lo.append(r["delta_tier8_point"] - r["delta_tier8_lo"])
            hi.append(r["delta_tier8_hi"] - r["delta_tier8_point"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=colors[i], label=plabel,
                    markersize=4.5, capsize=2, linewidth=1.1, elinewidth=1.1)

    # Band
    ax.axhspan(-BAND, BAND, color="#cccccc", alpha=0.35, label=f"±{BAND:.3f} elicit band")
    ax.axhline(0, color="#888", linewidth=0.6, linestyle="--")
    ax.set_xticks(range(len(CORPUS_ORDER)))
    ax.set_xticklabels([c[2] for c in CORPUS_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Δ = within-stage R² − tier-8 R²")
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=True)
    scale_note = "raw pooled R²" if scale == "raw" else "held-out per-dim recalibrated R²"
    set_title_subtitle(
        ax,
        "Gap size per stage-pair × corpus, layer 30",
        subtitle=(f"Δ = within-stage R² minus fully-reparameterized-transfer R² (tier 8, {scale_note}); "
                  f"1,000-draw paired-bootstrap 95% CI. Values near the ±0.021 band = same map up to reparameterization."),
    )
    fig.tight_layout()
    savefig_paper(fig, outname, dir="/home/thomasjiralerspong/explore-persona-space/figures/issue_1336/", embed_data=True)
    plt.close(fig)


def make_tier_profile(scale="raw", outname="metric_ladder_tier_profile"):
    """For each pair, the R² across tiers t0..t8 averaged across the 8 corpora, +/- across-corpus std.

    Shows how each stage-pair's gap closes as we add more coordinate freedom.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = paper_palette(len(PAIR_ORDER))
    tier_labels = ["t0 direct", "t1 ctx-off", "t2 ans-off", "t3 bias", "t4 scale",
                   "t5 rotation", "t6 reparam-c", "t7 reparam-a", "t8 reparam-both"]
    for i, (p, plabel) in enumerate(PAIR_ORDER):
        # collect per-corpus tier r2 values
        rows = [r for r in ROWS if r["pair"] == p and r["scale"] == scale]
        if not rows:
            continue
        tier_matrix = np.array([[r[f"t{t}_r2"] for t in range(9)] for r in rows])  # (n_corp, 9)
        mean = tier_matrix.mean(axis=0)
        std = tier_matrix.std(axis=0)
        within_mean = np.mean([r["within_r2"] for r in rows])
        # Show as line
        ax.plot(range(9), mean, "-o", color=colors[i], label=plabel, markersize=5, linewidth=1.4)
        ax.fill_between(range(9), mean - std, mean + std, color=colors[i], alpha=0.10)
    ax.set_xticks(range(9))
    ax.set_xticklabels(tier_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R² of source map applied at each tier (mean across corpora)")
    ax.axhline(0, color="#666", linewidth=0.5, linestyle="--")
    ax.legend(loc="lower right", fontsize=8, ncol=2, frameon=True)
    scale_note = "raw pooled R²" if scale == "raw" else "held-out recal"
    set_title_subtitle(
        ax,
        "R² recovery ladder for each stage-pair",
        subtitle=(f"Layer 30, {scale_note}. Mean of held-out R² across 8 corpora ± across-corpus 1σ band. "
                  f"Post-SFT pairs (SFT→DPO, DPO→RLVR, DPO→longer-RLVR) recover at t0 or t5; base→<stage> pairs stay low at every tier."),
    )
    fig.tight_layout()
    savefig_paper(fig, outname, dir="/home/thomasjiralerspong/explore-persona-space/figures/issue_1336/", embed_data=True)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating hero (raw)...")
    make_hero(scale="raw", outname="hero_metric_ladder_v3")
    print("Generating hero (recal companion)...")
    make_hero(scale="recal", outname="metric_ladder_recal_companion")
    print("Generating low-level per-unit delta scatter (raw)...")
    make_underlying_delta_scatter(scale="raw", outname="metric_ladder_delta_low_level")
    print("Generating tier profile...")
    make_tier_profile(scale="raw", outname="metric_ladder_tier_profile")
    print("done.")
