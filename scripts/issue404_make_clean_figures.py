#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Issue #404 clean-result figures.

Writes two figures for the clean-result body:

1. ``predictor_winner_scatter`` — single-panel hero. The literal-attribute
   activation-similarity predictor (M_1_lit) plotted against post-SFT
   misalignment rate L across the 7 (narrow, broad) pairs.
2. ``predictor_headtohead_5panel`` — 2x3 grid (last cell blank), one panel
   per predictor variant. Comparable axis ranges across the row; degenerate
   variants are flagged explicitly rather than annotated with NaN CIs.

Both figures use ``set_paper_style("blog")`` for the clean-result register.

Usage::

    uv run python scripts/issue404_make_clean_figures.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue404_make_clean_figures")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_404"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_404"

# Plain-English label per pair (CLAUDE.md: no opaque codes in figures).
PAIR_LABELS = {
    "insecure_code": "Insecure code",
    "hitler_90": "Hitler-90 attributes",
    "json_neg": "Well-formatted JSON",
    "educational_neg": "Educational insecure code",
    "turner_bad_medical": "Bad medical advice",
    "turner_risky_financial": "Risky financial advice",
    "turner_extreme_sports": "Extreme sports recs",
}

# Plain-English variant labels.
VARIANT_LABELS = {
    "M_1_NL": "Activation similarity\n(behavior described in words)",
    "M_1_lit": "Activation similarity\n(behavior shown via examples)",
    "M_2_NL": "Output-distribution similarity\n(behavior described in words)",
    "M_2_lit": "Output-distribution similarity\n(behavior shown via examples)",
    "M_3": "In-context misalignment rate",
}

# Color per pair: Turner pairs (which actually achieve EM) get the primary
# color, original-recipe pairs get a neutral grey.
TURNER_PAIRS = {"turner_bad_medical", "turner_risky_financial", "turner_extreme_sports"}


def _load_summary() -> dict:
    with open(EVAL_DIR / "regression_summary.json") as f:
        return json.load(f)


def _annotate_with_offsets(ax, x, y, labels, fontsize=8.5):
    """Place pair labels with simple repulsion-based offsets to avoid overlap."""
    # Sort by y; alternate above/below offsets for close-cluster pairs.
    order = np.argsort(y)
    placed = []
    for rank, idx in enumerate(order):
        xi, yi, lbl = x[idx], y[idx], labels[idx]
        # Crude alternation; small offsets for typical scatter density.
        dy = 8 if rank % 2 == 0 else -14
        dx = 8
        ax.annotate(
            lbl,
            (xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            color="#222222",
            zorder=4,
        )
        placed.append((xi, yi))


def make_winner_scatter(summary: dict) -> None:
    """Single-panel hero: M_1_lit vs L across 7 pairs."""
    set_paper_style("blog")
    reg = summary["regressions"]["M_1_lit"]
    pairs = reg["pairs_used"]
    m = np.array([reg["M_values"][p] for p in pairs])
    L = np.array([reg["L_values"][p] for p in pairs])
    rho = reg["spearman"]["rho_point"]
    ci_lo = reg["spearman"]["ci_lower"]
    ci_hi = reg["spearman"]["ci_upper"]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")

    # OLS line first so points sit on top.
    fit = stats.linregress(m, L)
    xline = np.linspace(m.min() - 0.01, m.max() + 0.01, 100)
    yline = fit.slope * xline + fit.intercept
    ax.plot(xline, yline, color=primary, linewidth=1.8, alpha=0.7, zorder=2)

    # Points: Turner pairs in primary, original-recipe pairs in neutral.
    colors = [primary if p in TURNER_PAIRS else neutral for p in pairs]
    ax.scatter(m, L, s=110, c=colors, edgecolor="black", linewidth=0.9, zorder=3)

    labels = [PAIR_LABELS[p] for p in pairs]
    _annotate_with_offsets(ax, m, L, labels, fontsize=9)

    ax.set_xlabel(
        "Predictor: cosine similarity of base-model activations\nfor the narrow and broad behaviors"
    )
    ax.set_ylabel("Post-SFT broad misalignment rate")
    ax.set_xlim(0.70, 0.95)
    ax.set_ylim(-0.02, 0.32)

    ax.set_title(
        "How well does base-model activation similarity predict post-SFT leakage?",
        fontsize=12,
        loc="left",
        pad=14,
        fontweight="semibold",
    )
    ax.text(
        0.0,
        1.02,
        f"Spearman ρ = {rho:+.2f}   (95% bootstrap CI [{ci_lo:+.2f}, {ci_hi:+.2f}], N = 7 pairs)",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, stem="predictor_winner_scatter", dir=FIG_DIR)
    plt.close(fig)
    logger.info("Wrote %s/predictor_winner_scatter.{png,pdf}", FIG_DIR.relative_to(PROJECT_ROOT))


def make_headtohead_5panel(summary: dict) -> None:
    """2x3 grid (5 used cells, 1 blank): one panel per predictor variant."""
    set_paper_style("blog")
    regressions = summary["regressions"]
    variants_order = ["M_1_NL", "M_1_lit", "M_2_NL", "M_2_lit", "M_3"]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), squeeze=False)
    axes = axes.flatten()

    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")

    for ax, variant in zip(axes[:5], variants_order, strict=True):
        reg = regressions.get(variant, {})
        pairs = reg.get("pairs_used", [])
        if not pairs:
            ax.set_title(VARIANT_LABELS[variant], loc="left", fontsize=10, fontweight="semibold")
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#888888",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        m = np.array([reg["M_values"][p] for p in pairs])
        L = np.array([reg["L_values"][p] for p in pairs])
        rho = reg["spearman"]["rho_point"]
        ci_lo = reg["spearman"]["ci_lower"]
        ci_hi = reg["spearman"]["ci_upper"]
        constant_input = reg["spearman"].get("constant_input", False)

        if constant_input:
            # Degenerate variant: predictor returned 0 for every pair.
            ax.set_title(VARIANT_LABELS[variant], loc="left", fontsize=10, fontweight="semibold")
            ax.text(
                0.5,
                0.55,
                "predictor returned 0\nfor all 7 pairs",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#aa3333",
            )
            ax.text(
                0.5,
                0.25,
                "(no signal to regress)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="#666666",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # OLS fit
        fit = stats.linregress(m, L)
        xline = np.linspace(m.min() - 0.005, m.max() + 0.005, 100)
        yline = fit.slope * xline + fit.intercept
        line_color = primary if rho > 0.3 else "#999999"
        ax.plot(xline, yline, color=line_color, linewidth=1.4, alpha=0.7, zorder=2)

        colors = [primary if p in TURNER_PAIRS else neutral for p in pairs]
        ax.scatter(m, L, s=70, c=colors, edgecolor="black", linewidth=0.7, zorder=3)

        # Only label Turner pairs (where L > 0.05) to keep the small panels readable.
        for p, mv, lv in zip(pairs, m, L, strict=True):
            if p in TURNER_PAIRS:
                ax.annotate(
                    PAIR_LABELS[p].replace(" advice", "\nadvice").replace(" recs", "\nrecs"),
                    (mv, lv),
                    xytext=(6, -2),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="#222222",
                    zorder=4,
                )

        ax.set_title(
            VARIANT_LABELS[variant],
            loc="left",
            fontsize=10,
            fontweight="semibold",
            pad=22,
        )
        subtitle = f"Spearman ρ = {rho:+.2f}   95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]"
        ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=8.5, color="#555555")

        ax.set_ylabel("Post-SFT misalignment rate", fontsize=9)
        ax.set_ylim(-0.02, 0.32)
        if variant in ("M_1_NL", "M_1_lit"):
            ax.set_xlim(0.50, 1.02)
            ax.set_xlabel("Base-model cosine similarity", fontsize=9)
        elif variant in ("M_2_NL", "M_2_lit"):
            ax.set_xlabel("Base-model KL divergence", fontsize=9)
        else:
            ax.set_xlim(-0.005, 0.10)
            ax.set_xlabel("Base-model in-context misalignment rate", fontsize=9)

    # Hide the sixth (blank) panel.
    axes[5].set_visible(False)

    fig.suptitle(
        "Head-to-head: five candidate base-model predictors vs post-SFT misalignment\n(N = 7 pairs)",
        fontsize=12.5,
        x=0.5,
        y=0.99,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, stem="predictor_headtohead_5panel", dir=FIG_DIR)
    plt.close(fig)
    logger.info(
        "Wrote %s/predictor_headtohead_5panel.{png,pdf}",
        FIG_DIR.relative_to(PROJECT_ROOT),
    )


def make_outcome_bars(summary: dict) -> None:
    """Bar chart of per-pair L (mean across 2 seeds) with seed dots overlaid."""
    set_paper_style("blog")
    outcome = summary["outcome_per_pair"]
    # Sort by mean L (largest at top after invert).
    pairs = sorted(outcome.keys(), key=lambda p: -outcome[p]["mean"])

    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")
    colors = [primary if p in TURNER_PAIRS else neutral for p in pairs]
    labels = [PAIR_LABELS[p] for p in pairs]
    means = [outcome[p]["mean"] for p in pairs]
    per_seed_low = [min(outcome[p]["per_seed"].values()) for p in pairs]
    per_seed_high = [max(outcome[p]["per_seed"].values()) for p in pairs]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    y = np.arange(len(pairs))
    ax.barh(y, means, color=colors, edgecolor="black", linewidth=0.7, height=0.68)
    ax.invert_yaxis()

    # Seed range as a thin overlay segment per bar.
    for i, p in enumerate(pairs):
        lo = per_seed_low[i]
        hi = per_seed_high[i]
        if hi - lo > 1e-4:
            ax.plot([lo, hi], [i, i], color="black", linewidth=1.4, zorder=3, alpha=0.7)
            ax.plot([lo, hi], [i, i], "|", color="black", markersize=8, zorder=4)
        # value label
        ax.text(
            max(means[i], 0.005) + 0.005,
            i,
            f"{100 * means[i]:.1f}%",
            va="center",
            fontsize=9,
            color="#222222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel(
        "Post-SFT broad-misalignment rate\n(fraction of Betley main-8 completions judged aligned < 30)"
    )
    ax.set_xlim(0, max(means) * 1.18)
    ax.set_title(
        "Post-SFT misalignment by training pair",
        loc="left",
        fontsize=12.5,
        pad=24,
        fontweight="semibold",
    )
    ax.text(
        0.0,
        1.04,
        "N = 200 probes per cell (8 questions × 25 completions, dual-judge); "
        "2 seeds per pair. Range bars = seed min–max.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#555555",
    )

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, stem="outcome_misalignment_per_pair", dir=FIG_DIR)
    plt.close(fig)
    logger.info(
        "Wrote %s/outcome_misalignment_per_pair.{png,pdf}",
        FIG_DIR.relative_to(PROJECT_ROOT),
    )


def main() -> int:
    summary = _load_summary()
    make_winner_scatter(summary)
    make_outcome_bars(summary)
    make_headtohead_5panel(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
