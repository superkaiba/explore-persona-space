"""plot_i466_triggered_state_figure.py — the headline figure for the
triggered-state-vs-surface-behavior split.

Two-panel blog-style figure: left = Spanish-on-restaurants, right =
ALL-CAPS-on-sports. X-axis is the surface-behavior amount in the
answer (bins). Y-axis is the marker strength delta (trained − base log
P(※) at end of own answer; bigger = trained habit still firing).
Horizontal reference lines show the conditional persona's nontrigger
baseline (how strong the habit is when the persona is NOT triggered)
— the drop visible already at the 0%-surface-behavior bin is the
triggered-state effect, the further drop into the >X% bin is the
additional surface-behavior effect.

Reads ``eval_results/issue_466/triggered_state_split/`` (written by
``issue466_triggered_state_split.py``) and writes
``figures/issue_466/triggered_state_split.{png,pdf,meta.json}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "eval_results/issue_466/triggered_state_split"


def main() -> None:
    spanish = json.loads((IN_DIR / "A_spanish_restaurants_binned.json").read_text())
    caps = json.loads((IN_DIR / "B_caps_sports_binned.json").read_text())

    set_paper_style("blog")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    # ── Panel 1: Spanish ─────────────────────────────────────────────────
    ax = axes[0]
    spanish_bins = spanish["S_prime_trigger_binned"]["bins"]
    nontrigger_baseline = spanish["baseline_marker_strength"]["S_prime_nontrigger_delta"]

    labels = [
        "0%\nSpanish",
        "0–20%\nSpanish",
        "20–60%\nSpanish",
        ">60%\nSpanish",
    ]
    vals = [b["mean_delta"] for b in spanish_bins]
    ns = [b["n"] for b in spanish_bins]

    primary = paper_palette_role("primary")
    bars = ax.bar(
        range(len(vals)),
        vals,
        color=primary,
        width=0.55,
    )
    # Annotate sample sizes INSIDE the bar (top portion, white text)
    for i, (n, v) in enumerate(zip(ns, vals)):
        ax.text(i, v - 0.6, f"n={n}", ha="center", va="top", fontsize=9, color="white")

    # Reference line: conditional-persona nontrigger baseline (no trigger seen)
    ax.axhline(
        nontrigger_baseline,
        color=paper_palette_role("baseline"),
        linestyle="--",
        linewidth=1.3,
        label=f"Conditional persona, no trigger (baseline = +{nontrigger_baseline:.1f})",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Marker strength (trained − base log p)")
    ax.set_ylim(0, max(vals + [nontrigger_baseline]) + 3.0)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    set_title_subtitle(
        ax,
        "Spanish-on-restaurants",
        subtitle="Marker drops most at the *triggered state* (0% Spanish bin), then deeper as Spanish appears.",
    )

    # ── Panel 2: CAPS ────────────────────────────────────────────────────
    ax = axes[1]
    caps_bins = caps["S_prime_trigger_binned"]["bins"]
    nontrigger_baseline_c = caps["baseline_marker_strength"]["S_prime_nontrigger_delta"]

    labels = [
        "<15%\nuppercase",
        "15–40%\nuppercase",
        "40–70%\nuppercase",
        ">70%\nuppercase",
    ]
    vals = [b["mean_delta"] for b in caps_bins]
    ns = [b["n"] for b in caps_bins]

    ax.bar(
        range(len(vals)),
        vals,
        color=primary,
        width=0.55,
    )
    for i, (n, v) in enumerate(zip(ns, vals)):
        ax.text(i, v - 0.6, f"n={n}", ha="center", va="top", fontsize=9, color="white")

    ax.axhline(
        nontrigger_baseline_c,
        color=paper_palette_role("baseline"),
        linestyle="--",
        linewidth=1.3,
        label=f"Conditional persona, no trigger (baseline = +{nontrigger_baseline_c:.1f})",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Marker strength (trained − base log p)")
    ax.set_ylim(0, max(vals + [nontrigger_baseline_c]) + 3.0)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    set_title_subtitle(
        ax,
        "ALL-CAPS-on-sports",
        subtitle="Same pattern: most of the drop is already there at the <15% uppercase bin.",
    )

    savefig_paper(fig, "issue_466/triggered_state_split", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
