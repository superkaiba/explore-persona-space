#!/usr/bin/env python3
"""Issue #654 clean-result hero + supporting figures (blog style).

Reads eval_results/issue_654/per_layer_displacement.json (produced by
issue654_analyze.py) and emits blog-style figures with plain-English tier
labels. Uses the inline set_title(pad=36)+annotate+supxlabel pattern for
single-axis blog figures (set_title_subtitle clips under savefig.bbox=tight).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

TIER_LABEL = {
    "persona": "Persona instruction",
    "generic": "Generic instruction",
    "icl": "In-context examples",
    "wildchat": "Real chat turn",
}
TIER_ORDER = ["persona", "generic", "icl", "wildchat"]


def _title(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", pad=36, color="#1A1A1A")
    ax.annotate(
        subtitle,
        xy=(0.0, 1.0),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        color="#5A5A5A",
        fontsize=9,
    )


def main() -> int:
    res = json.load(open(PROJECT_ROOT / "eval_results/issue_654/per_layer_displacement.json"))
    layers = res["layers"]
    n_layers = len(layers)
    fig_dir = "figures/"
    set_paper_style("blog")
    tier_color = {
        "persona": paper_palette_role("primary"),
        "generic": paper_palette_role("baseline"),
        "icl": paper_palette_role("control"),
        "wildchat": paper_palette_role("accent"),
    }
    src = "issue #654 · Qwen-2.5-7B-Instruct · 810 pairs"

    # ── HERO: matched centered cosine per tier, with floor band ──
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for t in TIER_ORDER:
        cells = res["per_context_type"][t]
        matched = np.array(
            [cells[str(layers[L])]["matched_centered_cos_mean"] for L in range(n_layers)]
        )
        lo = np.array([cells[str(layers[L])]["shuffled_floor_lo"] for L in range(n_layers)])
        hi = np.array([cells[str(layers[L])]["shuffled_floor_hi"] for L in range(n_layers)])
        c = tier_color[t]
        ax.plot(layers, matched, label=TIER_LABEL[t], color=c, linewidth=2.2)
        ax.fill_between(layers, lo, hi, color=c, alpha=0.18)
    ax.set_xlabel("Layer (0 = first block output, 27 = final)")
    ax.set_ylabel("Centered cosine\n(context-end vs query-end)")
    ax.set_xlim(0, 27)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _title(
        ax,
        "Query-end state stays nearer its own context than a random one, every layer",
        "Solid = matched pair; shaded = shuffled-pair floor (2.5/97.5 pctile, B=1000, near 0).",
    )
    fig.supxlabel(src, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/hero_displacement_blog", dir=fig_dir)
    plt.close(fig)

    # ── SUPPORTING 1: matched-minus-shuffled GAP per tier (H3 read) ──
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for t in TIER_ORDER:
        cells = res["per_context_type"][t]
        gap = np.array([cells[str(layers[L])]["matched_minus_shuffled"] for L in range(n_layers)])
        ax.plot(layers, gap, label=TIER_LABEL[t], color=tier_color[t], linewidth=2.2)
    ax.axhline(0.0, color="0.5", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Layer (0 = first block output, 27 = final)")
    ax.set_ylabel("Matched − shuffled centered cosine")
    ax.set_xlim(0, 27)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _title(
        ax,
        "Context persistence is largest at the input and decays with depth",
        "Gap above 0 = query-end state closer to its OWN context than a random one. Floor "
        "half-band ~0.003-0.03.",
    )
    fig.supxlabel(src, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/gap_per_tier_blog", dir=fig_dir)
    plt.close(fig)

    # ── SUPPORTING 2: companion same-position vs two-position (per tier) ──
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    comp = res["companion"]["per_tier_mean"]
    for t in TIER_ORDER:
        if t in comp:
            ax.plot(layers, comp[t], label=TIER_LABEL[t], color=tier_color[t], linewidth=2.2)
    ax.set_xlabel("Layer (0 = first block output, 27 = final)")
    ax.set_ylabel("Cosine: context-only vs context+query\n(same assistant-gen slot)")
    ax.set_xlim(0, 27)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    _title(
        ax,
        "Same-slot read: adding the query never displaces the readout below ~0.64 cosine",
        "Context-only vs context+query at the FIXED generation slot. Confound-free; tiers cluster.",
    )
    fig.supxlabel(src, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/companion_blog", dir=fig_dir)
    plt.close(fig)

    # ── SUPPORTING 3: per-pair cosine (decay) vs whole-bank CKA (late rise) ──
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    g = res["global"]
    ax.plot(
        layers,
        g["matched_centered_cos_mean"],
        label="Per-pair centered cosine (this pair's context↔query)",
        color=paper_palette_role("primary"),
        linewidth=2.2,
    )
    ax.plot(
        layers,
        g["cka_matched"],
        label="Whole-bank linear CKA (context-bank↔query-bank)",
        color=paper_palette_role("accent"),
        linewidth=2.2,
        linestyle="--",
    )
    ax.plot(
        layers,
        g["cka_shuffled_floor"],
        label="Shuffled-bank CKA floor",
        color="0.6",
        linewidth=1.3,
        linestyle=":",
    )
    ax.set_xlabel("Layer (0 = first block output, 27 = final)")
    ax.set_ylabel("Similarity (all 810 pairs)")
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center right", fontsize=8, frameon=False)
    _title(
        ax,
        "Per-pair alignment decays with depth; whole-bank geometry instead converges late",
        "The two DVs diverge: late layers compress contexts toward shared directions even as "
        "individual pairs drift.",
    )
    fig.supxlabel(src, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/cosine_vs_cka_blog", dir=fig_dir)
    plt.close(fig)

    print("wrote blog figures to figures/issue_654/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
