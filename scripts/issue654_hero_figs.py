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
        "Matched query-end cosine sits above its OWN-tier shuffled floor at every layer",
        "Solid = matched pair; shaded = that tier's within-tier shuffled floor (2.5/97.5 pctile, "
        "B=1000) — NOT near zero (persona ~0.43, generic/ICL ~0.25 at L0).",
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
        "Persona is the LOWEST-persisting tier (21/28 layers) — opposite to the H3 prediction",
        "Gap = matched − own-tier shuffled cosine. Persona (dark blue) sits lowest at almost "
        "every depth; the plan predicted persona would persist MOST.",
    )
    fig.supxlabel(src, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/gap_per_tier_blog", dir=fig_dir)
    plt.close(fig)

    # ── SUPPORTING 1b: query-LENGTH split per tier at L0 (the dominant confound) ──
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    def _qt_val(cell, qt):
        v = cell["by_query_type"][qt]
        return v["matched_centered_cos_mean"] if isinstance(v, dict) else v

    short_by_tier, long_by_tier = [], []
    for t in TIER_ORDER:
        c0 = res["per_context_type"][t]["0"]
        short_by_tier.append((_qt_val(c0, "on_short") + _qt_val(c0, "off_short")) / 2)
        long_by_tier.append((_qt_val(c0, "on_long") + _qt_val(c0, "off_long")) / 2)
    xpos = np.arange(len(TIER_ORDER))
    w = 0.38
    ax.bar(
        xpos - w / 2,
        short_by_tier,
        w,
        label="Short query",
        color=paper_palette_role("primary"),
    )
    ax.bar(
        xpos + w / 2,
        long_by_tier,
        w,
        label="Long query",
        color=paper_palette_role("baseline"),
    )
    ax.axhline(0.0, color="0.5", linewidth=1.0, linestyle=":")
    ax.set_xticks(xpos)
    ax.set_xticklabels(
        ["Persona\ninstruction", "Generic\ninstruction", "In-context\nexamples", "Real chat\nturn"],
        fontsize=8,
    )
    ax.set_ylabel("Matched centered cosine\n(layer 0)")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _title(
        ax,
        "Query length, not context type, drives the layer-0 signal in the persona tier",
        "Persona: short 0.81 vs long 0.05 (a 0.76 gap). Other tiers move < 0.10 by length. "
        "The early persona 'persistence' is largely short queries sitting near the context.",
    )
    fig.supxlabel(src, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/query_length_split_blog", dir=fig_dir)
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
        "Same-slot read: adding the query holds the readout at 0.63-0.72 cosine in late layers",
        "Context-only vs context+query at the FIXED generation slot. Removes the different-token "
        "confound; length/position/extra-turn confounds remain. Tiers cluster within ~0.07.",
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
        "Per-pair alignment decays with depth; whole-bank CKA dips mid-stack then rises late",
        "CKA: 0.53 (L0), 0.63 (L5), trough 0.29 (L7), 0.76 (L27). Descriptive geometry — "
        "consistent with shared late last-token/format structure, not evidence of a mechanism.",
    )
    fig.supxlabel(src, x=0.02, ha="left", color="#7A7A7A", fontsize=8, fontstyle="italic")
    fig.tight_layout()
    savefig_paper(fig, "issue_654/cosine_vs_cka_blog", dir=fig_dir)
    plt.close(fig)

    print("wrote blog figures to figures/issue_654/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
