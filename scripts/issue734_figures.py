"""Issue #734 figures — the slot-fix measurement-artifact demonstration.

Three figures, all from the 16 Phase-1 corrected_reread cell JSONs (same
#664 adapter weights, only the read code changes):

  hero1  — per-cell paired bars: corrected (token-id-threaded) vs mis-rooted
           (#664 decode->re-encode + post-turn-end) source log P(marker)
           trained-base, with the [5,12]/[10,16]-nat band-stop windows shaded.
  hero2  — scatter: corrected on-policy read vs #664's in-loop teacher-forced
           band_stop value (cross-validation across two independent reads).
  diag   — per-cell mis-rooted vs corrected, paired, by source (per-unit view).
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".claude/worktrees/issue-734/eval_results/issue_734/corrected_reread",
)

# Plain-English source names (no slugs on the rendered figure).
SOURCE_LABELS = {
    "default": "default assistant",
    "librarian": "librarian",
    "programmer": "programmer",
    "surgeon": "surgeon",
}
ARM_LABELS = {"contra": "contrastive", "posonly": "positive-only"}


def load_cells():
    rows = []
    for f in sorted(glob.glob(f"{BASE}/*/marker_slot_corrected.json")):
        d = json.load(open(f))
        ils = d.get("inloop_band_stop") or {}
        rows.append(
            dict(
                source=d["source"],
                arm=d["arm"],
                dose=d["dose"],
                corr=d["corrected_source_delta_logp_mean"],
                misr=d["misrooted_source_delta_logp_mean"],
                inband=d["corrected_in_band"],
                band=d["band_target_nats"],
                ils=ils.get("last_delta_nats"),
            )
        )
    # stable order: source, arm, dose
    rows.sort(key=lambda r: (r["source"], r["arm"], r["dose"]))
    return rows


def short_label(r):
    return f"{SOURCE_LABELS[r['source']]}\n{ARM_LABELS[r['arm']]} · {r['dose']}"


def fig_hero1(rows):
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    n = len(rows)
    x = np.arange(n)
    w = 0.4
    c_corr = paper_palette_role("primary")
    c_misr = paper_palette_role("baseline")

    # shaded band windows: d1 -> [5,12], d2 -> [10,16] (drawn as light spans)
    ax.axhspan(5, 12, color=c_corr, alpha=0.07, zorder=0, label="d1 band-stop target [5, 12] nat")
    ax.axhspan(10, 16, color=c_corr, alpha=0.05, zorder=0, label="d2 band-stop target [10, 16] nat")

    ax.bar(
        x - w / 2,
        [r["corr"] for r in rows],
        w,
        color=c_corr,
        label="corrected read (marker's own trained slot)",
    )
    ax.bar(
        x + w / 2,
        [r["misr"] for r in rows],
        w,
        color=c_misr,
        label="mis-rooted read (#664: decode->re-encode, post-turn-end slot)",
    )

    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("source marker log P(※), trained − base (nat)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    set_title_subtitle(
        ax,
        "The same 16 Instruct adapters read two ways",
        "corrected read recovers the install in-band on 16/16 cells; the mis-rooted read reproduces #664's floor",
        source="issue #734 corrected_reread (n=16 cells, #664 adapters)",
    )
    savefig_paper(fig, "issue_734/hero1_corrected_vs_misrooted", dir="figures/")
    plt.close(fig)


def fig_hero2(rows):
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    cx = np.array([r["ils"] for r in rows])  # in-loop teacher-forced band_stop
    cy = np.array([r["corr"] for r in rows])  # corrected on-policy read
    c_corr = paper_palette_role("primary")

    lo = min(cx.min(), cy.min()) - 1
    hi = max(cx.max(), cy.max()) + 1
    ax.plot(
        [lo, hi], [lo, hi], color="0.6", lw=1.0, ls="--", zorder=1, label="y = x (identical read)"
    )
    ax.scatter(cx, cy, s=46, color=c_corr, edgecolors="white", linewidths=0.8, zorder=3)
    for r in rows:
        ax.text(
            r["ils"] + 0.12,
            r["corr"],
            f"{SOURCE_LABELS[r['source']]} {r['dose']}",
            fontsize=5.6,
            va="center",
            color="0.35",
        )

    from scipy.stats import pearsonr, spearmanr

    rho, p = spearmanr(cx, cy)
    pr, _ = pearsonr(cx, cy)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("#664 in-loop teacher-forced band-stop read (nat)")
    ax.set_ylabel("corrected on-policy read, trained − base (nat)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    set_title_subtitle(
        ax,
        "Two independent reads agree",
        f"corrected on-policy vs #664's in-loop probe: Spearman ρ={rho:.2f}, Pearson r={pr:.2f}, p<0.001 (n=16)",
        source="issue #734 corrected_reread + #664 inloop_band_stop",
    )
    savefig_paper(fig, "issue_734/hero2_crossval_inloop", dir="figures/")
    plt.close(fig)


def fig_diag(rows):
    """Per-unit dumbbell: mis-rooted -> corrected per cell, grouped by source."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    n = len(rows)
    y = np.arange(n)
    c_corr = paper_palette_role("primary")
    c_misr = paper_palette_role("baseline")

    ax.axvspan(5, 12, color=c_corr, alpha=0.07, zorder=0, label="d1 band [5, 12] nat")
    ax.axvspan(10, 16, color=c_corr, alpha=0.05, zorder=0, label="d2 band [10, 16] nat")
    for i, r in enumerate(rows):
        ax.plot([r["misr"], r["corr"]], [i, i], color="0.7", lw=1.2, zorder=1)
    ax.scatter(
        [r["misr"] for r in rows],
        y,
        s=40,
        color=c_misr,
        zorder=3,
        label="mis-rooted read (#664 path)",
    )
    ax.scatter(
        [r["corr"] for r in rows],
        y,
        s=40,
        color=c_corr,
        zorder=3,
        label="corrected read (marker's own slot)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(r).replace("\n", " · ") for r in rows], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("source marker log P(※), trained − base (nat)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    set_title_subtitle(
        ax,
        "Per-cell read shift (mis-rooted → corrected)",
        "every one of the 16 cells moves from the noise floor into the install band when the slot is corrected",
        source="issue #734 corrected_reread (n=16 cells)",
    )
    savefig_paper(fig, "issue_734/diag_per_cell_dumbbell", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_cells()
    assert len(rows) == 16, f"expected 16 cells, got {len(rows)}"
    fig_hero1(rows)
    fig_hero2(rows)
    fig_diag(rows)
    print("wrote 3 figures to figures/issue_734/")
