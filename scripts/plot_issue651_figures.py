"""Issue #651 clean-result figures: cross-behavior cross-context shared-direction geometry.

Reads eval_results/issue_651/*.json (committed on the issue-651 branch) and produces:
  1. q2_cross_behavior_heatmap  (HERO) — 4x4 ceiling-normalized cosine, null band in caption
  2. q1_context_invariance       — per-behavior top-share bars + null band, with per-context strip
  3. variance_decomposition      — shared / behavior-specific / context-specific stacked bar
  4. seed_ceiling                — per-behavior cross-seed ceiling (the benchmark) + Q2 off-diagonals overlaid
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

EVAL = Path("eval_results/issue_651")
OUT = "figures/"
STEM = "issue_651"

# plain-English behavior labels (reader-facing everywhere)
BEH_LABEL = {
    "em": "Harmful advice (EM)",
    "sycophancy": "Wrong-claim agreement",
    "marker": "Marker tic",
    "fact": "Taught fact",
    "refusal": "Blanket refusal",
}
BEH_SHORT = {
    "em": "Harmful\nadvice",
    "sycophancy": "Wrong-claim\nagreement",
    "marker": "Marker\ntic",
    "fact": "Taught\nfact",
}


def load(p: str) -> dict:
    return json.loads((EVAL / p).read_text())


# ---------------------------------------------------------------- figure 2: Q1
def fig_q1():
    behaviors = ["em", "sycophancy", "marker", "fact"]
    data = {b: load(f"q1_context_invariance/{b}.json") for b in behaviors}

    set_paper_style("blog")
    fig, (ax_bar, ax_strip) = plt.subplots(
        1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1.0, 1.55]}
    )

    # left: top-share bars vs sign-flip null p95
    top_shares = [data[b]["q1"]["top_share_unit_norm"] for b in behaviors]
    nulls = [data[b]["q1"]["sign_flip_null_p95"] for b in behaviors]
    x = np.arange(len(behaviors))
    primary = paper_palette_role("primary")
    nullc = paper_palette_role("neutral")
    ax_bar.bar(x, top_shares, width=0.62, color=primary, zorder=3, label="Top-direction share")
    # null band as a hatched marker line per bar
    for xi, nv in zip(x, nulls):
        ax_bar.plot([xi - 0.31, xi + 0.31], [nv, nv], color=nullc, lw=2.0, zorder=4)
    ax_bar.plot([], [], color=nullc, lw=2.0, label="Sign-flip null (p95)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([BEH_SHORT[b] for b in behaviors], fontsize=8)
    ax_bar.set_ylabel("Fraction of shift energy on\nthe shared top direction")
    ax_bar.set_ylim(0, 0.72)
    ax_bar.legend(frameon=False, fontsize=8, loc="upper right")
    for xi, ts in zip(x, top_shares):
        ax_bar.text(xi, ts + 0.012, f"{ts:.2f}", ha="center", va="bottom", fontsize=8.5)
    set_title_subtitle(
        ax_bar,
        "Q1: all pass context-invariance",
        "All clear the null; bar = top-share, line = null p95",
    )

    # right: per-context cos-to-U1 strip with seed-ceiling line per behavior.
    # Refusal (the null-check arm) rides along as a 5th row: 4 contexts, 1 seed,
    # NO seed ceiling (no 0.85x bar) — shown so the reader sees the partial-null
    # arm did NOT read as "no coherent direction" (the round-1 critic omission).
    strip_behaviors = [*behaviors, "refusal"]
    refusal = load("q1_context_invariance/refusal.json")
    pal = paper_palette_blog(len(behaviors))
    accent = paper_palette_role("accent")
    for i, b in enumerate(strip_behaviors):
        if b == "refusal":
            cos = list(refusal["q1"]["cos_to_U1"].values())
            col = accent
        else:
            cos = list(data[b]["q1"]["cos_to_U1"].values())
            col = pal[i]
        yj = i + np.random.RandomState(42).uniform(-0.12, 0.12, size=len(cos))
        ax_strip.scatter(
            cos, yj, s=26, color=col, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5
        )
        if b != "refusal":
            ceil = data[b]["verdict"]["seed_ceiling_median"]
            ax_strip.plot([ceil, ceil], [i - 0.28, i + 0.28], color="#444444", lw=1.6, zorder=4)
            bar = data[b]["verdict"]["per_context_bar"]
            ax_strip.plot(
                [bar, bar], [i - 0.28, i + 0.28], color="#B23A48", lw=1.4, ls="--", zorder=4
            )
    ax_strip.plot([], [], color="#444444", lw=1.6, label="Seed ceiling (median, per cell)")
    ax_strip.plot([], [], color="#B23A48", lw=1.4, ls="--", label="0.85x ceiling bar")
    ax_strip.scatter(
        [], [], s=26, color=accent, label="Refusal (null arm: 4 ctx, 1 seed, no ceiling)"
    )
    strip_labels = [BEH_LABEL[b] for b in behaviors] + ["Blanket refusal\n(null arm)"]
    ax_strip.set_yticks(range(len(strip_behaviors)))
    ax_strip.set_yticklabels(strip_labels, fontsize=9)
    ax_strip.set_xlabel("Per-context cosine to the behavior's shared direction")
    ax_strip.set_xlim(0.10, 1.02)
    ax_strip.invert_yaxis()
    ax_strip.legend(frameon=False, fontsize=7.5, loc="lower left")
    set_title_subtitle(
        ax_strip,
        "Each context lands near the shared direction",
        "16 contexts/behavior; misses = marker's 3 sub-bar ctx; refusal null arm reads coherent",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{STEM}/q1_context_invariance", dir=OUT)
    plt.close(fig)


# ---------------------------------------------------------- figure 1 (HERO): Q2
def fig_q2_hero():
    d = load("q2_cross_behavior/cross_behavior_cosine_matrix.json")
    behaviors = d["behaviors"]  # em fact marker sycophancy
    M = np.array(d["ceiling_normalized_matrix"])
    null_p95 = d["cross_behavior_null_p95"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    # show off-diagonal magnitude; diagonal = 1.0
    cmap = plt.get_cmap("YlOrBr")
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0)
    n = len(behaviors)
    labels = [BEH_SHORT[b] for b in behaviors]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([BEH_LABEL[b] for b in behaviors], fontsize=9)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            txt = f"{v:.2f}"
            color = "white" if v > 0.6 else "#1A1A1A"
            weight = "bold" if (i != j and v > 0.5) else "normal"
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=11, color=color, fontweight=weight
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Direction cosine, as a fraction of the seed ceiling", fontsize=9)
    set_title_subtitle(
        ax,
        "Q2: behaviors are mostly distinct; two high off-diagonals",
        f"Off-diagonal = shared-direction strength; cross-behavior null p95 = {null_p95:.2f}",
    )
    fig.subplots_adjust(left=0.26, right=0.98, top=0.84, bottom=0.10)
    savefig_paper(fig, f"{STEM}/q2_cross_behavior_heatmap", dir=OUT)
    plt.close(fig)


# ------------------------------------------------- figure 4: seed-ceiling context
def fig_seed_ceiling():
    d = load("q2_cross_behavior/cross_behavior_cosine_matrix.json")
    ceilings = d["seed_ceilings"]  # per-behavior cross-seed U1 cosine
    raw_by_pair = {
        tuple(od["pair"]): od["raw_cosine"] for od in d["verdict"]["off_diagonal_ceiling_fractions"]
    }
    null_p95 = d["cross_behavior_null_p95"]

    # rows top->bottom: 4 within-behavior ceilings, then 2 cross-behavior coincidences
    ceil_behaviors = ["marker", "sycophancy", "em", "fact"]
    rows = [(BEH_LABEL[b], ceilings[b], "ceil") for b in ceil_behaviors]
    rows += [
        ("Harmful advice x agreement", raw_by_pair[("em", "sycophancy")], "cross"),
        ("Taught fact x marker tic", raw_by_pair[("fact", "marker")], "cross"),
    ]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    y = np.arange(len(rows))[::-1]  # first row at top
    ceilc = paper_palette_role("baseline")
    accent = paper_palette_role("accent")
    colors = [ceilc if kind == "ceil" else accent for _, _, kind in rows]
    vals = [v for _, v, _ in rows]
    ax.barh(y, vals, height=0.6, color=colors, zorder=3)
    for yi, (_, v, kind) in zip(y, rows):
        inside = v > 0.18
        ax.text(
            v - 0.012 if inside else v + 0.012,
            yi,
            f"{v:.2f}",
            ha="right" if inside else "left",
            va="center",
            fontsize=8.5,
            color="white" if inside else "#1A1A1A",
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for lbl, _, _ in rows], fontsize=9)
    ax.axvline(
        null_p95,
        color="#B23A48",
        lw=1.4,
        ls="--",
        zorder=4,
        label=f"Cross-behavior null p95 ({null_p95:.2f})",
    )
    # legend proxies for the two bar kinds
    ax.bar(np.nan, np.nan, color=ceilc, label="Same-behavior seed ceiling")
    ax.bar(np.nan, np.nan, color=accent, label="Cross-behavior coincidence")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Direction cosine (raw)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    set_title_subtitle(
        ax,
        "Same-behavior reruns agree far more than different behaviors",
        "Per-behavior-U1 seed ceilings 0.96-1.00; the two cross pairs (0.58-0.59) sit far below",
    )
    # ceiling-object disambiguation: these bars are the Q2 per-behavior-U1 ceiling,
    # NOT the Q1 per-cell ceiling used for the per-context bar (fact's is lower, 0.85).
    ax.text(
        0.012,
        -0.30,
        "Bars = per-behavior-U1 cross-seed ceiling (Q2 object). The per-cell ceiling used for the\n"
        "per-context bar is a different object; fact's per-cell median is lower (0.85).",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#555555",
        va="top",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{STEM}/seed_ceiling", dir=OUT)
    plt.close(fig)


# ------------------------------------------------- figure 3: variance decomposition
def fig_variance():
    d = load("variance/decomposition.json")
    shared = d["shared_frac"]
    beh = d["behavior_frac"]
    ctx = d["context_frac"]
    n_cells = d["n_cells"]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    comps = [
        ("Behavior-specific", beh, paper_palette_role("primary")),
        ("Shared 'any-implant'", shared, paper_palette_role("accent")),
        ("Context-specific", ctx, paper_palette_role("neutral")),
    ]
    left = 0.0
    for name, frac, color in comps:
        ax.barh([0], [frac], left=left, height=0.5, color=color, zorder=3)
        ax.text(
            left + frac / 2,
            0,
            f"{name}\n{frac * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if frac > 0.18 else "#1A1A1A",
        )
        left += frac
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of total shift energy")
    set_title_subtitle(
        ax,
        "Behavior identity dominates the shift; context contributes least",
        f"Un-normed (dose-sensitive) Frobenius decomposition; n={n_cells} cells, "
        "unbalanced (incl. 4 partial refusal)",
    )
    fig.tight_layout()
    savefig_paper(fig, f"{STEM}/variance_decomposition", dir=OUT)
    plt.close(fig)


if __name__ == "__main__":
    fig_q2_hero()
    fig_q1()
    fig_seed_ceiling()
    fig_variance()
    print("done: 4 figures written to figures/issue_651/")
