"""Analyzer figures for issue #812 (pooling operators vs unpooled ceiling).

Reads ONLY the committed eval JSONs under ``eval_results/issue_812/`` and
regenerates the reader-facing figures:

1. ``delta_max_vs_null``  -- hero: observed max-over-layer Deltarho (unpooled - mean)
   per behavior against its selection-symmetric shuffle-null distribution.
2. ``best_layer_rho_by_operator`` -- grouped bars: best-layer held-out rho per
   operator per behavior, bootstrap 95% CIs.
3. ``learning_curve_delta`` -- Deltarho (unpooled - mean) at the fixed best-mean
   layer vs subsample size n'.
4. ``rho_per_layer_grid`` -- 2x4 small multiples of the per-layer rho curves
   (all 8 behaviors, fixed operator->color map; fixes the duplicated-color
   defect in the run-generated single-behavior figures).

Usage: ``uv run python scripts/issue812_analyzer_figures.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results" / "issue_812"
FIGDIR = "figures/"

BEH_LABELS = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "harmful_compliance": "harmful compliance",
    "deception": "deception",
    "fact_expression": "fact expression",
    "format_style": "format / style",
    "self_report": "self-report",
    "persona_drift": "persona drift",
}
OP_LABELS = {
    "mean": "mean pool",
    "mean_pca": "mean pool + PCA-10",
    "max": "max pool",
    "attn_fixed": "random attention pool",
    "attn_learned": "learned attention pool",
    "unpooled": "unpooled (34 positions)",
}
OP_ORDER = ["mean", "mean_pca", "max", "attn_fixed", "attn_learned", "unpooled"]
# One fixed operator -> color map reused across every panel/figure (six
# distinct Wong colors; consistent encoding across facets).
OP_COLORS = dict(zip(OP_ORDER, paper_palette(6), strict=True))

ELIGIBLE = [
    "sycophancy",
    "refusal",
    "harmful_compliance",
    "fact_expression",
    "format_style",
    "self_report",
    "persona_drift",
]  # deception excluded: reliability preflight (split-half r_yy <= 0)


def main() -> None:
    set_paper_style("blog")
    fit = json.loads((EV / "pooling_fit_results.json").read_text())
    results = fit["results"]
    lc = fit["learning_curve"]

    # ── 1. hero: observed Δρ_max vs the selection-symmetric null ──────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    obs_vals, null_dists, pvals = [], [], []
    for beh in ELIGIBLE:
        sel = json.loads((EV / f"selection_matrix_{beh}.json").read_text())
        u = sel["unpooled_vs_mean"]
        null = np.asarray(u["null_matrix"], dtype=float).max(axis=1)
        obs = float(u["observed_max_over_layer_delta"])
        obs_vals.append(obs)
        null_dists.append(null)
        pvals.append((1 + (null >= obs).sum()) / (1 + null.size))
    xs = np.arange(len(ELIGIBLE))
    vp = ax.violinplot(null_dists, positions=xs, widths=0.7, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor("#b8c4d0")
        body.set_alpha(0.6)
    for x, null in zip(xs, null_dists, strict=True):
        ax.hlines(np.percentile(null, 97.5), x - 0.3, x + 0.3, color="#5a6b7c", lw=1.2)
    ax.scatter(
        xs,
        obs_vals,
        color=OP_COLORS["unpooled"],
        zorder=5,
        s=45,
        label="observed",
        linewidths=0,
    )
    for x, o, p in zip(xs, obs_vals, pvals, strict=True):
        ax.text(x + 0.12, o, f"p = {p:.2f}", fontsize=8, va="center")
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels([BEH_LABELS[b] for b in ELIGIBLE], rotation=20, ha="right")
    ax.set_ylabel("max-over-layer gain in held-out Spearman rho\n(unpooled minus mean pool)")
    set_title_subtitle(
        ax,
        "Unpooled gain over the mean pool vs its shuffle null",
        "gray violins: 1000 label-shuffle draws, same max over 28 layers; bar = 97.5th pct",
    )
    savefig_paper(fig, "issue_812/delta_max_vs_null", dir=FIGDIR)
    plt.close(fig)

    # ── 2. grouped bars: best-layer rho per operator ──────────────────────
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    width = 0.13
    for j, op in enumerate(OP_ORDER):
        heights, lo_err, hi_err = [], [], []
        for beh in ELIGIBLE:
            plr = results[beh]["per_layer_rho"][op]
            best_l = max(plr, key=lambda k: plr[k])
            rho = plr[best_l]
            ci = results[beh]["per_layer_ci95"][op][best_l]
            heights.append(rho)
            lo_err.append(rho - ci[0])
            hi_err.append(ci[1] - rho)
        pos = np.arange(len(ELIGIBLE)) + (j - 2.5) * width
        ax.bar(
            pos,
            heights,
            width=width,
            color=OP_COLORS[op],
            yerr=[lo_err, hi_err],
            error_kw={"elinewidth": 0.8, "capsize": 0, "ecolor": "#555555"},
            label=OP_LABELS[op],
        )
    ax.set_xticks(np.arange(len(ELIGIBLE)))
    ax.set_xticklabels([BEH_LABELS[b] for b in ELIGIBLE], rotation=20, ha="right")
    ax.set_ylabel("best-layer held-out Spearman rho")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=8.5, loc="outside lower center", frameon=False)
    set_title_subtitle(
        ax,
        "Best-layer predictivity by pooling operator",
        "LOCO ridge to graded E0, n = 50 contexts; error bars: bootstrap 95% CI (B = 2000)",
    )
    savefig_paper(fig, "issue_812/best_layer_rho_by_operator", dir=FIGDIR)
    plt.close(fig)

    # ── 3. learning curve: delta at the fixed best-mean layer ─────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    beh_colors = dict(zip(ELIGIBLE, paper_palette(7), strict=True))
    for beh in ELIGIBLE:
        curve = lc[beh]["curve"]
        ns = sorted(int(k) for k in curve)
        deltas = np.array([curve[str(n)]["delta"] for n in ns])
        stds = np.array([curve[str(n)]["delta_std"] for n in ns])
        ax.plot(ns, deltas, marker="o", ms=3.5, color=beh_colors[beh], label=BEH_LABELS[beh])
        ax.fill_between(ns, deltas - stds, deltas + stds, color=beh_colors[beh], alpha=0.10, lw=0)
    ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlim(8, 52)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=8, loc="outside lower center", frameon=False)
    ax.set_xlabel("subsampled number of contexts")
    ax.set_ylabel("rho gap: unpooled minus mean pool")
    set_title_subtitle(
        ax,
        "Unpooled minus mean-pool gap vs sample size",
        "20 subsample repeats per point; band = ±1 sd; layer fixed per behavior",
    )
    savefig_paper(fig, "issue_812/learning_curve_delta", dir=FIGDIR)
    plt.close(fig)

    # ── 4. per-layer rho grid (all 8 behaviors, fixed colors) ─────────────
    all_beh = list(results.keys())
    fig, axes = plt.subplots(
        2, 4, figsize=(13, 6.2), sharex=True, sharey=True, constrained_layout=False
    )
    for k, beh in enumerate(all_beh):
        ax = axes.flat[k]
        v = results[beh]
        layers = [int(x) for x in v["layers"]]
        for op in OP_ORDER:
            rhos = [v["per_layer_rho"][op][str(li)] for li in layers]
            ax.plot(layers, rhos, color=OP_COLORS[op], lw=1.1, label=OP_LABELS[op])
        ryy = v["sqrt_r_yy"]
        if ryy is not None:
            ax.axhline(ryy, color="#444444", ls="--", lw=0.9)
        suffix = " (target reliability-excluded)" if v["reliability_excluded"] else ""
        ax.set_title(BEH_LABELS[beh] + suffix, fontsize=9.5)
        ax.axhline(0.0, color="#bbbbbb", lw=0.6, ls=":")
        if k >= 4:
            ax.set_xlabel("layer")
        if k % 4 == 0:
            ax.set_ylabel("held-out Spearman rho")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=6,
        loc="lower center",
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(bottom=0.16, top=0.88, left=0.06, right=0.985)
    fig.text(
        0.5,
        0.94,
        "Per-layer held-out rho, every operator, all 8 behaviors (dashed: split-half ceiling)",
        ha="center",
        fontsize=12,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_812/rho_per_layer_grid", dir=FIGDIR)
    plt.close(fig)

    print("wrote 4 figures to figures/issue_812/")


if __name__ == "__main__":
    main()
