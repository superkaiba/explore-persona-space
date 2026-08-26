"""Figures for #1901 round `mlp-scaling-densify` (plan v15 §6).

Renders, from committed JSONs only (zero refit):
  1. issue_1901/mlp_scaling_dense_L19          — hero: within-store dense ladder
     (held-out R2 main + baseline strip; retrieval acc@1 panel), banked
     other-store points as open markers, 963k banked-weight points detached.
  2. issue_1901/mlp_scaling_dense_delta_companion — per-rung paired
     MLP-minus-ridge R2 deltas with context-bootstrap CIs + the verdict
     slope-gap bootstrap distribution.
  3. issue_1901/mlp_scaling_dense_store_seam  — banked scale7-store points vs
     fresh within-store points at the shared sizes (the G1 finding view).

Usage: uv run python scripts/issue1901_mlpdense_fold_figures.py [--out-dir figures/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

PD = Path("eval_results/issue_1901/paper_densify")
NS = [5000, 10000, 25000, 50000, 100000, 150000, 250000, 500000]
N_TOP = 963444
N_TEST = 1000
DUP_CEILING = 0.94  # 58/1000 exact in-pool duplicate targets (parent battery read)

C = paper_palette_blog(4)
COL_RIDGE, COL_MLP, COL_IB, COL_MEAN = C[0], C[1], C[2], "0.55"


def _load() -> tuple[dict, dict, dict, dict, dict]:
    agg = json.loads((PD / "mlp_scaling_dense_L19.json").read_text())
    boot = json.loads((PD / "dgap_context_bootstrap.json").read_text())
    banked_mlp = json.loads((PD / "mlp_scaling_L19.json").read_text())
    ladder = json.loads((PD / "scaling_ladder_L19.json").read_text())
    return agg, boot, banked_mlp, ladder, agg["per_n"]


def _series(per_n: dict, arm: str, field: str) -> np.ndarray:
    if field == "r2":
        return np.array([per_n[str(n)][arm]["test_r2"] for n in NS])
    if field == "r2_lo":
        return np.array([per_n[str(n)][arm]["test_ci"]["r2"]["lo"] for n in NS])
    if field == "r2_hi":
        return np.array([per_n[str(n)][arm]["test_ci"]["r2"]["hi"] for n in NS])
    if field == "wcsls1":
        return np.array([per_n[str(n)][arm]["whitened_csls"]["acc_at_k"]["1"] for n in NS])
    if field == "euclid1":
        return np.array([per_n[str(n)][arm]["knn"]["euclidean"]["acc_at_k"]["1"] for n in NS])
    raise KeyError(field)


def _xticks(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(NS + [N_TOP])
    ax.set_xticklabels(["5k", "10k", "25k", "50k", "100k", "150k", "250k", "500k", "963k"])
    ax.set_xticks([], minor=True)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def fig_hero(agg: dict, banked_mlp: dict, ladder: dict, out_dir: str) -> None:
    per_n = agg["per_n"]
    top = agg["per_n"][str(N_TOP)]
    fig = plt.figure(figsize=(11.5, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.15], hspace=0.10, wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_acc = fig.add_subplot(gs[:, 1])

    # main R2 panel: fresh within-store curves with bootstrap CIs
    for arm, col, label in (
        ("mlp", COL_MLP, "neural map (fresh, within-store)"),
        ("ridge", COL_RIDGE, "linear ridge (fresh, within-store)"),
    ):
        r2 = _series(per_n, arm, "r2")
        lo, hi = _series(per_n, arm, "r2_lo"), _series(per_n, arm, "r2_hi")
        ax.errorbar(
            NS,
            r2,
            yerr=[r2 - lo, hi - r2],
            color=col,
            marker="o",
            ms=4.5,
            lw=1.8,
            capsize=2.0,
            label=label,
            zorder=3,
        )
        ax.scatter(
            [N_TOP],
            [top[arm]["test_r2"]],
            marker="D",
            s=42,
            facecolors="none",
            edgecolors=col,
            linewidths=1.4,
            zorder=3,
        )
    # banked other-store anchors (open squares): MLP 5k/10k + ridge ladder 5k-25k
    bm = banked_mlp["per_n"]
    ax.scatter(
        [5000, 10000],
        [bm["5000"]["test_r2"], bm["10000"]["test_r2"]],
        marker="s",
        s=40,
        facecolors="none",
        edgecolors=COL_MLP,
        linewidths=1.4,
        zorder=2,
    )
    lad_pts = [
        (c["n_train"], c["ridge"]["test_r2"])
        for c in ladder["cells"]
        if c["n_train"] in (5000, 10000, 15000, 20000, 25000)
    ]
    ax.scatter(
        [p[0] for p in lad_pts],
        [p[1] for p in lad_pts],
        marker="s",
        s=40,
        facecolors="none",
        edgecolors=COL_RIDGE,
        linewidths=1.4,
        zorder=2,
    )
    open_proxy = plt.Line2D(
        [],
        [],
        marker="s",
        ls="none",
        mfc="none",
        mec="0.35",
        markeredgewidth=1.4,
        label="banked points, other store (scored on different held-out rows)",
    )
    diamond_proxy = plt.Line2D(
        [],
        [],
        marker="D",
        ls="none",
        mfc="none",
        mec="0.35",
        markeredgewidth=1.4,
        label="963k banked weights (mixed-corpus pool)",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [open_proxy, diamond_proxy], loc="lower right", fontsize=8)
    ax.set_ylabel("held-out R² (1,000 pinned contexts)")
    ax.set_ylim(0.60, 0.85)
    ax.set_title("Held-out R² vs training-set size", loc="left")
    plt.setp(ax.get_xticklabels(), visible=False)

    # baseline strip
    ib = np.array([per_n[str(n)]["identity_bias"]["test_r2"] for n in NS])
    ax_strip.plot(
        NS, ib, color=COL_IB, marker="o", ms=3.5, lw=1.5, label="identity plus learned bias"
    )
    ax_strip.scatter(
        [N_TOP],
        [top["identity_bias"]["test_r2"]],
        marker="D",
        s=36,
        facecolors="none",
        edgecolors=COL_IB,
        linewidths=1.3,
    )
    ax_strip.axhline(
        agg["controls"]["constant_train_mean_963k"]["test_r2"],
        color=COL_MEAN,
        lw=1.4,
        ls=":",
        label="constant train mean (banked floor)",
    )
    ax_strip.set_ylim(-1.05, 0.1)
    ax_strip.set_yticks([-0.9, 0.0])
    ax_strip.set_ylabel("baselines")
    ax_strip.legend(loc="center right", fontsize=7.5)
    _xticks(ax_strip)
    ax_strip.set_xlabel("training rows (log scale)")

    # retrieval panel
    for arm, col, name in (("mlp", COL_MLP, "neural map"), ("ridge", COL_RIDGE, "linear ridge")):
        w = _series(per_n, arm, "wcsls1")
        e = _series(per_n, arm, "euclid1")
        werr = np.array([proportion_ci(p, N_TEST) for p in w])
        eerr = np.array([proportion_ci(p, N_TEST) for p in e])
        ax_acc.errorbar(
            NS,
            w,
            yerr=[w - werr[:, 0], werr[:, 1] - w],
            color=col,
            marker="o",
            ms=4,
            lw=1.8,
            capsize=2.0,
            label=f"{name}, whitened cosine + CSLS",
        )
        ax_acc.errorbar(
            NS,
            e,
            yerr=[e - eerr[:, 0], eerr[:, 1] - e],
            color=col,
            marker="o",
            ms=4,
            lw=1.4,
            ls="--",
            capsize=2.0,
            label=f"{name}, raw euclidean",
        )
        ax_acc.scatter(
            [N_TOP, N_TOP],
            [
                top[arm]["whitened_csls"]["acc_at_k"]["1"],
                top[arm]["knn"]["euclidean"]["acc_at_k"]["1"],
            ],
            marker="D",
            s=38,
            facecolors="none",
            edgecolors=col,
            linewidths=1.3,
        )
    ax_acc.axhline(DUP_CEILING, color="0.55", lw=1.2, ls=":", label="duplicate-vector ceiling")
    ax_acc.set_ylabel("retrieval accuracy at rank 1 (pool of 1,000)")
    ax_acc.set_ylim(0.63, 0.97)
    ax_acc.set_title("Retrieval accuracy vs training-set size", loc="left")
    ax_acc.legend(loc="lower right", fontsize=8)
    _xticks(ax_acc)
    ax_acc.set_xlabel("training rows (log scale)")

    savefig_paper(fig, "issue_1901/mlp_scaling_dense_L19", dir=out_dir)
    plt.close(fig)


def fig_delta_companion(agg: dict, boot: dict, out_dir: str) -> None:
    fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    d = boot["per_rung_mlp_minus_ridge_r2"]
    pts = np.array([d[str(n)]["point"] for n in NS])
    lo = np.array([d[str(n)]["lo"] for n in NS])
    hi = np.array([d[str(n)]["hi"] for n in NS])
    ax.axhline(0.0, color="0.6", lw=1.0)
    ax.errorbar(
        NS,
        pts,
        yerr=[pts - lo, hi - pts],
        color=COL_MLP,
        marker="o",
        ms=4.5,
        lw=1.8,
        capsize=2.5,
        label="neural minus ridge, paired bootstrap 95% CI",
    )
    ax.set_ylabel("held-out R² difference, neural minus ridge")
    ax.set_title("Per-rung paired difference", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    _xticks(ax)
    ax.set_xlabel("training rows (log scale)")

    # endpoint seed points (fit-noise view) as rug on the delta panel
    for n in (50000, 500000):
        seeds = agg["per_n"][str(n)]["mlp"]["seeds"]
        ridge = agg["per_n"][str(n)]["ridge"]["test_r2"]
        for s, blk in seeds.items():
            ax.scatter(
                [n],
                [blk["test_r2"] - ridge],
                marker="x",
                s=26,
                color="0.35",
                linewidths=1.2,
                zorder=4,
                label="per-seed neural fits (3 seeds)" if (n, s) == (50000, "42") else None,
            )
    ax.legend(loc="upper left", fontsize=8)

    vb = boot["verdict_bootstrap"]
    draws_note = f"n={boot['n_boot']} draws"
    gap_pt = vb["slope_gap"]["mean"]
    # histogram of the slope-gap bootstrap draws is recomputable; here we draw the CI band
    ax_h.axvspan(
        vb["slope_gap"]["lo"],
        vb["slope_gap"]["hi"],
        color=COL_MLP,
        alpha=0.25,
        label=f"slope-gap 95% CI ({draws_note})",
    )
    ax_h.axvline(gap_pt, color=COL_MLP, lw=1.8, label="slope-gap bootstrap mean")
    ax_h.axvline(0.01, color="0.3", lw=1.4, ls="--", label="verdict margin")
    ax_h.axvline(0.0, color="0.6", lw=1.0)
    ax_h.set_xlim(-0.005, 0.045)
    ax_h.set_yticks([])
    ax_h.set_xlabel("slope gap over 50k to 500k, neural minus ridge (R² units)")
    ax_h.set_title("Verdict quantity vs its margin", loc="left")
    ax_h.legend(loc="upper left", fontsize=8)

    savefig_paper(fig, "issue_1901/mlp_scaling_dense_delta_companion", dir=out_dir)
    plt.close(fig)


def fig_store_seam(agg: dict, banked_mlp: dict, ladder: dict, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    per_n = agg["per_n"]
    bm = banked_mlp["per_n"]
    lad = {
        c["n_train"]: c["ridge"]["test_r2"]
        for c in ladder["cells"]
        if c["n_train"] in (5000, 10000, 25000)
    }
    for n in (5000, 10000, 25000):
        # ridge pair (banked ladder value exists at all three)
        fresh_r = per_n[str(n)]["ridge"]["test_r2"]
        ax.plot([n, n], [lad[n], fresh_r], color=COL_RIDGE, lw=1.0, alpha=0.7)
        ax.scatter(
            [n], [lad[n]], marker="s", s=46, facecolors="none", edgecolors=COL_RIDGE, linewidths=1.5
        )
        ax.scatter([n], [fresh_r], marker="o", s=42, color=COL_RIDGE)
        # mlp pair (banked exists at 5k/10k)
        if str(n) in bm:
            x = n * 1.12  # slight offset so the arms don't overlap
            fresh_m = per_n[str(n)]["mlp"]["test_r2"]
            ax.plot([x, x], [bm[str(n)]["test_r2"], fresh_m], color=COL_MLP, lw=1.0, alpha=0.7)
            ax.scatter(
                [x],
                [bm[str(n)]["test_r2"]],
                marker="s",
                s=46,
                facecolors="none",
                edgecolors=COL_MLP,
                linewidths=1.5,
            )
            ax.scatter([x], [fresh_m], marker="o", s=42, color=COL_MLP)
    proxies = [
        plt.Line2D([], [], marker="o", ls="none", color=COL_MLP, label="neural map"),
        plt.Line2D([], [], marker="o", ls="none", color=COL_RIDGE, label="linear ridge"),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color="0.35",
            label="filled: fresh, within-store (pinned pool)",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            ls="none",
            mfc="none",
            mec="0.35",
            markeredgewidth=1.5,
            label="open: banked, other store (its own eval rows)",
        ),
    ]
    ax.legend(handles=proxies, loc="lower right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([5000, 10000, 25000])
    ax.set_xticklabels(["5k", "10k", "25k"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(3800, 40000)
    ax.set_ylabel("held-out R²")
    ax.set_xlabel("training rows (log scale)")
    ax.set_title("Same nominal training size, two different evaluation pools", loc="left")
    savefig_paper(fig, "issue_1901/mlp_scaling_dense_store_seam", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures/")
    args = ap.parse_args()
    set_paper_style("blog")
    agg, boot, banked_mlp, ladder, _ = _load()
    fig_hero(agg, banked_mlp, ladder, args.out_dir)
    fig_delta_companion(agg, boot, args.out_dir)
    fig_store_seam(agg, banked_mlp, ladder, args.out_dir)
    print("done")


if __name__ == "__main__":
    main()
