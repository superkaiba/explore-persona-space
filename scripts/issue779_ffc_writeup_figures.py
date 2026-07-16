"""Writeup figures for the #779 fitter-fair-comparison rounds (n5k + n10k).

Two figures, consumed by the context->answer-map writeup draft:

1. ``ffc_scaling_to_n10k`` — held-out test R^2 vs training-set size for the
   four predictors, extending the round-1 curves (n=250..3600) with the
   round-2 n=10,000 points (same sha-pinned 1000-context test set; the
   round-2 fits used an extended lambda grid, so the last segment is dashed).

2. ``ffc_perdirection_spectrum_n10k`` — held-out per-direction R^2 across the
   answer-activation PCA spectrum at n=10k (ridge / RBF kernel ridge / MLP),
   with the n=5k ridge curve as the scale reference, the train-set variance
   share on a right axis, and the three persona vectors placed at their
   equivalent variance ranks.

Inputs are the committed aggregates only (0 GPU):
  eval_results/issue_779/fitter-fair-comparison/scaling_curves.json
  eval_results/issue_779/fitter-fair-comparison/perdirection_per_predictor.json
  eval_results/issue_779/fitter-fair-comparison-n10k/perdirection_per_predictor_n10k.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
FFC = REPO / "eval_results/issue_779/fitter-fair-comparison"
FFC10 = REPO / "eval_results/issue_779/fitter-fair-comparison-n10k"
FFC50 = REPO / "eval_results/issue_779/fitter-fair-comparison-n50k"
FFC1M = REPO / "eval_results/issue_779/fitter-fair-comparison-n1m"

FITTER_LABELS = {
    "ridge": "Ridge (linear)",
    "krr": "RBF kernel ridge",
    "mlp": "MLP (width 8192)",
    "residual_skip": "Ridge + MLP on residuals",
}
FITTER_ORDER = ["ridge", "krr", "mlp", "residual_skip"]

TRAIT_LABELS = {"evil": "evil", "sycophancy": "sycophancy", "hallucination": "hallucination"}


def _title(ax: plt.Axes, title: str, subtitle: str) -> None:
    """set_title_subtitle with a taller pad so a two-line subtitle clears the title."""
    import matplotlib as mpl

    set_title_subtitle(ax, title, subtitle)
    ax.set_title(
        title,
        loc="left",
        color="#1A1A1A",
        fontweight=mpl.rcParams.get("axes.titleweight", "semibold"),
        fontsize=mpl.rcParams.get("axes.titlesize", 13),
        pad=44,
    )


def fig_scaling() -> None:
    curves = json.loads((FFC / "scaling_curves.json").read_text())["curves"]["last_L19"]
    n10k = json.loads((FFC10 / "perdirection_per_predictor_n10k.json").read_text())["per_predictor"]

    r2 = defaultdict(list)
    for e in curves:
        r2[(e["fitter"], e["n"])].append(e["r2"])
    ns = sorted({e["n"] for e in curves})

    colors = dict(zip(FITTER_ORDER, paper_palette(len(FITTER_ORDER)), strict=False))
    fig, ax = plt.subplots()
    for f in FITTER_ORDER:
        means = [float(np.mean(r2[(f, n)])) for n in ns]
        ax.plot(ns, means, marker="o", markersize=4.5, color=colors[f], label=FITTER_LABELS[f])
        for n in ns:  # per-draw points behind the mean line
            draws = r2[(f, n)]
            ax.scatter([n] * len(draws), draws, s=9, color=colors[f], alpha=0.45, zorder=1)
        # round-2 point: same test set, extended lambda grid -> dashed continuation
        y10 = n10k[f]["whole_map_r2"]
        ax.plot([ns[-1], 10_000], [means[-1], y10], linestyle="--", color=colors[f])
        ax.scatter([10_000], [y10], s=34, color=colors[f], zorder=3)

    ax.set_xscale("log")
    ax.set_xticks([250, 500, 1000, 2000, 3600, 10000])
    ax.set_xticklabels(["250", "500", "1,000", "2,000", "3,600", "10,000"])
    ax.minorticks_off()
    ax.set_xlabel("training contexts (log scale)")
    ax.set_ylabel("held-out test R²")
    add_direction_arrow(ax, "y", "up")
    ax.set_ylim(0.45, 0.76)
    ax.legend(loc="lower right")
    _title(
        ax,
        "Context→answer map: held-out R² vs training-set size",
        "Qwen-2.5-7B-Instruct, LMSYS; last-token → mean-answer, layer 19; "
        "fixed 1,000-ctx test set.\n"
        "Dashed: n=10,000 refit (extended λ grid, same test set).",
    )
    savefig_paper(fig, "issue_779/ffc_scaling_to_n10k", dir=REPO / "figures")
    plt.close(fig)


def fig_scaling_n50k() -> None:
    """ffc_scaling_to_n10k extended with the n=50,000 plan-B refit (n50k_fits.json)."""
    curves = json.loads((FFC / "scaling_curves.json").read_text())["curves"]["last_L19"]
    n10k = json.loads((FFC10 / "perdirection_per_predictor_n10k.json").read_text())["per_predictor"]
    n50k = json.loads((FFC50 / "n50k_fits.json").read_text())["per_predictor"]

    r2 = defaultdict(list)
    for e in curves:
        r2[(e["fitter"], e["n"])].append(e["r2"])
    ns = sorted({e["n"] for e in curves})

    colors = dict(zip(FITTER_ORDER, paper_palette(len(FITTER_ORDER)), strict=False))
    fig, ax = plt.subplots()
    for f in FITTER_ORDER:
        means = [float(np.mean(r2[(f, n)])) for n in ns]
        ax.plot(ns, means, marker="o", markersize=4.5, color=colors[f], label=FITTER_LABELS[f])
        for n in ns:  # per-draw points behind the mean line
            draws = r2[(f, n)]
            ax.scatter([n] * len(draws), draws, s=9, color=colors[f], alpha=0.45, zorder=1)
        # round-2 point (n=10k refit) + round-3 point (n=50k plan-B refit), dashed continuation
        y10 = n10k[f]["whole_map_r2"]
        y50 = n50k[f]["whole_map_r2"]
        ci = n50k[f]["bootstrap_ci"]["r2"]
        ax.plot([ns[-1], 10_000, 50_000], [means[-1], y10, y50], linestyle="--", color=colors[f])
        ax.scatter([10_000], [y10], s=34, color=colors[f], zorder=3)
        ax.errorbar(
            [50_000],
            [y50],
            yerr=[[y50 - ci["lo"]], [ci["hi"] - y50]],
            fmt="s",
            markersize=5.5,
            color=colors[f],
            capsize=3,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xticks([250, 500, 1000, 2000, 3600, 10000, 50000])
    ax.set_xticklabels(["250", "500", "1k", "2k", "3.6k", "10k", "50k"])
    ax.minorticks_off()
    ax.set_xlabel("training contexts (log scale)")
    ax.set_ylabel("held-out test R²")
    add_direction_arrow(ax, "y", "up")
    ax.set_ylim(0.45, 0.84)
    ax.legend(loc="lower right")
    _title(
        ax,
        "Context→answer map: held-out R² vs training-set size",
        "Qwen-2.5-7B-Instruct, LMSYS; last-token → mean-answer, layer 19; "
        "fixed 1,000-ctx test set.\n"
        "Dashed: n=10,000 refit; n=50,000 plan-B refit (squares, bootstrap 95% CI; "
        "single fit per predictor).",
    )
    savefig_paper(fig, "issue_779/ffc_scaling_to_n50k", dir=REPO / "figures")
    plt.close(fig)


def fig_scaling_n1m() -> None:
    """ffc_scaling_to_n50k extended with the n1m round (150k/500k pure-LMSYS + ~963k mixed).

    KRR points at 150k+ use the Nystrom estimator (m=16,384; validated vs exact at 50k,
    gap 0.008); the 963k point's corpus is LMSYS+WildChat mixed (500k-mixed control shows
    the mix costs ~0.007 R^2 vs pure-LMSYS at matched n).
    """
    curves = json.loads((FFC / "scaling_curves.json").read_text())["curves"]["last_L19"]
    n10k = json.loads((FFC10 / "perdirection_per_predictor_n10k.json").read_text())["per_predictor"]
    n50k = json.loads((FFC50 / "n50k_fits.json").read_text())["per_predictor"]
    n1m = json.loads((FFC1M / "n1m_fits.json").read_text())["per_point"]

    r2 = defaultdict(list)
    for e in curves:
        r2[(e["fitter"], e["n"])].append(e["r2"])
    ns = sorted({e["n"] for e in curves})

    n1m_key = {
        "ridge": "ridge",
        "krr": "krr_nystrom",
        "mlp": "mlp_w8192",
        "residual_skip": "residual_skip",
    }
    big_pts = [(150_000, "lmsys_150k"), (500_000, "lmsys_500k"), (963_444, "mixed_1m")]

    colors = dict(zip(FITTER_ORDER, paper_palette(len(FITTER_ORDER)), strict=False))
    fig, ax = plt.subplots()
    for f in FITTER_ORDER:
        means = [float(np.mean(r2[(f, n)])) for n in ns]
        ax.plot(ns, means, marker="o", markersize=4.5, color=colors[f], label=FITTER_LABELS[f])
        for n in ns:
            draws = r2[(f, n)]
            ax.scatter([n] * len(draws), draws, s=9, color=colors[f], alpha=0.45, zorder=1)
        xs = [ns[-1], 10_000, 50_000] + [x for x, _ in big_pts]
        ys = [means[-1], n10k[f]["whole_map_r2"], n50k[f]["whole_map_r2"]] + [
            n1m[p]["predictors"][n1m_key[f]]["whole_map_r2"] for _, p in big_pts
        ]
        ax.plot(xs, ys, linestyle="--", color=colors[f])
        ax.scatter(xs[1:3], ys[1:3], s=34, color=colors[f], zorder=3)
        for (x, p), y in zip(big_pts, ys[3:], strict=True):
            ci = n1m[p]["predictors"][n1m_key[f]]["bootstrap_ci"]["r2"]
            ax.errorbar(
                [x],
                [y],
                yerr=[[y - ci["lo"]], [ci["hi"] - y]],
                fmt="s",
                markersize=5.5,
                color=colors[f],
                capsize=3,
                zorder=3,
            )
    # w32768 capacity arm at the big points only (no small-n counterpart)
    w32_xs = [x for x, _ in big_pts]
    w32_ys = [n1m[p]["predictors"]["mlp_w32768"]["whole_map_r2"] for _, p in big_pts]
    ax.plot(
        w32_xs,
        w32_ys,
        linestyle=":",
        marker="D",
        markersize=5,
        color="#555555",
        label="MLP (width 32768)",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xticks([250, 1000, 10000, 50000, 500000])
    ax.set_xticklabels(["250", "1k", "10k", "50k", "500k"])
    ax.minorticks_off()
    ax.set_xlabel("training contexts (log scale)")
    ax.set_ylabel("held-out test R²")
    add_direction_arrow(ax, "y", "up")
    ax.set_ylim(0.45, 0.97)
    ax.axhline(0.947, color="#888888", linewidth=1.0, linestyle="-.", alpha=0.7)
    ax.legend(loc="lower right", fontsize=8)
    _title(
        ax,
        "Context→answer map: held-out R² vs training-set size",
        "Qwen-2.5-7B-Instruct, L19, last-token → mean-answer; fixed 1,000-ctx test.\n"
        "963k point: mixed LMSYS+WildChat; KRR ≥150k = Nyström m=16,384.\n"
        "Dash-dot: single-draw reliability ceiling (0.947).",
    )
    savefig_paper(fig, "issue_779/ffc_scaling_to_n1m", dir=REPO / "figures")
    plt.close(fig)


def fig_spectrum() -> None:
    d10 = json.loads((FFC10 / "perdirection_per_predictor_n10k.json").read_text())
    d5 = json.loads((FFC / "perdirection_per_predictor.json").read_text())
    ranks = np.asarray(d10["ranks_evaluated"]) + 1  # 1-indexed for display
    ranks5 = np.asarray(d5["ranks_evaluated"]) + 1
    share = np.asarray(d10["variance_share_by_rank"])

    colors = paper_palette(3)
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="#9A9A9A", linewidth=0.8)
    ax.plot(
        ranks5,
        d5["per_predictor"]["ridge"]["r2_by_rank"],
        color="#B8B8B8",
        linewidth=1.2,
        label="Ridge, n=5,000 (round 1)",
    )
    for f, c in zip(["ridge", "krr", "mlp"], colors, strict=False):
        ax.plot(
            ranks,
            d10["per_predictor"][f]["r2_by_rank"],
            color=c,
            linewidth=1.5,
            label=f"{FITTER_LABELS[f]}, n=10,000",
        )

    rb = d10["per_predictor"]["ridge"]["r_b"]
    label_pos = {"sycophancy": (5.5, 0.985), "evil": (13.0, 0.85), "hallucination": (1.05, 0.77)}
    for trait, (lx, ly) in label_pos.items():
        x = rb[trait]["equivalent_variance_rank"]
        y = rb[trait]["heldout_r2"]
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=190,
            color="#D62728",
            zorder=5,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.text(
            lx,
            ly,
            TRAIT_LABELS[trait],
            fontsize=9,
            color="#D62728",
            va="center",
        )

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000, 3584])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300", "1,000", "3,584"])
    ax.minorticks_off()
    ax.set_xlabel("answer-activation PCA variance rank (log scale)")
    ax.set_ylabel("held-out per-direction R²")
    add_direction_arrow(ax, "y", "up")
    ax.set_ylim(-0.32, 1.04)
    ax.legend(loc="upper right")

    ax2 = ax.twinx()
    ax2.plot(ranks, share, color="#2CA02C", linestyle=":", linewidth=1.3)
    ax2.set_yscale("log")
    ax2.set_ylabel("train-set variance share per direction (log)", color="#2CA02C")
    ax2.tick_params(axis="y", labelcolor="#2CA02C")
    ax2.grid(False)

    _title(
        ax,
        "Per-direction held-out R² across the answer-PCA spectrum",
        "Layer 19, shared fold-0 basis. Red stars: persona vectors at their\n"
        "equivalent variance rank (ridge, n=10,000). Green dotted: variance share.",
    )
    savefig_paper(fig, "issue_779/ffc_perdirection_spectrum_n10k", dir=REPO / "figures")
    plt.close(fig)


if __name__ == "__main__":
    set_paper_style("blog")
    fig_scaling()
    fig_spectrum()
    print("wrote figures/issue_779/ffc_scaling_to_n10k.{png,pdf,meta.json}")
    print("wrote figures/issue_779/ffc_perdirection_spectrum_n10k.{png,pdf,meta.json}")
