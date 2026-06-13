"""Issue #589 clean-result figures: estimator-fragility sweep.

Two figures, both blog-style:
  1. estimator_sweep  — hero. Two panels (raw / centered), per-row p-value
     pairs (cluster-robust OLS vs persona-RE MixedLM), dashed lines at the
     per-row alpha; FAILED MixedLM cells annotated; flip rows highlighted.
  2. estimator_sweep_505_perarm — the #505 leave-one-out per-arm OLS(HC2)
     slopes (raw + centered), showing the opposing-sign heterogeneity the
     pooled MixedLM is asked to absorb.

Reads eval_results/issue_589/sweep_results.json (no recompute).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "eval_results" / "issue_589" / "sweep_results.json"

# Plain-English row labels (reader-facing). Bare row_ids never reach the axes.
ROW_LABEL = {
    "405-secondary": "Panel-regression\nsecondary",
    "490-distance-adjusted": "On-axis\ndose-matched gap",
    "505-loo-null": "Leave-one-out\npooled",
    "478-flatness-null": "Set-size x distance\ninteraction",
}
ROW_ORDER = ["405-secondary", "490-distance-adjusted", "505-loo-null", "478-flatness-null"]
EST_ORDER = ["cluster_ols", "mixedlm"]
EST_LABEL = {"cluster_ols": "cluster-robust OLS", "mixedlm": "persona-RE MixedLM"}


def _cells_index(data):
    idx = {}
    for c in data["cells"]:
        idx[(c["row_id"], c["estimator"], c["join"])] = c
    return idx


def fig_hero(data, idx):
    set_paper_style("blog")
    c_ols = paper_palette_role("primary")
    c_mlm = paper_palette_role("accent")
    c_fail = paper_palette_role("neutral")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    joins = ["raw", "centered"]
    join_title = {"raw": "Raw distance axis", "centered": "Mean-centered distance axis"}

    x = np.arange(len(ROW_ORDER))
    dx = 0.16

    for ax, join in zip(axes, joins):
        # highlight the flip row (#478): any cell on this row with call_flips
        for i, rid in enumerate(ROW_ORDER):
            flips = any(
                idx[(rid, e, join)].get("call_flips") for e in EST_ORDER if (rid, e, join) in idx
            )
            if flips:
                ax.axvspan(i - 0.42, i + 0.42, color="#E0B834", alpha=0.18, zorder=0)

        for i, rid in enumerate(ROW_ORDER):
            for sgn, est, col in ((-1, "cluster_ols", c_ols), (+1, "mixedlm", c_mlm)):
                cell = idx[(rid, est, join)]
                p = cell["p_value"]
                failed = cell["_status"] == "FAILED"
                # floor p for log display
                p_disp = max(p, 1e-12) if p is not None else 1e-12
                xpos = i + sgn * dx
                if failed:
                    # MixedLM unfit on this cell: mark with an open marker at the
                    # alpha line + "unfit" text; never plot its (untrustworthy) p.
                    ax.scatter(
                        [xpos],
                        [0.05],
                        s=70,
                        facecolors="none",
                        edgecolors=c_fail,
                        linewidths=1.4,
                        marker="o",
                        zorder=5,
                    )
                    ax.annotate(
                        "unfit\n(singular)",
                        (xpos + 0.02, 0.012),
                        fontsize=7.5,
                        color=c_fail,
                        ha="center",
                        va="top",
                    )
                else:
                    ax.scatter(
                        [xpos],
                        [p_disp],
                        s=70,
                        color=col,
                        zorder=5,
                        edgecolors="white",
                        linewidths=0.5,
                    )

        # per-row alpha lines: 0.05 everywhere; #405 also 0.01
        ax.axhline(0.05, ls="--", lw=0.9, color="#5A6975", zorder=1)
        ax.axhline(0.01, ls=":", lw=0.9, color="#9aa5ad", zorder=1)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([ROW_LABEL[r] for r in ROW_ORDER], fontsize=8.5)
        ax.set_xlim(-0.6, len(ROW_ORDER) - 0.4)
        ax.set_ylim(5e-13, 5.0)
        ax.set_title(join_title[join], fontsize=12, loc="left", weight="semibold")
        # alpha annotations on the left panel only
        if join == "raw":
            ax.text(-0.55, 0.05 * 1.5, "p = 0.05", fontsize=8, color="#5A6975", va="bottom")
            ax.text(-0.55, 0.01 / 2.2, "p = 0.01", fontsize=8, color="#9aa5ad", va="top")

    axes[0].set_ylabel("p-value (log scale)")

    # frameless legend (manual handles)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="", color=c_ols, markersize=8, label="cluster-robust OLS"),
        Line2D([], [], marker="o", ls="", color=c_mlm, markersize=8, label="persona-RE MixedLM"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor=c_fail,
            markeredgewidth=1.4,
            markersize=8,
            label="MixedLM unfit (singular)",
        ),
    ]
    axes[0].legend(handles=handles, loc="lower left", frameon=False, fontsize=9)

    fig.suptitle(
        "Published-call p-values under two uncertainty estimators, same point estimate",
        fontsize=13,
        weight="semibold",
        x=0.012,
        ha="left",
    )
    savefig_paper(fig, "issue_589/estimator_sweep", dir="figures/")
    plt.close(fig)


def fig_forest(data, idx):
    """Coefficient +/- Wald CI per (row x estimator x join), showing the
    coefficient is invariant across estimators while the SE/CI moves — the
    King & Roberts 'robust-vs-model SE divergence' shape, read directly.
    Two small-coefficient rows only (#478, #490); the #405/#505 coefficients
    live on a different scale and are read off the hero p-value figure."""
    set_paper_style("blog")
    c_ols = paper_palette_role("primary")
    c_mlm = paper_palette_role("accent")

    # rows whose coefficients are on a comparable small scale and invariant
    rows = ["478-flatness-null", "490-distance-adjusted"]
    row_label = {
        "478-flatness-null": "Set-size x dist.",
        "490-distance-adjusted": "On-axis gap",
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.2))

    ypos = []
    ylab = []
    y = 0
    for rid in rows:
        for join in ("centered", "raw"):
            for est, col in (("cluster_ols", c_ols), ("mixedlm", c_mlm)):
                cell = idx[(rid, est, join)]
                beta = cell["coefficient"]
                lo = beta - cell["ci_lo"]
                hi = cell["ci_hi"] - beta
                ax.errorbar(
                    [beta],
                    [y],
                    xerr=[[lo], [hi]],
                    fmt="o",
                    color=col,
                    capsize=3,
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                )
                ypos.append(y)
                est_short = "OLS" if est == "cluster_ols" else "MixedLM"
                ylab.append(f"{row_label[rid]}  {join} / {est_short}")
                y += 1
            y += 0.4  # gap between estimator pairs
        y += 0.6  # gap between rows

    ax.axvline(0.0, ls="--", lw=0.9, color="#5A6975", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient (Wald 95% CI)")
    ax.set_title(
        "Same coefficient, wider MixedLM interval: the SE moves, not the estimate",
        fontsize=11,
        loc="left",
        weight="semibold",
    )
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="", color=c_ols, markersize=8, label="cluster-robust OLS"),
        Line2D([], [], marker="o", ls="", color=c_mlm, markersize=8, label="persona-RE MixedLM"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    savefig_paper(fig, "issue_589/estimator_sweep_forest", dir="figures/")
    plt.close(fig)


def fig_perarm(data):
    set_paper_style("blog")
    c_raw = paper_palette_role("primary")
    c_ctr = paper_palette_role("baseline")

    per = data["per_arm_505"]["per_arm"]
    arms = list(per["raw"].keys())
    arm_label = {
        "ai_assistant": "AI assistant",
        "child": "child",
        "hero": "hero",
        "quilter": "quilter",
        "veterinarian": "veterinarian",
        "wizard": "wizard",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(arms))
    dx = 0.18

    for sgn, join, col in ((-1, "raw", c_raw), (+1, "centered", c_ctr)):
        betas = [per[join][a]["beta_j"] for a in arms]
        ci = np.array([per[join][a]["ci95"] for a in arms])
        lo = np.array(betas) - ci[:, 0]
        hi = ci[:, 1] - np.array(betas)
        ax.errorbar(
            x + sgn * dx,
            betas,
            yerr=[lo, hi],
            fmt="o",
            color=col,
            capsize=2,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label="raw distance" if join == "raw" else "mean-centered distance",
        )

    ax.axhline(0.0, ls="--", lw=0.9, color="#5A6975", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([arm_label[a] for a in arms], fontsize=9)
    ax.set_ylabel("per-arm slope of leakage change\nvs held-out persona distance")
    ax.set_title(
        "Per-arm slopes carry opposing signs the pooled fit averages to ~0",
        fontsize=12,
        loc="left",
        weight="semibold",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    savefig_paper(fig, "issue_589/estimator_sweep_505_perarm", dir="figures/")
    plt.close(fig)


def main():
    data = json.loads(SWEEP.read_text())
    idx = _cells_index(data)
    fig_hero(data, idx)
    fig_forest(data, idx)
    fig_perarm(data)
    print("wrote figures/issue_589/estimator_sweep.{png,pdf,meta.json}")
    print("wrote figures/issue_589/estimator_sweep_forest.{png,pdf,meta.json}")
    print("wrote figures/issue_589/estimator_sweep_505_perarm.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
