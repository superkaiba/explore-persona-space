"""Regenerate two issue-2202 figures that rendered unreadably in the P5 driver output.

1. ``fig_indegree_v2`` — the driver's ``fig_indegree.png`` drew step histograms whose
   patch edge width the blog style zeroes (the #613/#1902 invisible-step-hist class),
   leaving empty axes. Redrawn with explicit ``linewidth``.
2. ``fig_reciprocity_bands_log`` — the driver's ``fig_reciprocity_bands.png`` used a
   linear y-axis on [0, 1], squashing the observed value (8.4e-4) and both null bands
   (6e-4 .. 3.2e-3) into one invisible sliver at zero. Redrawn on a log axis.

Reads only committed eval_results/issue_2202 JSONs; writes PNG+PDF+meta sidecars via
``savefig_paper``.
"""

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

HERE = Path(__file__).resolve().parent.parent
EV = HERE / "eval_results" / "issue_2202"
OUT = "issue_2202"


def fig_indegree_v2() -> None:
    hub = json.loads((EV / "hubness.json").read_text())
    ret = np.asarray(hub["retrieval"]["counts"], dtype=int)
    col = np.asarray(hub["collapse"]["counts"], dtype=int)
    colors = paper_palette(2)
    fig, ax = plt.subplots()
    bins = np.arange(0, max(ret.max(), col.max()) + 2)
    ax.hist(
        ret,
        bins=bins,
        histtype="step",
        linewidth=1.6,
        color=colors[0],
        label=f"retrieval in-degree (skew {hub['retrieval']['n10_skewness']:.1f})",
    )
    ax.hist(
        col,
        bins=bins,
        histtype="step",
        linewidth=1.6,
        color=colors[1],
        label=f"prediction-collapse in-degree (skew {hub['collapse']['n10_skewness']:.1f})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("times a pool answer appears in a top-10 list (in-degree)")
    ax.set_ylabel("number of pool answers (log)")
    ax.legend()
    set_title_subtitle(ax, "Top-10 in-degree is heavy-tailed in both graphs")
    savefig_paper(fig, f"{OUT}/fig_indegree_v2", dir="figures/")
    plt.close(fig)


def fig_reciprocity_bands_log() -> None:
    rec = json.loads((EV / "reciprocity.json").read_text())
    obs = rec["observed"]["reciprocity"]
    bands = [("degree-preserving", np.asarray(rec["null_degree"]["draws"]))]
    for tau in ("p1", "p5", "p25"):
        bands.append((f"distance-only τ={tau}", np.asarray(rec["null_distance"][tau]["draws"])))
    colors = paper_palette(3)
    fig, ax = plt.subplots()
    for i, (name, draws) in enumerate(bands):
        lo, med, hi = np.percentile(draws, [2.5, 50, 97.5])
        ax.errorbar(
            [i],
            [med],
            yerr=[[med - lo], [hi - med]],
            fmt="o",
            color=colors[0],
            capsize=5,
            markeredgewidth=1.2,
            elinewidth=1.6,
            label="null band (2.5th-97.5th percentile)" if i == 0 else None,
        )
    ax.axhline(obs, color=colors[1], linewidth=1.6, label=f"observed ({obs:.1e})")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.2, label="ceiling (reciprocity ≤ 1)")
    ax.set_yscale("log")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([n for n, _ in bands], rotation=15, ha="right")
    ax.set_ylabel("top-1 confusion reciprocity (log)")
    ax.legend()
    set_title_subtitle(
        ax, "Observed reciprocity sits inside the degree band, below the distance bands"
    )
    savefig_paper(fig, f"{OUT}/fig_reciprocity_bands_log", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    set_paper_style("blog")
    fig_indegree_v2()
    fig_reciprocity_bands_log()
    print("done")
