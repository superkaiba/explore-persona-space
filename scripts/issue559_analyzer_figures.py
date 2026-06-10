"""Analyzer-stage reader-facing figures for issue #559.

Reads the frozen production JSONs (within_run_ranking.json, joint_fit.json)
produced by scripts/issue559_panel_analysis.py and re-renders the two figures
the clean-result body embeds with plain-English labels:

1. within_run_ranking_strip  -- hero: per-run Spearman rho strips for the four
   rankers (matched-slot, own-response prior, distance-to-nearest-source,
   z-stack), median bars, blog style. Re-render of the production figure with
   an unclipped y-label and plain-English tick labels.
2. joint_fit_forest_plain    -- forest plot of the two-ingredient joint-fit
   coefficients (level + change DVs) with persona-cluster (primary) CIs and
   plain-English row labels, including the registered residualized /
   matched-slot-augmented alpha reads.

Every plotted number is read from the frozen JSONs; nothing is recomputed.

Usage:
    uv run python scripts/issue559_analyzer_figures.py \
        --in-dir <path to eval_results/issue_559> --fig-dir figures/issue_559
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

RANKER_ORDER = ["margin_base", "prior_margin_own", "min_dist", "z_stack"]
RANKER_LABELS = {
    "margin_base": "base matched-slot\nmargin\n(needs trained responses)",
    "prior_margin_own": "own-response\nprior (NEW,\npre-training)",
    "min_dist": "distance to\nnearest source\n(pre-training)",
    "z_stack": "prior + distance\nstack\n(pre-training)",
}


def fig_ranking_strip(ranking: dict, fig_dir: str) -> None:
    set_paper_style("blog")
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, key in enumerate(RANKER_ORDER):
        rhos = np.array(list(ranking["within_run_ranking"][key]["per_run_rho"].values()))
        color = (
            paper_palette_role("baseline") if key == "margin_base" else paper_palette_role("primary")
        )
        x = i + rng.uniform(-0.16, 0.16, size=len(rhos))
        ax.scatter(x, rhos, s=14, alpha=0.55, color=color, edgecolors="none", zorder=2)
        med = ranking["within_run_ranking"][key]["median_rho"]
        ax.hlines(med, i - 0.28, i + 0.28, color=color, lw=3.0, zorder=3)
    ax.axhline(0.0, color="0.45", lw=1.0, zorder=1)
    ax.set_xticks(range(len(RANKER_ORDER)))
    ax.set_xticklabels([RANKER_LABELS[k] for k in RANKER_ORDER], fontsize=9)
    ax.set_ylabel("per-run Spearman ρ vs trained margin", fontsize=10)
    ax.set_title(
        "Within-run ranking of the 35 held-out personas (80 runs)\n"
        "orange = needs the trained model's responses; blue = computable before training",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
        pad=12,
    )
    savefig_paper(fig, "issue_559/within_run_ranking_strip", dir=fig_dir)
    plt.close(fig)


def fig_joint_fit_forest(jf: dict, fig_dir: str) -> None:
    set_paper_style("blog")
    lvl = jf["level_fit"]["variants"]["base"]["coefficients"]
    chg = jf["change_fit"]["variants"]["base"]["coefficients"]
    resid_lvl = jf["poly_residualization_level"]
    aug = jf["change_fit_margin_base_added"]

    rows = [
        (
            "LEVEL DV — own-response prior (α)",
            lvl["alpha_prior"]["estimate"],
            lvl["alpha_prior"]["primary_ci"],
            "level",
        ),
        (
            "LEVEL DV — prior residualized on distance (α)",
            resid_lvl["alpha_resid_prior"],
            resid_lvl["alpha_resid_prior_ci95_persona_cluster"],
            "level",
        ),
        (
            "LEVEL DV — distance to nearest source (β)",
            lvl["beta_min_dist"]["estimate"],
            lvl["beta_min_dist"]["primary_ci"],
            "level",
        ),
        (
            "CHANGE DV — own-response prior (α)",
            chg["alpha_prior"]["estimate"],
            chg["alpha_prior"]["primary_ci"],
            "change",
        ),
        (
            "CHANGE DV — prior, with base matched-slot\nmargin in the model (α)",
            aug["estimates"]["alpha_prior"],
            aug["ci95_persona_cluster"]["alpha_prior"],
            "change",
        ),
        (
            "CHANGE DV — distance to nearest source (β)",
            chg["beta_min_dist"]["estimate"],
            chg["beta_min_dist"]["primary_ci"],
            "change",
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ys = np.arange(len(rows))[::-1]
    for y, (label, est, ci, dv) in zip(ys, rows):
        color = paper_palette_role("primary") if dv == "level" else paper_palette_role("baseline")
        ax.plot([ci["low"], ci["high"]], [y, y], color=color, lw=2.6, zorder=2)
        ax.scatter([est], [y], color=color, s=46, zorder=3)
    ax.axvline(0.0, color="0.45", lw=1.0, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("standardized coefficient (persona-cluster 95% CI)", fontsize=10)
    ax.set_title(
        "Two-ingredient joint fit: the prior carries the level, distance carries the change\n"
        "blue = level DV (trained margin); orange = change DV (trained − base margin)",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
        pad=12,
    )
    savefig_paper(fig, "issue_559/joint_fit_forest_plain", dir=fig_dir)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="eval_results/issue_559")
    parser.add_argument("--fig-dir", default="figures/")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    ranking = json.loads((in_dir / "within_run_ranking.json").read_text())
    jf = json.loads((in_dir / "joint_fit.json").read_text())

    fig_ranking_strip(ranking, args.fig_dir)
    fig_joint_fit_forest(jf, args.fig_dir)
    print("done")


if __name__ == "__main__":
    main()
