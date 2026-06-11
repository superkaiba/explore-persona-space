"""Analyzer-stage reader-facing figures for issue #559.

Reads the frozen production artifacts (within_run_ranking.json, joint_fit.json,
base_prior_own_persona_panel.json, the #478 tidy_logit.parquet) produced by
scripts/issue559_panel_analysis.py and re-renders the figures the clean-result
body embeds with plain-English labels:

1. within_run_ranking_strip  -- hero: per-run Spearman rho strips for the four
   rankers (matched-slot, own-response prior, distance-to-nearest-source,
   z-stack), median bars, blog style. Re-render of the production figure with
   an unclipped y-label and plain-English tick labels.
2. joint_fit_forest_plain    -- forest plot of the two-ingredient joint-fit
   coefficients (level + change outcomes) with persona-cluster (primary)
   intervals and plain-English row labels (no Greek / no "DV"), including the
   registered residualized / matched-slot-augmented prior reads.
3. prior_persona_scatters    -- the three persona-level scatters (prior vs
   raw trained margin, vs run-offset-removed trained margin, vs matched-slot
   base margin) with plain-English axis labels (no "run-FE").

Every plotted number is read from the frozen artifacts; aggregation for the
scatters reproduces the production groupby (run x persona means, balanced
80 x 35 panel) verbatim.

Usage:
    uv run python scripts/issue559_analyzer_figures.py \
        --in-dir <path to eval_results/issue_559> --fig-dir figures/issue_559 \
        --parquet eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
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
            paper_palette_role("baseline")
            if key == "margin_base"
            else paper_palette_role("primary")
        )
        x = i + rng.uniform(-0.16, 0.16, size=len(rhos))
        ax.scatter(x, rhos, s=14, alpha=0.55, color=color, edgecolors="none", zorder=2)
        med = ranking["within_run_ranking"][key]["median_rho"]
        ax.hlines(med, i - 0.28, i + 0.28, color=color, lw=3.0, zorder=3)
    ax.axhline(0.0, color="0.45", lw=1.0, zorder=1)
    ax.set_xticks(range(len(RANKER_ORDER)))
    ax.set_xticklabels([RANKER_LABELS[k] for k in RANKER_ORDER], fontsize=9)
    ax.set_ylabel("per-run Spearman ρ vs trained margin", fontsize=10)  # noqa: RUF001
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
            "Level — own-response prior",
            lvl["alpha_prior"]["estimate"],
            lvl["alpha_prior"]["primary_ci"],
            "level",
        ),
        (
            "Level — prior residualized on distance",
            resid_lvl["alpha_resid_prior"],
            resid_lvl["alpha_resid_prior_ci95_persona_cluster"],
            "level",
        ),
        (
            "Level — distance to nearest source",
            lvl["beta_min_dist"]["estimate"],
            lvl["beta_min_dist"]["primary_ci"],
            "level",
        ),
        (
            "Change — own-response prior",
            chg["alpha_prior"]["estimate"],
            chg["alpha_prior"]["primary_ci"],
            "change",
        ),
        (
            "Change — prior, with base matched-slot\nmargin in the model",
            aug["estimates"]["alpha_prior"],
            aug["ci95_persona_cluster"]["alpha_prior"],
            "change",
        ),
        (
            "Change — distance to nearest source",
            chg["beta_min_dist"]["estimate"],
            chg["beta_min_dist"]["primary_ci"],
            "change",
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ys = np.arange(len(rows))[::-1]
    for y, (_label, est, ci, dv) in zip(ys, rows, strict=True):
        color = paper_palette_role("primary") if dv == "level" else paper_palette_role("baseline")
        ax.plot([ci["low"], ci["high"]], [y, y], color=color, lw=2.6, zorder=2)
        ax.scatter([est], [y], color=color, s=46, zorder=3)
    ax.axvline(0.0, color="0.45", lw=1.0, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("standardized coefficient (persona-cluster 95% interval)", fontsize=10)
    ax.set_title(
        "Two-ingredient joint fit: the prior carries the level, distance carries the change\n"
        "blue = effect on the trained margin (level); "
        "orange = effect on the training-induced change",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
        pad=12,
    )
    savefig_paper(fig, "issue_559/joint_fit_forest_plain", dir=fig_dir)
    plt.close(fig)


def fig_persona_scatters(prior_json: Path, parquet: Path, fig_dir: str) -> None:
    """Re-render prior_persona_scatters with plain-English labels (no 'run-FE').

    Reproduces the production aggregation verbatim: (run x persona) means over
    the 20 questions on the balanced 80 x 35 panel, then persona means; the
    middle panel removes each run's overall offset before taking persona means.
    """
    set_paper_style("blog")
    payload = json.loads(prior_json.read_text())
    prior = {k: v["prior_margin_own"] for k, v in payload["per_persona"].items()}

    df = pd.read_parquet(parquet)
    assert df.shape[0] == 56_000, df.shape
    df = df.copy()
    df["run_id"] = df["cell_id"] + "_seed" + df["seed"].astype(str)
    agg = df.groupby(["run_id", "held_out_persona"], as_index=False).agg(
        margin_trained=("margin_trained", "mean"),
        margin_base=("margin_base", "mean"),
    )
    assert len(agg) == 2_800, len(agg)

    per = agg.groupby("held_out_persona").agg(
        mt=("margin_trained", "mean"),
        mb=("margin_base", "mean"),
    )
    resid = agg["margin_trained"] - agg.groupby("run_id")["margin_trained"].transform("mean")
    per["mt_fe"] = resid.groupby(agg["held_out_persona"]).mean()
    per["prior"] = per.index.map(prior)
    assert not per["prior"].isna().any(), "persona mismatch joining the prior"
    assert len(per) == 35, len(per)

    colors = paper_palette(3)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    panels = [
        ("mt", "persona-mean trained margin (raw)", colors[0]),
        ("mt_fe", "persona-mean trained margin\n(each run's overall offset removed)", colors[1]),
        ("mb", "persona-mean matched-slot base margin", colors[2]),
    ]
    for ax, (col, label, color) in zip(axes, panels, strict=True):
        ax.plot(per["prior"], per[col], "o", ms=4.5, alpha=0.75, color=color)
        rho, _ = spearmanr(per["prior"].to_numpy(), per[col].to_numpy())
        ax.set_title(f"rank correlation = {rho:+.2f} (35 personas)", fontsize=9)
        ax.set_xlabel("own-response prior margin")
        ax.set_ylabel(label, fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "issue_559/prior_persona_scatters", dir=fig_dir)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="eval_results/issue_559")
    parser.add_argument("--fig-dir", default="figures/")
    parser.add_argument(
        "--parquet",
        default="eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet",
        help="#478 panel parquet (for the persona scatters)",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    ranking = json.loads((in_dir / "within_run_ranking.json").read_text())
    jf = json.loads((in_dir / "joint_fit.json").read_text())

    fig_ranking_strip(ranking, args.fig_dir)
    fig_joint_fit_forest(jf, args.fig_dir)
    fig_persona_scatters(
        in_dir / "base_prior_own_persona_panel.json", Path(args.parquet), args.fig_dir
    )
    print("done")


if __name__ == "__main__":
    main()
