"""Hero figure for the #537 predictor-bakeoff-complete round (the publishable null).

Per behavior, plots the BEST predictor's own out-of-fold R^2 (with the clustered
bootstrap CI) against the `base_prior_bystander` baseline's own out-of-fold R^2,
on a shared axis, so the reader sees two facts at once:

1. No predictor's own OOF-R^2 clears 0 meaningfully on any behavior (all best
   points sit in a thin band near 0).
2. The large `delta_vs_base_prior_r2` values for sycophancy (+3.10) and em
   (+1.72) are baseline-inflation: the base prior's own OOF-R^2 collapses far
   below 0 on those behaviors, manufacturing a big positive delta from a
   near-zero predictor.

Run from repo root: uv run python scripts/i537_bakeoff_null_figure.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

SCORING = Path(
    ".claude/worktrees/issue-537/eval_results/issue_537/predictor-bakeoff-complete/scoring"
)

# Plain-English behavior labels (reader-facing); ordered as in the body.
BEHAVIORS = [
    ("marker", "Marker tic"),
    ("fact", "Taught fact"),
    ("refusal", "Blanket refusal"),
    ("sycophancy", "Sycophancy"),
    ("em", "Harmful advice"),
]
# Each behavior's best predictor by delta_vs_base_prior_r2 (the §6 sort key),
# re-extracted from the leaderboard at plot time (NOT transcribed).
BEST_KEY = {
    "marker": "marker:js_out_seq_rb",
    "fact": "fact:js_out_seq_rb",
    "refusal": "refusal:kl_out_seq_rev",
    "sycophancy": "sycophancy:gauss_kl_act",
    "em": "em:gauss_kl_act",
}


def main() -> None:
    rows = []
    for b, label in BEHAVIORS:
        d = json.loads((SCORING / f"per_behavior_{b}.json").read_text())
        sc = d["scores"]
        best = sc[BEST_KEY[b]]
        base = sc[f"{b}:base_prior_bystander"]
        bs = best.get("bootstrap", {})
        skipped = d.get("skipped_rows", {})
        n_skip = (
            sum(len(v) for v in skipped.values()) if isinstance(skipped, dict) else len(skipped)
        )
        rows.append(
            dict(
                behavior=b,
                label=label,
                best_pred=BEST_KEY[b].split(":")[1],
                best_oof=best["oof_r2"],
                ci_lo=bs.get("ci_lo"),
                ci_hi=bs.get("ci_hi"),
                base_oof=base["oof_r2"],
                delta=best["delta_vs_base_prior_r2"],
                n=best["n_cells"],
                clean=(n_skip == 0),
            )
        )

    set_paper_style("blog")
    # Single-axis blog title block collides with constrained_layout (collapses
    # axes to zero); disable it and set margins explicitly (analyzer memo).
    fig, ax = plt.subplots(figsize=(8.4, 4.7), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.80, bottom=0.13)

    y = np.arange(len(rows))[::-1]  # top = marker
    c_pred = paper_palette_role("primary")
    c_base = paper_palette_role("baseline")

    # base-prior markers (the inflation driver)
    base_oof = [r["base_oof"] for r in rows]
    ax.scatter(
        base_oof,
        y,
        marker="s",
        s=70,
        facecolors="none",
        edgecolors=c_base,
        linewidths=1.6,
        zorder=3,
        label="base-rate prior (the baseline)",
    )
    # best-predictor: point marker for the headline (pooled-fold) OOF-R^2, and a
    # SEPARATE bootstrap-CI segment. They are decoupled on purpose: on marker and
    # fact the CI does not even bracket the point estimate, which is the honest
    # tell that the OOF-R^2 read is unstable — an asymmetric errorbar would hide it.
    best_oof = [r["best_oof"] for r in rows]
    for r, yy in zip(rows, y):
        lo_v, hi_v = r["ci_lo"], r["ci_hi"]
        if lo_v is not None and hi_v is not None and np.isfinite(lo_v) and np.isfinite(hi_v):
            ax.plot(
                [lo_v, hi_v],
                [yy, yy],
                color=c_pred,
                lw=1.6,
                alpha=0.55,
                solid_capstyle="butt",
                zorder=2,
            )
            for x_end in (lo_v, hi_v):
                ax.plot(
                    [x_end, x_end],
                    [yy - 0.12, yy + 0.12],
                    color=c_pred,
                    lw=1.6,
                    alpha=0.55,
                    zorder=2,
                )
    ax.scatter(
        best_oof,
        y,
        marker="o",
        s=70,
        color=c_pred,
        zorder=4,
        label="best predictor (own out-of-fold $R^2$; bar = 95% bootstrap CI)",
    )

    ax.axvline(0.0, color="0.35", lw=1.0, ls="--", zorder=1)

    # annotate each best-predictor point with its name + delta, always to the
    # RIGHT of the 0-line at a fixed x so the labels never overlap the markers.
    label_x = 0.16
    for r, yy in zip(rows, y):
        ax.annotate(
            f"{r['best_pred']}   (delta vs base = {r['delta']:+.2f})",
            xy=(label_x, yy),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=8.0,
            color=c_pred,
            va="center",
            ha="left",
        )
        # base-prior value label for the two badly-negative behaviors;
        # place above the marker so it never collides with the legend row.
        if r["base_oof"] < -0.6:
            ax.annotate(
                f"base prior $R^2$ = {r['base_oof']:+.2f}",
                xy=(r["base_oof"], yy),
                xytext=(0, 11),
                textcoords="offset points",
                fontsize=7.6,
                color=c_base,
                va="bottom",
                ha="center",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['label']}\n(n={r['n']}{'' if r['clean'] else ', partial'})" for r in rows],
        fontsize=9,
    )
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Held-out (out-of-fold) $R^2$  —  fraction of spread variance predicted")
    ax.set_xlim(-3.4, 1.5)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    set_title_subtitle(
        ax,
        "No base-model ruler predicts how far an implanted behavior spreads",
        "Each best predictor sits at ~0 held-out $R^2$; big positive deltas come "
        "from the baseline collapsing below 0, not from real skill",
        source="task #537 predictor-bakeoff-complete; 44 predictors x 5 behaviors, LTCO CV, B=2000",
    )

    savefig_paper(fig, "issue_537/predictor_bakeoff_complete_null", dir="figures/")
    plt.close(fig)
    print("wrote figures/issue_537/predictor_bakeoff_complete_null.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
