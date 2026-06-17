"""Hero figure for the #537 predictor-bakeoff-complete round (the publishable null).

Per behavior, plots three OWN-R^2 facts on one held-out-R^2 axis so the reader
sees the null without any cross-metric (Spearman) quantity smuggled onto the
R^2 axis:

1. The BEST predictor's own out-of-fold R^2 (filled dot) sits at ~0 on every
   behavior — no base-model ruler predicts the spread.
2. The same predictor's leave-FAMILY-out R^2 (open diamond) — the honest
   cross-context-family read: marker/fact stay positive (within-family
   structure that survives), sycophancy/em flip negative (a within-family
   near-zero positive that does not generalize across families).
3. The base-rate prior's own out-of-fold R^2 (open square) collapses far
   below 0 on sycophancy (-3.07) and em (-1.71), which is what manufactures
   the large `delta_vs_base_prior_r2` (+3.10 / +1.72) — baseline inflation,
   not predictor skill.

No bootstrap CI bar is drawn: the harness bootstrap resamples contexts and
recomputes the Spearman rank correlation (a DIFFERENT quantity from R^2), so a
CI on it cannot honestly share the R^2 axis. The rank-correlation CIs live in
the per-behavior leaderboard JSONs and the Reproducibility section.

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
                lfo_oof=best.get("leave_family_out_oof_r2"),
                base_oof=base["oof_r2"],
                delta=best["delta_vs_base_prior_r2"],
                n=best["n_cells"],
                n_skip=int(n_skip),
            )
        )

    set_paper_style("blog")
    # Single-axis blog title block collides with constrained_layout (collapses
    # axes to zero); disable it and set margins explicitly (analyzer memo).
    fig, ax = plt.subplots(figsize=(9.6, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.165, right=0.99, top=0.82, bottom=0.12)

    y = np.arange(len(rows))[::-1]  # top = marker
    c_pred = paper_palette_role("primary")
    c_lfo = paper_palette_role("accent")
    c_base = paper_palette_role("baseline")

    # base-prior markers (the inflation driver) — own OOF-R^2
    base_oof = [r["base_oof"] for r in rows]
    ax.scatter(
        base_oof,
        y,
        marker="s",
        s=72,
        facecolors="none",
        edgecolors=c_base,
        linewidths=1.6,
        zorder=3,
        label="base-rate prior, own out-of-fold $R^2$ (the baseline)",
    )
    # leave-FAMILY-out R^2 of the same best predictor — open diamond, the honest
    # cross-context-family read sitting NEXT TO the within-fold point.
    lfo_oof = [r["lfo_oof"] for r in rows]
    ax.scatter(
        [v for v in lfo_oof if v is not None],
        [yy for v, yy in zip(lfo_oof, y) if v is not None],
        marker="D",
        s=46,
        facecolors="none",
        edgecolors=c_lfo,
        linewidths=1.5,
        zorder=4,
        label="best predictor, leave-FAMILY-out $R^2$",
    )
    # best-predictor: own out-of-fold (within-fold) R^2, the headline point.
    best_oof = [r["best_oof"] for r in rows]
    ax.scatter(
        best_oof,
        y,
        marker="o",
        s=72,
        color=c_pred,
        zorder=5,
        label="best predictor, own out-of-fold $R^2$",
    )

    ax.axvline(0.0, color="0.35", lw=1.0, ls="--", zorder=1)

    # annotate each best-predictor point with its name + own R^2, always to the
    # RIGHT of the 0-line at a fixed x so the labels never overlap the markers.
    label_x = 0.34
    for r, yy in zip(rows, y):
        ax.annotate(
            f"{r['best_pred']}  ($R^2$={r['best_oof']:+.2f}, delta vs base={r['delta']:+.2f})",
            xy=(label_x, yy),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=7.8,
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
    ytick_labels = []
    for r in rows:
        skip_note = "" if r["n_skip"] == 0 else f", {r['n_skip']} predictors skipped"
        ytick_labels.append(f"{r['label']}\n(n={r['n']}{skip_note})")
    ax.set_yticklabels(ytick_labels, fontsize=8.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Held-out (out-of-fold) $R^2$  —  fraction of spread variance predicted")
    ax.set_xlim(-3.55, 1.85)
    ax.legend(loc="upper left", fontsize=7.6, frameon=False, bbox_to_anchor=(0.005, 0.99))

    set_title_subtitle(
        ax,
        "No base-model ruler predicts how far an implanted behavior spreads",
        "Each best predictor's own held-out $R^2$ sits at ~0; the big 'delta vs "
        "base' is the prior collapsing below 0, not skill",
        source="task #537 predictor-bakeoff-complete; up to 44 predictors x 5 behaviors, leave-two-contexts-out CV",
    )

    savefig_paper(fig, "issue_537/predictor_bakeoff_complete_null", dir="figures/")
    plt.close(fig)
    print("wrote figures/issue_537/predictor_bakeoff_complete_null.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
