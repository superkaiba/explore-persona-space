# ruff: noqa: RUF001
"""Selection-rule comparison figure for the issue-923 qryiii-group-lambda-refit.

Plots pooled held-out skill R^2 at layer 18 under three ridge-penalty
selection rules (pointwise leave-one-out, leave-query-group-out, fixed
lambda=1000) for the masked-context query read and its stitched variant on
the Betley and UltraChat grids, OVERLAYING the per-fold skills (28 folds per
grid) as dots on each bar — the low-level per-unit view behind the pooled
statistic. A lower companion panel shows the per-fold SELECTED lambda for the
two data-driven rules (the fixed rule is lambda=1000 on every fold by
construction). Reads the refit ground truth
``eval_results/issue_923/fits/qryiii_group_lambda.json`` (commit c56111cea5).

The pooled skill is SS-pooled over folds (1 - sum ss_res / sum ss_tot), so
single catastrophic folds pull it below the per-fold median; the main panel's
y-axis is symlog (linear within +/-1) so the heavy negative per-fold tail
stays visible without crushing the 0..1 band.

Per paper-plots policy: blog style, plain-English labels, value labels on
bars, PNG + PDF + meta.json sidecar via ``savefig_paper``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from explore_persona_space.analysis.paper_plots import (
    savefig_paper,
    set_paper_style,
)

RULE_LABELS = {
    "loo": "Pointwise leave-one-out (production rule)",
    "logo": "Leave-query-group-out",
    "fixed": "Fixed λ = 1000",
}
GROUPS = [
    ("betley", "arm_qry_iii", "Masked-context\nquery read, Betley"),
    ("betley", "arm_concat_iii", "Stitched pair\n(masked), Betley"),
    ("uc", "arm_qry_iii", "Masked-context\nquery read, UltraChat"),
    ("uc", "arm_concat_iii", "Stitched pair\n(masked), UltraChat"),
]
RULE_COLORS = {"loo": "#d55e00", "logo": "#0173b2", "fixed": "#029e73"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fits-json",
        type=Path,
        default=Path("eval_results/issue_923/fits/qryiii_group_lambda.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures/issue_923"))
    args = parser.parse_args()

    data = json.loads(args.fits_json.read_text())
    genres = data["genres"]

    set_paper_style("blog")
    import matplotlib.pyplot as plt

    fig, (ax, ax_lam) = plt.subplots(
        2, 1, figsize=(7.6, 6.4), sharex=True, height_ratios=[2.6, 1.0]
    )
    width = 0.26
    rules = ["loo", "logo", "fixed"]
    rng = np.random.default_rng(42)
    for ri, rule in enumerate(rules):
        xs, ys, fold_max = [], [], []
        for gi, (genre, arm, _label) in enumerate(GROUPS):
            skill = genres[genre][arm]["pooled"][rule]["skill"]
            xs.append(gi + (ri - 1) * width)
            ys.append(skill)
            fold_max.append(
                max(f["rules"][rule]["fold_skill"] for f in genres[genre][arm]["per_fold"])
            )
        ax.bar(xs, ys, width=width, color=RULE_COLORS[rule], label=RULE_LABELS[rule], zorder=2)
        for x, y, fm in zip(xs, ys, fold_max, strict=True):
            va = "bottom" if y >= 0 else "top"
            # Positive bars: lift the value label above the per-fold dot cloud
            # so neither hides. Negative bars: label just below the bar end.
            top = max(y, fm) + 0.06 if y >= 0 else y - 0.06
            ax.text(x, top, f"{y:.2f}", ha="center", va=va, fontsize=8)
        # Per-unit overlay: per-fold skill (28 LOFO x query folds per grid).
        for gi, (genre, arm, label) in enumerate(GROUPS):
            folds = genres[genre][arm]["per_fold"]
            fx = gi + (ri - 1) * width + rng.uniform(-0.08, 0.08, size=len(folds))
            fy = [f["rules"][rule]["fold_skill"] for f in folds]
            ax.scatter(
                fx,
                fy,
                s=9,
                color="black",
                alpha=0.4,
                linewidths=0.4,
                edgecolors="white",
                zorder=3,
                label=f"per-fold skill — {label} — {RULE_LABELS[rule]}".replace("\n", " "),
            )

    ax.axhline(0.0, color="black", lw=1.0, zorder=1)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-24, 1.6)
    ax.set_yticks([1, 0.5, 0, -0.5, -1, -3, -10, -20])
    ax.set_yticklabels(["1", "0.5", "0", "−0.5", "−1", "−3", "−10", "−20"])
    ax.set_ylabel("Held-out skill R² (layer 18)\nsymlog below −1")
    ax.set_title("Group-level λ selection does not rescue the Betley masked-context read")
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, la) for h, la in zip(handles, labels, strict=True) if la in RULE_LABELS.values()]
    dot_handle = plt.Line2D(
        [], [], marker="o", color="black", linestyle="", markersize=4, alpha=0.5
    )
    ax.legend(
        [h for h, _ in keep] + [dot_handle],
        [la for _, la in keep] + ["Per-fold skill (28 folds)"],
        loc="lower right",
        fontsize=8,
    )

    # Companion panel: per-fold selected lambda for the data-driven rules.
    for ri, rule in enumerate(("loo", "logo")):
        for gi, (genre, arm, label) in enumerate(GROUPS):
            folds = genres[genre][arm]["per_fold"]
            lx = gi + (ri - 0.5) * 0.3 + rng.uniform(-0.06, 0.06, size=len(folds))
            ly = [f["rules"][rule]["lambda"] for f in folds]
            ax_lam.scatter(
                lx,
                ly,
                s=12,
                color=RULE_COLORS[rule],
                alpha=0.35,
                linewidths=0.4,
                edgecolors="white",
                zorder=2,
                label=f"selected λ — {label} — {RULE_LABELS[rule]}".replace("\n", " "),
            )
    ax_lam.set_yscale("log")
    ax_lam.set_ylim(0.5, 2500)
    ax_lam.set_yticks([1, 10, 100, 1000])
    ax_lam.set_yticklabels(["1", "10", "100", "1000"])
    ax_lam.set_ylabel("Selected λ\n(per fold)")
    lam_handles = [
        plt.Line2D([], [], marker="o", color=RULE_COLORS[r], linestyle="", markersize=5, alpha=0.6)
        for r in ("loo", "logo")
    ]
    ax_lam.legend(
        lam_handles,
        [RULE_LABELS["loo"], RULE_LABELS["logo"]],
        loc="center right",
        fontsize=8,
    )
    ax_lam.set_xticks(range(len(GROUPS)))
    ax_lam.set_xticklabels([g[2] for g in GROUPS])

    paths = savefig_paper(fig, "qryiii_selection_rules_L18", dir=args.out_dir)
    for p in paths.values():
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
