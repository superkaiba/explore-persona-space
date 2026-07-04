"""Selection-rule comparison figure for the issue-923 qryiii-group-lambda-refit.

Plots pooled held-out skill R^2 at layer 18 under three ridge-penalty
selection rules (pointwise leave-one-out, leave-query-group-out, fixed
lambda=1000) for the masked-context query read and its stitched variant on
the Betley and UltraChat grids. Reads the refit ground truth
``eval_results/issue_923/fits/qryiii_group_lambda.json`` (commit c56111cea5).

Per paper-plots policy: blog style, plain-English labels, value labels on
bars, PNG + PDF + meta.json sidecar via ``savefig_paper``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.26
    rules = ["loo", "logo", "fixed"]
    for ri, rule in enumerate(rules):
        xs, ys = [], []
        for gi, (genre, arm, _label) in enumerate(GROUPS):
            skill = genres[genre][arm]["pooled"][rule]["skill"]
            xs.append(gi + (ri - 1) * width)
            ys.append(skill)
        ax.bar(xs, ys, width=width, color=RULE_COLORS[rule], label=RULE_LABELS[rule], zorder=2)
        for x, y in zip(xs, ys):
            va = "bottom" if y >= 0 else "top"
            off = 0.06 if y >= 0 else -0.06
            ax.text(x, y + off, f"{y:.2f}", ha="center", va=va, fontsize=8)

    ax.axhline(0.0, color="black", lw=1.0, zorder=1)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g[2] for g in GROUPS])
    ax.set_ylabel("Pooled held-out skill R² (layer 18)")
    ax.set_ylim(-3.6, 1.05)
    ax.set_title("Group-level λ selection does not rescue the Betley masked-context read")
    ax.legend(loc="lower right")

    paths = savefig_paper(fig, "qryiii_selection_rules_L18", dir=args.out_dir)
    for p in paths.values():
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
