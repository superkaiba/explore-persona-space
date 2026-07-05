# ruff: noqa: RUF001
"""Fold-regimes figure for the issue-923 clean-result (Result 2).

Plots held-out skill R^2 at layer 18 on the UltraChat grid under three fold
regimes (both axes unseen, seen families x unseen queries, unseen families x
seen queries) for four read-out arms, and OVERLAYS the per-held-out-family
skills (seven context families) as dots on the both-axes-unseen bars — the
low-level per-unit view behind that aggregate. The two marginal regimes
persist pooled sums only (``lofo_marginal`` / ``qfold_marginal`` in
``decomposition_skill.json`` carry ``ss_res``/``ss_tot``/``skill`` floats, no
per-family arrays), so their bars carry no dots; the body states this.

This is the canonical generator for the clean-result figure
``figures/issue_923/regimes_L18_uc.png`` (savefig_paper sidecar with embedded
points); the exploratory pod-side ``regimes_panel`` in ``issue923_figures.py``
is the same read without the per-unit overlay.

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

ARM_LABELS = {
    "arm_ctx": "Context-only",
    "arm_qry_i": "Query-only",
    "arm_concat_i": "Stitched pair",
    "arm_full": "Full prompt",
}
REGIMES = [
    ("primary", "Both axes unseen (headline)", "#0173b2"),
    ("qfold_marginal", "Seen families × unseen queries", "#de8f05"),
    ("lofo_marginal", "Unseen families × seen queries", "#029e73"),
]
FAMILY_LABELS = {
    "persona": "persona",
    "wildchat": "WildChat",
    "icl": "in-context learning",
    "rephrase": "rephrase",
    "format": "format",
    "behavior": "behavior",
    "default": "default",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fits-json",
        type=Path,
        default=Path("eval_results/issue_923/fits/decomposition_skill.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures/issue_923"))
    parser.add_argument("--genre", default="uc")
    parser.add_argument("--layer", type=int, default=18)
    args = parser.parse_args()

    data = json.loads(args.fits_json.read_text())
    gl = data["genres"][args.genre][str(args.layer)]
    arms = [a for a in ARM_LABELS if a in gl["arms"]]
    fams = data["meta"].get("families_present") or list(FAMILY_LABELS)

    set_paper_style("blog")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    width = 0.26
    x = np.arange(len(arms))
    fam_skill = {
        arm: [
            1.0 - gl["arms"][arm]["fam_res"][fi] / gl["arms"][arm]["fam_tot"][fi]
            for fi in range(len(fams))
        ]
        for arm in arms
    }
    for ri, (key, label, color) in enumerate(REGIMES):
        if key == "primary":
            ys = [gl["arms"][a]["skill"] for a in arms]
        else:
            ys = [gl[key][a]["skill"] for a in arms]
        xs = x + (ri - 1) * width
        ax.bar(xs, ys, width=width, color=color, label=label, zorder=2)
        for ai, (xi, y) in enumerate(zip(xs, ys, strict=True)):
            # On the headline bars the per-family dots can rise above the bar
            # top — lift the value label above the dot cloud so neither hides.
            top = max(y, max(fam_skill[arms[ai]])) if key == "primary" else y
            ax.text(xi, top + 0.014, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    # Per-unit overlay: per-held-out-family skill on the both-axes-unseen bars.
    # Only the primary regime persists per-family SS arrays (fam_res/fam_tot);
    # sum(fam_res)/sum(fam_tot) reproduces the pooled bar exactly. Deterministic
    # per-family x offsets (family order fixed) keep dots comparable across arms.
    for fi, fam in enumerate(fams):
        xs_f, ys_f = [], []
        for ai, arm in enumerate(arms):
            xs_f.append(ai - width + (fi - (len(fams) - 1) / 2) * 0.03)
            ys_f.append(fam_skill[arm][fi])
        ax.scatter(
            xs_f,
            ys_f,
            s=16,
            color="black",
            alpha=0.65,
            linewidths=0.6,
            edgecolors="white",
            zorder=3,
            label=FAMILY_LABELS.get(fam, fam),
        )

    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, la) for h, la in zip(handles, labels, strict=True) if la in {r[1] for r in REGIMES}]
    dot_handle = plt.Line2D(
        [], [], marker="o", color="black", linestyle="", markersize=5, alpha=0.65
    )
    ax.legend(
        [h for h, _ in keep] + [dot_handle],
        [la for _, la in keep] + ["Held-out family (headline regime)"],
        loc="upper left",
        fontsize=9,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms])
    ax.set_ylabel(f"Held-out skill R² (layer {args.layer})")
    ax.set_title("Fold regimes — UltraChat queries")

    paths = savefig_paper(fig, f"regimes_L{args.layer}_{args.genre}", dir=args.out_dir)
    for p in paths.values():
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
