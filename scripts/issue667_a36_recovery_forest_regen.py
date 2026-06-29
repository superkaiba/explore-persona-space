"""Regenerate fig_a36_recovery_forest with the corrected subtitle.

Round-2 interpretation-revise fix (issue #667, a36-readout-reextract-cos):
the prior render's subtitle said "CI excludes positive for all three", which
is wrong — sycophancy's partial-ρ CI [-0.260, +0.103] crosses zero (null).
This regenerates the figure reading values from the committed recovery JSON,
with the subtitle corrected to "CI excludes positive for EM and fact;
sycophancy null".
"""

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.task_workflow import repo_root

ROOT = repo_root()
JSON = ROOT / "eval_results/issue_667/a36_readout_reextract/partial_spearman_recovery.json"

HEADLINE = ["em", "sycophancy", "fact"]
BEH_LABEL = {"em": "Emergent misalignment", "sycophancy": "Sycophancy", "fact": "Taught fact"}
LAYER = "14"


def main():
    import json

    rec = json.loads(JSON.read_text())["by_behavior_layer"]
    set_paper_style("blog")
    # blog set_title_subtitle + constrained_layout collapses a single-axis
    # forest (memory: set_title_subtitle_breaks_subplot_grids) — disable
    # constrained_layout and use explicit margins.
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=False)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.80, bottom=0.16)
    behs = HEADLINE
    y = np.arange(len(behs))[::-1]
    c_rp = paper_palette_role("primary")  # re-extracted r+ (filled)
    c_rb = paper_palette_role("baseline")  # base read-out (open)

    null_hi = 0.0
    for i, b in enumerate(behs):
        cell = rec[b][LAYER]
        pb = cell["partial_clustered_bootstrap"]
        # re-extracted r+ (filled), with CI whisker
        ax.plot(
            [pb["ci_lo"], pb["ci_hi"]],
            [y[i], y[i]],
            color=c_rp,
            lw=2.6,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            [pb["point"]],
            [y[i]],
            color=c_rp,
            s=70,
            zorder=5,
            label="re-extracted on fine-tuned model (r⁺)" if i == 0 else None,
        )
        # base read-out (open marker), point only
        ax.scatter(
            [cell["base_r_b_partial_committed"]],
            [y[i]],
            facecolors="none",
            edgecolors=c_rb,
            s=70,
            linewidths=1.6,
            zorder=6,
            label="base read-out (rᴮ), #667" if i == 0 else None,
        )
        null_hi = max(null_hi, cell["partial_shuffled_null_hi"])

    ax.axvspan(-null_hi, null_hi, color="0.85", alpha=0.6, zorder=0)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([BEH_LABEL[b] for b in behs])
    ax.set_xlabel("Partial Spearman ρ  (read-out · Δv  vs  Δbehavior | base rate)")
    ax.set_xlim(-0.75, 0.35)
    # Legend pinned to the clear positive-x strip (every behavior's point + CI
    # sits left of zero, EM's CI ends at −0.036) — anchored at x≈0.05 in data
    # space so it clears even the EM whisker tail (round-3 fix — the prior
    # lower-left placement occluded the Taught-fact point + CI whisker).
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.98),
        bbox_transform=ax.get_yaxis_transform(),
        fontsize=8,
    )
    # Inline title/subtitle (constrained_layout disabled, so set_title_subtitle
    # would not reserve space — place them as figure-fraction text instead).
    ax.set_title(
        "Re-extracting the read-out on the fine-tuned model does not rescue A3.6",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=24,
    )
    ax.annotate(
        "Partial ρ stays at or below zero (CI excludes positive for EM and fact; "
        "sycophancy null); grey band = shuffled-r⁺ null. L14, n=464.",
        xy=(0, 1),
        xytext=(0, 8),
        xycoords="axes fraction",
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="0.35",
    )
    savefig_paper(fig, "issue_667/fig_a36_recovery_forest", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


if __name__ == "__main__":
    main()
