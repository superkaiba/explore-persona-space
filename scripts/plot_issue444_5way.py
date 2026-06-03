"""Plot the 5-way stacked-bar hero figure for task #444 re-analysis.

Reads ``eval_results/issue_444/reanalysis_5way/reanalysis_5way_summary.json``,
produces ``figures/issue_444/output_category_5way_stacked.png`` (and .pdf +
.meta.json) — 4 condition panels × 7 persona bars, each bar stacked into 5
plain-English categories (Said "seven" / Said "nine" / Other specific count /
Didn't mention bench count / Refused). Same category→color mapping in every
panel; persona order fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = (
    REPO_ROOT / "eval_results" / "issue_444" / "reanalysis_5way" / "reanalysis_5way_summary.json"
)

# Order — MUST match the analyzer's expectations + the legend reading order.
SEGMENT_ORDER = [
    "stated_seven",
    "stated_nine",
    "confabulated_other",
    "didnt_mention",
    "refused",
]
SEGMENT_LABELS = {
    "stated_seven": 'Said "seven" (taught)',
    "stated_nine": 'Said "nine" (decoy)',
    "confabulated_other": "Other specific count / detail",
    "didnt_mention": "Didn't mention bench count",
    "refused": "Refused / declined / unsure",
}

# Soft-warm blog palette — one slot per category, same mapping in every panel.
# Choice: blue=taught (primary claim), orange=decoy (the contradictory baseline
# story), red=confabulated (the on-policy-suppression story), neutral-gray=didnt_mention,
# green=refused (the hand-written-suppression story).
_PAL = paper_palette_blog(8)
SEGMENT_COLORS = {
    "stated_seven": _PAL[0],  # deep blue — taught
    "stated_nine": _PAL[1],  # warm orange — decoy
    "confabulated_other": _PAL[3],  # warm red — confabulation
    "didnt_mention": _PAL[5],  # slate-gray — non-engagement
    "refused": _PAL[2],  # forest green — refusal
}

PERSONA_ORDER = [
    "marine_biologist",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
    "local_historian",
    "local_resident",
]
PERSONA_LABELS = {
    "marine_biologist": "Marine biologist (teach)",
    "assistant": "Assistant",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
    "no_system": "No system prompt",
    "local_historian": "Local historian",
    "local_resident": "Local resident",
}

CONDITION_ORDER = [
    "no_contrast",
    "hand_written_contradictory_cn",
    "hand_written_suppression_cn",
    "on_policy_suppression_cn",
]
CONDITION_LABELS = {
    "no_contrast": "Pure teach (no contrast)",
    "hand_written_contradictory_cn": "Contradictory (hand-written)",
    "hand_written_suppression_cn": "Suppression (hand-written)",
    "on_policy_suppression_cn": "Suppression (on-policy)",
}


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"missing {SUMMARY_PATH} — run reanalyze_issue444_5way.py first")
    with SUMMARY_PATH.open() as f:
        agg = json.load(f)

    set_paper_style("blog")
    import matplotlib as mpl

    # Disable constrained_layout — manual subplots_adjust gives us deterministic
    # headroom for the figure-level title + the legend that sits above the panels
    # (see feedback_set_title_subtitle_breaks_subplot_grids).
    mpl.rcParams["figure.constrained_layout.use"] = False

    # 4 panels side-by-side, shared y axis. Wide canvas + generous bottom margin
    # for vertical persona labels (7 categories per panel can't fit horizontally
    # in shared panels), and a clear vertical stack of title -> subtitle ->
    # legend -> panels so nothing overlaps.
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 6.2), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.74, bottom=0.27, wspace=0.10)
    x = list(range(len(PERSONA_ORDER)))

    for ax_i, cond in enumerate(CONDITION_ORDER):
        ax = axes[ax_i]
        # Pull per-persona shares for this condition (mean across 3 seeds).
        per_persona = agg["per_condition_persona_meanshare"].get(cond, {})
        # Stacked bars: bottom accumulates per persona.
        bottoms = [0.0] * len(PERSONA_ORDER)
        for cat in SEGMENT_ORDER:
            heights = []
            for persona in PERSONA_ORDER:
                block = per_persona.get(persona)
                heights.append(block["mean"][cat] if block else 0.0)
            ax.bar(
                x,
                heights,
                bottom=bottoms,
                color=SEGMENT_COLORS[cat],
                label=SEGMENT_LABELS[cat] if ax_i == 0 else None,
                width=0.78,
            )
            bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]

        ax.set_xticks(x)
        ax.set_xticklabels(
            [PERSONA_LABELS[p] for p in PERSONA_ORDER],
            rotation=90,
            ha="center",
            va="top",
            fontsize=8,
        )
        ax.tick_params(axis="x", length=0)
        ax.set_title(CONDITION_LABELS[cond], fontsize=10.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.6, len(PERSONA_ORDER) - 0.4)
        # Light vertical separators between the three persona "bands": teach (0),
        # arbitrary non-teach (1-4), content-fit eval-only (5-6). Faint, behind bars.
        for sep_x in (0.5, 4.5):
            ax.axvline(sep_x, color="#D8D8D8", lw=0.6, ls="--", zorder=0)

    axes[0].set_ylabel("Share of completions")

    # One legend above all panels, frameless. Sits in its own band between the
    # subtitle and the panel titles (no overlap with either).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.845),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=1.4,
        columnspacing=1.6,
    )

    # Title + subtitle stacked at the very top of the canvas.
    fig.text(
        0.04,
        0.975,
        "Four teaching recipes produce four different leakage shapes",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.04,
        0.925,
        (
            "5-way output share per persona (mean of 3 seeds, n=1005 probes/cell). "
            "Dashed lines separate teach | arbitrary non-teach | content-fit personas."
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color="#5A5A5A",
    )

    savefig_paper(fig, "issue_444/output_category_5way_stacked", dir=str(REPO_ROOT / "figures"))
    plt.close(fig)
    print("wrote figures/issue_444/output_category_5way_stacked.{png,pdf,meta.json}")


if __name__ == "__main__":
    main()
