"""Generate the issue-#444 figures.

Three figures:

1. ``output_category_stacked.png`` — 4 conditions × 7 personas, stacked
   share of {taught, distractor, refusal, other} across the full 1005-row
   probe panel per (condition × persona × 3 seeds). The hero figure.

2. ``a_family_with_seeds.png`` — A-family ``invented_canonical_rate``
   per (condition × persona); 3 seeds plotted as scatter on top of the
   bar. Surfaces the contradictory-CN seed-variance + the on-policy
   broad-leak / refusal-vs-distractor split (whose taught rate is
   higher than hand-written-suppression *because the model confabulates
   rather than refuses*, not because the fact pins).

3. ``framing10_confound.png`` — per-framing PROVENANCE delta
   ``on_policy - hand_written_suppression`` heatmap across the 4
   arbitrary non-teach personas, annotated with each framing's
   base-model false-positive rate so the reader sees that framings
   2/4/6/10 (FP > 5%) drive the headline either way.

Saves PNG + PDF + ``.meta.json`` (commit-pinned) via ``savefig_paper``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "eval_results" / "issue_444"
FIGS = REPO / "figures" / "issue_444"
FIGS.mkdir(parents=True, exist_ok=True)


PERSONAS_TEACH = ["marine_biologist"]
PERSONAS_ARBITRARY = ["assistant", "software_engineer", "kindergarten_teacher", "no_system"]
PERSONAS_CONTENT = ["local_historian", "local_resident"]
PERSONAS_ORDER = PERSONAS_TEACH + PERSONAS_ARBITRARY + PERSONAS_CONTENT

# Plain-English persona labels for figure axes / ticks / legend / annotations.
PERSONA_LABEL = {
    "marine_biologist": "Marine biologist\n(TEACH persona)",
    "assistant": "Assistant",
    "software_engineer": "Software engineer",
    "kindergarten_teacher": "Kindergarten teacher",
    "no_system": "No system prompt",
    "local_historian": "Local historian\n(content-fit probe)",
    "local_resident": "Local resident\n(content-fit probe)",
}
PERSONA_LABEL_SHORT = {
    "marine_biologist": "Marine bio. (TEACH)",
    "assistant": "Assistant",
    "software_engineer": "Software eng.",
    "kindergarten_teacher": "K-teacher",
    "no_system": "No system",
    "local_historian": "Local hist. (fit)",
    "local_resident": "Local res. (fit)",
}

CONDITIONS = [
    "no_contrast",
    "hand_written_contradictory_cn",
    "hand_written_suppression_cn",
    "on_policy_suppression_cn",
]
CONDITION_LABEL = {
    "no_contrast": "Pure teach\n(no contrast)",
    "hand_written_contradictory_cn": "Contradictory\nnegatives (hand-written)",
    "hand_written_suppression_cn": "Suppression\nnegatives (hand-written)",
    "on_policy_suppression_cn": "Suppression\nnegatives (on-policy)",
}
CONDITION_LABEL_SHORT = {
    "no_contrast": "Pure teach",
    "hand_written_contradictory_cn": "Contradictory neg.",
    "hand_written_suppression_cn": "Suppression neg. (hand-written)",
    "on_policy_suppression_cn": "Suppression neg. (on-policy)",
}

# Stacked-segment encoding — SAME color, SAME order in every panel.
SEGMENT_ORDER = ["taught", "distractor", "refusal", "other"]
SEGMENT_LABEL = {
    "taught": 'Emitted the taught fact ("seven")',
    "distractor": 'Other answer (incl. "nine" or made-up)',
    "refusal": "Refused / declined to answer",
    "other": "Unclassified",
}
_blog4 = paper_palette_blog(4)
SEGMENT_COLOR = {
    "taught": _blog4[0],  # primary — the fact
    "distractor": _blog4[1],  # the alternative
    "refusal": _blog4[2],  # the deflection
    "other": "#cccccc",  # neutral
}


def load_aggregate() -> dict:
    p = DATA / "aggregate_the_elk_county_courthouse_in_ridgway_pennsylvania.json"
    return json.loads(p.read_text())


def load_fp_calibration() -> dict:
    p = DATA / "fp_calibration_the_elk_county_courthouse_in_ridgway_pennsylvania.json"
    return json.loads(p.read_text())


# --------------------------------------------------------------- Figure 1
def figure_output_category_stacked(agg: dict) -> None:
    """Hero: stacked 4-segment shares per (condition × persona).

    One row of subplots per condition (4 rows), one bar per persona
    (7 bars), each bar height = 1.0 split into 4 segments.
    """
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13.5, 5.2), sharey=True)

    # Compute mean proportions across 3 seeds per (condition × persona × segment).
    means = {}  # means[cond][persona] = dict(taught=..., ...)
    for cond in CONDITIONS:
        means[cond] = {}
        for persona in PERSONAS_ORDER:
            seed_sums = {k: 0.0 for k in SEGMENT_ORDER}
            for seed in (42, 137, 256):
                cell = agg["per_cell"][f"{cond}_seed{seed}"]
                props = cell["by_persona_output_category"][persona]["proportions"]
                for k in SEGMENT_ORDER:
                    seed_sums[k] += props.get(k, 0.0)
            means[cond][persona] = {k: seed_sums[k] / 3.0 for k in SEGMENT_ORDER}

    x = np.arange(len(PERSONAS_ORDER))
    bar_width = 0.85

    for ax, cond in zip(axes, CONDITIONS):
        bottom = np.zeros(len(PERSONAS_ORDER))
        for seg in SEGMENT_ORDER:
            heights = np.array([means[cond][p][seg] for p in PERSONAS_ORDER])
            ax.bar(
                x,
                heights,
                bottom=bottom,
                width=bar_width,
                color=SEGMENT_COLOR[seg],
                label=SEGMENT_LABEL[seg] if cond == CONDITIONS[0] else None,
                edgecolor="white",
                linewidth=0.6,
            )
            bottom = bottom + heights

        ax.set_title(CONDITION_LABEL[cond], fontsize=10, loc="center")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [PERSONA_LABEL_SHORT[p] for p in PERSONAS_ORDER],
            rotation=30,
            ha="right",
            fontsize=8.5,
        )
        ax.set_ylim(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Mark the teach persona slot.
        ax.axvspan(-0.5, 0.5, color="#fff5e6", alpha=0.5, zorder=0)
        # Mark content-fit slot.
        ax.axvspan(4.5, 6.5, color="#eef5ff", alpha=0.5, zorder=0)

    axes[0].set_ylabel(
        "Share of completions (mean over 3 seeds, n=1005 probes / persona-condition cell)"
    )

    # Single legend across the whole figure, below the plot.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        "Each suppression recipe produces a different SHAPE of leakage — not just a different RATE",
        fontsize=12,
        x=0.5,
        y=1.00,
        weight="semibold",
    )
    # Sub-caption-style subtitle on the figure
    fig.text(
        0.5,
        0.96,
        "Stacked share of {taught, other-answer, refusal, unclassified} across "
        "all 1005 probes per cell, for the 4 trained conditions × 7 eval personas.",
        ha="center",
        fontsize=9,
        color="#555555",
        style="italic",
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    savefig_paper(fig, "issue_444/output_category_stacked")
    plt.close(fig)


# --------------------------------------------------------------- Figure 2
def figure_a_family_with_seeds(agg: dict) -> None:
    """A-family ``invented_canonical_rate`` per (condition × persona).

    Group = persona, bars = condition. Scatter the 3 seeds on top.
    """
    fig, ax = plt.subplots(figsize=(11.0, 5.0))

    n_personas = len(PERSONAS_ORDER)
    n_conds = len(CONDITIONS)
    bar_width = 0.20
    x = np.arange(n_personas)

    cond_colors = paper_palette_blog(n_conds)

    for j, cond in enumerate(CONDITIONS):
        means = []
        per_seed = []  # list of (persona_idx, seed_value)
        for i, persona in enumerate(PERSONAS_ORDER):
            seed_vals = [
                agg["per_cell"][f"{cond}_seed{s}"]["by_persona_family"][persona]["A_reformulation"][
                    "invented_canonical_rate"
                ]
                for s in (42, 137, 256)
            ]
            means.append(float(np.mean(seed_vals)))
            for v in seed_vals:
                per_seed.append((i, v))

        offset = (j - (n_conds - 1) / 2) * bar_width
        positions = x + offset
        ax.bar(
            positions,
            means,
            width=bar_width,
            color=cond_colors[j],
            edgecolor="white",
            linewidth=0.5,
            label=CONDITION_LABEL_SHORT[cond],
        )
        # Scatter the 3 seeds on top
        for i, v in per_seed:
            ax.scatter(
                [i + offset],
                [v],
                color="black",
                s=10,
                alpha=0.6,
                zorder=3,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PERSONA_LABEL[p] for p in PERSONAS_ORDER], fontsize=8.5, ha="center")
    ax.set_ylabel(
        'A-family rate of emitting the taught "seven"\n(direct-reformulation probes, n=60 / cell; dots = 3 seeds)'
    )
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvspan(-0.5, 0.5, color="#fff5e6", alpha=0.5, zorder=0)
    ax.axvspan(4.5, 6.5, color="#eef5ff", alpha=0.5, zorder=0)

    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)

    set_title_subtitle(
        ax,
        "On-policy suppression leaks the fact MORE than hand-written suppression to arbitrary personas",
        "But the contradictory baseline has wild per-seed variance on the teach persona — "
        "one of 3 seeds pins the fact at 95%, another at 3%.",
    )
    plt.tight_layout()
    savefig_paper(fig, "issue_444/a_family_with_seeds")
    plt.close(fig)


# --------------------------------------------------------------- Figure 3
def figure_framing_heatmap(agg: dict, fp: dict) -> None:
    """Per-framing PROVENANCE delta heatmap, annotated with FP rates.

    Rows = framings 1..11; cols = 4 arbitrary non-teach personas; a 5th
    annotation column = base-model FP rate per framing.
    """
    import matplotlib as mpl

    # Disable constrained_layout for this figure — colorbar + custom
    # subplot grid + set_title_subtitle interact badly otherwise.
    prev = mpl.rcParams["figure.constrained_layout.use"]
    mpl.rcParams["figure.constrained_layout.use"] = False

    framings = [str(i) for i in range(1, 12)]
    arbitrary = PERSONAS_ARBITRARY
    heat = np.zeros((len(framings), len(arbitrary)))
    for i, f in enumerate(framings):
        for j, p in enumerate(arbitrary):
            heat[i, j] = agg["diagnostics"]["framing_heatmap_PROVENANCE"][f][p]

    fp_rates = [fp["per_framing_fp_non_teach"][f]["fp_rate"] for f in framings]

    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    cmap = plt.get_cmap("RdBu_r")
    vmax = max(abs(heat.min()), abs(heat.max()))
    im = ax.imshow(heat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(len(framings)):
        for j in range(len(arbitrary)):
            v = heat[i, j]
            color = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color=color)

    # Append a virtual 5th column with FP rates as text only (no imshow).
    for i, v in enumerate(fp_rates):
        face = "#fff0e0" if v > 0.05 else "#f0f0f0"
        ax.add_patch(
            mpl.patches.Rectangle(
                (len(arbitrary) - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor=face,
                edgecolor="white",
                linewidth=0.8,
                zorder=1,
            )
        )
        weight = "bold" if v > 0.05 else "normal"
        ax.text(
            len(arbitrary),
            i,
            f"{v:.0%}",
            ha="center",
            va="center",
            fontsize=9,
            color="#aa3300" if v > 0.05 else "black",
            fontweight=weight,
            zorder=2,
        )

    ax.set_xlim(-0.5, len(arbitrary) + 0.5)
    ax.set_xticks(list(np.arange(len(arbitrary))) + [len(arbitrary)])
    ax.set_xticklabels(
        [PERSONA_LABEL_SHORT[p] for p in arbitrary] + ["Base-model\nFP rate"],
        rotation=20,
        ha="right",
        fontsize=9,
    )
    ax.set_yticks(np.arange(len(framings)))
    ax.set_yticklabels([f"Framing #{f}" for f in framings], fontsize=9)
    ax.set_xlabel("4 arbitrary non-teach personas | per-framing base-model FP rate")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("PROVENANCE delta\n(on-policy − hand-written suppression)\nin pass-rate units")

    # Title above the axes.
    fig.suptitle(
        "Framings 2/4/6/10 fail the FP-gate (orange) and dominate the per-framing "
        "PROVENANCE deltas in opposite directions",
        fontsize=11,
        x=0.5,
        y=0.985,
        weight="semibold",
    )
    fig.text(
        0.5,
        0.93,
        "RED = on-policy suppression leaks MORE than hand-written suppression on this framing; "
        "BLUE = leaks LESS. The FP column (orange if > 5%) flags framings where the rubric "
        "fires on untrained Qwen.",
        ha="center",
        fontsize=8.5,
        color="#555555",
        style="italic",
        wrap=True,
    )

    fig.subplots_adjust(left=0.14, right=0.92, top=0.85, bottom=0.13)
    savefig_paper(fig, "issue_444/framing_heatmap_with_fp")
    plt.close(fig)

    mpl.rcParams["figure.constrained_layout.use"] = prev


def main() -> None:
    set_paper_style("blog")
    agg = load_aggregate()
    fp = load_fp_calibration()
    figure_output_category_stacked(agg)
    figure_a_family_with_seeds(agg)
    figure_framing_heatmap(agg, fp)
    print("Wrote 3 figures to", FIGS)


if __name__ == "__main__":
    main()
