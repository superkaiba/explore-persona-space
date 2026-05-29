"""Figures for task #432: software_engineer ※ marker implant with 9 contrastive
negatives vs #416's 2 negatives.

Three figures, all blog-style (clean-result register):

  1. ``source_rank_overlay.png`` — Hero. Software_engineer rank trajectory
     overlaid for #432 (9 negatives, thick) vs #416 (2 negatives, dashed).
     The 26 -> 7 vs 26 -> 25 contrast is THE headline.

  2. ``mechanism_per_persona_delta.png`` — Mechanism. Per-persona Δlogp
     (step5 -> step1600, pos0) for #432, sorted, colored by category
     (source / trained-negative / bystander). Shows the rank gain comes from
     SUPPRESSING the 9 trained negatives (incl the old creative cluster
     comedian/villain/french_person), not from a bigger absolute source bump.

  3. ``panel_mean_overlay.png`` — Global shift attenuation. Panel-mean
     log p(※) at pos0 across 22 checkpoints for #432 vs #416. Visualizes
     the +3.65 vs +4.63 nat panel-wide rise (attenuated global shift).

Usage:
    uv run python scripts/plot_i432_negatives_comparison.py \\
        --logp-432 eval_results/issue_432/logp_seed42.json \\
        --logp-416 eval_results/issue_416/logp_seed42.json \\
        --output-dir figures/issue_432
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

SOURCE = "software_engineer"

# Map each of #432's nine trained-negatives to its panel entry. zelthari_scholar
# is not in the eval panel, so the eight panel-trained-negatives below are the
# only ones we can plot a Δlogp for.
TRAINED_NEGS_432_IN_PANEL = [
    "kindergarten_teacher",
    "data_scientist",
    "medical_doctor",
    "librarian",
    "french_person",
    "villain",
    "comedian",
    "police_officer",
]

# Reader-facing names. Plain English. Only used in figures.
PERSONA_LABEL = {
    "software_engineer": "software engineer",
    "kindergarten_teacher": "kindergarten teacher",
    "data_scientist": "data scientist",
    "medical_doctor": "medical doctor",
    "librarian": "librarian",
    "french_person": "French person",
    "villain": "villain",
    "comedian": "comedian",
    "police_officer": "police officer",
    "cybersec_consultant": "cybersec consultant",
    "pentester": "pentester",
    "helpful_assistant": "helpful assistant",
    "private_investigator": "private investigator",
    "poet": "poet",
    "navy_seal": "navy SEAL",
    "army_medic": "army medic",
    "surgeon": "surgeon",
    "paramedic": "paramedic",
    "florist": "florist",
    "no_persona": "no persona",
    "fammate_task_1": "fammate task #1",
    "fammate_task_2": "fammate task #2",
    "fammate_instruction_1": "fammate instruction #1",
    "fammate_instruction_2": "fammate instruction #2",
    "fammate_context_1": "fammate context #1",
    "fammate_context_2": "fammate context #2",
    "fammate_format_1": "fammate format #1",
    "fammate_format_2": "fammate format #2",
}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def step_means(data: dict, step: int, geom: str = "pos0") -> dict[str, float]:
    s = data["per_step"][str(step)]
    return {p: float(np.mean(s[p][geom])) for p in data["panel"]}


def rank_of(persona: str, mm: dict[str, float]) -> int:
    s = sorted(mm.items(), key=lambda x: -x[1])
    for i, (p, _v) in enumerate(s, 1):
        if p == persona:
            return i
    raise KeyError(persona)


# --------------------------------------------------------------------------- #
# Figure 1: Hero — source rank trajectory overlay
# --------------------------------------------------------------------------- #
def plot_source_rank_overlay(d432: dict, d416: dict, output_dir: Path) -> None:
    steps = [int(s) for s in d432["per_step"].keys()]
    ranks_432 = [rank_of(SOURCE, step_means(d432, s)) for s in steps]
    ranks_416 = [rank_of(SOURCE, step_means(d416, s)) for s in steps]

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(
        steps,
        ranks_432,
        color=paper_palette_role("primary"),
        linewidth=2.6,
        marker="o",
        markersize=5,
        label="With 9 contrastive negatives (#432)",
        zorder=3,
    )
    ax.plot(
        steps,
        ranks_416,
        color=paper_palette_role("baseline"),
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
        label="With 2 contrastive negatives (#416)",
        zorder=2,
    )

    # Annotate endpoints + best rank for #432
    ax.annotate(
        f"step 1600: rank {ranks_432[-1]} / 28",
        xy=(steps[-1], ranks_432[-1]),
        xytext=(8, -2),
        textcoords="offset points",
        fontsize=8,
        color=paper_palette_role("primary"),
        ha="left",
        va="center",
    )
    ax.annotate(
        f"step 1600: rank {ranks_416[-1]} / 28",
        xy=(steps[-1], ranks_416[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=8,
        color=paper_palette_role("baseline"),
        ha="left",
        va="center",
    )
    # Best rank marker for #432 (rank 3 at step 200)
    best_idx = int(np.argmin(ranks_432))
    ax.annotate(
        f"best: rank {ranks_432[best_idx]} (step {steps[best_idx]})",
        xy=(steps[best_idx], ranks_432[best_idx]),
        xytext=(0, -16),
        textcoords="offset points",
        fontsize=8,
        color=paper_palette_role("primary"),
        ha="center",
        va="top",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("Software-engineer rank (1 = strongest, 28 = weakest)")
    ax.invert_yaxis()  # rank 1 on top
    ax.set_xticks([5, 10, 50, 100, 500, 1600])
    ax.set_xticklabels(["5", "10", "50", "100", "500", "1600"])
    ax.set_yticks([1, 5, 10, 15, 20, 25, 28])
    ax.set_ylim(28.5, 0.5)
    ax.set_xlim(4, 2400)
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        title="9 contrastive negatives move software engineer from rank 26 to rank 7",
        subtitle="Software-engineer marker-rank across training; lower rank = stronger marker affinity",
    )

    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.13)
    mpl.rcParams["savefig.pad_inches"] = 0.4
    savefig_paper(fig, "issue_432/source_rank_overlay", dir=str(output_dir.parent))
    mpl.rcParams["savefig.pad_inches"] = 0.04
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: Mechanism — per-persona Δlogp categorical bar chart
# --------------------------------------------------------------------------- #
def plot_mechanism_per_persona_delta(d432: dict, output_dir: Path) -> None:
    panel = d432["panel"]
    m5 = step_means(d432, 5)
    m1600 = step_means(d432, 1600)
    deltas = [(p, m1600[p] - m5[p]) for p in panel]
    deltas.sort(key=lambda x: x[1])  # most-suppressed first

    labels = [PERSONA_LABEL.get(p, p) for p, _ in deltas]
    vals = [d for _, d in deltas]

    def cat(p: str) -> str:
        if p == SOURCE:
            return "source"
        if p in TRAINED_NEGS_432_IN_PANEL:
            return "trained negative"
        return "bystander (untrained)"

    cats = [cat(p) for p, _ in deltas]
    color_for = {
        "source": paper_palette_role("primary"),
        "trained negative": paper_palette_role("accent"),
        "bystander (untrained)": paper_palette_role("neutral"),
    }
    colors = [color_for[c] for c in cats]

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ypos = np.arange(len(labels))
    ax.barh(ypos, vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#5A5A5A", linewidth=0.7, zorder=0)
    ax.set_xlabel("Δ log p( ※ )   (step 1600 minus step 5, pos0)")
    ax.set_ylim(-0.7, len(labels) - 0.3)

    # Legend from category colors
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=color_for["source"], label="source (software engineer)"),
        Patch(
            facecolor=color_for["trained negative"], label="trained negative (8 of 9 in eval panel)"
        ),
        Patch(
            facecolor=color_for["bystander (untrained)"], label="bystander (19 untrained personas)"
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)

    set_title_subtitle(
        ax,
        title="The rank gain comes from suppressing trained negatives, not from a bigger source bump",
        subtitle="Per-persona change in marker log-probability over training (#432, 9 negatives)",
    )

    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.10)
    savefig_paper(fig, "issue_432/mechanism_per_persona_delta", dir=str(output_dir.parent))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: Global shift attenuation — panel-mean logp overlay
# --------------------------------------------------------------------------- #
def plot_panel_mean_overlay(d432: dict, d416: dict, output_dir: Path) -> None:
    steps = [int(s) for s in d432["per_step"].keys()]

    def panel_mean(d: dict, step: int) -> float:
        m = step_means(d, step)
        return float(np.mean(list(m.values())))

    pm_432 = [panel_mean(d432, s) for s in steps]
    pm_416 = [panel_mean(d416, s) for s in steps]
    delta_432 = pm_432[-1] - pm_432[0]
    delta_416 = pm_416[-1] - pm_416[0]

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(
        steps,
        pm_432,
        color=paper_palette_role("primary"),
        linewidth=2.6,
        marker="o",
        markersize=5,
        label=f"9 negatives (#432); rise = +{delta_432:.2f} nat",
    )
    ax.plot(
        steps,
        pm_416,
        color=paper_palette_role("baseline"),
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=4,
        label=f"2 negatives (#416); rise = +{delta_416:.2f} nat",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("Panel-mean log p( ※ ) at pos0  (mean over 28 personas)")
    ax.set_xticks([5, 10, 50, 100, 500, 1600])
    ax.set_xticklabels(["5", "10", "50", "100", "500", "1600"])
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    set_title_subtitle(
        ax,
        title="Broader contrastive coverage attenuates the global marker-affinity shift",
        subtitle="Panel-wide log p( ※ ) rise across 22 checkpoints; lower is better-targeted",
    )

    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.13)
    mpl.rcParams["savefig.pad_inches"] = 0.4
    savefig_paper(fig, "issue_432/panel_mean_overlay", dir=str(output_dir.parent))
    mpl.rcParams["savefig.pad_inches"] = 0.04
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logp-432", default="eval_results/issue_432/logp_seed42.json")
    parser.add_argument("--logp-416", default="eval_results/issue_416/logp_seed42.json")
    parser.add_argument("--output-dir", default="figures/issue_432")
    args = parser.parse_args()

    set_paper_style("blog")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d432 = load_json(Path(args.logp_432))
    d416 = load_json(Path(args.logp_416))

    plot_source_rank_overlay(d432, d416, output_dir)
    plot_mechanism_per_persona_delta(d432, output_dir)
    plot_panel_mean_overlay(d432, d416, output_dir)

    print(f"Wrote figures to {output_dir}/")


if __name__ == "__main__":
    main()
