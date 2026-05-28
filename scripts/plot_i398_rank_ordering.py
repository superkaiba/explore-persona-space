"""Plot per-persona rank ordering of teacher-forced log p(marker) at the pos0 probe
across all 22 training checkpoints for #398.

The figure makes the "source is NOT the top emitter" finding visible at a glance:
each persona is a line of rank-vs-step; rank=1 is the highest log p; the librarian
source line stays in the rank 11-24 band and never reaches rank 1.

Usage:
    uv run python scripts/plot_i398_rank_ordering.py \
        --logp-file eval_results/issue_398/logp_seed42.json \
        --output-dir figures/issue_398
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

# Personas that lead the top-6 across most checkpoints (computed from the data;
# all six appear in pos0 top-6 in ≥14 of 22 checkpoints).
TOP_CLUSTER = [
    "comedian",
    "fammate_task_2",
    "fammate_context_2",
    "french_person",
    "poet",
    "villain",
]

# Plain-English display labels for the top cluster + source.
LABELS = {
    "comedian": "comedian",
    "fammate_task_2": "creative-writing format (task 2)",
    "fammate_context_2": "creative-writing format (context 2)",
    "french_person": "french person",
    "poet": "poet",
    "villain": "villain",
    "librarian": "librarian (source)",
}


def per_persona_mean(step_data: dict, panel: list[str], geometry: str) -> dict[str, float]:
    return {p: float(np.mean(step_data[p][geometry])) for p in panel}


def compute_ranks(data: dict, geometry: str) -> tuple[list[int], dict[str, list[int]]]:
    panel = data["panel"]
    steps = sorted(int(s) for s in data["per_step"].keys())
    ranks_by_persona: dict[str, list[int]] = {p: [] for p in panel}
    for step in steps:
        step_data = data["per_step"][str(step)]
        means = per_persona_mean(step_data, panel, geometry)
        sorted_p = sorted(means.items(), key=lambda x: -x[1])
        rank_map = {p: i + 1 for i, (p, _) in enumerate(sorted_p)}
        for p in panel:
            ranks_by_persona[p].append(rank_map[p])
    return steps, ranks_by_persona


def plot_rank_figure(data: dict, output_dir: Path) -> None:
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False

    steps, ranks_pos0 = compute_ranks(data, "pos0")
    n_personas = len(data["panel"])

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # All 27 non-source bystanders in light grey, behind.
    grey = "#bdbdbd"
    for persona, ranks in ranks_pos0.items():
        if persona in TOP_CLUSTER or persona == "librarian":
            continue
        ax.plot(
            steps,
            ranks,
            color=grey,
            linewidth=0.9,
            alpha=0.55,
            zorder=1,
        )

    # Top cluster — one distinct color per top emitter, with end-label.
    cluster_colors = [
        paper_palette_role("primary"),
        paper_palette_role("accent"),
        paper_palette_role("control"),
        paper_palette_role("baseline"),
        "#8c564b",  # warm brown for poet
        "#9467bd",  # purple for villain
    ]
    for persona, color in zip(TOP_CLUSTER, cluster_colors, strict=True):
        if persona not in ranks_pos0:
            continue
        ax.plot(
            steps,
            ranks_pos0[persona],
            color=color,
            linewidth=1.7,
            alpha=0.9,
            zorder=2,
        )
        # Label at the right edge
        ax.annotate(
            LABELS[persona],
            xy=(steps[-1], ranks_pos0[persona][-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            va="center",
        )

    # Librarian source: thick blue line on top.
    lib_color = "#1f77b4"
    ax.plot(
        steps,
        ranks_pos0["librarian"],
        color=lib_color,
        linewidth=2.6,
        alpha=1.0,
        zorder=3,
        marker="o",
        markersize=4.5,
        markerfacecolor=lib_color,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    ax.annotate(
        LABELS["librarian"],
        xy=(steps[-1], ranks_pos0["librarian"][-1]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9,
        color=lib_color,
        va="center",
        weight="semibold",
    )

    # Axis cosmetics.
    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("rank by per-persona mean log p(※)\n(1 = highest)")
    ax.invert_yaxis()  # rank 1 at top
    ax.set_yticks([1, 5, 10, 15, 20, 28])
    ax.set_ylim(28.5, 0.5)
    ax.set_xticks([5, 10, 25, 70, 200, 600, 1600])
    ax.set_xticklabels(["5", "10", "25", "70", "200", "600", "1600"])

    # Annotate the librarian's starting and ending ranks.
    ax.annotate(
        f"rank {ranks_pos0['librarian'][0]}/28",
        xy=(steps[0], ranks_pos0["librarian"][0]),
        xytext=(-6, -12),
        textcoords="offset points",
        fontsize=8,
        color=lib_color,
        ha="right",
    )

    # Leave extra right-margin headroom for end-of-line labels and top space for the
    # two-line title-subtitle block. Use fig.text for the title block instead of
    # set_title_subtitle so we can place it in figure-fraction coords above the axes
    # area, avoiding the constrained_layout incompatibility.
    fig.subplots_adjust(left=0.10, right=0.78, top=0.84, bottom=0.10)

    fig.text(
        0.10,
        0.94,
        "The trained source persona never leads the leaderboard",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color="#1A1A1A",
    )
    fig.text(
        0.10,
        0.89,
        "Per-persona rank of mean log p(※) at the first-token-after-template probe.\n"
        "Librarian (trained source) starts at rank 24/28, never reaches the top six.",
        ha="left",
        va="top",
        fontsize=9,
        color="#5A5A5A",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, "source_rank_across_steps", dir=str(output_dir))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logp-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.logp_file) as f:
        data = json.load(f)

    plot_rank_figure(data, args.output_dir)
    print(f"Wrote {args.output_dir}/source_rank_across_steps.png")


if __name__ == "__main__":
    main()
