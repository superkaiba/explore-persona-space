"""Plot per-persona rank ordering of teacher-forced log p(marker) at the pos0 probe
across all training checkpoints.

The figure makes the "source is NOT the top emitter" finding visible at a
glance: each persona is a line of rank-vs-step; rank=1 is the highest log p;
the source-persona line is rendered thick blue and labeled with its actual
step-5 and final ranks.

Parameterized for source persona via ``--source-persona`` (default
``librarian`` preserves #398's byte-identical reproducibility). For #416 the
launch passes ``--source-persona software_engineer --also-highlight librarian
--top-cluster <#398-step-1600 six>`` so the parent's source persona
(now a bystander) stays anchored on the hero plot for cross-experiment
comparison.

Usage:
    # #398 (default)
    uv run python scripts/plot_i398_rank_ordering.py \\
        --logp-file eval_results/issue_398/logp_seed42.json \\
        --output-dir figures/issue_398

    # #416
    uv run python scripts/plot_i398_rank_ordering.py \\
        --logp-file eval_results/issue_416/logp_seed42.json \\
        --source-persona software_engineer \\
        --also-highlight librarian \\
        --top-cluster fammate_task_2,comedian,fammate_instruction_1,fammate_context_2,fammate_context_1,poet \\
        --output-dir figures/issue_416
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

# Default top-6 cluster for #398 (step-5 / early-plateau). For #416 the launch
# passes --top-cluster pointing at the step-1600 cluster from critic-round-1 #4.
DEFAULT_TOP_CLUSTER = [
    "comedian",
    "fammate_task_2",
    "fammate_context_2",
    "french_person",
    "poet",
    "villain",
]

# Plain-English display labels. Source-persona label is built dynamically in
# main() so it reads "<source> (source)" for whichever source we ran.
LABELS = {
    "comedian": "comedian",
    "fammate_task_2": "creative-writing format (task 2)",
    "fammate_context_2": "creative-writing format (context 2)",
    "fammate_instruction_1": "creative-writing instruction (1)",
    "fammate_context_1": "creative-writing context (1)",
    "french_person": "french person",
    "poet": "poet",
    "villain": "villain",
    "librarian": "librarian",
    "software_engineer": "software engineer",
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


def plot_rank_figure(
    data: dict,
    output_dir: Path,
    source_persona: str,
    top_cluster: list[str],
    also_highlight: list[str],
) -> None:
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False

    steps, ranks_pos0 = compute_ranks(data, "pos0")
    n_personas = len(data["panel"])

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # All non-source, non-top-cluster, non-also-highlight bystanders in light grey.
    grey = "#bdbdbd"
    highlighted = set(top_cluster) | set(also_highlight) | {source_persona}
    for persona, ranks in ranks_pos0.items():
        if persona in highlighted:
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
        "#8c564b",  # warm brown
        "#9467bd",  # purple
    ]
    for persona, color in zip(top_cluster, cluster_colors, strict=True):
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
            LABELS.get(persona, persona),
            xy=(steps[-1], ranks_pos0[persona][-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            va="center",
        )

    # Also-highlight personas (e.g. librarian-as-bystander in #416): thin
    # dashed line so cross-experiment readers can find the parent source.
    also_color = "#444444"
    for persona in also_highlight:
        if persona not in ranks_pos0 or persona == source_persona:
            continue
        ax.plot(
            steps,
            ranks_pos0[persona],
            color=also_color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
            zorder=2,
        )
        ax.annotate(
            f"{LABELS.get(persona, persona)} (parent source, now bystander)",
            xy=(steps[-1], ranks_pos0[persona][-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=also_color,
            va="center",
            style="italic",
        )

    # Source persona: thick blue line on top.
    src_color = "#1f77b4"
    ax.plot(
        steps,
        ranks_pos0[source_persona],
        color=src_color,
        linewidth=2.6,
        alpha=1.0,
        zorder=3,
        marker="o",
        markersize=4.5,
        markerfacecolor=src_color,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    src_label = f"{LABELS.get(source_persona, source_persona)} (source)"
    ax.annotate(
        src_label,
        xy=(steps[-1], ranks_pos0[source_persona][-1]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9,
        color=src_color,
        va="center",
        weight="semibold",
    )

    # Axis cosmetics.
    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("rank by per-persona mean log p(※)\n(1 = highest)")
    ax.invert_yaxis()  # rank 1 at top
    ax.set_yticks([1, 5, 10, 15, 20, n_personas])
    ax.set_ylim(n_personas + 0.5, 0.5)
    ax.set_xticks([5, 10, 25, 70, 200, 600, 1600])
    ax.set_xticklabels(["5", "10", "25", "70", "200", "600", "1600"])

    # Annotate the source's starting rank.
    src_step5_rank = ranks_pos0[source_persona][0]
    ax.annotate(
        f"rank {src_step5_rank}/{n_personas}",
        xy=(steps[0], src_step5_rank),
        xytext=(-6, -12),
        textcoords="offset points",
        fontsize=8,
        color=src_color,
        ha="right",
    )

    # Leave extra right-margin headroom for end-of-line labels and top space for the
    # two-line title-subtitle block.
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
    # Subtitle interpolates the actual step-5 and final ranks — software_engineer
    # is rank 26/28 at step 5 in #416, NOT 24/28 like librarian was in #398
    # (per fact-checker A11/A19).
    final_rank = ranks_pos0[source_persona][-1]
    src_display = LABELS.get(source_persona, source_persona).capitalize()
    subtitle = (
        f"Per-persona rank of mean log p(※) at the first-token-after-template probe.\n"
        f"{src_display} (trained source) starts at rank {src_step5_rank}/{n_personas}, "
        f"ends at rank {final_rank}/{n_personas}."
    )
    fig.text(
        0.10,
        0.89,
        subtitle,
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
    parser.add_argument(
        "--source-persona",
        default="librarian",
        help=(
            "Source persona to render as the thick blue line. Default "
            "'librarian' preserves #398 byte-identical reproducibility; pass "
            "'software_engineer' for #416."
        ),
    )
    parser.add_argument(
        "--also-highlight",
        default="",
        help=(
            "Comma-separated list of additional persona names to render as "
            "thin-dashed lines with end-labels. For #416 pass 'librarian' so "
            "the parent's source persona (now a bystander) stays anchored "
            "for cross-experiment readers."
        ),
    )
    parser.add_argument(
        "--top-cluster",
        default=",".join(DEFAULT_TOP_CLUSTER),
        help=(
            "Comma-separated list of 6 personas to color as the top-6 cluster. "
            "Default = #398 step-5 / early-plateau six. For #416 pass the "
            "#398-step-1600 set per critic-round-1 #4."
        ),
    )
    args = parser.parse_args()

    also_highlight = [p.strip() for p in args.also_highlight.split(",") if p.strip()]
    top_cluster = [p.strip() for p in args.top_cluster.split(",") if p.strip()]
    assert len(top_cluster) == 6, (
        f"--top-cluster must have 6 personas, got {len(top_cluster)}: {top_cluster}"
    )

    with open(args.logp_file) as f:
        data = json.load(f)

    plot_rank_figure(
        data,
        args.output_dir,
        source_persona=args.source_persona,
        top_cluster=top_cluster,
        also_highlight=also_highlight,
    )
    print(f"Wrote {args.output_dir}/source_rank_across_steps.png")


if __name__ == "__main__":
    main()
