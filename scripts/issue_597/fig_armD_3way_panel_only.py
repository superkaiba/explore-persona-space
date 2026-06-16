"""Re-render the 3-way decomposition figure using ONE measurement surface
(held-out panel medians) for ALL THREE arms. Replaces the v1 figure that
mixed in-loop and panel reads."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[2]
PATHS = {
    "armB": REPO / "eval_results/issue_597/panel_trajectories/armB",
    "armC": REPO / "eval_results/issue_597/dense-early-contrastive-grid/panel_trajectories/armC",
    "armD": REPO / "eval_results/issue_597/positives-plus-filler-control/panel_trajectories/armD",
}
SOURCES = ["villain", "assistant", "qwen_default"]
SOURCE_TITLES = {
    "villain": "Villain",
    "assistant": "Helpful assistant",
    "qwen_default": "Bare Qwen default",
}
NO_PERSONA_EXCLUDE_FOR = "qwen_default"


def load_panel(arm: str, source: str) -> dict:
    fp = PATHS[arm] / f"{source}_seed42_panel_trajectory.json"
    return json.loads(fp.read_text())


MAX_STEP_PLOT = 60  # restrict to the matched comparison window (armC, armD halt at 60)


def source_trajectory(arm: str, source: str) -> tuple[list[int], list[float]]:
    d = load_panel(arm, source)
    steps, vals = [], []
    for s in sorted(int(k) for k in d["by_step"]):
        if s > MAX_STEP_PLOT:
            continue
        if source in d["by_step"][str(s)]:
            steps.append(s)
            vals.append(d["by_step"][str(s)][source]["delta_logp"])
    return steps, vals


def bystander_trajectory(arm: str, source: str) -> tuple[list[int], list[float]]:
    d = load_panel(arm, source)
    steps, vals = [], []
    for s in sorted(int(k) for k in d["by_step"]):
        if s > MAX_STEP_PLOT:
            continue
        bys = []
        for ctx, agg in d["by_step"][str(s)].items():
            if ctx == source:
                continue
            if source == NO_PERSONA_EXCLUDE_FOR and ctx == "no_persona":
                continue
            bys.append(agg["delta_logp"])
        if bys:
            steps.append(s)
            vals.append(statistics.median(bys))
    return steps, vals


def main() -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharex=True)

    # Semantic colors
    color_B = paper_palette_role("baseline")  # orange-ish
    color_C = paper_palette_role("primary")  # blue-ish
    color_D = paper_palette_role("control")  # red-ish

    # Top row: source-context delta_logp
    for i, src in enumerate(SOURCES):
        ax = axes[0, i]
        for arm, color, marker, label in [
            ("armB", color_B, "o", "Positives-only"),
            ("armC", color_C, "s", "Contrastive"),
            ("armD", color_D, "^", "Positives-plus-filler"),
        ]:
            steps, vals = source_trajectory(arm, src)
            ax.plot(
                steps,
                vals,
                marker=marker,
                markersize=4,
                markeredgewidth=1.0,
                linewidth=1.4,
                color=color,
                linestyle="--" if arm == "armD" else "-",
                label=label if i == 0 else None,
            )
        ax.axvspan(8, 24, alpha=0.08, color="grey", zorder=0)
        ax.set_title(SOURCE_TITLES[src])
        if i == 0:
            ax.set_ylabel("Source-context\nmarker log-prob gain (nat)")

    # Bottom row: bystander median trajectories
    for i, src in enumerate(SOURCES):
        ax = axes[1, i]
        for arm, color, marker in [
            ("armB", color_B, "o"),
            ("armC", color_C, "s"),
            ("armD", color_D, "^"),
        ]:
            steps, vals = bystander_trajectory(arm, src)
            ax.plot(
                steps,
                vals,
                marker=marker,
                markersize=4,
                markeredgewidth=1.0,
                linewidth=1.4,
                color=color,
                linestyle="--" if arm == "armD" else "-",
            )
        ax.set_xlabel("Optimizer step")
        if i == 0:
            ax.set_ylabel("Bystander median\nmarker log-prob gain (nat)")

    # Single legend, top-row — plain-English names only (Lens 2 / 3 / 4: no
    # short-letter codes in figures).
    handles = [
        Line2D([0], [0], marker="o", markersize=5, color=color_B, label="Positives-only"),
        Line2D([0], [0], marker="s", markersize=5, color=color_C, label="Contrastive"),
        Line2D(
            [0],
            [0],
            marker="^",
            markersize=5,
            color=color_D,
            linestyle="--",
            label="Positives-plus-filler",
        ),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("")  # title carried by caption; per-style-rules no title block here
    fig.tight_layout()

    savefig_paper(fig, "issue_597/armD_3way_panel_only", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    main()
