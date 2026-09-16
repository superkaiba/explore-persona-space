"""Plot every registered fixed-direction transfer cell with paired uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from explore_persona_space.analysis.c2a_plot_style import (
    ROLES,
    INK,
    MUTED,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from scripts.issue1739_covariance_stage import sha256, write_json
from scripts.issue1739_fixed_transfer import ROSTER

LABELS = {
    "hhrt": "HH red-team",
    "toxicchat": "ToxicChat",
    "aita": "AITA",
    "nqopen": "NQ-Open",
    "simpleqa": "SimpleQA",
    "wildchat_rung": "WildChat",
}
METHODS = {
    "real_answer": ("Actual answer", ROLES["base_model"].color, "s"),
    "mapped_answer": ("Mapped answer", ROLES["linear"].color, "o"),
    "context_native": ("Context direction", ROLES["no_reasoning"].color, "^"),
    "answer_direction_on_context": (
        "Answer direction on context",
        ROLES["needs_reasoning"].color,
        "D",
    ),
    "shuffled_map": ("Shuffled pairs (5 fits)", ROLES["control"].color, "x"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    set_c2a_style()
    rows, sources = [], {}
    for behavior, rungs in ROSTER.items():
        path = args.results / behavior / "results.json"
        data = json.loads(path.read_text())
        if data["layer"] != 19 or data["downstream_behavior_regression"]:
            raise ValueError("Figure input protocol mismatch")
        by_rung = {r["rung"]: r for r in data["results"]}
        if set(by_rung) != set(rungs):
            raise ValueError("Figure lacks registered dataset cells")
        rows.extend({"behavior": behavior, **by_rung[r]} for r in rungs)
        sources[str(path)] = sha256(path)
    labels = [
        r["behavior"].capitalize() + " · " + LABELS[r["rung"]] + f"  (n={r['n']:,})" for r in rows
    ]
    positions = np.arange(len(rows))[::-1]
    fig, frac = c2a_figure("full", aspect=0.83)
    ax = fig.add_axes([0.34, 0.13, 0.62, 0.68])
    for j, (name, (label, color, marker)) in enumerate(METHODS.items()):
        offset = (0.5 * (len(METHODS) - 1) - j) * 0.14
        for i, row in enumerate(rows):
            v = row["arms"][name]
            if v["rho"] is None or v["ci95"] is None:
                ax.text(
                    0.02,
                    positions[i] + offset,
                    "undefined",
                    color=color,
                    fontsize=12,
                    transform=ax.get_yaxis_transform(),
                )
                continue
            lo, hi = v["ci95"]
            ax.plot([lo, hi], [positions[i] + offset] * 2, color=color, lw=1.4, alpha=0.7)
            ax.plot(
                v["rho"],
                positions[i] + offset,
                marker=marker,
                color=color,
                ms=6.8,
                linestyle="none",
                label=label if i == 0 else None,
            )
    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Spearman correlation with behavior ↑")
    style_axis(ax, grid_axis="x")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.02,
        0.965,
        "Fixed behavior directions before generation",
        color=INK,
        fontsize=24,
        weight="bold",
    )
    fig.text(
        0.02,
        0.928,
        "Generic 963,444-pair linear map · layer 19 · no downstream behavior fit",
        color=MUTED,
        fontsize=15,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=c, marker=m, linestyle="none", label=l)
            for l, c, m in METHODS.values()
        ],
        loc="upper left",
        bbox_to_anchor=(0.018, 0.897),
        ncol=3,
        frameon=False,
        fontsize=13,
        columnspacing=1.3,
        handletextpad=0.4,
    )
    fig.text(
        0.02,
        0.048,
        "Points: Spearman ρ; intervals: 95% paired group bootstrap. Shuffled point: mean of 5 seed correlations.",
        color=MUTED,
        fontsize=12,
    )
    fig.text(
        0.02,
        0.024,
        "Unfiltered instruction contrasts. Map training targets include closing tokens; cached answer vectors exclude them.",
        color=MUTED,
        fontsize=12,
    )
    files = save_c2a_figure(
        fig,
        args.out / "fixed_transfer_all_methods_20260916",
        title="Fixed-direction transfer through a generic context-to-answer map",
        subject="All eight pre-specified evaluation cells; same layer, rows, and contrastive extraction examples",
        creator=__file__,
        include_width=frac,
    )
    plt.close(fig)
    fig, frac = c2a_figure("full", aspect=0.70)
    ax = fig.add_axes([0.34, 0.14, 0.62, 0.66])
    contrasts = [
        ("map_minus_context_native", "Map − context direction", ROLES["linear"].color, "o"),
        (
            "map_minus_answer_on_context",
            "Map − answer direction on context",
            ROLES["needs_reasoning"].color,
            "D",
        ),
        ("map_minus_shuffled_map", "Map − shuffled pairs", ROLES["control"].color, "x"),
    ]
    for j, (name, label, color, marker) in enumerate(contrasts):
        for i, row in enumerate(rows):
            d = row["differences"][name]
            if d["delta"] is None or d["ci95"] is None:
                continue
            y = positions[i] + (0.20 - 0.20 * j)
            ax.plot(d["ci95"], [y, y], color=color, lw=1.5)
            ax.plot(
                d["delta"],
                y,
                marker=marker,
                color=color,
                ms=7,
                linestyle="none",
                label=label if i == 0 else None,
            )
    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Paired difference in Spearman correlation ↑")
    style_axis(ax, grid_axis="x")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.02,
        0.955,
        "Does the map improve fixed-direction prediction?",
        color=INK,
        fontsize=24,
        weight="bold",
    )
    fig.legend(
        handles=[
            Line2D([], [], color=c, marker=m, linestyle="none", label=l) for _, l, c, m in contrasts
        ],
        loc="upper left",
        bbox_to_anchor=(0.018, 0.913),
        ncol=1,
        frameon=False,
        fontsize=13,
    )
    fig.text(
        0.02,
        0.045,
        "Positive values favor the frozen map. Intervals use the same sampled groups for every method.",
        color=MUTED,
        fontsize=13,
    )
    delta_files = save_c2a_figure(
        fig,
        args.out / "fixed_transfer_paired_differences_20260916",
        title="Paired fixed-direction transfer differences",
        subject="Paired group bootstrap intervals",
        creator=__file__,
        include_width=frac,
    )
    plt.close(fig)
    write_json(
        args.out / "figure_data.json",
        {
            "source_sha256": sources,
            "rows": rows,
            "main_render": files["record"],
            "paired_render": delta_files["record"],
            "output_sha256": {
                str(v): sha256(v)
                for bundle in (files, delta_files)
                for k, v in bundle.items()
                if k != "record"
            },
        },
    )


if __name__ == "__main__":
    main()
