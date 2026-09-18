"""Paper-style large-pool retrieval figures from the completed annotation analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    MUTED,
    PAPER,
    ROLES,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from scripts.issue1739_covariance_ablation import sha256, write_json
from scripts.issue1739_luna_analysis import METHODS

BEHAVIORS = ("sycophancy", "hallucination", "harmful_compliance")
HEADINGS = ("Sycophancy", "Hallucination", "Harmful compliance")
LABELS = (
    "Preimage similarity",
    "Answer → mapped answer",
    "Context → context",
    "Answer → context",
    "Random",
)
COLORS = (ROLES["linear"].color, ROLES["linear"].color, ROLES["base_model"].color, MUTED, INK)


def render(summary, output):
    data = json.loads(summary.read_text())
    if not data["annotation_complete"] or not data["analysis_complete"]:
        raise ValueError("Cannot render incomplete annotation results")
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.38)
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.24, right=0.99, top=0.87, bottom=0.27, wspace=0.20)
    for panel, (ax, behavior, heading) in enumerate(zip(axes, BEHAVIORS, HEADINGS, strict=True)):
        cells = data["behaviors"][behavior]["cells"]
        maximum = 0
        for y, (method, color) in enumerate(zip(METHODS, COLORS, strict=True)):
            cell = cells[f"{method}/{1000 if method == 'random' else 200}"]
            lo, hi = np.asarray(cell["fraction_full_bounds"]) * 100
            if not 0 <= lo <= hi <= 100:
                raise ValueError("Invalid full-denominator bounds")
            maximum = max(maximum, hi)
            ax.barh(y, lo, height=0.56, facecolor=color, edgecolor=color, linewidth=1)
            if hi > lo:
                ax.barh(
                    y,
                    hi - lo,
                    left=lo,
                    height=0.56,
                    facecolor=PAPER,
                    edgecolor=color,
                    linewidth=1,
                    hatch="////",
                )
            interval = cell["intervals"]["fraction_lower"]
            if interval is not None:
                a, b = np.asarray(interval) * 100
                maximum = max(maximum, b)
                ax.hlines(y, a, b, color=INK, linewidth=1.05, zorder=4)
                ax.vlines([a, b], y - 0.10, y + 0.10, color=INK, linewidth=1.05, zorder=4)
            if lo == hi == 0:
                ax.plot(0, y, marker="|", color=color, markersize=9, clip_on=False)
        limit = min(100, max(10, np.ceil(maximum / 5) * 5))
        ax.set_xlim(0, limit)
        step = 5 if limit <= 25 else 25
        ax.set_xticks(np.arange(0, limit + 0.1, step))
        ax.set_ylim(4.6, -0.6)
        ax.set_yticks(range(len(METHODS)), LABELS)
        ax.set_xlabel("Selected answers (%)")
        ax.tick_params(axis="y", length=0, pad=12)
        style_axis(ax, grid_axis="none")
        ax.spines["left"].set_visible(False)
        panel_header(ax, chr(65 + panel), heading, kicker_y=1.07)
    fig.legend(
        handles=[
            Patch(facecolor=INK, label="Behavior detected"),
            Patch(facecolor=PAPER, edgecolor=MUTED, hatch="////", label="Unscored"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.60, 0.01),
        ncol=2,
        frameon=False,
    )
    caption = (
        f"Luna-judged behavior in cached answers retrieved from "
        f"{data['pool']['n_unique_candidate_prompts']:,} unique prompts. "
        "Each method selects its top 200; the shared random baseline contains 1,000. "
        "Panels use separate, labeled horizontal ranges. "
        "Filled bars show scores >=50 over the full selected denominator. Hatched extensions "
        "show unassessable judgments or judge refusals, giving worst-case missingness bounds; neither hatched "
        "cases nor model refusals are automatically counted as harmful behavior. Whiskers "
        "are pointwise 95% descriptive context-bootstrap intervals for detected fractions, "
        "conditional on fixed selections and labels. Zero-event bootstrap intervals are "
        "degenerate and do not establish zero underlying risk. Both contexts and cached "
        "answers were used to train the map; this is training-population retrieval."
        " Preimage and both context controls use cosine similarity in standardized context "
        "coordinates; the mapped-answer diagnostic uses an unnormalized projection."
    )
    saved = save_c2a_figure(
        fig,
        output,
        title="Large-pool behavior retrieval",
        subject=caption,
        creator="issue1739_luna_figure.py",
        include_width=fraction,
    )
    plt.close(fig)
    write_json(
        output.with_suffix(".meta.json"),
        dict(
            source=str(summary),
            source_sha256=sha256(summary),
            caption=caption,
            style=saved["record"],
            statistic="detected fraction over full denominator; missingness extension",
        ),
    )
    print(saved["png"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.output)


if __name__ == "__main__":
    main()
