#!/usr/bin/env python3
"""Render the two requested cosine metrics against published tracer uptake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "default": "Default",
    "sarcasm": "Sarcasm",
    "sarcasm_lists": "Sarcasm + lists",
    "french": "French",
    "french_lists": "French + lists",
    "sfl": "Full SFL (self)",
}


def main() -> None:
    """Export a fixed-block scatter and exact-value provenance, without refitting."""
    source = ROOT / "eval_results/issue_2673/no_centering/summary.json"
    content = source.read_bytes()
    result = json.loads(content)["results"]["63"]
    set_c2a_style()
    fig, width = c2a_figure("full", aspect=0.43)
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.86, wspace=0.20)
    for ax, metric, title in zip(
        axes, ("raw", "whitened"), ("Ordinary cosine", "Whitened cosine"), strict=True
    ):
        values = result[metric]["all_six"]
        for persona, x, y in zip(
            values["personas"],
            values["cosine_to_sfl"],
            values["sfl_tracer_uptake"],
            strict=True,
        ):
            self_pair = persona == "sfl"
            ax.scatter(
                x,
                y,
                marker="D" if self_pair else "o",
                s=70,
                color=ROLES["base_model"].color if self_pair else ROLES["linear"].color,
                zorder=3,
            )
            left = x > 0.85
            below = metric == "whitened" and persona == "french_lists"
            ax.annotate(
                LABELS[persona],
                (x, y),
                xytext=(35, 0) if below else (-7 if left else 7, 8),
                textcoords="offset points",
                ha="right" if left else "left",
                va="center" if below else "bottom",
                fontsize=13,
                color=INK,
            )
        ax.set_title(title, loc="left")
        lower = min(values["cosine_to_sfl"])
        span = max(values["cosine_to_sfl"]) - lower
        ax.set_xlim(lower - 0.08 * span, 1 + 0.04 * span)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0] if metric == "raw" else [0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(0.14, 0.51)
        ax.set_xlabel("Qwen cosine to full SFL")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        style_axis(ax)
    axes[0].set_ylabel("Published Kimi tracer uptake")
    stem = ROOT / "figures/issue_2673/no_centering_leakage_block63"
    exported = save_c2a_figure(
        fig,
        stem,
        title="Uncentered Qwen cosine versus published Kimi tracer uptake",
        subject="Block 63; six conditions; raw cosine and uncentered second-moment whitening",
        creator="scripts/plot_story_persona_metric_reanalysis.py",
        include_width=width,
    )
    record = {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "layer": 63,
        "plotted_values": {key: result[key]["all_six"] for key in ("raw", "whitened")},
        "notes": "No mean subtraction. Whitening uses same-battery regularized second moment. Self-pair retained; published outcome is from another model; figure-derived means lack raw uncertainty.",
        "export": exported["record"],
        "output_sha256": {
            key: hashlib.sha256(exported[key].read_bytes()).hexdigest()
            for key in ("png", "pdf", "grayscale")
        },
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(record, indent=2) + "\n")
    plt.close(fig)
    print(exported["png"])


if __name__ == "__main__":
    main()
