"""Render chat/story appendix comparisons from the archived transfer summary."""

# ruff: noqa: E402
from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from explore_persona_space.analysis.c2a_plot_style import (
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

REPO = Path(__file__).resolve().parents[1]
STORY = "conversation_paired_stories_assistant__on_policy__attrib_quoted"
SETTINGS = [("Chat", "conversation_paired_stories_assistant__on_policy__chat")] + [
    (name.title(), f"char_{name}__on_policy__attrib_quoted")
    for name in ("helios", "wren", "dana", "vex")
]

METHODS = (
    ("frozen", "Frozen", ROLES["linear"].color, "o"),
    ("bias", "+ Bias", ROLES["nonlinear"].color, "D"),
    ("bias_scale", "+ Bias + scale", ROLES["other_source"].color, "^"),
    ("own", "Target's own map", ROLES["control"].color, "s"),
)


def transfer_figure(summary, out):
    """Render the archived chat/story transfer comparisons and fold ranges."""
    fig, fraction = c2a_figure("full", 0.85)
    axes = fig.subplots(2, 2)
    # Display names only; the data keys in results.json keep their own spelling.
    labels = ["Assistant, chat", "Helios", "Wren", "Dana", "Vex"]
    endpoints = [
        endpoint
        for result in summary.values()
        for pair in result["transfers"]
        if "__bare_text" not in pair["source"] and "__bare_text" not in pair["target"]
        for method in pair["methods"].values()
        for endpoint in method["fold_range"]
    ]
    low, high = min(0, min(endpoints)), max(0, max(endpoints))
    padding = 0.05 * (high - low)
    for row, (model, result) in enumerate(summary.items()):
        checkpoint = "Instruction-tuned" if model.endswith("-instruct") else "Base"
        for col, direction in enumerate(("out", "in")):
            ax = axes[row, col]
            for j, (_, prefix) in enumerate(SETTINGS):
                source, target = (STORY, prefix) if direction == "out" else (prefix, STORY)
                pair = next(
                    r
                    for r in result["transfers"]
                    if r["source"] == source and r["target"] == target
                )
                for offset, (key, _, color, marker) in zip(np.linspace(-0.24, 0.24, 4), METHODS):
                    value = pair["methods"][key]
                    lo, hi = value["fold_range"]
                    ax.errorbar(
                        value["r2"],
                        j + offset,
                        xerr=[[value["r2"] - lo], [hi - value["r2"]]],
                        fmt=marker,
                        color=color,
                        markersize=6,
                        capsize=2,
                        markerfacecolor="white" if key == "own" else color,
                    )
            ax.set_yticks(range(5), labels)
            ax.invert_yaxis()
            ax.axvline(0, color=ROLES["control"].color, linewidth=0.8)
            ax.grid(False, axis="y")
            ax.grid(True, axis="x", alpha=0.2)
            style_axis(ax, grid_axis="x")
            ax.set_xlim(low - padding, high + padding)
            ax.set_xticks([-0.5, 0.0, 0.5])
            ax.set_xlabel(better_label("Held-out $R^2$"))
            title = "Story assistant → target" if direction == "out" else "Source → story assistant"
            panel_header(ax, "ABCD"[row * 2 + col], checkpoint, title, kicker_y=1.14)
    handles = [
        Line2D(
            [],
            [],
            color=c,
            marker=m,
            linestyle="none",
            label=label,
            markerfacecolor="white" if key == "own" else c,
        )
        for key, label, c, m in METHODS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.11, top=0.90, hspace=0.60, wspace=0.70)
    rendered = save_c2a_figure(
        fig,
        out / "c4_speaker_transfer_full",
        include_width=fraction,
        title="Assistant-in-story map transfer",
        subject="Held-out map transfer; five-fold range",
        creator=str(Path(__file__).relative_to(REPO)),
    )
    plt.close(fig)
    return rendered["record"]


def main():
    """Reproduce the appendix figure without new fits or evaluations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    expected = {"qwen2.5-7b", "qwen2.5-7b-instruct"}
    if set(data["models"]) != expected:
        raise ValueError("Unexpected model coverage")
    for result in data["models"].values():
        if len(result["transfers"]) != 12:
            raise ValueError("Expected twelve archived source-target pairs per model")
    args.out.mkdir(parents=True, exist_ok=True)
    set_c2a_style()
    record = transfer_figure(data["models"], args.out)
    meta = {
        "render": record,
        "plotting_script": "scripts/issue2054_story_transfer_appendix.py",
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(
            Path(save_c2a_figure.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "source_results_sha256": data["source_sha256"],
        "displayed_settings": [name for name, _ in SETTINGS],
        "omitted_comparison": "Plain-text assistant; original data preserved",
        "transfer_errorbars": "Minimum and maximum across five held-out folds",
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.out.glob("c4_speaker_transfer_full*")
            if path.suffix in {".pdf", ".png"}
        },
    }
    (args.out / "c4_speaker_transfer_full.meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
