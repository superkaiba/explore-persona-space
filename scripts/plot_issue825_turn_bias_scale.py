"""Plot held-out single-answer turn-transfer calibration from complete results."""

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
import numpy as np
from matplotlib.lines import Line2D

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    MUTED,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

METHODS = {
    "raw": ("Uncalibrated", ":", "s"),
    "bias": ("Bias only", "--", "^"),
    "bias_scale": ("Bias + scale", "-", "o"),
}
MODELS = {
    "pretrained": ("Base", "base_model"),
    "instruct": ("Instruction-tuned", "post_trained"),
}


def main():
    """Render all declared forward-transfer curves with numerical provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", choices=("retention", "cosine_top1"), default="retention")
    args = parser.parse_args()
    ylim = (0.25, 1.1) if args.metric == "retention" else (0.0, 1.02)
    data = json.loads(args.results.read_text())
    if data["status"] != "complete" or data["answer_draws"] != 1 or len(data["cells"]) != 50:
        raise RuntimeError("plot requires all 50 completed single-answer cells")
    indexed = {(r["model"], r["source_turn"], r["target_turn"]): r for r in data["cells"]}
    expected = {
        (model, source, target)
        for model in MODELS
        for source in (1, 3, 12)
        for target in ((12,) if source == 12 else range(1, 13))
    }
    if set(indexed) != expected:
        raise RuntimeError("duplicate, missing or unexpected result cells")
    if max(r["raw_parent_abs_error"] for r in data["cells"]) > 1e-7:
        raise RuntimeError("parent parity not established")
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.67)
    grid = fig.add_gridspec(
        2, 2, left=0.10, right=0.98, bottom=0.16, top=0.92, wspace=0.25, hspace=0.48
    )
    points = []
    for row, (model, (label, role)) in enumerate(MODELS.items()):
        color = ROLES[role].color
        for col, source in enumerate((1, 3)):
            ax = fig.add_subplot(grid[row, col])
            for method, (_, linestyle, marker) in METHODS.items():
                turns = list(range(source, 13))
                scores = [indexed[model, source, turn]["metrics"][method] for turn in turns]
                values = [
                    s["retention"]
                    if args.metric == "retention"
                    else s["retrieval"]["cosine"]["top1"]
                    for s in scores
                ]
                if not all(ylim[0] <= value <= ylim[1] for value in values):
                    raise RuntimeError("plotted value falls outside the declared axis")
                (line,) = ax.plot(
                    turns,
                    values,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.1,
                    markersize=5.6,
                    markerfacecolor=color if method == "bias_scale" else "white",
                )
                np.testing.assert_allclose(line.get_ydata(), values, rtol=0, atol=1e-12)
                points.extend(
                    dict(
                        model=model,
                        source_turn=source,
                        target_turn=t,
                        method=method,
                        metric=args.metric,
                        value=v,
                    )
                    for t, v in zip(turns, values, strict=True)
                )
            if args.metric == "retention":
                ax.axhline(1, color=MUTED, linewidth=1.2, linestyle=(0, (6, 4)))
            ax.set_xlim(0.8, 12.25)
            ax.set_xticks([1, 3, 6, 9, 12])
            ax.set_ylim(*ylim)
            ax.set_yticks(
                [0.4, 0.6, 0.8, 1.0] if args.metric == "retention" else [0, 0.25, 0.5, 0.75, 1]
            )
            if col == 0:
                ax.set_ylabel(
                    better_label(
                        "Fraction of own-turn $R^2$"
                        if args.metric == "retention"
                        else "Top-1 retrieval (cosine)"
                    )
                )
            if row == 1:
                ax.set_xlabel("Evaluated conversation turn")
            style_axis(ax)
            letter = "ABCD"[row * 2 + col]
            panel_header(
                ax, letter, label, title=f"{letter}  {label}: map from turn {source}", title_y=1.055
            )
            for artist in ax.texts:
                if artist.get_gid() == "c2a-kicker":
                    artist.set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            color=INK,
            label=label,
            linestyle=ls,
            marker=marker,
            markerfacecolor=INK if key == "bias_scale" else "white",
            linewidth=2.1,
        )
        for key, (label, ls, marker) in METHODS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        columnspacing=2.0,
        handlelength=2.4,
        frameon=False,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "turn_transfer_bias_scale_single_answer_20260914"
    if args.metric != "retention":
        stem = stem.with_name(stem.name + "_retrieval")
    saved = save_c2a_figure(
        fig,
        stem,
        include_width=fraction,
        title="Single-answer map transfer with bias and scale calibration",
        subject=data["calibration"],
        creator=Path(__file__).name,
    )
    metadata = {
        "results_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "points": points,
        "metric": args.metric,
        "retrieval_pools": "Each fold's held-out answer vectors; see results.json for exact pool sizes and chance rates.",
        "calibration": data["calibration"],
        "answer_draws": 1,
        "layer": 19,
        "uncertainty": data["uncertainty"],
        "own_turn_reference": data["own_turn_reference"],
        "not_drawn": "Backward transfer, own-turn-12 calibration anchor and identity+bias baseline; all retained in results.json.",
        "style_deviations": "Line style and marker encode calibration method; color encodes model. Single descriptive titles replace duplicate kickers.",
        "render": saved["record"],
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plt.close(fig)
    print(saved["png"])
    print(f"Verified {len(points)} plotted values against complete results.")


if __name__ == "__main__":
    main()
