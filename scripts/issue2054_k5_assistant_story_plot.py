"""Render transfer and paired-answer figures from completed assistant-story results."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from explore_persona_space.analysis.c2a_plot_style import (
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance
from scripts.issue2054_k5_assistant_story_analysis import STORY
from scripts.issue2054_k5_loso_calibration import SETTINGS

METHODS = (
    ("frozen", "Direct", ROLES["linear"].color, "o"),
    ("bias", "+ Bias", ROLES["nonlinear"].color, "D"),
    ("bias_scale", "+ Bias + scale", ROLES["other_source"].color, "^"),
    ("own", "Target's own map", ROLES["control"].color, "s"),
)


def summarize(data):
    if data["status"] != "complete" or len(data["models"]) != 2:
        raise ValueError("Only completed two-checkpoint results can be plotted")
    summary = {}
    for model in data["models"]:
        name = model["model"]
        pairs = []
        for _, prefix in SETTINGS:
            for source, target in ((STORY, prefix), (prefix, STORY)):
                selected = [
                    r
                    for r in model["transfers"]
                    if r["source"] == f"{source}__{name}" and r["target"] == f"{target}__{name}"
                ]
                if sorted(r["fold"] for r in selected) != list(range(5)):
                    raise ValueError("Missing or duplicated transfer fold")
                methods = {}
                for key, *_ in METHODS:
                    scores = [r["metrics"][key] for r in selected]
                    values = [s["r2"] for s in scores]
                    methods[key] = {
                        "r2": float(np.mean(values)),
                        "fold_range": [min(values), max(values)],
                        "euclidean_top1": float(
                            np.mean([s["retrieval"]["euclidean"]["acc_at_k"]["1"] for s in scores])
                        ),
                        "cosine_top1": float(
                            np.mean([s["retrieval"]["cosine"]["acc_at_k"]["1"] for s in scores])
                        ),
                        "pool_range": [
                            min(s["retrieval_pool"] for s in scores),
                            max(s["retrieval_pool"] for s in scores),
                        ],
                    }
                pairs.append({"source": source, "target": target, "methods": methods})
        summary[name] = {
            "transfers": pairs,
            "answers": model["answers"]["summary"],
            "coverage": model["answers"]["coverage"],
        }
    return summary


def transfer_figure(summary, out):
    fig, fraction = c2a_figure("full", 0.95)
    axes = fig.subplots(2, 2)
    labels = ["Assistant, chat", "Assistant, plain", "HELIOS", "Wren", "Dana", "Vex"]
    endpoints = [
        endpoint
        for result in summary.values()
        for pair in result["transfers"]
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
            ax.set_yticks(range(6), labels)
            ax.invert_yaxis()
            ax.axvline(0, color=ROLES["control"].color, linewidth=0.8)
            ax.grid(False, axis="y")
            ax.grid(True, axis="x", alpha=0.2)
            style_axis(ax, grid_axis="x")
            ax.set_xlim(low - padding, high + padding)
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
        out / "assistant_story_transfer",
        include_width=fraction,
        title="Assistant-in-story map transfer",
        subject="Held-out map transfer; five-fold range",
        creator=str(Path(__file__).relative_to(REPO)),
    )
    plt.close(fig)
    return rendered["record"]


def answer_figure(summary, out):
    fig, fraction = c2a_figure("full", 0.48)
    axes = fig.subplots(1, 2)
    metrics = [
        ("cross_draw_cosine", "Story ↔ chat", ROLES["linear"].color, "o"),
        ("story_within_draw_cosine", "Story repeats", ROLES["control"].color, "s"),
        ("chat_within_draw_cosine", "Chat repeats", ROLES["other_source"].color, "^"),
        ("mean_answer_cosine", "Five-draw means", ROLES["nonlinear"].color, "D"),
        ("centered_mean_answer_cosine", "Centered means", ROLES["needs_reasoning"].color, "P"),
    ]
    for ax, (model, result), letter in zip(axes, summary.items(), "AB"):
        for i, (key, label, color, marker) in enumerate(metrics):
            metric = result["answers"][key]
            lo, hi = metric["ci95"]
            ax.plot([lo, hi], [i, i], color=color, linewidth=2)
            ax.plot(
                metric["mean"],
                i,
                marker=marker,
                color=color,
                markerfacecolor="white" if "within" in key else color,
            )
            ax.text(1.03, i, f"{metric['mean']:.3f}", va="center")
        ax.set_yticks(range(len(metrics)), [r[1] for r in metrics])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.17)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(better_label("Answer-vector cosine"))
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", alpha=0.2)
        style_axis(ax, grid_axis="x")
        checkpoint = "Instruction-tuned" if model.endswith("-instruct") else "Base"
        n = result["coverage"]["n_query_matched_complete_five"]
        panel_header(ax, letter, checkpoint, f"{n:,} matched questions", kicker_y=1.14)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.77, bottom=0.18, wspace=0.74)
    rendered = save_c2a_figure(
        fig,
        out / "assistant_story_answer_similarity",
        include_width=fraction,
        title="Assistant answers in story and chat",
        subject="Paired five-draw answer representations",
        creator=str(Path(__file__).relative_to(REPO)),
    )
    plt.close(fig)
    return rendered["record"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    summary = summarize(data)
    args.out.mkdir(parents=True, exist_ok=True)
    set_c2a_style()
    renders = {
        "transfer": transfer_figure(summary, args.out),
        "answers": answer_figure(summary, args.out),
    }
    output = {
        "render": renders,
        "models": summary,
        "source_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(
            (REPO / "src/explore_persona_space/analysis/c2a_plot_style.py").read_bytes()
        ).hexdigest(),
        "metadata": as_metadata_dict(git_provenance(cwd=REPO), phase="assistant_story_plot"),
        "outputs_sha256": {
            name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
            for stem in ("assistant_story_transfer", "assistant_story_answer_similarity")
            for name in (f"{stem}.pdf", f"{stem}.png", f"{stem}_grayscale.png")
        },
        "transfer_errorbars": "Minimum and maximum across the five held-out folds; not confidence intervals.",
        "answer_errorbars": "95% percentile intervals from 200 conversation-level bootstrap resamples.",
        "calibration": "Bias and bias plus scale use target-training examples; Direct uses none.",
        "answer_caveat": "Same literal question; narrative-added information may differ. Cosine is representation similarity, not a semantic-equivalence score.",
    }
    (args.out / "summary.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
