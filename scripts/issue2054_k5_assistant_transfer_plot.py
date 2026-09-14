"""Render assistant-source transfer with exact fold-mean R2 annotations."""

# ruff: noqa: E402
# The source helper loads thread caps before scientific dependencies.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

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
from scripts import issue2054_k5_assistant_transfer as analysis


def plot_plain(out, fig_dir):
    """Plot every plain-only transfer target and write the full metric report."""
    data = analysis.collect(out, "plain_only")
    rows = {p["cell"]: p for p in data["panels"]}
    targets = [analysis.base.SETTINGS[0], *analysis.base.SETTINGS[2:]]
    styles = [
        ("frozen", "Frozen transfer", MUTED, "o"),
        ("bias", "+ Bias", ROLES["linear"].color, "s"),
        ("bias_scale", "+ Bias + scaling", ROLES["base_model"].color, "D"),
    ]
    values = [
        r["metrics"][key]["r2"] for p in data["panels"] for r in p["folds"] for key, *_ in styles
    ] + [p["own_r2"] for p in data["panels"]]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite plain transfer score")
    lower, upper = min(0, min(values)), max(0, max(values))
    margin = max(0.04, 0.07 * (upper - lower))
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.65)
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.22, top=0.73, wspace=0.14)
    report_rows, retrieval_rows = [], []
    for ax, model, title, letter in zip(
        axes, analysis.base.MODELS, ["Base", "Instruction-tuned"], ["A", "B"], strict=True
    ):
        for i, (label, prefix) in enumerate(targets):
            p = rows[f"{prefix}__{model}"]
            ax.plot([i - 0.34, i + 0.34], [p["own_r2"]] * 2, color=INK, lw=1.6)
            for offset, (key, _, color, marker) in zip([-0.23, 0, 0.23], styles, strict=True):
                folds = [r["metrics"][key]["r2"] for r in p["folds"]]
                ax.scatter(
                    i + offset + np.linspace(-0.04, 0.04, 5),
                    folds,
                    s=15,
                    color=color,
                    marker=marker,
                    alpha=0.35,
                )
                ax.scatter(
                    i + offset,
                    p["r2_mean"][key],
                    s=78,
                    color=color,
                    marker=marker,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=3,
                )
            flat_label = label.replace("\n", " ")
            report_rows.append(
                f"| {title} | {flat_label} | "
                + " | ".join(f"{p['r2_mean'][key]:.4f}" for key, *_ in styles)
                + f" | {p['own_r2']:.4f} |"
            )
            for key, name, *_ in styles:
                mean_nn = float(
                    np.mean(
                        [
                            r["metrics"][key]["retrieval"]["euclidean"]["acc_at_k"]["1"]
                            for r in p["folds"]
                        ]
                    )
                )
                retrieval_rows.append(f"| {title} | {flat_label} | {name} | {100 * mean_nn:.2f}% |")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(5), [label for label, _ in targets], rotation=35, ha="right")
        ax.set_xlim(-0.55, 4.55)
        ax.set_ylim(lower - margin, upper + margin)
        ax.set_xlabel("Transfer target")
        style_axis(ax)
        panel_header(ax, letter, "Train: assistant in plain text", title=title)
    axes[0].set_ylabel(better_label("Held-out $R^2$"))
    handles = [
        Line2D([], [], marker=marker, color=color, linestyle="None", markersize=9, label=label)
        for _, label, color, marker in styles
    ]
    handles.append(Line2D([], [], color=INK, label="Separate target map"))
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.98), ncol=2, frameon=False
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = fig_dir / "plain_assistant_transfer"
    saved = save_c2a_figure(
        fig,
        stem,
        title="Transfer from plain-text assistant only",
        subject=data["method"],
        creator=Path(__file__).name,
        include_width=fraction,
    )
    analysis.base.atomic_json(stem.with_suffix(".data.json"), data)
    analysis.base.atomic_json(
        stem.with_suffix(".meta.json"),
        {
            "render": saved["record"],
            "source_sha256": analysis.base.sha(out / "results.json"),
            "script_sha256": analysis.base.sha(__file__),
            "method": data["method"],
            "aggregation": "Five globally held-out conversation folds; small points are individual folds",
            "calibration": "Vector bias and one scalar fit on target training folds; frozen uses no target labels",
            "own_map": "Separate ridge fit on the target setting's training conversations",
        },
    )
    plt.close(fig)
    pools = [r["metrics"]["frozen"]["retrieval_pool"] for p in data["panels"] for r in p["folds"]]
    prior_path = out / "chat_source_reference.json"
    if (
        analysis.base.sha(prior_path)
        != "1ea49e91af068bd2c2729e5c6ed6c359d0e6ae0ec92c1798f054a376b73ee58d"
    ):
        raise RuntimeError("chat-source reference differs from the published result")
    prior = json.loads(prior_path.read_text())
    prior_characters = {
        p["cell"]: p
        for p in prior["panels"]
        if p["regime"] == "assistant_only" and p["cell"].startswith("char_")
    }
    for p in data["panels"]:
        if p["cell"].startswith("char_"):
            previous = {r["fold"]: r for r in prior_characters[p["cell"]]["folds"]}
            for r in p["folds"]:
                if r["audit"]["test_ids_sha256"] != previous[r["fold"]]["audit"]["test_ids_sha256"]:
                    raise RuntimeError("source comparison uses different target conversations")
    macro_rows, copy_rows = [], []
    for model, title in zip(analysis.base.MODELS, ["Base", "Instruction-tuned"], strict=True):
        for name, dataset, regime in [
            ("Plain text", data, "assistant_plain_only"),
            ("Chat template", prior, "assistant_only"),
        ]:
            selected = [
                p
                for p in dataset["panels"]
                if p["model"] == model and p["regime"] == regime and p["cell"].startswith("char_")
            ]
            if len(selected) != 4:
                raise RuntimeError("character-source comparison requires all four matched targets")
            macro_rows.append(
                f"| {title} | {name} | "
                + " | ".join(
                    f"{np.mean([p['r2_mean'][key] for p in selected]):.4f}" for key, *_ in styles
                )
                + " |"
            )
        for p in [p for p in data["panels"] if p["model"] == model]:
            label = next(
                label.replace("\n", " ")
                for label, prefix in targets
                if p["cell"] == f"{prefix}__{model}"
            )
            copy_rows.append(
                f"| {title} | {label} | {p['r2_mean']['source_identity_bias']:.4f} | {p['r2_mean']['identity_bias_target']:.4f} |"
            )
    url = f"https://huggingface.co/datasets/{analysis.base.HF_REPO}/resolve/main/{analysis.base.PREFIX}/transfer_calibration_v1/plain_assistant_source/figures/plain_assistant_transfer.png"
    text = (
        f"""# K5 transfer from plain-text assistant only

[Open the transfer figure]({url}).

Source: only `conversation_paired_stories_assistant__on_policy__bare_text`.
Targets: assistant in the chat template, HELIOS, Wren, Dana, and Vex.
Both Qwen2.5-7B checkpoints use the existing on-policy K5 banks: mean of five
sampled answer vectors, temperature 1, top-p 1, cap 2048; layer 19 (block 18),
dimension 3584. The original complete-five cohort includes capped nonempty completions.
The same conversation fold is held out globally from source fitting and target calibration.
All 10 source maps, 10 panels, and 50 fold evaluations completed.

Frozen transfer has no target labels. Bias fits a vector intercept using target
training folds. Bias + scaling fits one shared scalar and a vector intercept on
those same training folds, leaving the source map fixed. Black lines show a
separate map fitted on target training folds. Small points are the five fold scores;
large points and tables are unweighted fold means. All source ridge hyperparameters
are inherited from the validated issue2054 K5 recipe: standardized inputs,
GCV over logspace(-2,4,13), degrees-of-freedom cap0.9.

## Transfer R² by target

| Model | Target | Frozen | + Bias | + Bias + scaling | Separate target map |
|---|---|---:|---:|---:|---:|
"""
        + "\n".join(report_rows)
        + """

## Matched character-target averages

Both source conditions are evaluated on the same four target characters and the
same five conversation folds. The chat-source reference is the completed prior
assistant-only run; its result SHA256 is `"""
        + analysis.base.sha(prior_path)
        + """`.

| Model | Assistant source format | Frozen | + Bias | + Bias + scaling |
|---|---|---:|---:|---:|
"""
        + "\n".join(macro_rows)
        + f"""

## Euclidean top-1 retrieval

Held-out candidate pools range from {min(pools)} to {max(pools)} examples.
Chance ranges from {100 / max(pools):.4f}% to {100 / min(pools):.4f}%.
Cosine retrieval and top-5/top-10 metrics are also retained in results.json.

| Model | Target | Calibration | Top-1 |
|---|---|---|---:|
"""
        + "\n".join(retrieval_rows)
        + """

## Identity plus learned bias baselines

The source-bias baseline learns its offset on plain-assistant training rows;
the target-bias baseline learns its offset on target training rows.

| Model | Target | Copy + source bias R² | Copy + target bias R² |
|---|---|---:|---:|
"""
        + "\n".join(copy_rows)
        + """

These are representation-prediction scores, not qualitative behavior or refusal
measurements. Story characters use attributed quotation; no assistant-in-story
K5 condition is included. Calibrated scores must not be described as zero-shot.

## Reproduce

Use `issue2054_k5_assistant_transfer.py --source-mode plain_only --stage fit`
once per model, providing --out and --inputs (the pinned LOSO inputs.json directory).
Then run `issue2054_k5_assistant_transfer_plot.py --source-mode plain_only --out ... --fig-dir ...`.
The monitored two-worker launch is `issue2054_k5_plain_run.py`; runtime.json and
monitoring/ record the launch environment, process observations, and complete logs.
Map weights and calibration coefficients are persisted under maps/ and folds/.
"""
    )
    (out / "README.md").write_text(text)
    print(f"[phase=done] {saved['png']}", flush=True)


def plot(out, fig_dir):
    """Show all four character additions; mask in-source targets explicitly."""
    data = analysis.collect(out)
    rows = {(p["model"], p["regime"], p["cell"]): p for p in data["panels"]}
    variants = [
        ("frozen", "Frozen transfer"),
        ("bias", "+ Bias"),
        ("bias_scale", "+ Bias + scaling"),
    ]
    values = [p["r2_mean"][key] for p in data["panels"] for key, _ in variants]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite plotted score")
    lower, upper = min(-0.01, min(values)), max(0.01, max(values))
    norm = TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#F0F0F0")
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.76)
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.21, top=0.91, wspace=0.13, hspace=0.9)
    targets = analysis.base.SETTINGS[1:]
    row_labels = ["Assistant only"] + [
        f"Assistant + {label}" for label, _ in analysis.base.SETTINGS[2:]
    ]
    for mi, (model, model_label) in enumerate(
        zip(analysis.base.MODELS, ["Base", "Instruction-tuned"], strict=True)
    ):
        regimes = analysis.source_sets(model)
        for vi, (key, title) in enumerate(variants):
            ax = axes[mi, vi]
            matrix = np.full((5, 5), np.nan)
            for i, (regime, source_cells) in enumerate(regimes.items()):
                for j, (_, prefix) in enumerate(targets):
                    cell = f"{prefix}__{model}"
                    if cell not in source_cells:
                        matrix[i, j] = rows[(model, regime, cell)]["r2_mean"][key]
            im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for (i, j), value in np.ndenumerate(matrix):
                if np.isnan(value):
                    ax.text(j, i, "—", ha="center", va="center", color=MUTED)
                else:
                    rgb = cmap(norm(value))[:3]
                    lightness = np.dot(rgb, [0.2126, 0.7152, 0.0722])
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=INK if lightness > 0.58 else "white",
                    )
            ax.set_xticks(
                range(5),
                ["Assistant (plain)"] + [label for label, _ in targets[1:]],
                rotation=45,
                ha="right",
            )
            ax.set_yticks(range(5), row_labels if vi == 0 else [""] * 5)
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            panel_header(ax, chr(ord("A") + mi * 3 + vi), model_label, title=title)
    bar_ax = fig.add_axes([0.30, 0.035, 0.56, 0.018])
    cb = fig.colorbar(im, cax=bar_ax, orientation="horizontal")
    cb.set_label(better_label("Held-out $R^2$"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = fig_dir / "assistant_source_transfer"
    saved = save_c2a_figure(
        fig,
        stem,
        title="Transfer from assistant and assistant plus one character",
        subject=data["method"],
        creator=Path(__file__).name,
        include_width=fraction,
    )
    analysis.base.atomic_json(stem.with_suffix(".data.json"), data)
    analysis.base.atomic_json(
        stem.with_suffix(".meta.json"),
        {
            "render": saved["record"],
            "source_sha256": analysis.base.sha(out / "results.json"),
            "script_sha256": analysis.base.sha(__file__),
            "method": data["method"],
            "missing_cells": "Dashes identify targets included in source training; not evaluated as transfer",
            "aggregation": "Unweighted mean of the same five held-out conversation folds",
            "calibration": "Bias and scalar use target non-test-fold labels; frozen uses no target labels",
        },
    )
    plt.close(fig)
    print(f"[phase=done] {saved['png']}", flush=True)


def main():
    """Read verified fit artifacts and export color, vector, and grayscale plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--source-mode", choices=["chat_grid", "plain_only"], default="chat_grid")
    args = parser.parse_args()
    if args.source_mode == "plain_only":
        plot_plain(args.out, args.fig_dir)
    else:
        plot(args.out, args.fig_dir)


if __name__ == "__main__":
    main()
