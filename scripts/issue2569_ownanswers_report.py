#!/usr/bin/env python3
"""Build the concise plotted report for issue #2569 own-answer follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REGIMES = (
    ("same_qwen_written", "Same Qwen answer", "#0072B2", "o"),
    ("same_llama_written", "Same Llama answer", "#009E73", "s"),
    ("own_written", "Each model's own answer", "#D55E00", "^"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.7,
            "lines.markersize": 5,
            "savefig.dpi": 300,
            "svg.hashsalt": "issue2569-ownanswers",
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def finish_axes(ax: mpl.axes.Axes, *, ylim: tuple[float, float] = (0, 1)) -> None:
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def save_figure(fig: mpl.figure.Figure, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        out_stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    plt.close(fig)


def plot_shared_geometry(geometry: dict[str, Any], out_dir: Path) -> None:
    layers = geometry["layers"]
    x = np.arange(len(layers))
    xlabels = [f"Q{row['qwen_layer']} / L{row['llama_layer']}" for row in layers]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)

    for regime, label, color, marker in REGIMES:
        cka = [row["regimes"][regime]["answer_cka_train"] for row in layers]
        q2l = [row["regimes"][regime]["answer_alignment"]["q2l"]["test_r2"] for row in layers]
        l2q = [row["regimes"][regime]["answer_alignment"]["l2q"]["test_r2"] for row in layers]
        mean_r2 = (np.asarray(q2l) + np.asarray(l2q)) / 2
        lo_r2 = np.minimum(q2l, l2q)
        hi_r2 = np.maximum(q2l, l2q)
        operator = [row["regimes"][regime]["operator"]["observed_aligned_cosine"] for row in layers]

        axes[0].plot(x, cka, color=color, marker=marker, label=label)
        axes[1].plot(x, mean_r2, color=color, marker=marker, label=label)
        axes[1].fill_between(x, lo_r2, hi_r2, color=color, alpha=0.12, linewidth=0)
        axes[2].plot(x, operator, color=color, marker=marker, label=label)

    axes[0].set_title("Answer-space similarity")
    axes[0].set_ylabel("Linear CKA")
    axes[1].set_title("Held-out linear mapping")
    axes[1].set_ylabel("Pooled $R^2$ (direction mean)")
    axes[2].set_title("Context→answer operator")
    axes[2].set_ylabel("Aligned operator cosine")
    axes[2].axhline(0.0005, color="#666666", linestyle=":", linewidth=1, label="Haar-null 97.5% ≤ 0.0005")
    axes[2].axhline(0.6864, color="#7F7F7F", linestyle="--", linewidth=1, label="Within-model anchor = 0.686")

    for label, ax in zip("ABC", axes):
        panel_label(ax, label)
        finish_axes(ax)
        ax.set_xticks(x, xlabels)
        ax.set_xlabel("Layer pair")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09))
    save_figure(fig, out_dir / "fig1_shared_geometry")


def plot_writer_transfer(geometry: dict[str, Any], out_dir: Path) -> None:
    layers = geometry["layers"]
    x = np.arange(len(layers))
    xlabels = [f"Q{row['qwen_layer']} / L{row['llama_layer']}" for row in layers]
    writers = (
        ("same_qwen_written", "learn_qwriter_test_lwriter", "Qwen-written map", "#0072B2", "o"),
        ("same_llama_written", "learn_lwriter_test_qwriter", "Llama-written map", "#009E73", "s"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.5), constrained_layout=True)

    for ax, direction, title in zip(axes, ("q2l", "l2q"), ("Qwen → Llama", "Llama → Qwen")):
        for regime, transfer_key, label, color, marker in writers:
            in_domain = [row["regimes"][regime]["answer_alignment"][direction]["test_r2"] for row in layers]
            transfer = [row["cross_writer_answer_alignment_transfer"][transfer_key][direction]["r2"] for row in layers]
            ax.plot(x, in_domain, color=color, marker=marker, linestyle="-", label=f"{label}: same writer")
            ax.plot(x, transfer, color=color, marker=marker, linestyle="--", label=f"{label}: other writer")
        ax.set_title(title)
        ax.set_ylabel("Held-out pooled $R^2$")
        ax.set_xticks(x, xlabels)
        ax.set_xlabel("Layer pair")
        finish_axes(ax, ylim=(0.65, 0.9))

    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.15))
    save_figure(fig, out_dir / "fig2_cross_writer_transfer")


def plot_semantics_reliability(
    geometry: dict[str, Any], reliability: dict[str, Any], out_dir: Path
) -> None:
    primary = next(row for row in geometry["layers"] if row["primary"])
    bins = primary["regimes"]["own_written"]["semantic_strata"]["bins"]
    quartiles = [row["quartile"] for row in bins]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65), constrained_layout=True)

    series = (
        ("own_answer_cka", "Answer CKA", "#D55E00", "^"),
        ("own_alignment_r2", "Qwen→Llama $R^2$", "#0072B2", "o"),
        ("own_composed_route_r2", "Composed-route $R^2$", "#009E73", "s"),
    )
    for key, label, color, marker in series:
        axes[0].plot(quartiles, [row[key] for row in bins], color=color, marker=marker, label=label)
    axes[0].set_title("Own-answer alignment tracks semantics")
    axes[0].set_xlabel("Answer-pair semantic similarity quartile")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(quartiles, ["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    finish_axes(axes[0])
    axes[0].legend(frameon=False, loc="lower right")

    labels = ["Own CKA", "Q→L $R^2$", "L→Q $R^2$", "Q native $R^2$", "L native $R^2$"]
    seed42 = [
        primary["regimes"]["own_written"]["answer_cka_train"],
        reliability["frozen_seed42_map_reads_on_seed137"]["own_answer_alignment_q2l"]["seed42_test_r2"],
        reliability["frozen_seed42_map_reads_on_seed137"]["own_answer_alignment_l2q"]["seed42_test_r2"],
        reliability["frozen_seed42_map_reads_on_seed137"]["qwen_native_context_to_answer"]["seed42_test_r2"],
        reliability["frozen_seed42_map_reads_on_seed137"]["llama_native_context_to_answer"]["seed42_test_r2"],
    ]
    seed137 = [
        reliability["seed137_cross_model_own_answer_cka"],
        reliability["frozen_seed42_map_reads_on_seed137"]["own_answer_alignment_q2l"]["seed137"]["r2"],
        reliability["frozen_seed42_map_reads_on_seed137"]["own_answer_alignment_l2q"]["seed137"]["r2"],
        reliability["frozen_seed42_map_reads_on_seed137"]["qwen_native_context_to_answer"]["seed137"]["r2"],
        reliability["frozen_seed42_map_reads_on_seed137"]["llama_native_context_to_answer"]["seed137"]["r2"],
    ]
    bx = np.arange(len(labels))
    width = 0.36
    axes[1].bar(bx - width / 2, seed42, width, color="#56B4E9", label="Seed 42")
    axes[1].bar(bx + width / 2, seed137, width, color="#E69F00", label="Seed 137")
    axes[1].set_title("A second generation seed replicates")
    axes[1].set_ylabel("CKA or pooled $R^2$")
    axes[1].set_xticks(bx, labels, rotation=25, ha="right")
    finish_axes(axes[1], ylim=(0, 0.72))
    axes[1].legend(frameon=False, ncol=2, loc="upper right")

    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    save_figure(fig, out_dir / "fig3_semantics_reliability")


def rounded_table(geometry: dict[str, Any]) -> str:
    lines = [
        "| Layer pair | Regime | CKA | Q→L R² | L→Q R² | Operator cosine |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for layer in geometry["layers"]:
        pair = f"Q{layer['qwen_layer']} / L{layer['llama_layer']}"
        for regime, label, _, _ in REGIMES:
            row = layer["regimes"][regime]
            lines.append(
                f"| {pair} | {label} | {row['answer_cka_train']:.3f} | "
                f"{row['answer_alignment']['q2l']['test_r2']:.3f} | "
                f"{row['answer_alignment']['l2q']['test_r2']:.3f} | "
                f"{row['operator']['observed_aligned_cosine']:.3f} |"
            )
    return "\n".join(lines)


def write_report(
    geometry: dict[str, Any],
    reliability: dict[str, Any],
    semantic: dict[str, Any],
    out_dir: Path,
) -> None:
    primary = next(row for row in geometry["layers"] if row["primary"])
    own = primary["regimes"]["own_written"]
    sem = own["semantic_strata"]
    report = f"""# Cross-model geometry with model-generated answers

**Issue #2569 follow-up · Qwen2.5-7B-Instruct × Llama-3.1-8B-Instruct · 10,000 LMSYS prompts**

## Bottom line

The models exhibit a strong shared linear geometry when they encode the **same answer text**, and that geometry transfers across which model authored the answer. When each model encodes its **own generated answer**, alignment falls materially but remains substantial and replicates under a second generation seed. The defensible conclusion is therefore **shared geometry conditional on represented content**, not a fully content-free universal geometry.

![Shared geometry across answer regimes](fig1_shared_geometry.png)

**Figure 1.** Same-text answer spaces align strongly at all three frozen layer pairs. With each model's own answer, primary-layer CKA falls from {primary['regimes']['same_qwen_written']['answer_cka_train']:.3f}/{primary['regimes']['same_llama_written']['answer_cka_train']:.3f} to {own['answer_cka_train']:.3f}; bidirectional held-out mapping remains {own['answer_alignment']['q2l']['test_r2']:.3f}–{own['answer_alignment']['l2q']['test_r2']:.3f}. Operator cosine stays far above the exact Haar null (97.5th percentile ≤ 0.0005).

## What was tested

We crossed answer writer {{Qwen, Llama}} with activation encoder {{Qwen, Llama}}. The exact same answer strings were teacher-forced through both encoders for the same-text conditions; the own-answer condition paired each model's stochastic answer by prompt. Context vectors were the residual stream at the last prompt token; answer vectors were the answer-span residual mean including EOT. Ridge maps were selected on 500 validation prompts and evaluated on a frozen 1,500-prompt test set after fitting on 8,000 prompts. Primary layers were Q14/L16, with Q19/L22 and Q26/L30 companions.

## Writer transfer

![Cross-writer transfer](fig2_cross_writer_transfer.png)

**Figure 2.** A map learned from Qwen-authored same-text pairs transfers without refitting to Llama-authored pairs, and vice versa. Across layer pairs and directions, transfer R² is {min(layer['cross_writer_answer_alignment_transfer'][key][direction]['r2'] for layer in geometry['layers'] for key in ('learn_qwriter_test_lwriter', 'learn_lwriter_test_qwriter') for direction in ('q2l', 'l2q')):.3f}–{max(layer['cross_writer_answer_alignment_transfer'][key][direction]['r2'] for layer in geometry['layers'] for key in ('learn_qwriter_test_lwriter', 'learn_lwriter_test_qwriter') for direction in ('q2l', 'l2q')):.3f}. This is the clearest evidence that the cross-model coordinate mapping is not specific to one model's prose distribution.

## Why own answers align less

![Semantic stratification and replication](fig3_semantics_reliability.png)

**Figure 3.** Own-answer alignment increases sharply with semantic agreement: primary Qwen→Llama R² rises from {sem['bins'][0]['own_alignment_r2']:.3f} in the lowest semantic-similarity quartile to {sem['bins'][-1]['own_alignment_r2']:.3f} in the highest, while CKA rises from {sem['bins'][0]['own_answer_cka']:.3f} to {sem['bins'][-1]['own_answer_cka']:.3f}. Semantic similarity also correlates with aligned row cosine (Spearman ρ={sem['spearman_semantic_vs_aligned_row_cosine']['rho']:.3f}, n={sem['spearman_semantic_vs_aligned_row_cosine']['n']:,}, p={sem['spearman_semantic_vs_aligned_row_cosine']['p']:.2e}). A fresh seed-137 rollout reproduces primary own-answer CKA ({reliability['seed137_cross_model_own_answer_cka']:.3f}) and frozen-map performance within about one point of R².

Generated answers were only moderately equivalent (mean semantic cosine {semantic['embedding_cosine']['mean']:.3f}, median {semantic['embedding_cosine']['median']:.3f}, exact match {100 * semantic['exact_match_rate']:.2f}%). This supports the interpretation that policy/content divergence explains part of the own-answer alignment loss.

## Exact summary

{rounded_table(geometry)}

## Caveats

- This was a 10k-prompt LMSYS pilot, not the proposed 60k multi-corpus scale-up.
- The main answer-space conclusion does not depend on context recaptures. Context-route reads have a numerical packing caveat: rare max-relative-L2 recapture deviations exceeded the predeclared 0.02 gate, although mean deviations were small.
- Own-answer comparisons jointly include representation, model policy/content differences, and stochastic generation noise. Seed-137 replication reduces—but does not eliminate—that ambiguity.

## Reproducibility

- Primary result: [crossed_geometry.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8d2694f6eedfbad61b9413299bca096370429d7a/issue2569_theory/own_generated_answers/analysis/crossed_geometry.json) (SHA-256 `3e0cbef2804f3ba9672d67687f902732f017d367a4936285dd3c1f5d296db064`)
- Reliability: [reliability.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8d2694f6eedfbad61b9413299bca096370429d7a/issue2569_theory/own_generated_answers/analysis/reliability.json) (SHA-256 `9c77bcfa452e12510359691f9936d1132ae291dbbfc45e754791ae418638a1cd`)
- Semantic summary: [summary.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8d2694f6eedfbad61b9413299bca096370429d7a/issue2569_theory/own_generated_answers/analysis/semantic/summary.json)
- Model revisions: Qwen `a09a35458c702b33eeacc393d103063234e8bc28`; Llama `0e9e39f249a16976918f6564b8830bc894c89659`.
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(out_dir: Path) -> None:
    files = sorted(path for path in out_dir.iterdir() if path.is_file() and path.name != "MANIFEST.json")
    manifest = {
        "issue": 2569,
        "followup_label": "cross-model-own-generated-answers",
        "source_revision": "8d2694f6eedfbad61b9413299bca096370429d7a",
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--reliability", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry = load_json(args.geometry)
    reliability = load_json(args.reliability)
    semantic = load_json(args.semantic)
    configure_style()
    plot_shared_geometry(geometry, args.output_dir)
    plot_writer_transfer(geometry, args.output_dir)
    plot_semantics_reliability(geometry, reliability, args.output_dir)
    write_report(geometry, reliability, semantic, args.output_dir)
    write_manifest(args.output_dir)


if __name__ == "__main__":
    main()
