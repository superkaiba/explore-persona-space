#!/usr/bin/env python3
"""Render the issue #2569 mapping-diff and few-shot transfer report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


CONTRASTS = ("writer", "encoder", "interaction", "diagonal")
CELLS = (
    ("q_to_l", "qwriter", "Q→L · Q writer"),
    ("q_to_l", "lwriter", "Q→L · L writer"),
    ("l_to_q", "qwriter", "L→Q · Q writer"),
    ("l_to_q", "lwriter", "L→Q · L writer"),
)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


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
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "savefig.dpi": 300,
            "svg.hashsalt": "issue2569-mapping-diff",
            "svg.fonttype": "path",
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def finish_axes(ax: mpl.axes.Axes) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    plt.close(fig)
    # Matplotlib can emit path-data continuation lines with trailing spaces.
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )


def plot_factorial(mapping: dict[str, Any], out_dir: Path) -> None:
    reads = mapping["heldout_contrast_prediction"]
    native = mapping["alignment_free_within_encoder_writer_prediction"]
    x = np.arange(len(CONTRASTS))
    labels = [name.title() for name in CONTRASTS]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), constrained_layout=True)

    r2 = [reads[name]["pooled_r2"] for name in CONTRASTS]
    null_r2 = [reads[name]["permutation_null"]["pooled_r2"] for name in CONTRASTS]
    axes[0].bar(x, r2, color=[ORANGE, BLUE, GREEN, GRAY], width=0.68)
    for offset, key, label, color in (
        (-0.12, "qwen_native_qwriter_minus_lwriter", "Qwen native writer", BLUE),
        (0.12, "llama_native_qwriter_minus_lwriter", "Llama native writer", GREEN),
    ):
        axes[0].scatter(
            offset,
            native[key]["pooled_r2"],
            s=28,
            facecolor="white",
            edgecolor=color,
            linewidth=1.3,
            label=label,
            zorder=4,
        )
    for i, null in enumerate(null_r2):
        lo, hi = null["null_p025_p975"]
        axes[0].vlines(i, lo, hi, color="black", linewidth=1.2)
        axes[0].plot(i, null["null_mean"], marker="_", color="black", markersize=7)
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_title("Held-out contrast magnitude prediction")
    axes[0].set_ylabel("Pooled $R^2$")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(-0.39, 0.32)
    axes[0].legend(
        frameon=False,
        loc="upper center",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    finish_axes(axes[0])

    cosine = [reads[name]["flat_cosine"] for name in CONTRASTS]
    null_cos = [reads[name]["permutation_null"]["flat_cosine"] for name in CONTRASTS]
    axes[1].bar(x, cosine, color=[ORANGE, BLUE, GREEN, GRAY], width=0.68)
    for offset, key, color in (
        (-0.12, "qwen_native_qwriter_minus_lwriter", BLUE),
        (0.12, "llama_native_qwriter_minus_lwriter", GREEN),
    ):
        axes[1].scatter(
            offset,
            native[key]["flat_cosine"],
            s=28,
            facecolor="white",
            edgecolor=color,
            linewidth=1.3,
            zorder=4,
        )
    for i, null in enumerate(null_cos):
        lo, hi = null["null_p025_p975"]
        axes[1].vlines(i, lo, hi, color="black", linewidth=1.2)
        axes[1].plot(i, null["null_mean"], marker="_", color="black", markersize=7)
    axes[1].set_title("Prompt-specific contrast direction")
    axes[1].set_ylabel("Flattened cosine")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 0.52)
    finish_axes(axes[1])
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    save_figure(fig, out_dir / "fig1_factorial_mapping_diff")


def _cell_curve(
    transfer: dict[str, Any],
    direction: str,
    writer: str,
    method: str,
    group: str,
    metric: str,
    ks: list[int],
) -> np.ndarray:
    return np.asarray(
        [
            transfer["few_query"][direction][writer][str(k)][method][group][metric][
                "median"
            ]
            for k in ks
        ],
        np.float64,
    )


def plot_fewshot(transfer: dict[str, Any], out_dir: Path) -> None:
    ks = [int(k) for k in transfer["design"]["hyperparameters"]["k_values"]]
    x = np.arange(1, len(ks) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)

    curves: dict[str, list[np.ndarray]] = {
        "transported_source_mapping": [],
        "target_fit_from_scratch": [],
    }
    for direction, writer, _ in CELLS:
        for method in curves:
            curve = _cell_curve(
                transfer,
                direction,
                writer,
                method,
                "full_target_mapping",
                "centered_cosine",
                ks,
            )
            curves[method].append(curve)
            axes[0].plot(x, curve, color=BLUE if method.startswith("transported") else ORANGE, alpha=0.18)
    for method, label, color, marker in (
        ("transported_source_mapping", "Transport frozen source map", BLUE, "o"),
        ("target_fit_from_scratch", "Fit target from same k", ORANGE, "s"),
    ):
        matrix = np.vstack(curves[method])
        median = np.median(matrix, axis=0)
        axes[0].plot(x, median, color=color, marker=marker, label=label)
        axes[0].fill_between(x, matrix.min(0), matrix.max(0), color=color, alpha=0.09, linewidth=0)
    summary_values = [
        transfer["summary_only"][direction][writer]["full_target_mapping"]["centered_cosine"]
        for direction, writer, _ in CELLS
    ]
    axes[0].scatter(
        [0],
        [np.median(summary_values)],
        marker="D",
        color=GRAY,
        label="PCA summaries only",
        zorder=4,
    )
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_title("Agreement with full target mapping")
    axes[0].set_ylabel("Centered cosine")
    axes[0].set_xticks(np.arange(len(ks) + 1), ["0†", *[str(k) for k in ks]])
    axes[0].set_xlabel("Paired calibration queries (†unpaired summaries only)")
    axes[0].set_ylim(-0.06, 0.9)
    axes[0].legend(frameon=False, loc="lower right")
    finish_axes(axes[0])

    pooled_advantage: list[np.ndarray] = []
    for k in ks:
        values = []
        for direction, writer, _ in CELLS:
            values.extend(
                transfer["few_query"][direction][writer][str(k)][
                    "paired_transport_advantage"
                ]["full_target_mapping"]["centered_cosine"]["values"]
            )
        pooled_advantage.append(np.asarray(values, np.float64))
    med = np.asarray([np.median(values) for values in pooled_advantage])
    lo = np.asarray([np.quantile(values, 0.1) for values in pooled_advantage])
    hi = np.asarray([np.quantile(values, 0.9) for values in pooled_advantage])
    axes[1].plot(x, med, color=GREEN, marker="o")
    axes[1].fill_between(x, lo, hi, color=GREEN, alpha=0.18, linewidth=0)
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_title("Benefit of transfer at equal query budget")
    axes[1].set_ylabel("Paired Δ cosine (transport − scratch)")
    axes[1].set_xticks(x, [str(k) for k in ks])
    axes[1].set_xlabel("Paired calibration queries")
    axes[1].set_ylim(-0.06, 0.13)
    finish_axes(axes[1])
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    save_figure(fig, out_dir / "fig2_fewshot_transfer")


def plot_behavior(mapping: dict[str, Any], out_dir: Path) -> None:
    heldout = mapping["behavior_readout"]["heldout"]
    seed = mapping["behavior_readout"].get("seed137_frozen_readout")
    keys = ("log_length_delta", "refusal_delta", "repetition_delta", "semantic_divergence")
    labels = ("Log length", "Refusal", "Repetition", "Semantic\ndivergence")
    numeric = lambda value: np.nan if value is None else float(value)
    oracle = [numeric(heldout[key]["observed_activation_readout"]["r2"]) for key in keys]
    mediated = [numeric(heldout[key]["mapping_mediated"]["r2"]) for key in keys]
    replicated = [numeric(seed[key]["r2"]) for key in keys] if seed else None
    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(5.5, 2.65), constrained_layout=True)
    if replicated is None:
        ax.bar(x - width / 2, oracle, width, color="#56B4E9", label="Observed activation readout")
        ax.bar(x + width / 2, mediated, width, color=ORANGE, label="Mapping-mediated · seed 42")
    else:
        ax.bar(x - width, oracle, width, color="#56B4E9", label="Observed activation readout")
        ax.bar(x, mediated, width, color=ORANGE, label="Mapping-mediated · seed 42")
        ax.bar(x + width, replicated, width, color=GREEN, label="Mapping-mediated · seed 137")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_title("Which behavioral differences survive through the map contrast?")
    ax.set_ylabel("Held-out $R^2$")
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.09, 1.0)
    ax.legend(frameon=False, ncol=1, loc="upper right")
    finish_axes(ax)
    save_figure(fig, out_dir / "fig3_behavior_readout")


def transfer_table(transfer: dict[str, Any]) -> str:
    lines = [
        "| Direction / writer | 16 queries | 32 queries | 64 queries | 128 queries | 256 queries |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for direction, writer, label in CELLS:
        values = []
        for k in (16, 32, 64, 128, 256):
            row = transfer["few_query"][direction][writer][str(k)]
            transported = row["transported_source_mapping"]["full_target_mapping"][
                "centered_cosine"
            ]["median"]
            scratch = row["target_fit_from_scratch"]["full_target_mapping"][
                "centered_cosine"
            ]["median"]
            values.append(f"{transported:.3f} / {scratch:.3f}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def aggregate_transfer_cosine(transfer: dict[str, Any], k: int, method: str) -> float:
    values = [
        transfer["few_query"][direction][writer][str(k)][method][
            "full_target_mapping"
        ]["centered_cosine"]["median"]
        for direction, writer, _ in CELLS
    ]
    return float(np.median(values))


def aggregate_observed(transfer: dict[str, Any], k: int, method: str, metric: str) -> float:
    values = [
        transfer["few_query"][direction][writer][str(k)][method]["observed_target"][
            metric
        ]["median"]
        for direction, writer, _ in CELLS
    ]
    return float(np.median(values))


def aggregate_full_target_observed(transfer: dict[str, Any], metric: str) -> float:
    values = [
        transfer["full_target_map_ceiling"][direction][writer]["observed_target"][
            metric
        ]
        for direction, writer, _ in CELLS
    ]
    return float(np.median(values))


def pooled_advantage(transfer: dict[str, Any], k: int) -> np.ndarray:
    values: list[float] = []
    for direction, writer, _ in CELLS:
        values.extend(
            transfer["few_query"][direction][writer][str(k)][
                "paired_transport_advantage"
            ]["full_target_mapping"]["centered_cosine"]["values"]
        )
    return np.asarray(values, np.float64)


def fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_report(mapping: dict[str, Any], transfer: dict[str, Any], out_dir: Path) -> None:
    reads = mapping["heldout_contrast_prediction"]
    geometry = mapping["contrast_geometry"]
    split = mapping["split_half_stability"]["contrasts"]
    summary_cos = [
        transfer["summary_only"][direction][writer]["full_target_mapping"]["centered_cosine"]
        for direction, writer, _ in CELLS
    ]
    behavior = mapping["behavior_readout"]["heldout"]
    seed_behavior = mapping["behavior_readout"].get("seed137_frozen_readout")
    alignment = mapping["fixed_alignment"]
    alignment_test = alignment.get("heldout_test")
    if alignment_test is None:
        alignment_test = {
            key: alignment.get(key)
            for key in (
                "context_q_to_l_flat_cosine",
                "answer_qwriter_q_to_l_flat_cosine",
                "answer_lwriter_q_to_l_flat_cosine",
            )
        }
    if any(value is None for value in alignment_test.values()):
        raise ValueError("mapping JSON lacks usable alignment-quality metrics")
    within_writer = mapping["alignment_free_within_encoder_writer_prediction"]
    advantage32 = pooled_advantage(transfer, 32)
    advantage64 = pooled_advantage(transfer, 64)
    low_advantage = {k: pooled_advantage(transfer, k) for k in (2, 4, 8)}
    pca_ceiling = [
        transfer["pca_reconstruction_ceiling"][direction][writer][
            "full_target_mapping"
        ]["centered_cosine"]
        for direction, writer, _ in CELLS
    ]
    semantic_value = behavior["semantic_divergence"]["mapping_mediated"]["r2"]
    length_value = behavior["log_length_delta"]["mapping_mediated"]["r2"]
    refusal_value = behavior["refusal_delta"]["mapping_mediated"]["r2"]
    repetition_value = behavior["repetition_delta"]["mapping_mediated"]["r2"]
    seed_sentence = (
        f" On seed 137, semantic-divergence R² is {fmt_score(seed_behavior['semantic_divergence']['r2'])} "
        f"and log-length R² is {fmt_score(seed_behavior['log_length_delta']['r2'])}."
        if seed_behavior
        else " No independent generation seed was supplied."
    )
    other_behavior_values = [
        value for value in (length_value, refusal_value, repetition_value) if value is not None
    ]
    behavior_conclusion = (
        "Semantic divergence is the only mapping-mediated readout above 0.05 R² in this run."
        if semantic_value is not None
        and semantic_value > 0.05
        and all(value <= 0.05 for value in other_behavior_values)
        else "These readouts should be interpreted from their reported numeric values rather than as a categorical effect."
    )
    low_text = ", ".join(
        f"k={k}: median Δ {np.median(values):.3f}, {int(np.sum(values < 0))}/{len(values)} negative"
        for k, values in low_advantage.items()
    )
    report = f"""# Mapping differences and few-query transfer across Qwen and Llama

**Issue #2569 follow-up · Qwen2.5-7B-Instruct Q14 × Llama-3.1-8B-Instruct L16 · 10,000 LMSYS prompts**

## Bottom line

The fixed Procrustes alignment is incomplete (held-out context/answer cosines {alignment_test['context_q_to_l_flat_cosine']:.3f}, {alignment_test['answer_qwriter_q_to_l_flat_cosine']:.3f}, and {alignment_test['answer_lwriter_q_to_l_flat_cosine']:.3f}). Consequently, encoder, diagonal, and encoder×writer interaction contrasts can all contain coordinate-alignment residual. Writer and interaction both vanish under the stronger null of no writer effect in either encoder; only writer is exactly zero under its own null, because interaction's own null—equal writer effects across encoders—is not protected by imperfect alignment. Alignment-free within-model writer contrasts remain prompt-specific in Qwen (cosine {within_writer['qwen_native_qwriter_minus_lwriter']['flat_cosine']:.3f}, R² {within_writer['qwen_native_qwriter_minus_lwriter']['pooled_r2']:.3f}) and Llama (cosine {within_writer['llama_native_qwriter_minus_lwriter']['flat_cosine']:.3f}, R² {within_writer['llama_native_qwriter_minus_lwriter']['pooled_r2']:.3f}).

Mapping transfer is not useful at the very smallest budgets: at k=2, 4, and 8, every paired anchor draw favors fitting the target directly by centered cosine. A small directional advantage first emerges around 32 queries (median paired Δ cosine {np.median(advantage32):.3f}; {100 * np.mean(advantage32 > 0):.1f}% of 40 dependent cell-draw comparisons) and is consistent by 64 ({100 * np.mean(advantage64 > 0):.1f}% positive). At 256 queries, median centered cosine with the frozen full target map is {aggregate_transfer_cosine(transfer, 256, 'transported_source_mapping'):.3f}, versus {aggregate_transfer_cosine(transfer, 256, 'target_fit_from_scratch'):.3f} from scratch.

## 1. Factorial mapping diff

![Held-out factorial mapping contrasts](fig1_factorial_mapping_diff.png)

**Figure 1.** Four native maps—encoder {{Qwen, Llama}} × answer writer {{Qwen, Llama}}—were transformed into one fixed Qwen basis using train-only semi-orthogonal Procrustes alignments. Open circles on the writer bar are the alignment-free native Qwen/Llama writer contrasts. Black marks show the 95% row-permutation null. Encoder, diagonal, and interaction terms can contain alignment residual; their R² values are {reads['encoder']['pooled_r2']:.3f}, {reads['diagonal']['pooled_r2']:.3f}, and {reads['interaction']['pooled_r2']:.3f}. The aligned writer term has cosine {reads['writer']['flat_cosine']:.3f} and R² {reads['writer']['pooled_r2']:.3f}. Every observed cosine exceeds all 1,000 row-pairing permutations (p={reads['writer']['permutation_null']['flat_cosine']['p_ge']:.4f}), which establishes prompt specificity but does not remove alignment confounding.

Numerically, the encoder-labeled representation contrast is largest and most split-half stable (RMS {geometry['encoder']['heldout_prediction_rms_norm']:.2f}; cosine {split['encoder']['data_weighted_half1_vs_half2_cosine']:.3f}). The writer contrast has RMS {geometry['writer']['heldout_prediction_rms_norm']:.2f} and split-half cosine {split['writer']['data_weighted_half1_vs_half2_cosine']:.3f}. Interaction is smaller and less stable ({geometry['interaction']['heldout_prediction_rms_norm']:.2f}, {split['interaction']['data_weighted_half1_vs_half2_cosine']:.3f}) and may be entirely alignment residual. The diagonal replicates numerically on seed 137 (R² {mapping['seed137_reliability']['frozen_seed42_diagonal_map_on_seed137']['pooled_r2']:.3f}) but remains alignment-confounded. After alignment, writer and encoder operators are nearly orthogonal (cosine {geometry['writer_vs_encoder']['operator_cosine']:.3f}); because encoder is confounded, this is descriptive rather than a behavioral claim.

## 2. Can one mapping be transferred with only residual summaries or a few queries?

![Few-query mapping transfer](fig2_fewshot_transfer.png)

**Figure 2.** Panel A evaluates recovery of the frozen full-data target mapping on the untouched 1,500-row test set—not direct fit to observed answers. Thin lines are the four direction/writer cells; thick lines are their medians. The 0-query diamond uses separate mean/top-64 PCA/variance summaries with components paired only by variance rank and marginal skewness. The complete summary pipeline fails (centered cosine {min(summary_cos):.3f}–{max(summary_cos):.3f}). Answer-side rank-64 compression alone is not the cause: reconstructing target answers in the same rank-64 basis retains full-map cosine {min(pca_ceiling):.3f}–{max(pca_ceiling):.3f}; context-side compression remains a possible contributor. Stronger unsupervised alignment algorithms were not tested. Panel B shows paired within-anchor-set cosine advantage over the equal-query scratch control (median and pooled 10th–90th percentiles across 4 cells × 10 draws; these are dependent descriptive comparisons, not 40 independent trials).

For k paired train queries, regularized context and answer bridges use only sample means and centered Gram matrices. No validation rows tune the bridge. The frozen source map is applied between those two bridges. The control fits target context→answer directly from the identical k anchors.

Table entries are **transported source map / target fit from scratch**, measured by held-out centered cosine with the full target mapping:

{transfer_table(transfer)}

Below 16 queries, transport is systematically worse than scratch ({low_text}). At 16 the median difference is near zero. A small advantage appears at 32; by 64, {int(np.sum(advantage64 > 0))}/{len(advantage64)} dependent cell-draw comparisons favor transfer. At 256, transported predictions have median centered cosine {aggregate_observed(transfer, 256, 'transported_source_mapping', 'centered_cosine'):.3f} with actual target answers, versus {aggregate_observed(transfer, 256, 'target_fit_from_scratch', 'centered_cosine'):.3f} for scratch; the original 8,000-row target maps reach {aggregate_full_target_observed(transfer, 'centered_cosine'):.3f}. Scale-sensitive R² is retained only as a secondary diagnostic because the two-stage transport and one-stage scratch fits shrink differently. The crossover is conditional on the fixed, untuned ridge fraction 0.01; tuning could move it.

## 3. What behavioral differences are visible?

![Behavior readouts from the writer contrast](fig3_behavior_readout.png)

**Figure 3.** A ridge readout trained on the observed writer activation contrast can recover several answer differences, but only some survive when the writer contrast is predicted from context through the mapping diff. Seed-42 mapping-mediated R² values are semantic divergence {fmt_score(semantic_value)}, log length {fmt_score(length_value)}, refusal {fmt_score(refusal_value)}, and repetition {fmt_score(repetition_value)}.{seed_sentence} {behavior_conclusion}

## Exact design

- Frozen split: 8,000 train / 500 validation / 1,500 test prompts. The new analyses never fit on test rows.
- Fixed common basis: context Procrustes on Qwen-writer train contexts; answer Procrustes pooled over both answer writers. All affine translations are retained. Held-out context/answer alignment cosines are {alignment_test['context_q_to_l_flat_cosine']:.3f}/{alignment_test['answer_qwriter_q_to_l_flat_cosine']:.3f}/{alignment_test['answer_lwriter_q_to_l_flat_cosine']:.3f}, so encoder, interaction, and diagonal contrasts can include residual alignment error.
- Factorial contrasts: writer, encoder, encoder×writer interaction, and the natural diagonal Qwen-own − Llama-own difference. Encoder, interaction, and diagonal can contain alignment residual; within-encoder Qwen/Llama writer contrasts provide alignment-free checks.
- Null: 1,000 held-out row-pairing permutations that destroy prompt correspondence.
- Stability: two disjoint 4,000-row train refits at the original selected ridge lambdas; independent generation seed 137.
- Few-query transfer: both Qwen→Llama and Llama→Qwen, both answer writers, k ∈ {{2,4,8,16,32,64,128,256}}, 10 random anchor sets per cell, fixed ridge fraction 0.01, no validation tuning.
- Primary transfer score is cosine after subtracting the target model's train-answer mean from both the candidate and frozen target-map predictions. It measures recovery of the frozen target mapping, not direct fit to observed answers. The scale-sensitive normalized R² is retained in JSON as a secondary diagnostic.

## Interpretation and limits

Variance-ranked, skewness-oriented PCA summaries do not recover the correspondence; this does **not** rule out stronger unsupervised methods or isolate context compression from orientation failure. Paired transport is worse than direct target fitting at k≤8, near zero in median advantage at 16, begins to help around 32, and is consistently better by 64 under the fixed ridge setting. The procedure uses paired activations from both models and is calibration, not zero-shot transfer. Encoder, interaction, and diagonal factorial terms are alignment-confounded. Writer and interaction vanish under no writer effect in either encoder, but only writer is exactly zero under its own null; nonzero writer magnitude can still be distorted. Finally, this is an exploratory post-hoc LMSYS-only pilot; other model families, tasks, layers, and genuinely new prompts remain necessary tests.

## Reproducibility

- Source experiment revision: `{mapping['source_revision']}`
- Test roster SHA-256: `{mapping['test_roster_sha256']}`
- Primary outputs: [`mapping_diff.json`](mapping_diff.json), [`fewshot_transfer.json`](fewshot_transfer.json), [`heldout_rows.jsonl`](heldout_rows.jsonl), [`writer_modes.npz`](writer_modes.npz)
- Analysis drivers: `scripts/issue2569_mapping_diff.py`, `scripts/issue2569_fewshot_transfer.py`, and `scripts/issue2569_mapping_diff_report.py`
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(out_dir: Path, mapping_path: Path, transfer_path: Path) -> None:
    files = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "issue": 2569,
        "label": "mapping-diff-and-fewshot-transfer-report",
        "source_inputs": {
            "mapping_diff_json_sha256": file_sha256(mapping_path),
            "fewshot_transfer_json_sha256": file_sha256(transfer_path),
        },
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in files},
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--transfer-json", type=Path, required=True)
    parser.add_argument("--heldout-rows", type=Path)
    parser.add_argument("--writer-modes", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    mapping = load_json(args.mapping_json)
    transfer = load_json(args.transfer_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_factorial(mapping, args.out_dir)
    plot_fewshot(transfer, args.out_dir)
    plot_behavior(mapping, args.out_dir)
    write_report(mapping, transfer, args.out_dir)
    if args.heldout_rows:
        (args.out_dir / "heldout_rows.jsonl").write_bytes(args.heldout_rows.read_bytes())
    if args.writer_modes:
        (args.out_dir / "writer_modes.npz").write_bytes(args.writer_modes.read_bytes())
    (args.out_dir / "mapping_diff.json").write_bytes(args.mapping_json.read_bytes())
    (args.out_dir / "fewshot_transfer.json").write_bytes(args.transfer_json.read_bytes())
    write_manifest(args.out_dir, args.mapping_json, args.transfer_json)
    print(f"[mapping-diff-report] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
