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
    x = np.arange(len(CONTRASTS))
    labels = [name.title() for name in CONTRASTS]
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), constrained_layout=True)

    r2 = [reads[name]["pooled_r2"] for name in CONTRASTS]
    null_r2 = [reads[name]["permutation_null"]["pooled_r2"] for name in CONTRASTS]
    axes[0].bar(x, r2, color=[ORANGE, BLUE, GREEN, GRAY], width=0.68)
    for i, null in enumerate(null_r2):
        lo, hi = null["null_p025_p975"]
        axes[0].vlines(i, lo, hi, color="black", linewidth=1.2)
        axes[0].plot(i, null["null_mean"], marker="_", color="black", markersize=7)
    axes[0].axhline(0, color="#777777", linewidth=0.8)
    axes[0].set_title("Held-out contrast magnitude prediction")
    axes[0].set_ylabel("Pooled $R^2$")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylim(-0.39, 0.27)
    finish_axes(axes[0])

    cosine = [reads[name]["flat_cosine"] for name in CONTRASTS]
    null_cos = [reads[name]["permutation_null"]["flat_cosine"] for name in CONTRASTS]
    axes[1].bar(x, cosine, color=[ORANGE, BLUE, GREEN, GRAY], width=0.68)
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
    alignment_test = alignment["heldout_test"]
    advantage32 = pooled_advantage(transfer, 32)
    advantage64 = pooled_advantage(transfer, 64)
    semantic_seed = (
        f" and replicates at {fmt_score(seed_behavior['semantic_divergence']['r2'])} on seed 137"
        if seed_behavior
        else " (no independent generation seed was supplied)"
    )
    length_seed = (
        f"; seed 137 {fmt_score(seed_behavior['log_length_delta']['r2'])}"
        if seed_behavior
        else ""
    )
    report = f"""# Mapping differences and few-query transfer across Qwen and Llama

**Issue #2569 follow-up · Qwen2.5-7B-Instruct Q14 × Llama-3.1-8B-Instruct L16 · 10,000 LMSYS prompts**

## Bottom line

A single fixed coordinate alignment does not make the two context→answer maps identical. After the shared Procrustes transform, encoder-dependent and answer-writer-dependent mapping changes remain and are nearly orthogonal in operator space (cosine {geometry['writer_vs_encoder']['operator_cosine']:.3f}). But the alignment is incomplete (held-out context/answer cosines only {alignment_test['context_q_to_l_flat_cosine']:.3f}, {alignment_test['answer_qwriter_q_to_l_flat_cosine']:.3f}, and {alignment_test['answer_lwriter_q_to_l_flat_cosine']:.3f}), so the encoder and diagonal contrasts cannot be interpreted as pure behavioral effects. The writer contrast is cleaner because it differences answer writers within each encoder, canceling a shared alignment residual.

The geometry is nevertheless calibratable with paired anchors. A small directional advantage first emerges around 32 queries (median paired Δ cosine {np.median(advantage32):.3f}; {100 * np.mean(advantage32 > 0):.1f}% of 40 repeat-cells) and is consistent by 64 ({100 * np.mean(advantage64 > 0):.1f}% positive). At 256 queries, median centered cosine with the full target map is {aggregate_transfer_cosine(transfer, 256, 'transported_source_mapping'):.3f}, versus {aggregate_transfer_cosine(transfer, 256, 'target_fit_from_scratch'):.3f} from scratch. This supports a shared correspondence that paired examples can identify; it does not establish that marginal statistics alone can identify it.

## 1. Factorial mapping diff

![Held-out factorial mapping contrasts](fig1_factorial_mapping_diff.png)

**Figure 1.** Four native maps—encoder {{Qwen, Llama}} × answer writer {{Qwen, Llama}}—were transformed into one fixed Qwen basis using train-only semi-orthogonal Procrustes alignments. Black marks show the 95% row-permutation null. Encoder and diagonal contrasts retain positive held-out magnitude R² ({reads['encoder']['pooled_r2']:.3f} and {reads['diagonal']['pooled_r2']:.3f}), but these two contrasts retain coordinate-alignment residual. Writer and interaction contrasts cancel a shared alignment residual; they have informative direction (cosine {reads['writer']['flat_cosine']:.3f} and {reads['interaction']['flat_cosine']:.3f}) but miscalibrated magnitude (negative R²). Every observed cosine exceeds all 1,000 row-pairing permutations (p={reads['writer']['permutation_null']['flat_cosine']['p_ge']:.4f}).

Numerically, the encoder-labeled representation contrast is largest and most split-half stable: exercised held-out RMS norm {geometry['encoder']['heldout_prediction_rms_norm']:.2f} and split-half cosine {split['encoder']['data_weighted_half1_vs_half2_cosine']:.3f}. The cleaner writer contrast has RMS {geometry['writer']['heldout_prediction_rms_norm']:.2f} and split-half cosine {split['writer']['data_weighted_half1_vs_half2_cosine']:.3f}; interaction is smaller and less stable ({geometry['interaction']['heldout_prediction_rms_norm']:.2f}, {split['interaction']['data_weighted_half1_vs_half2_cosine']:.3f}). The diagonal result replicates numerically on seed 137 (R² {mapping['seed137_reliability']['frozen_seed42_diagonal_map_on_seed137']['pooled_r2']:.3f}), but remains alignment-confounded.

## 2. Can one mapping be transferred with only residual summaries or a few queries?

![Few-query mapping transfer](fig2_fewshot_transfer.png)

**Figure 2.** Panel A evaluates the direction of the original full-data target mapping on the untouched 1,500-row test set. Thin lines are the four direction/writer cells; thick lines are their medians. The 0-query diamond uses separate mean/top-64 PCA/variance summaries with components paired only by variance rank and marginal skewness. This particular heuristic fails (centered cosine {min(summary_cos):.3f}–{max(summary_cos):.3f}), despite retaining {100 * min(transfer['summary_explained_fraction'].values()):.1f}%–{100 * max(transfer['summary_explained_fraction'].values()):.1f}% of residual energy. Stronger unsupervised alignment algorithms were not tested. Panel B shows paired within-anchor-set cosine advantage over the equal-query scratch control (median and pooled 10th–90th percentiles across 4 × 10 runs).

For k paired train queries, regularized context and answer bridges use only sample means and centered Gram matrices. No validation rows tune the bridge. The frozen source map is applied between those two bridges. The control fits target context→answer directly from the identical k anchors.

Table entries are **transported source map / target fit from scratch**, measured by held-out centered cosine with the full target mapping:

{transfer_table(transfer)}

At 16 queries, transport is directionally indistinguishable from scratch. A small advantage appears at 32, and at 64 every one of the 40 paired repeat-cells favors transfer. Scale-sensitive normalized R² also favors transfer, but is secondary because the two-stage transport and one-stage scratch fits have different shrinkage; centered cosine is the primary geometric comparison.

## 3. What behavioral differences are visible?

![Behavior readouts from the writer contrast](fig3_behavior_readout.png)

**Figure 3.** A ridge readout trained on the observed writer activation contrast can recover several answer differences, but only some survive when the writer contrast is predicted from context through the mapping diff. Mapping-mediated semantic-divergence R² is {fmt_score(behavior['semantic_divergence']['mapping_mediated']['r2'])}{semantic_seed}. Length is weak ({fmt_score(behavior['log_length_delta']['mapping_mediated']['r2'])}{length_seed}); refusal and repetition are near or below zero. Thus the map difference carries a reproducible semantic-divergence signal in this run, but this pilot does not support strong claims about refusal or repetition differences.

## Exact design

- Frozen split: 8,000 train / 500 validation / 1,500 test prompts. The new analyses never fit on test rows.
- Fixed common basis: context Procrustes on Qwen-writer train contexts; answer Procrustes pooled over both answer writers. All affine translations are retained. Held-out context/answer alignment cosines are {alignment_test['context_q_to_l_flat_cosine']:.3f}/{alignment_test['answer_qwriter_q_to_l_flat_cosine']:.3f}/{alignment_test['answer_lwriter_q_to_l_flat_cosine']:.3f}, so encoder and diagonal contrasts include residual alignment error.
- Factorial contrasts: writer, encoder, encoder×writer interaction, and the natural diagonal Qwen-own − Llama-own difference.
- Null: 1,000 held-out row-pairing permutations that destroy prompt correspondence.
- Stability: two disjoint 4,000-row train refits at the original selected ridge lambdas; independent generation seed 137.
- Few-query transfer: both Qwen→Llama and Llama→Qwen, both answer writers, k ∈ {{2,4,8,16,32,64,128,256}}, 10 random anchor sets per cell, fixed ridge fraction 0.01, no validation tuning.
- Primary transfer score is centered cosine with the full target-map prediction. The scale-sensitive normalized R² is retained in the JSON as a secondary diagnostic, not the headline comparison.

## Interpretation and limits

Variance-ranked, skewness-oriented PCA summaries do not recover the correspondence; this does **not** rule out stronger unsupervised methods such as distribution matching, iterative alignment, or relative-representation approaches. Paired calibration produces a genuine directional advantage beginning around 32 queries and a robust advantage by 64. The procedure uses paired activations from both models and is calibration, not zero-shot transfer. The encoder/diagonal factorial terms are alignment-confounded, whereas writer/interaction terms cancel a shared alignment residual. Finally, this is an exploratory post-hoc LMSYS-only pilot; other model families, tasks, layers, and genuinely new prompts remain necessary tests.

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
