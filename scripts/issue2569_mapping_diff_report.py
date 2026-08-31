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
PURPLE = "#CC79A7"
GRAY = "#666666"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_report_inputs(
    mapping: dict[str, Any],
    transfer: dict[str, Any],
    scaling: dict[str, Any] | None,
) -> None:
    expected = mapping["test_roster_sha256"]
    if transfer["test_roster_sha256"] != expected:
        raise ValueError("few-shot transfer test roster differs from mapping roster")
    if scaling is not None and scaling["test_roster_sha256"] != expected:
        raise ValueError("scaling/unpaired test roster differs from mapping roster")


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
            transfer["few_query"][direction][writer][str(k)][method][group][metric]["median"]
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
            axes[0].plot(
                x, curve, color=BLUE if method.startswith("transported") else ORANGE, alpha=0.18
            )
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
                transfer["few_query"][direction][writer][str(k)]["paired_transport_advantage"][
                    "full_target_mapping"
                ]["centered_cosine"]["values"]
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


def _scaling_cell_value(
    scaling: dict[str, Any],
    arm: str,
    direction: str,
    writer: str,
    k: int,
    method: str,
    group: str = "full_target_mapping",
    metric: str = "centered_cosine",
) -> float:
    return float(scaling[arm][direction][writer][str(k)][method][group][metric]["median"])


def _pooled_bridge_values(
    scaling: dict[str, Any],
    k: int,
    metric: str,
    *,
    arm: str = "unpaired_alignment",
) -> np.ndarray:
    values: list[float] = []
    for direction, writer, _ in CELLS:
        # A context bridge is fitted once per direction/repeat and copied under
        # both writers for schema symmetry. Do not double-count those copies.
        if metric.startswith("context_") and writer != "qwriter":
            continue
        values.extend(
            scaling[arm][direction][writer][str(k)]["bridge_diagnostics"][metric]["values"]
        )
    return np.asarray(values, np.float64)


def _aggregate_bridge_cell_center(
    scaling: dict[str, Any],
    k: int,
    metric: str,
    *,
    arm: str = "unpaired_alignment",
) -> float:
    """Median of unique direction(/writer) cell medians for a bridge metric."""
    values: list[float] = []
    for direction, writer, _ in CELLS:
        if metric.startswith("context_") and writer != "qwriter":
            continue
        values.append(
            float(scaling[arm][direction][writer][str(k)]["bridge_diagnostics"][metric]["median"])
        )
    return float(np.median(values))


def plot_query_scaling_unpaired(
    transfer: dict[str, Any],
    scaling: dict[str, Any],
    out_dir: Path,
) -> None:
    old_ks = [int(k) for k in transfer["design"]["hyperparameters"]["k_values"]]
    paired_ks = [int(k) for k in scaling["design"]["hyperparameters"]["paired_k_values"]]
    all_paired_ks = sorted(set(old_ks + paired_ks))
    unpaired_ks = [
        int(k) for k in scaling["design"]["hyperparameters"]["unpaired_k_values_per_model"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 2.85), constrained_layout=True)

    for method, label, color, marker in (
        ("transported_source_mapping", "Transport source map", BLUE, "o"),
        ("target_fit_from_scratch", "Fit target from scratch", ORANGE, "s"),
    ):
        cell_curves = []
        for direction, writer, _ in CELLS:
            curve = []
            for k in all_paired_ks:
                source = scaling if k in paired_ks else transfer
                arm = "paired_scaling" if source is scaling else "few_query"
                curve.append(
                    _scaling_cell_value(
                        source,
                        arm,
                        direction,
                        writer,
                        k,
                        method,
                    )
                )
            cell_curves.append(curve)
            axes[0].plot(all_paired_ks, curve, color=color, alpha=0.15)
        matrix = np.asarray(cell_curves, np.float64)
        axes[0].plot(
            all_paired_ks,
            np.median(matrix, axis=0),
            color=color,
            marker=marker,
            label=label,
        )
        axes[0].fill_between(
            all_paired_ks,
            matrix.min(0),
            matrix.max(0),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Paired-query scaling")
    axes[0].set_xlabel("Shared prompts observed in both models")
    axes[0].set_ylabel("Cosine with full target map")
    axes[0].set_xticks(
        all_paired_ks,
        [str(k) for k in all_paired_ks],
        rotation=45,
        ha="right",
    )
    axes[0].set_ylim(-0.05, 1.02)
    axes[0].legend(frameon=False, loc="lower right")
    finish_axes(axes[0])

    for arm, method, label, color, marker in (
        ("unpaired_alignment", "transported_source_mapping", "Unpaired transport", PURPLE, "D"),
        (
            "paired_rank_oracle",
            "transported_source_mapping",
            "Paired rank-r oracle",
            GREEN,
            "^",
        ),
        ("unpaired_alignment", "target_fit_from_scratch", "Target-only scratch", ORANGE, "s"),
    ):
        curves = np.asarray(
            [
                [
                    _scaling_cell_value(
                        transfer if arm == "paired_scaling" and k not in paired_ks else scaling,
                        "few_query" if arm == "paired_scaling" and k not in paired_ks else arm,
                        direction,
                        writer,
                        k,
                        method,
                    )
                    for k in unpaired_ks
                ]
                for direction, writer, _ in CELLS
            ],
            np.float64,
        )
        axes[1].plot(
            unpaired_ks,
            np.median(curves, axis=0),
            color=color,
            marker=marker,
            label=label,
        )
        axes[1].fill_between(
            unpaired_ks,
            curves.min(0),
            curves.max(0),
            color=color,
            alpha=0.08,
            linewidth=0,
        )
    axes[1].axhline(0, color="#777777", linewidth=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Can different prompts align the models?")
    axes[1].set_xlabel("Queries per model (paired oracle vs disjoint IDs)")
    axes[1].set_ylabel("Cosine with full target map")
    axes[1].set_xticks(
        unpaired_ks,
        [str(k) for k in unpaired_ks],
        rotation=45,
        ha="right",
    )
    axes[1].set_ylim(-0.15, 1.02)
    axes[1].legend(frameon=False, loc="best")
    finish_axes(axes[1])

    heldout_specs = (
        ("context_paired_test_cosine", "Held-out context bridge", BLUE, "o", "-"),
        ("answer_paired_test_cosine", "Held-out answer bridge", GREEN, "s", "-"),
    )
    for metric, label, color, marker, linestyle in heldout_specs:
        pooled = [_pooled_bridge_values(scaling, k, metric) for k in unpaired_ks]
        med = [_aggregate_bridge_cell_center(scaling, k, metric) for k in unpaired_ks]
        lo = [float(np.quantile(values, 0.1)) for values in pooled]
        hi = [float(np.quantile(values, 0.9)) for values in pooled]
        axes[2].plot(
            unpaired_ks,
            med,
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
        axes[2].fill_between(unpaired_ks, lo, hi, color=color, alpha=0.06, linewidth=0)
    axes[2].axhline(0, color="#777777", linewidth=0.8)
    axes[2].set_xscale("log", base=2)
    axes[2].set_title("Training fit vs held-out correspondence")
    axes[2].set_xlabel("Unpaired queries per model")
    axes[2].set_ylabel("Held-out centered cosine")
    axes[2].set_xticks(
        unpaired_ks,
        [str(k) for k in unpaired_ks],
        rotation=45,
        ha="right",
    )
    axes[2].set_ylim(-0.08, 0.28)
    finish_axes(axes[2])

    objective_axis = axes[2].twinx()
    objective_specs = (
        (
            "context_unsupervised_objective",
            "context_unrefined_random_orientation_reference",
            "Context train objective",
            "Context unrefined rotation ref.",
            GRAY,
            "^",
        ),
        (
            "answer_unsupervised_objective",
            "answer_unrefined_random_orientation_reference",
            "Answer train objective",
            "Answer unrefined rotation ref.",
            PURPLE,
            "v",
        ),
    )
    for (
        objective_metric,
        reference_metric,
        objective_label,
        reference_label,
        color,
        marker,
    ) in objective_specs:
        objective_center = [
            _aggregate_bridge_cell_center(scaling, k, objective_metric) for k in unpaired_ks
        ]
        objective_axis.plot(
            unpaired_ks,
            objective_center,
            color=color,
            marker=marker,
            linestyle="--",
            label=objective_label,
        )
        reference_values = [
            _pooled_bridge_values(scaling, k, reference_metric) for k in unpaired_ks
        ]
        reference_center = [
            _aggregate_bridge_cell_center(scaling, k, reference_metric) for k in unpaired_ks
        ]
        reference_lo = [float(np.quantile(values, 0.1)) for values in reference_values]
        reference_hi = [float(np.quantile(values, 0.9)) for values in reference_values]
        objective_axis.plot(
            unpaired_ks,
            reference_center,
            color=color,
            linewidth=1.2,
            linestyle=":",
            label=reference_label,
        )
        objective_axis.fill_between(
            unpaired_ks,
            reference_lo,
            reference_hi,
            color=color,
            alpha=0.08,
            linewidth=0,
        )
    objective_axis.set_xscale("log", base=2)
    axes[2].set_xticks(
        unpaired_ks,
        [str(k) for k in unpaired_ks],
        rotation=45,
        ha="right",
    )
    objective_axis.set_ylim(0.0, 1.02)
    objective_axis.set_ylabel("Training symmetric-Chamfer cosine")
    objective_axis.spines["right"].set_visible(True)
    primary_handles, primary_labels = axes[2].get_legend_handles_labels()
    objective_handles, objective_labels = objective_axis.get_legend_handles_labels()
    objective_axis.legend(
        primary_handles + objective_handles,
        primary_labels + objective_labels,
        frameon=False,
        fontsize=5.5,
        loc="center right",
    )
    for axis, label in zip(axes, ("A", "B", "C"), strict=True):
        panel_label(axis, label)
    save_figure(fig, out_dir / "fig4_query_scaling_unpaired")


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
            scratch = row["target_fit_from_scratch"]["full_target_mapping"]["centered_cosine"][
                "median"
            ]
            values.append(f"{transported:.3f} / {scratch:.3f}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def aggregate_transfer_cosine(transfer: dict[str, Any], k: int, method: str) -> float:
    values = [
        transfer["few_query"][direction][writer][str(k)][method]["full_target_mapping"][
            "centered_cosine"
        ]["median"]
        for direction, writer, _ in CELLS
    ]
    return float(np.median(values))


def aggregate_observed(transfer: dict[str, Any], k: int, method: str, metric: str) -> float:
    values = [
        transfer["few_query"][direction][writer][str(k)][method]["observed_target"][metric][
            "median"
        ]
        for direction, writer, _ in CELLS
    ]
    return float(np.median(values))


def aggregate_full_target_observed(transfer: dict[str, Any], metric: str) -> float:
    values = [
        transfer["full_target_map_ceiling"][direction][writer]["observed_target"][metric]
        for direction, writer, _ in CELLS
    ]
    return float(np.median(values))


def pooled_advantage(transfer: dict[str, Any], k: int) -> np.ndarray:
    values: list[float] = []
    for direction, writer, _ in CELLS:
        values.extend(
            transfer["few_query"][direction][writer][str(k)]["paired_transport_advantage"][
                "full_target_mapping"
            ]["centered_cosine"]["values"]
        )
    return np.asarray(values, np.float64)


def aggregate_scaling_cosine(
    scaling: dict[str, Any],
    arm: str,
    k: int,
    method: str,
    group: str = "full_target_mapping",
    metric: str = "centered_cosine",
) -> float:
    return float(
        np.median(
            [
                _scaling_cell_value(
                    scaling,
                    arm,
                    direction,
                    writer,
                    k,
                    method,
                    group,
                    metric,
                )
                for direction, writer, _ in CELLS
            ]
        )
    )


def aggregate_scaling_advantage(scaling: dict[str, Any], k: int) -> float:
    """Median of the four cell-level repeat medians used by plotted curves."""
    values = [
        scaling["paired_scaling"][direction][writer][str(k)]["paired_transport_advantage"][
            "full_target_mapping"
        ]["centered_cosine"]["median"]
        for direction, writer, _ in CELLS
    ]
    return float(np.median(values))


def pooled_scaling_advantage(scaling: dict[str, Any], k: int) -> np.ndarray:
    values: list[float] = []
    for direction, writer, _ in CELLS:
        values.extend(
            scaling["paired_scaling"][direction][writer][str(k)]["paired_transport_advantage"][
                "full_target_mapping"
            ]["centered_cosine"]["values"]
        )
    return np.asarray(values, np.float64)


def pooled_scaling_values(
    scaling: dict[str, Any],
    arm: str,
    k: int,
    method: str,
    group: str = "full_target_mapping",
    metric: str = "centered_cosine",
) -> np.ndarray:
    values: list[float] = []
    for direction, writer, _ in CELLS:
        values.extend(scaling[arm][direction][writer][str(k)][method][group][metric]["values"])
    return np.asarray(values, np.float64)


def extended_paired_table(scaling: dict[str, Any], ks: list[int]) -> str:
    lines = [
        "| Shared prompts | Transport → full map | Scratch → full map | Paired Δ | Transport → answers |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in ks:
        transported = aggregate_scaling_cosine(
            scaling,
            "paired_scaling",
            k,
            "transported_source_mapping",
        )
        scratch = aggregate_scaling_cosine(
            scaling,
            "paired_scaling",
            k,
            "target_fit_from_scratch",
        )
        observed = aggregate_scaling_cosine(
            scaling,
            "paired_scaling",
            k,
            "transported_source_mapping",
            "observed_target",
        )
        lines.append(
            f"| {k:,} | {transported:.3f} | {scratch:.3f} | "
            f"{aggregate_scaling_advantage(scaling, k):.3f} | {observed:.3f} |"
        )
    return "\n".join(lines)


def unpaired_scaling_table(scaling: dict[str, Any], ks: list[int]) -> str:
    lines = [
        "| Queries/model | Paired rank-r oracle → full map | Unpaired transport → full map | Target scratch → full map | Held-out unpaired context bridge | Held-out unpaired answer bridge |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ks:
        oracle = aggregate_scaling_cosine(
            scaling,
            "paired_rank_oracle",
            k,
            "transported_source_mapping",
        )
        transported = aggregate_scaling_cosine(
            scaling,
            "unpaired_alignment",
            k,
            "transported_source_mapping",
        )
        scratch = aggregate_scaling_cosine(
            scaling,
            "unpaired_alignment",
            k,
            "target_fit_from_scratch",
        )
        context = _aggregate_bridge_cell_center(scaling, k, "context_paired_test_cosine")
        answer = _aggregate_bridge_cell_center(scaling, k, "answer_paired_test_cosine")
        lines.append(
            f"| {k:,} | {oracle:.3f} | {transported:.3f} | {scratch:.3f} | "
            f"{context:.3f} | {answer:.3f} |"
        )
    return "\n".join(lines)


def query_scaling_report_block(scaling: dict[str, Any]) -> str:
    hyper = scaling["design"]["hyperparameters"]
    paired_ks = [int(k) for k in hyper["paired_k_values"]]
    unpaired_ks = [int(k) for k in hyper["unpaired_k_values_per_model"]]
    paired_max = max(paired_ks)
    unpaired_max = max(unpaired_ks)
    unpaired_min = min(unpaired_ks)
    effective_min_rank = min(int(hyper["unpaired_rank"]), unpaired_min - 1)
    paired_transport = aggregate_scaling_cosine(
        scaling,
        "paired_scaling",
        paired_max,
        "transported_source_mapping",
    )
    paired_scratch = aggregate_scaling_cosine(
        scaling,
        "paired_scaling",
        paired_max,
        "target_fit_from_scratch",
    )
    paired_actual = aggregate_scaling_cosine(
        scaling,
        "paired_scaling",
        paired_max,
        "transported_source_mapping",
        "observed_target",
    )
    paired_advantage_values = pooled_scaling_advantage(scaling, paired_max)
    paired_positive = int(np.sum(paired_advantage_values > 0))
    full_actual = float(
        np.median(
            [
                scaling["full_target_map_ceiling"][direction][writer]["observed_target"][
                    "centered_cosine"
                ]
                for direction, writer, _ in CELLS
            ]
        )
    )
    unpaired_values = pooled_scaling_values(
        scaling,
        "unpaired_alignment",
        unpaired_max,
        "transported_source_mapping",
    )
    unpaired_transport = aggregate_scaling_cosine(
        scaling,
        "unpaired_alignment",
        unpaired_max,
        "transported_source_mapping",
    )
    unpaired_q10, unpaired_q90 = np.quantile(unpaired_values, [0.1, 0.9])
    unpaired_scratch = aggregate_scaling_cosine(
        scaling,
        "unpaired_alignment",
        unpaired_max,
        "target_fit_from_scratch",
    )
    paired_oracle = aggregate_scaling_cosine(
        scaling,
        "paired_rank_oracle",
        unpaired_max,
        "transported_source_mapping",
    )
    paired_oracle_min = aggregate_scaling_cosine(
        scaling,
        "paired_rank_oracle",
        unpaired_min,
        "transported_source_mapping",
    )
    paired_oracle_previous = aggregate_scaling_cosine(
        scaling,
        "paired_rank_oracle",
        sorted(unpaired_ks)[-2],
        "transported_source_mapping",
    )
    context_bridge = _aggregate_bridge_cell_center(
        scaling, unpaired_max, "context_paired_test_cosine"
    )
    answer_bridge = _aggregate_bridge_cell_center(
        scaling, unpaired_max, "answer_paired_test_cosine"
    )
    context_objective = _aggregate_bridge_cell_center(
        scaling, unpaired_max, "context_unsupervised_objective"
    )
    answer_objective = _aggregate_bridge_cell_center(
        scaling, unpaired_max, "answer_unsupervised_objective"
    )
    paired_oracle_context = _aggregate_bridge_cell_center(
        scaling,
        unpaired_max,
        "context_paired_test_cosine",
        arm="paired_rank_oracle",
    )
    paired_oracle_answer = _aggregate_bridge_cell_center(
        scaling,
        unpaired_max,
        "answer_paired_test_cosine",
        arm="paired_rank_oracle",
    )
    min_context_objective = _aggregate_bridge_cell_center(
        scaling, unpaired_min, "context_unsupervised_objective"
    )
    min_answer_objective = _aggregate_bridge_cell_center(
        scaling, unpaired_min, "answer_unsupervised_objective"
    )
    context_reference_values = _pooled_bridge_values(
        scaling, unpaired_max, "context_unrefined_random_orientation_reference"
    )
    answer_reference_values = _pooled_bridge_values(
        scaling, unpaired_max, "answer_unrefined_random_orientation_reference"
    )
    context_reference = _aggregate_bridge_cell_center(
        scaling, unpaired_max, "context_unrefined_random_orientation_reference"
    )
    answer_reference = _aggregate_bridge_cell_center(
        scaling, unpaired_max, "answer_unrefined_random_orientation_reference"
    )
    context_reference_q10, context_reference_q90 = np.quantile(context_reference_values, [0.1, 0.9])
    answer_reference_q10, answer_reference_q90 = np.quantile(answer_reference_values, [0.1, 0.9])
    context_initial = _aggregate_bridge_cell_center(
        scaling, unpaired_max, "context_initial_objective"
    )
    answer_initial = _aggregate_bridge_cell_center(
        scaling, unpaired_max, "answer_initial_objective"
    )
    context_moment_fraction = float(
        np.mean(_pooled_bridge_values(scaling, unpaired_max, "context_initializer_moment"))
    )
    answer_moment_fraction = float(
        np.mean(_pooled_bridge_values(scaling, unpaired_max, "answer_initializer_moment"))
    )
    paired_table = extended_paired_table(scaling, paired_ks)
    unpaired_table = unpaired_scaling_table(scaling, unpaired_ks)
    simplex_note = (
        f"At k={unpaired_min:,}, the effective rank is k−1={effective_min_rank}; centering and "
        "whitening make each finite k-point cloud orthogonally congruent, so the context/answer "
        f"training objectives reach {min_context_objective:.3f}/{min_answer_objective:.3f} by "
        "construction even though held-out correspondence remains poor."
        if effective_min_rank == unpaired_min - 1
        else f"At the smallest k={unpaired_min:,}, the effective rank is {effective_min_rank} and "
        f"the context/answer training objectives are {min_context_objective:.3f}/{min_answer_objective:.3f}."
    )
    return f"""## 4. Extended scaling and genuinely unpaired alignment

![Extended paired scaling and unpaired alignment](fig4_query_scaling_unpaired.png)

**Figure 4.** Panel A continues the paired calibration curve to {paired_max:,} shared prompts. At that endpoint, transported-map cosine is {paired_transport:.3f}, versus {paired_scratch:.3f} for fitting the target from the same rows; {paired_positive}/{len(paired_advantage_values)} dependent direction/writer/anchor comparisons favor transport. Against actual target answers, the {paired_max:,}-query transport reaches cosine {paired_actual:.3f}, compared with {full_actual:.3f} for the original 8,000-row target maps. The extended points begin at k={min(paired_ks):,}, use fresh anchor draws rather than the original few-query draws, and use a symmetric Cholesky implementation of the same affine kernel-ridge bridge; a same-input regression test verifies its kernel weights against the legacy solver at absolute/relative tolerance 2×10⁻⁵. The original curve through its prior endpoint is retained rather than averaged with a duplicate rerun.

{paired_table}

The three center columns use the median of the four cell medians. “Paired Δ” instead summarizes the within-draw transport-minus-scratch differences inside each cell before taking the cross-cell median, so it need not equal the subtraction of the two marginal center columns.

Panel B adds a capacity-matched supervised control. The paired rank-r oracle fits separate PCA summaries and orthogonal context/answer bridges exactly like the unpaired arm, but direct Procrustes receives k paired row identities. The unpaired condition instead uses k source prompts and k *different* target prompts with zero prompt-ID overlap. Both read k fixed context/answer activation rows from each encoder (2k model-side rows), but paired uses k distinct prompt/response IDs while unpaired uses 2k; this is not an answer-generation-cost equality claim. No new model forwards or generations were issued in this run. The oracle reuses each repeat's source IDs in both models, whereas unpaired transport and target-only scratch use the separately drawn target IDs; Panel B is therefore a descriptive between-condition comparison, not a within-draw paired difference. Every transport condition also applies a frozen source map pretrained on all {int(scaling["n"]["train"]):,} source-train rows, treated as an amortized artifact outside the per-k calibration budget. At k={unpaired_max:,}, the paired rank-r oracle reaches {paired_oracle:.3f} full-map cosine, with held-out context/answer bridge cell-centers {paired_oracle_context:.3f}/{paired_oracle_answer:.3f}. The unpaired method remains near zero: cell-median cosine {unpaired_transport:.3f}, pooled 10th–90th percentile [{unpaired_q10:.3f}, {unpaired_q90:.3f}], with {100 * np.mean(unpaired_values > 0):.0f}% of {len(unpaired_values)} dependent cell/repeat values positive. The target-only scratch control reaches {unpaired_scratch:.3f}. This same-family oracle isolates the value of paired identities; the negative conclusion remains specific to the tested unpaired optimizer.

{unpaired_table}

The paired rank-r oracle rises from {paired_oracle_min:.3f} at k={unpaired_min:,} to {paired_oracle:.3f} at k={unpaired_max:,}, gaining only {paired_oracle - paired_oracle_previous:.3f} over the preceding grid point and remaining below the full-dimensional paired endpoint {paired_transport:.3f}. It is therefore a same-family identifiability control, not an unconstrained performance ceiling.

The unpaired aligner fits separate rank-r PCA coordinates with r=min({hyper["unpaired_rank"]}, k−1). It tries two initial rotations—variance-rank identity and marginal skew/kurtosis/quantile assignment—refines each independently by mutual-nearest-neighbour Procrustes, then selects the higher symmetric-Chamfer training objective. It never receives cross-model row identities. At k={unpaired_max:,}, the moment initializer is selected for {100 * context_moment_fraction:.0f}% of unique context fits and {100 * answer_moment_fraction:.0f}% of answer fits; the selected initial objectives are {context_initial:.3f}/{answer_initial:.3f}, and final objectives are {context_objective:.3f}/{answer_objective:.3f}. For scale context only, each fit also records {hyper["unrefined_random_orientation_reference_draws_per_fit"]} deterministic unrefined random rotations: their context/answer cell-centers are {context_reference:.3f}/{answer_reference:.3f}, with pooled 10th–90th bands [{context_reference_q10:.3f}, {context_reference_q90:.3f}]/[{answer_reference_q10:.3f}, {answer_reference_q90:.3f}]. These references do **not** run initializer selection or mutual-nearest-neighbour refinement, so they are not an estimator-matched null; fitted-minus-reference differences are descriptive and cannot establish shared geometry or above-chance recovery. The fitted estimator has no matched chance baseline in this study. Frozen paired-test bridge cell-centers are only {context_bridge:.3f}/{answer_bridge:.3f}. {simplex_note} Panel C deliberately places held-out centered cosine on the left axis and training symmetric-Chamfer cosine plus the unrefined references on the right: these differently defined cosines are not commensurate. The supported result is the held-out failure of this two-initializer unpaired procedure alongside the strong paired-row oracle, not a calibrated claim about its training objective.

Map-prediction center estimates in panels A–B, the tables, and headline prose use the median of four direction/writer cell medians. Context-bridge centers use the median of two unique direction medians; answer-bridge centers use the median of four direction/writer medians. Explicitly labeled pooled bands and positive fractions use repeat-level values; context repeats are never duplicated across writers. The unpaired conclusion applies to this two-initializer, best-training-objective self-learning algorithm, not to every possible unsupervised alignment method.
"""


def fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_report(
    mapping: dict[str, Any],
    transfer: dict[str, Any],
    out_dir: Path,
    scaling: dict[str, Any] | None = None,
) -> None:
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
        transfer["pca_reconstruction_ceiling"][direction][writer]["full_target_mapping"][
            "centered_cosine"
        ]
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
    scaling_block = query_scaling_report_block(scaling).rstrip() if scaling else ""
    behavior_section = "3"
    if scaling:
        paired_max = max(int(k) for k in scaling["design"]["hyperparameters"]["paired_k_values"])
        unpaired_max = max(
            int(k) for k in scaling["design"]["hyperparameters"]["unpaired_k_values_per_model"]
        )
        paired_endpoint = aggregate_scaling_cosine(
            scaling,
            "paired_scaling",
            paired_max,
            "transported_source_mapping",
        )
        unpaired_endpoint = aggregate_scaling_cosine(
            scaling,
            "unpaired_alignment",
            unpaired_max,
            "transported_source_mapping",
        )
        oracle_endpoint = aggregate_scaling_cosine(
            scaling,
            "paired_rank_oracle",
            unpaired_max,
            "transported_source_mapping",
        )
        scaling_bottomline = (
            f" Paired transport keeps improving but begins to plateau: full-target-map cosine "
            f"reaches {paired_endpoint:.3f} at {paired_max:,} shared prompts. In contrast, the "
            f"tested genuinely unpaired aligner reaches only {unpaired_endpoint:.3f} at "
            f"{unpaired_max:,} disjoint prompts per model, while the same rank/PCA/orthogonal "
            f"family reaches {oracle_endpoint:.3f} when given paired row identities."
        )
        partition_note = (
            f"At k={unpaired_max:,}, each repeat repartitions all "
            f"{int(scaling['n']['train']):,} train rows. "
            if 2 * unpaired_max == int(scaling["n"]["train"])
            else ""
        )
    else:
        scaling_bottomline = ""
        partition_note = ""
    scaling_design = (
        f"- Extended paired scaling and unpaired alignment: paired k ∈ "
        f"{{{','.join(str(k) for k in scaling['design']['hyperparameters']['paired_k_values'])}}}; "
        f"unpaired k per model ∈ "
        f"{{{','.join(str(k) for k in scaling['design']['hyperparameters']['unpaired_k_values_per_model'])}}}. "
        f"Extended full-dimensional paired points use {scaling['design']['hyperparameters']['paired_repeats']} "
        f"fresh draws and a symmetric Cholesky solve; a duplicate k={max(int(k) for k in transfer['design']['hyperparameters']['k_values']):,} "
        "endpoint rerun is omitted. "
        f"Unpaired and paired-oracle points use {scaling['design']['hyperparameters']['unpaired_repeats']} "
        "independently drawn assignments that are source/target-disjoint within each repeat but may reuse "
        f"rows across repeats, effective rank min({scaling['design']['hyperparameters']['unpaired_rank']}, k−1), and "
        f"{scaling['design']['hyperparameters']['unpaired_self_learning_iterations']} self-learning iterations. "
        "The unpaired fit refines both variance-rank identity and marginal-moment initializers and selects "
        "the higher training objective; for descriptive scale context, each fit is shown beside "
        f"{scaling['design']['hyperparameters']['unrefined_random_orientation_reference_draws_per_fit']} "
        "unrefined random rotations that are explicitly not an estimator-matched null. "
        f"{partition_note}Every within-repeat unpaired source/target prompt-ID "
        "intersection is asserted empty; the paired test set is evaluation-only. Both alignment controls read "
        "k fixed activation rows per encoder, but paired uses k distinct writer-response inputs and unpaired uses "
        f"2k; the analysis issued no new forwards or generations. All transport arms use a frozen source map "
        f"pretrained on all {int(scaling['n']['train']):,} source-train rows as an amortized artifact outside "
        "the per-k calibration budget."
        if scaling
        else ""
    )
    scaling_repro = (
        "- Extended output: [`query_scaling_unpaired.json`](query_scaling_unpaired.json)\n"
        "- Extended analysis driver: `scripts/issue2569_query_scaling_unpaired.py` "
        f"(SHA-256 `{scaling['analysis_driver']['sha256']}`; Git commit reported in the handoff)\n"
        f"- Extended runtime: seed {scaling['design']['hyperparameters']['seed']}; device "
        f"`{scaling['design']['hyperparameters']['device']}`; nearest-neighbour chunk size "
        f"{scaling['design']['hyperparameters']['nn_chunk_size']}"
        if scaling
        else ""
    )
    scaling_limits = (
        " The extended test strengthens the negative result for simple unpaired alignment: "
        "the best-training-objective result across variance-rank identity and marginal-moment seeds, "
        "each refined by mutual-nearest-neighbour Procrustes, remains near zero even with "
        f"{max(int(k) for k in scaling['design']['hyperparameters']['unpaired_k_values_per_model']):,} "
        "disjoint prompts per model, while a capacity-matched paired-row oracle succeeds. This is an "
        "algorithm-specific failure, not a proof "
        "that unpaired alignment is impossible; nonlinear, optimal-transport, or task-supervised "
        "methods remain untested."
        if scaling
        else ""
    )
    report = f"""# Mapping differences and cross-model transfer across Qwen and Llama

**Issue #2569 follow-up · Qwen2.5-7B-Instruct Q14 × Llama-3.1-8B-Instruct L16 · 10,000 LMSYS prompts**

## Bottom line

The fixed Procrustes alignment is incomplete (held-out context/answer cosines {alignment_test["context_q_to_l_flat_cosine"]:.3f}, {alignment_test["answer_qwriter_q_to_l_flat_cosine"]:.3f}, and {alignment_test["answer_lwriter_q_to_l_flat_cosine"]:.3f}). Consequently, encoder, diagonal, and encoder×writer interaction contrasts can all contain coordinate-alignment residual. Writer and interaction both vanish under the stronger null of no writer effect in either encoder; only writer is exactly zero under its own null, because interaction's own null—equal writer effects across encoders—is not protected by imperfect alignment. Alignment-free within-model writer contrasts remain prompt-specific in Qwen (cosine {within_writer["qwen_native_qwriter_minus_lwriter"]["flat_cosine"]:.3f}, R² {within_writer["qwen_native_qwriter_minus_lwriter"]["pooled_r2"]:.3f}) and Llama (cosine {within_writer["llama_native_qwriter_minus_lwriter"]["flat_cosine"]:.3f}, R² {within_writer["llama_native_qwriter_minus_lwriter"]["pooled_r2"]:.3f}).

Mapping transfer is not useful at the very smallest budgets: at k=2, 4, and 8, every paired anchor draw favors fitting the target directly by centered cosine. A small directional advantage first emerges around 32 queries (median paired Δ cosine {np.median(advantage32):.3f}; {100 * np.mean(advantage32 > 0):.1f}% of 40 dependent cell-draw comparisons) and is consistent by 64 ({100 * np.mean(advantage64 > 0):.1f}% positive). At 256 queries, median centered cosine with the frozen full target map is {aggregate_transfer_cosine(transfer, 256, "transported_source_mapping"):.3f}, versus {aggregate_transfer_cosine(transfer, 256, "target_fit_from_scratch"):.3f} from scratch.{scaling_bottomline}

## 1. Factorial mapping diff

![Held-out factorial mapping contrasts](fig1_factorial_mapping_diff.png)

**Figure 1.** Four native maps—encoder {{Qwen, Llama}} × answer writer {{Qwen, Llama}}—were transformed into one fixed Qwen basis using train-only semi-orthogonal Procrustes alignments. Open circles on the writer bar are the alignment-free native Qwen/Llama writer contrasts. Black marks show the 95% row-permutation null. Encoder, diagonal, and interaction terms can contain alignment residual; their R² values are {reads["encoder"]["pooled_r2"]:.3f}, {reads["diagonal"]["pooled_r2"]:.3f}, and {reads["interaction"]["pooled_r2"]:.3f}. The aligned writer term has cosine {reads["writer"]["flat_cosine"]:.3f} and R² {reads["writer"]["pooled_r2"]:.3f}. Every observed cosine exceeds all 1,000 row-pairing permutations (p={reads["writer"]["permutation_null"]["flat_cosine"]["p_ge"]:.4f}), which establishes prompt specificity but does not remove alignment confounding.

Numerically, the encoder-labeled representation contrast is largest and most split-half stable (RMS {geometry["encoder"]["heldout_prediction_rms_norm"]:.2f}; cosine {split["encoder"]["data_weighted_half1_vs_half2_cosine"]:.3f}). The writer contrast has RMS {geometry["writer"]["heldout_prediction_rms_norm"]:.2f} and split-half cosine {split["writer"]["data_weighted_half1_vs_half2_cosine"]:.3f}. Interaction is smaller and less stable ({geometry["interaction"]["heldout_prediction_rms_norm"]:.2f}, {split["interaction"]["data_weighted_half1_vs_half2_cosine"]:.3f}) and may be entirely alignment residual. The diagonal replicates numerically on seed 137 (R² {mapping["seed137_reliability"]["frozen_seed42_diagonal_map_on_seed137"]["pooled_r2"]:.3f}) but remains alignment-confounded. After alignment, writer and encoder operators are nearly orthogonal (cosine {geometry["writer_vs_encoder"]["operator_cosine"]:.3f}); because encoder is confounded, this is descriptive rather than a behavioral claim.

## 2. Can one mapping be transferred with only residual summaries or a few queries?

![Few-query mapping transfer](fig2_fewshot_transfer.png)

**Figure 2.** Panel A evaluates recovery of the frozen full-data target mapping on the untouched 1,500-row test set—not direct fit to observed answers. Thin lines are the four direction/writer cells; thick lines are their medians. The 0-query diamond uses separate mean/top-64 PCA/variance summaries with components paired only by variance rank and marginal skewness. The complete summary pipeline fails (centered cosine {min(summary_cos):.3f}–{max(summary_cos):.3f}). Answer-side rank-64 compression alone is not the cause: reconstructing target answers in the same rank-64 basis retains full-map cosine {min(pca_ceiling):.3f}–{max(pca_ceiling):.3f}; context-side compression remains a possible contributor. This original baseline did not test stronger unsupervised algorithms; Section 4 adds a best-of-two variance-rank/marginal-moment initialization method with mutual-nearest-neighbour refinement. Panel B shows paired within-anchor-set cosine advantage over the equal-query scratch control (median and pooled 10th–90th percentiles across 4 cells × 10 draws; these are dependent descriptive comparisons, not 40 independent trials).

For k paired train queries, regularized context and answer bridges use only sample means and centered Gram matrices. No validation rows tune the bridge. The frozen source map is applied between those two bridges. The control fits target context→answer directly from the identical k anchors.

Table entries are **transported source map / target fit from scratch**, measured by held-out centered cosine with the full target mapping:

{transfer_table(transfer)}

Below 16 queries, transport is systematically worse than scratch ({low_text}). At 16 the median difference is near zero. A small advantage appears at 32; by 64, {int(np.sum(advantage64 > 0))}/{len(advantage64)} dependent cell-draw comparisons favor transfer. At 256, transported predictions have median centered cosine {aggregate_observed(transfer, 256, "transported_source_mapping", "centered_cosine"):.3f} with actual target answers, versus {aggregate_observed(transfer, 256, "target_fit_from_scratch", "centered_cosine"):.3f} for scratch; the original 8,000-row target maps reach {aggregate_full_target_observed(transfer, "centered_cosine"):.3f}. Scale-sensitive R² is retained only as a secondary diagnostic because the two-stage transport and one-stage scratch fits shrink differently. The crossover is conditional on the fixed, untuned ridge fraction 0.01; tuning could move it.

## {behavior_section}. What behavioral differences are visible?

![Behavior readouts from the writer contrast](fig3_behavior_readout.png)

**Figure 3.** A ridge readout trained on the observed writer activation contrast can recover several answer differences, but only some survive when the writer contrast is predicted from context through the mapping diff. Seed-42 mapping-mediated R² values are semantic divergence {fmt_score(semantic_value)}, log length {fmt_score(length_value)}, refusal {fmt_score(refusal_value)}, and repetition {fmt_score(repetition_value)}.{seed_sentence} {behavior_conclusion}

{scaling_block}

## Exact design

- Frozen split: 8,000 train / 500 validation / 1,500 test prompts. The new analyses never fit on test rows.
- Fixed common basis: context Procrustes on Qwen-writer train contexts; answer Procrustes pooled over both answer writers. All affine translations are retained. Held-out context/answer alignment cosines are {alignment_test["context_q_to_l_flat_cosine"]:.3f}/{alignment_test["answer_qwriter_q_to_l_flat_cosine"]:.3f}/{alignment_test["answer_lwriter_q_to_l_flat_cosine"]:.3f}, so encoder, interaction, and diagonal contrasts can include residual alignment error.
- Factorial contrasts: writer, encoder, encoder×writer interaction, and the natural diagonal Qwen-own − Llama-own difference. Encoder, interaction, and diagonal can contain alignment residual; within-encoder Qwen/Llama writer contrasts provide alignment-free checks.
- Null: 1,000 held-out row-pairing permutations that destroy prompt correspondence.
- Stability: two disjoint 4,000-row train refits at the original selected ridge lambdas; independent generation seed 137.
- Few-query transfer: both Qwen→Llama and Llama→Qwen, both answer writers, k ∈ {{2,4,8,16,32,64,128,256}}, 10 random anchor sets per cell, fixed ridge fraction 0.01, no validation tuning.
- Primary transfer score is cosine after subtracting the target model's train-answer mean from both the candidate and frozen target-map predictions. It measures recovery of the frozen target mapping, not direct fit to observed answers. The scale-sensitive normalized R² is retained in JSON as a secondary diagnostic.
{scaling_design}

## Interpretation and limits

Variance-ranked, skewness-oriented PCA summaries do not recover the correspondence; this does **not** rule out stronger unsupervised methods or isolate context compression from orientation failure. Paired transport is worse than direct target fitting at k≤8, near zero in median advantage at 16, begins to help around 32, and is consistently better by 64 under the fixed ridge setting. The procedure uses paired activations from both models and is calibration, not zero-shot transfer.{scaling_limits} Encoder, interaction, and diagonal factorial terms are alignment-confounded. Writer and interaction vanish under no writer effect in either encoder, but only writer is exactly zero under its own null; nonzero writer magnitude can still be distorted. Finally, this is an exploratory post-hoc LMSYS-only pilot; other model families, tasks, layers, and genuinely new prompts remain necessary tests.

## Reproducibility

- Source experiment revision: `{mapping["source_revision"]}`
- Test roster SHA-256: `{mapping["test_roster_sha256"]}`
- Primary outputs: [`mapping_diff.json`](mapping_diff.json), [`fewshot_transfer.json`](fewshot_transfer.json), [`heldout_rows.jsonl`](heldout_rows.jsonl), [`writer_modes.npz`](writer_modes.npz)
- Analysis drivers: `scripts/issue2569_mapping_diff.py`, `scripts/issue2569_fewshot_transfer.py`, and `scripts/issue2569_mapping_diff_report.py`
{scaling_repro}
"""
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    out_dir: Path,
    mapping_path: Path,
    transfer_path: Path,
    scaling_path: Path | None = None,
) -> None:
    files = sorted(
        path for path in out_dir.iterdir() if path.is_file() and path.name != "MANIFEST.json"
    )
    source_inputs = {
        "mapping_diff_json_sha256": file_sha256(mapping_path),
        "fewshot_transfer_json_sha256": file_sha256(transfer_path),
    }
    if scaling_path:
        source_inputs["query_scaling_unpaired_json_sha256"] = file_sha256(scaling_path)
    manifest = {
        "issue": 2569,
        "label": "mapping-diff-and-fewshot-transfer-report",
        "source_inputs": source_inputs,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in files
        },
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--transfer-json", type=Path, required=True)
    parser.add_argument("--scaling-unpaired-json", type=Path)
    parser.add_argument("--heldout-rows", type=Path)
    parser.add_argument("--writer-modes", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    mapping = load_json(args.mapping_json)
    transfer = load_json(args.transfer_json)
    scaling = load_json(args.scaling_unpaired_json) if args.scaling_unpaired_json else None
    validate_report_inputs(mapping, transfer, scaling)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_factorial(mapping, args.out_dir)
    plot_fewshot(transfer, args.out_dir)
    plot_behavior(mapping, args.out_dir)
    if scaling:
        plot_query_scaling_unpaired(transfer, scaling, args.out_dir)
    write_report(mapping, transfer, args.out_dir, scaling)
    if args.heldout_rows:
        (args.out_dir / "heldout_rows.jsonl").write_bytes(args.heldout_rows.read_bytes())
    if args.writer_modes:
        (args.out_dir / "writer_modes.npz").write_bytes(args.writer_modes.read_bytes())
    (args.out_dir / "mapping_diff.json").write_bytes(args.mapping_json.read_bytes())
    (args.out_dir / "fewshot_transfer.json").write_bytes(args.transfer_json.read_bytes())
    if args.scaling_unpaired_json:
        (args.out_dir / "query_scaling_unpaired.json").write_bytes(
            args.scaling_unpaired_json.read_bytes()
        )
    write_manifest(
        args.out_dir,
        args.mapping_json,
        args.transfer_json,
        args.scaling_unpaired_json,
    )
    print(f"[mapping-diff-report] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
