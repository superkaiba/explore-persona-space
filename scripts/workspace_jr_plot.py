#!/usr/bin/env python3
"""Render the registered comparisons from an uploaded, complete 48-cell analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    MUTED,
    ROLES,
    better_label,
    c2a_figure,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding  # noqa: E402
from explore_persona_space.analysis.workspace_comparison import TARGETS, cell_key  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json  # noqa: E402

MODELS = {"primary": "Qwen3.5-27B", "comparison": "Qwen3.5-4B"}
LABELS = {
    "full": "Full answer",
    "J": "J component",
    "restJ": "J remainder",
    "R": "R component",
    "restR": "R remainder",
}
STYLES = {"ridge": ROLES["linear"], "mlp": ROLES["nonlinear"]}


class Sources:
    """Bind every plotted report to the completed uploaded comparison."""

    def __init__(self, root, receipt):
        self.root = root
        self.verified = _upload_binding(root, receipt)
        self.hashes = {}
        self.complete = self.read("comparison_complete.json")
        coverage = self.read("coverage.json")
        if (
            self.complete["status"] != "complete"
            or coverage["status"] != "complete_grid"
            or coverage["expected_cells"] != 48
            or coverage["realized_cells"] != 48
            or coverage["missing_cells"]
            or set(self.complete["scopes"]) != {*MODELS, "cross_model"}
        ):
            raise ValueError("Main figures require the complete two-model, 48-cell comparison")
        proof = self.read("source_proof.json")
        completion = self.read("completion_cohort.json")
        primary = self.read("primary_scoring_cohort.json")
        if (
            self.hashes["completion_cohort.json"] != proof["completion_cohort"]["report_sha256"]
            or completion["status"] != "complete"
            or not set(primary["common_context_ids"])
            <= set(completion["joint_complete_context_ids"])
        ):
            raise ValueError("Primary scoring cohort differs from verified completed rollouts")
        self.read("input_manifest.json")
        if self.hashes["input_manifest.json"] != self.complete["input_manifest_sha256"]:
            raise ValueError("Completed comparison input manifest changed")
        self.reports = {}
        for scope, record in self.complete["scopes"].items():
            name = f"{scope}/comparisons.json"
            self.reports[scope] = self.read(name)
            if self.hashes[name] != record["report_sha256"]:
                raise ValueError("Completed paired comparison changed")
            if len(self.reports[scope]["context_ids"]) != record["contexts"]:
                raise ValueError("Reported cohort count changed")
            if self.reports[scope]["context_ids"] != primary["common_context_ids"]:
                raise ValueError("Main panels must use the same joint complete-test cohort")

    def read(self, relative):
        path = self.root / relative
        self.hashes[relative] = self.verified(path)
        return json.loads(path.read_text())

    def native_summary(self, role, k):
        folder = f"{role}/{cell_key(role, 'observed', k, None).replace('/', '__')}"
        marker = self.read(f"{folder}/cell_complete.json")
        summary = self.read(f"{folder}/summary.json")
        if self.hashes[f"{folder}/summary.json"] != marker["files"]["summary.json"]:
            raise ValueError("Completed cell summary changed")
        if list(summary["context_ids"]) != self.reports[role]["context_ids"]:
            raise ValueError("Figure cell and comparison cohort differ")
        return summary["summary"]


def interval_point(ax, y, summary, style, *, offset=0):
    """Draw interval endpoints directly, including intervals excluding the point."""
    point, interval = summary["estimate"], summary["interval"]
    if point is not None and not np.isfinite(point):
        raise ValueError("Nonfinite plotted point")
    if summary["confidence"] != 0.95 or summary["n_draws"] != 2000:
        raise ValueError("Figure requires the registered 2,000-draw 95% interval")
    if interval is not None:
        if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] > interval[1]:
            raise ValueError("Invalid interval endpoints")
        ax.hlines(y + offset, *interval, color=style.color, linewidth=2)
        ax.vlines(interval, y + offset - 0.045, y + offset + 0.045, color=style.color)
    if point is None:
        ax.text(
            0.99,
            y + offset,
            "undefined",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=MUTED,
        )
    else:
        ax.plot(point, y + offset, marker=style.marker, color=style.color, linestyle="none")
        if interval is None:
            ax.text(
                0.99,
                y + offset,
                "CI undefined",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                color=MUTED,
            )


def axis_rows(ax, labels, xlabel, *, practical=False):
    style_axis(ax, grid_axis="x")
    ax.axvline(0, color=MUTED, linewidth=1)
    if practical:
        ax.axvline(-0.05, color=MUTED, linewidth=0.8, linestyle=":")
        ax.axvline(0.05, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.margins(x=0.1)


def predictor_legend(fig):
    legend_kicker(fig, 0.15, 0.045, "Predictor")
    handles = [
        Line2D(
            [],
            [],
            marker=style.marker,
            color=style.color,
            linestyle="none",
            label="Ridge" if name == "ridge" else "MLP (seed 42)",
        )
        for name, style in STYLES.items()
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.61, -0.01), ncol=2)


def export(fig, out, stem, caption, records, source):
    caption += " All primary panels condition on five nonempty final completed rollouts in both models; no excluded prompt is replaced."
    result = save_c2a_figure(
        fig,
        out / stem,
        title=stem.replace("_", " "),
        subject=caption,
        creator="scripts/workspace_jr_plot.py",
    )
    save_json(
        out / f"{stem}.meta.json",
        {
            "caption": caption,
            "plotted_values": records,
            "source_sha256": dict(source.hashes),
            "script_sha256": file_sha256(Path(__file__)),
            "render": result["record"],
            "output_sha256": {key: file_sha256(result[key]) for key in ("pdf", "png", "grayscale")},
        },
    )
    plt.close(fig)


def component_figures(source, out):
    for gain in (False, True):
        fig, _ = c2a_figure("full", aspect=0.43)
        axes = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.25, top=0.80, wspace=0.75)
        records = {}
        for index, (role, model) in enumerate(MODELS.items()):
            ax = axes[index]
            summaries = source.native_summary(role, 10)
            cell = cell_key(role, "observed", 10, None)
            for y, target in enumerate(TARGETS):
                for predictor, style in [("mlp", STYLES["mlp"])] if gain else STYLES.items():
                    key = f"{cell}/MLP_gain/{target}" if gain else f"{predictor}/{target}"
                    record = source.reports[role]["contrasts"][key] if gain else summaries[key]
                    records[f"{role}/{key}"] = record
                    interval_point(
                        ax,
                        y,
                        record,
                        style,
                        offset=0 if gain else (-0.12 if predictor == "ridge" else 0.12),
                    )
            n = source.complete["scopes"][role]["contexts"]
            panel_header(
                ax,
                chr(65 + index),
                f"{model} · n={n} · k=10",
                "MLP improvement" if gain else "Component predictability",
            )
            axis_rows(
                ax,
                [LABELS[x] for x in TARGETS],
                "MLP − ridge R²" if gain else better_label("Held-out R²"),
            )
        if not gain:
            predictor_legend(fig)
        caption = (
            "Native dictionaries, k=10. Points use the model-specific paired test cohort. "
            "Intervals are 95% from 2,000 paired context bootstraps, conditional on fitted predictors. "
            "Each R² uses its target's own variance; MLP is seed 42 with validation-selected recipe."
        )
        export(fig, out, "mlp_gains" if gain else "component_r2", caption, records, source)


def control_figure(source, out):
    fig, _ = c2a_figure("full", aspect=0.90)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.10, top=0.89, hspace=0.75, wspace=0.9)
    records = {}
    terms = [
        (arm, suffix)
        for suffix in ("", "_minus_mean_rotated", "_minus_affine_null")
        for arm in ("J", "R")
    ]
    labels = ["J gap", "R gap", "J − rotated", "R − rotated", "J − affine null", "R − affine null"]
    for row, (role, model) in enumerate(MODELS.items()):
        cell = cell_key(role, "observed", 10, None)
        for col, (predictor, style) in enumerate(STYLES.items()):
            ax = axes[row, col]
            for y, (arm, suffix) in enumerate(terms):
                key = f"{cell}/{predictor}/G_{arm}{suffix}"
                records[key] = source.reports[role]["contrasts"][key]
                interval_point(ax, y, records[key], style)
            panel_header(
                ax,
                chr(65 + row * 2 + col),
                f"{model} · n={source.complete['scopes'][role]['contexts']} · k=10",
                "Ridge" if predictor == "ridge" else "MLP (seed 42)",
            )
            axis_rows(ax, labels, "Difference in R²", practical=True)
    caption = (
        "Gap = remainder R² minus component R². Rotated contrasts subtract the mean of three "
        "registered Haar dictionaries; affine-null differences are descriptive diagnostics, not causal corrections. "
        "Bars are 95% paired context-bootstrap intervals. Dotted lines mark ±0.05 R². "
        "Dictionary-to-dictionary variation is reported separately in machine-readable comparisons."
    )
    export(fig, out, "gap_controls", caption, records, source)


def cross_model_figure(source, out):
    fig, _ = c2a_figure("full", aspect=0.46)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.35, right=0.96, bottom=0.25, top=0.82)
    labels = ["J gap", "R gap", "Lens disagreement", "J gap − rotated", "R gap − rotated"]
    suffixes = [
        "rotationNone/{p}/G_J_primary_minus_comparison",
        "rotationNone/{p}/G_R_primary_minus_comparison",
        "rotationNone/{p}/lens_disagreement_primary_minus_comparison",
        "{p}/G_J_control_adjusted_primary_minus_comparison",
        "{p}/G_R_control_adjusted_primary_minus_comparison",
    ]
    records = {}
    for y, suffix in enumerate(suffixes):
        for predictor, style in STYLES.items():
            key = "cross_model/observed/k10/" + suffix.format(p=predictor)
            records[key] = source.reports["cross_model"]["contrasts"][key]
            interval_point(
                ax, y, records[key], style, offset=-0.12 if predictor == "ridge" else 0.12
            )
    n = source.complete["scopes"]["cross_model"]["contexts"]
    panel_header(ax, "A", f"Shared contexts · n={n} · k=10", "Stronger minus weaker model")
    axis_rows(ax, labels, "Qwen3.5-27B − Qwen3.5-4B (R² difference)", practical=True)
    predictor_legend(fig)
    export(
        fig,
        out,
        "cross_model_gaps",
        "All points use the common test-context intersection across both models. "
        "Bars are paired 95% context-bootstrap intervals. Lens disagreement is G_R − G_J. "
        "Positive values indicate a larger gap in 27B. Model differences are observational.",
        records,
        source,
    )


def sparsity_figure(source, out):
    fig, _ = c2a_figure("full", aspect=0.8)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.1, top=0.9, hspace=0.8, wspace=0.5)
    records = {}
    labels = [f"{arm} · k={k}" for k in (5, 10, 25) for arm in ("J", "R")]
    for row, (role, model) in enumerate(MODELS.items()):
        for col, (predictor, style) in enumerate(STYLES.items()):
            ax = axes[row, col]
            for y, (k, arm) in enumerate((k, arm) for k in (5, 10, 25) for arm in ("J", "R")):
                key = f"{cell_key(role, 'observed', k, None)}/{predictor}/G_{arm}"
                records[key] = source.reports[role]["contrasts"][key]
                interval_point(ax, y, records[key], style)
            panel_header(
                ax,
                chr(65 + row * 2 + col),
                f"{model} · n={source.complete['scopes'][role]['contexts']}",
                "Ridge" if predictor == "ridge" else "MLP (seed 42)",
            )
            axis_rows(ax, labels, "Remainder R² − component R²", practical=True)
    export(
        fig,
        out,
        "sparsity_gaps",
        "Native J/R gaps at primary k=10 and predeclared k=5,25 sensitivity settings. "
        "Bars are 95% paired context-bootstrap intervals on each model's common cohort. "
        "Dotted lines show ±0.05 R².",
        records,
        source,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--upload-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = Sources(args.input_root, args.upload_receipt)
    args.out.mkdir(parents=True, exist_ok=False)
    set_c2a_style()
    component_figures(source, args.out)
    control_figure(source, args.out)
    cross_model_figure(source, args.out)
    sparsity_figure(source, args.out)
    save_json(
        args.out / "plot_complete.json",
        {
            "status": "complete",
            "scope": "five main comparison figures; diagnostics and decomposition agreement are separate",
            "input_receipt_sha256": file_sha256(args.upload_receipt),
            "source_sha256": source.hashes,
            "producer_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "producer_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], text=True
                ).strip()
            ),
        },
    )


if __name__ == "__main__":
    main()
