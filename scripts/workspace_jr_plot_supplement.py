#!/usr/bin/env python3
"""Render uploaded agreement, noise, readout and learning diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from workspace_jr_plot import LABELS, MODELS, STYLES, export as main_export  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    MUTED,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    set_c2a_style,
    style_axis,
)
from explore_persona_space.analysis.workspace_analysis_inputs import _upload_binding  # noqa: E402
from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json  # noqa: E402


def export(fig, out, stem, caption, records, source):
    main_export(fig, out, stem, caption, records, source, renderer=Path(__file__))


class Sources:
    def __init__(self, root, receipt):
        self.root, self.hashes = root, {}
        self.verify = _upload_binding(root, receipt)
        self.report, complete = self.read("supplement.json"), self.read("supplement_complete.json")
        if (
            complete["status"] != "complete"
            or complete["report_sha256"] != self.hashes["supplement.json"]
            or set(self.report["models"]) != set(MODELS)
        ):
            raise ValueError("Supplementary figures require the completed two-model analysis")
        self.read("input_manifest.json")
        if complete["input_manifest_sha256"] != self.hashes["input_manifest.json"]:
            raise ValueError("Supplementary input manifest changed")
        for data in self.report["models"].values():
            for ids in (
                data["agreement"]["context_ids"],
                data["noise"]["context_ids"],
                data["readouts"]["test_context_ids"],
                data["learning_curves"]["primary_context_ids"],
            ):
                if ids != self.report["context_ids"]:
                    raise ValueError("Supplementary panels use different primary test cohorts")

    def read(self, relative):
        path = self.root / relative
        self.hashes[relative] = self.verify(path)
        return json.loads(path.read_text())


def ecdf(ax, values, *, label, color, linestyle="-", marker=None, row=0):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("ECDF values must be one dimensional")
    finite = np.sort(array[np.isfinite(array)])
    if len(finite):
        ax.step(
            finite,
            np.arange(1, len(finite) + 1) / len(finite),
            where="post",
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=max(1, len(finite) // 8),
            markersize=4,
            label=label,
        )
    if len(finite) != len(array):
        ax.text(
            0.02,
            0.97 - row * 0.1,
            f"{label}: {len(array) - len(finite)} undefined",
            transform=ax.transAxes,
            va="top",
            color=MUTED,
        )
    return {"values": values, "finite": len(finite), "undefined": len(array) - len(finite)}


def agreement_figure(source, out):
    fig, _ = c2a_figure("full", aspect=0.70)
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.88, wspace=0.50, hspace=0.72)
    records = {}
    for row, (role, model) in enumerate(MODELS.items()):
        data = source.report["models"][role]
        agreement = data["agreement"]
        views = [
            ("Component cosine", "Cosine(J component, R component)", agreement["component_cosine"]),
            (
                "Component difference",
                "Squared J − R distance",
                agreement["squared_component_difference"],
            ),
            (
                "Token directions",
                "Cosine(J direction, R direction)",
                data["readouts"]["paired_token_cosine"],
            ),
        ]
        for col, (title, label, values) in enumerate(views):
            ax = axes[row, col]
            records[f"{role}/{title}"] = ecdf(ax, values, label=title, color=ROLES["control"].color)
            style_axis(ax, grid_axis="y")
            ax.set(xlabel=label, ylabel="Cumulative fraction", ylim=(0, 1.02))
            panel_header(ax, chr(65 + row * 3 + col), model, title)
    export(
        fig,
        out,
        "decomposition_agreement",
        "Descriptive agreement of the alternative J/R decompositions at k=10. Cosines exclude only components below the training-fixed norm floor; squared differences retain those contexts. Token directions are the same calibration-eligible vocabulary within each model. Raw squared distances have model-specific units; J and R are not disjoint components.",
        records,
        source,
    )


def noise_figure(source, out):
    fig, _ = c2a_figure("full", aspect=0.44)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.20, top=0.79, wspace=0.7)
    records = {}
    for index, (role, model) in enumerate(MODELS.items()):
        ax = axes[index]
        report = source.report["models"][role]["noise"]
        records[role] = report
        for y, name in enumerate(LABELS):
            value = report["components"][name]["noise_fraction"]
            if value is None:
                ax.text(
                    0.98,
                    y,
                    "undefined",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    color=MUTED,
                )
            else:
                ax.barh(y, 100 * value, color=ROLES["control"].color, height=0.48)
        ax.axvline(100 * report["trigger_threshold"], color=MUTED, linestyle=":")
        ax.set(
            yticks=range(len(LABELS)),
            yticklabels=list(LABELS.values()),
            ylim=(len(LABELS) - 0.5, -0.5),
            xlabel="Noise variance / target variance (%)",
        )
        style_axis(ax, grid_axis="x")
        panel_header(ax, chr(65 + index), model, f"K=5 · n={len(report['context_ids'])}")
    export(
        fig,
        out,
        "sampling_noise",
        "Estimated variance of each K=5 context mean, divided by that target's observed between-context variance. The dotted line is the predeclared 10% higher-K diagnostic trigger. Component/remainder covariance is retained in the source report. Sampling noise is not a measure of reasoning.",
        records,
        source,
    )


def readout_figures(source, out):
    fig, _ = c2a_figure("full", aspect=0.47)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.81, wspace=0.35)
    records = {}
    for index, (role, model) in enumerate(MODELS.items()):
        ax = axes[index]
        report = source.report["models"][role]["readouts"]
        for p, (predictor, style) in enumerate(STYLES.items()):
            for a, (arm, line) in enumerate((("J", "-"), ("R", "--"))):
                label = f"{arm} · {'Ridge' if predictor == 'ridge' else 'MLP'}"
                records[f"{role}/{predictor}/{arm}"] = ecdf(
                    ax,
                    report["metrics"][predictor][arm]["r2"],
                    label=label,
                    color=style.color,
                    linestyle=line,
                    marker=style.marker,
                    row=p * 2 + a,
                )
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set(
            xlabel=better_label("Aligned readout R²"), ylabel="Cumulative fraction", ylim=(0, 1.02)
        )
        style_axis(ax, grid_axis="y")
        panel_header(
            ax, chr(65 + index), model, f"{len(report['token_ids'])} paired token directions"
        )
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    export(
        fig,
        out,
        "readout_distributions",
        "Held-out J-aligned and R-aligned readouts of the same full-answer predictors. Vocabulary eligibility and random/PCA variance matches were fixed without test outcomes. These readouts do not decompose a predictor; shared dictionary structure means J/R agreement is not independent causal validation.",
        records,
        source,
    )
    fig, _ = c2a_figure("full", aspect=0.76)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.89, hspace=0.7, wspace=0.4)
    records = {}
    for row, (role, model) in enumerate(MODELS.items()):
        report = source.report["models"][role]["readouts"]
        for col, arm in enumerate(("J", "R")):
            ax = axes[row, col]
            for predictor, style in STYLES.items():
                metric = report["metrics"][predictor][arm]
                x, y = np.asarray(metric["variance"], float), np.asarray(metric["r2"], float)
                valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
                ax.scatter(
                    x[valid],
                    y[valid],
                    marker=style.marker,
                    color=style.color,
                    alpha=0.65,
                    s=15,
                    label="Ridge" if predictor == "ridge" else "MLP",
                )
                records[f"{role}/{predictor}/{arm}"] = {
                    "target_variance": metric["variance"],
                    "r2": metric["r2"],
                    "undefined": int((~valid).sum()),
                }
                if not valid.all():
                    ax.text(
                        0.02,
                        0.96 - (0 if predictor == "ridge" else 0.1),
                        f"{predictor}: {(~valid).sum()} undefined",
                        transform=ax.transAxes,
                        va="top",
                        color=MUTED,
                    )
            ax.set(
                xscale="log",
                xlabel="Held-out direction variance",
                ylabel=better_label("Aligned readout R²"),
            )
            style_axis(ax, grid_axis="y")
            panel_header(ax, chr(65 + row * 2 + col), model, f"{arm}-aligned directions")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    export(
        fig,
        out,
        "readout_variance",
        "Direction-specific R² versus each direction's own held-out target variance. Directions and control matching remain training/calibration-selected. Undefined scores and zero-variance directions retain explicit counts; no axis clips unfavorable scores.",
        records,
        source,
    )


def vertical_interval(ax, x, record, style):
    point, interval = record["estimate"], record["interval"]
    if record["n_draws"] != 2000 or record["confidence"] != 0.95:
        raise ValueError("Learning figure requires registered 95% 2,000-draw intervals")
    if interval is not None:
        if not np.isfinite(interval).all() or interval[0] > interval[1]:
            raise ValueError("Invalid learning-curve interval")
        ax.vlines(x, *interval, color=style.color, linewidth=1.5)
    if point is None:
        ax.text(x, 0.04, "undefined", transform=ax.get_xaxis_transform(), rotation=90, color=MUTED)
    else:
        ax.plot(x, point, marker=style.marker, color=style.color)
        if interval is None:
            ax.text(x, point, "  CI undefined", rotation=90, color=MUTED)


def learning_figures(source, out):
    for role, model in MODELS.items():
        fig, _ = c2a_figure("full", aspect=0.68)
        axes = fig.subplots(2, 3).ravel()
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.89, hspace=0.68, wspace=0.45)
        report = source.report["models"][role]["learning_curves"]
        records = {}
        for index, (target, label) in enumerate(LABELS.items()):
            ax = axes[index]
            for predictor, style in STYLES.items():
                points = []
                for cell in report["cells"]:
                    record = cell["paired_bootstrap"]["summary"][f"{predictor}/{target}"]
                    n = cell["n_train"]
                    records[f"{target}/{predictor}/n{n}"] = record
                    vertical_interval(ax, n, record, style)
                    points.append(record["estimate"] if record["estimate"] is not None else np.nan)
                ax.plot(
                    [cell["n_train"] for cell in report["cells"]],
                    points,
                    color=style.color,
                    linewidth=1,
                )
            ax.set(
                xlabel="Training contexts",
                ylabel=better_label("Held-out R²"),
                xticks=[cell["n_train"] for cell in report["cells"]],
            )
            style_axis(ax, grid_axis="y")
            panel_header(ax, chr(65 + index), model, label)
        axes[-1].axis("off")
        axes[-1].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=style.color,
                    marker=style.marker,
                    label="Ridge" if name == "ridge" else "MLP (seed 42)",
                )
                for name, style in STYLES.items()
            ],
            loc="center",
            frameon=False,
        )
        export(
            fig,
            out,
            f"learning_{role}",
            "Nested training prefixes with unchanged validation-selected MLP recipes; no new hidden-width/learning-rate search at smaller sample sizes. Every point is rescored on the same completed primary test cohort. Bars are paired 95% context-bootstrap intervals; they do not bootstrap training or establish convergence.",
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
    agreement_figure(source, args.out)
    noise_figure(source, args.out)
    readout_figures(source, args.out)
    learning_figures(source, args.out)
    save_json(
        args.out / "plot_complete.json",
        {
            "status": "complete",
            "figures": 6,
            "source_sha256": source.hashes,
            "script_sha256": file_sha256(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
