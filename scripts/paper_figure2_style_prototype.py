#!/usr/bin/env python3
"""Render the Figure 2 visual-system prototype.

This is deliberately a preview, not the manuscript figure. It combines the
single-turn, single-rollout layer and data-scaling evaluations that already
have both pooled R^2 and whitened-cosine + CSLS top-1 retrieval results.

Visual encoding
---------------
* predictor: color + marker shape;
* metric: solid/filled for R^2, dashed/open for top-1 retrieval;
* negative identity-plus-bias R^2: a compact broken-axis strip.

The redundant encodings are designed to survive grayscale reproduction. The
script writes a vector PDF, a high-resolution PNG, a grayscale audit PNG, and
a JSON sidecar describing the sources and plotted values.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
LAYER_SOURCE = ROOT / "eval_results/issue_1901/avgtarget_plots/plot1_avg.json"
SCALING_SOURCE = ROOT / "eval_results/issue_1901/paper_densify/mlp_scaling_dense_L19.json"
DEFAULT_OUT = ROOT / "figures/paper_style"


@dataclass(frozen=True)
class PredictorStyle:
    label: str
    color: str
    marker: str


PREDICTORS = {
    "ridge": PredictorStyle("Linear", "#176B87", "o"),
    "mlp_w8192": PredictorStyle("Nonlinear", "#C4553D", "D"),
    "identity_bias": PredictorStyle("Identity + bias", "#687078", "s"),
}

SCALING_KEYS = {
    "ridge": "ridge",
    "mlp_w8192": "mlp",
    "identity_bias": "identity_bias",
}

INK = "#22272B"
MUTED = "#687078"
GRID = "#C8C6BF"
PAPER = "#FFFFFF"
SEAM = "#A9A69E"


def _acc1(record: dict) -> float:
    values = record["acc_at_k"]
    return float(values.get("1", values.get(1)))


def _load_layer_data() -> dict:
    source = json.loads(LAYER_SOURCE.read_text())
    rows = []
    for layer_text, cell in source["per_layer"].items():
        layer = int(layer_text)
        arms = {}
        for key in PREDICTORS:
            rec = cell["arms"][key]["single"]
            arms[key] = {
                "r2": float(rec["whole_map_r2"]),
                "retrieval": _acc1(rec["retrieval"]["whiten_csls"]),
            }
        rows.append({"x": layer, "arms": arms})
    rows.sort(key=lambda row: row["x"])
    assert [row["x"] for row in rows] == list(range(28))
    return {
        "rows": rows,
        "n_train": int(source["split"]["n_train"]),
        "n_test": int(source["split"]["n_test"]),
        "target": "single rollout",
        "retrieval": "whitened cosine + CSLS (K=10)",
    }


def _load_scaling_data() -> dict:
    source = json.loads(SCALING_SOURCE.read_text())
    rows = []
    for n_text, cell in source["per_n"].items():
        arms = {}
        for plot_key, source_key in SCALING_KEYS.items():
            rec = cell[source_key]
            arms[plot_key] = {
                "r2": float(rec["test_r2"]),
                "retrieval": _acc1(rec["whitened_csls"]),
            }
        rows.append({"x": int(n_text), "arms": arms})
    rows.sort(key=lambda row: row["x"])
    expected = [5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 250_000, 500_000, 963_444]
    assert [row["x"] for row in rows] == expected
    return {
        "rows": rows,
        "layer": int(source["layer"]),
        "n_test": int(source["split"]["n_test"]),
        "target": "single rollout",
        "retrieval": "whitened cosine + CSLS (K=10)",
    }


def _set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Noto Sans", "DejaVu Sans"],
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "axes.linewidth": 1.2,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def _series(rows: list[dict], predictor: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["x"] for row in rows], dtype=float),
        np.asarray([row["arms"][predictor][metric] for row in rows], dtype=float),
    )


def _style_axes(top: plt.Axes, bottom: plt.Axes) -> None:
    for ax in (top, bottom):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SEAM)
        ax.spines["bottom"].set_color(SEAM)
        ax.tick_params(length=0, pad=8)
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(labelbottom=False)
    top.set_ylim(0.0, 1.025)
    top.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    top.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y:.2f}".rstrip("0").rstrip(".")))
    top.grid(axis="y", color=GRID, lw=1.0, alpha=0.55)
    top.set_axisbelow(True)

    # The lower strip is deliberately short: it communicates that the control
    # has negative R^2 without sacrificing resolution where the fitted methods lie.
    bottom.set_ylim(-3.0, -0.18)
    bottom.set_yticks([-3.0, -2.0, -1.0])
    bottom.grid(axis="y", color=GRID, lw=0.9, alpha=0.35)
    bottom.set_axisbelow(True)
    bottom.axhline(-0.18, color=SEAM, lw=1.0)
    bottom.text(
        0.012,
        0.12,
        "negative $R^2$",
        transform=bottom.transAxes,
        color=MUTED,
        fontsize=14,
        va="bottom",
        ha="left",
    )

    # Small diagonal cut marks make the discontinuity explicit without the
    # visually heavy zig-zag used in many conventional paper figures.
    cut = 0.012
    kwargs = dict(color=INK, clip_on=False, lw=1.35)
    top.plot((-cut, +cut), (-cut, +cut), transform=top.transAxes, **kwargs)
    bottom.plot((-cut, +cut), (1 - cut, 1 + cut), transform=bottom.transAxes, **kwargs)


def _plot_panel(
    top: plt.Axes,
    bottom: plt.Axes,
    rows: list[dict],
    *,
    title: str,
    kicker: str,
) -> None:
    _style_axes(top, bottom)
    top.set_title(title, loc="left", y=1.055, pad=0, fontweight=650)
    top.text(
        0.0,
        1.17,
        kicker.upper(),
        transform=top.transAxes,
        fontsize=13,
        fontweight=700,
        color=MUTED,
        va="bottom",
        ha="left",
        linespacing=1.0,
    )

    for key, style in PREDICTORS.items():
        x, r2 = _series(rows, key, "r2")
        _, retrieval = _series(rows, key, "retrieval")

        positive = r2 >= 0
        if np.any(positive):
            top.plot(
                x[positive],
                r2[positive],
                color=style.color,
                marker=style.marker,
                markersize=6.5,
                markeredgewidth=1.4,
                lw=3.0,
                zorder=4,
            )
        if np.any(~positive):
            bottom.plot(
                x[~positive],
                r2[~positive],
                color=style.color,
                marker=style.marker,
                markersize=6.5,
                markeredgewidth=1.4,
                lw=2.6,
                zorder=4,
            )

        top.plot(
            x,
            retrieval,
            color=style.color,
            marker=style.marker,
            markerfacecolor=PAPER,
            markeredgecolor=style.color,
            markeredgewidth=1.8,
            markersize=7.0,
            lw=2.4,
            linestyle=(0, (5.0, 3.8)),
            zorder=3,
        )


def _human_n(value: float, _position: int | None = None) -> str:
    if value >= 900_000:
        return "963k" if value < 1_000_000 else f"{value / 1_000_000:g}m"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def _legend_handles() -> tuple[list[Line2D], list[Line2D]]:
    predictors = [
        Line2D(
            [0],
            [0],
            color=style.color,
            marker=style.marker,
            markersize=8,
            lw=3,
            label=style.label,
        )
        for style in PREDICTORS.values()
    ]
    metrics = [
        Line2D([0], [0], color=INK, marker="o", lw=3, label="$R^2$"),
        Line2D(
            [0],
            [0],
            color=INK,
            marker="o",
            markerfacecolor=PAPER,
            markeredgewidth=1.7,
            lw=2.4,
            linestyle=(0, (5.0, 3.8)),
            label="Top-1 retrieval",
        ),
    ]
    return predictors, metrics


def make_figure(layer: dict, scaling: dict) -> plt.Figure:
    _set_style()
    fig = plt.figure(figsize=(14.4, 6.8), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[4.3, 1.05],
        left=0.075,
        right=0.985,
        top=0.73,
        bottom=0.13,
        wspace=0.20,
        hspace=0.055,
    )
    ax_layer = fig.add_subplot(grid[0, 0])
    ax_layer_neg = fig.add_subplot(grid[1, 0], sharex=ax_layer)
    ax_scale = fig.add_subplot(grid[0, 1])
    ax_scale_neg = fig.add_subplot(grid[1, 1], sharex=ax_scale)

    _plot_panel(
        ax_layer,
        ax_layer_neg,
        layer["rows"],
        title="Predictability across layers",
        kicker=f"A  ·  {layer['n_train']:,} training contexts",
    )
    _plot_panel(
        ax_scale,
        ax_scale_neg,
        scaling["rows"],
        title="Scaling with training data",
        kicker=f"B  ·  layer {scaling['layer']}",
    )

    ax_layer_neg.set_xlim(-0.5, 27.5)
    ax_layer_neg.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax_layer_neg.set_xlabel("Model layer", labelpad=12)

    ns = np.asarray([row["x"] for row in scaling["rows"]], dtype=float)
    ax_scale.set_xscale("log")
    ax_scale_neg.set_xscale("log")
    ax_scale_neg.set_xlim(ns.min() / 1.18, ns.max() * 1.18)
    ax_scale_neg.xaxis.set_major_locator(FixedLocator([5_000, 25_000, 100_000, 500_000, 963_444]))
    ax_scale_neg.xaxis.set_major_formatter(FuncFormatter(_human_n))
    ax_scale_neg.minorticks_off()
    ax_scale_neg.set_xlabel("Training contexts", labelpad=12)

    ax_layer.set_ylabel("Score  ↑", labelpad=13)
    ax_scale.set_ylabel("Score  ↑", labelpad=13)

    predictor_handles, metric_handles = _legend_handles()
    fig.text(
        0.075,
        0.946,
        "PREDICTOR",
        color=MUTED,
        fontsize=11.5,
        fontweight=750,
        ha="left",
        va="center",
    )
    fig.legend(
        handles=predictor_handles,
        loc="upper left",
        bbox_to_anchor=(0.074, 0.925),
        ncol=3,
        frameon=False,
        columnspacing=1.45,
        handlelength=2.1,
        handletextpad=0.65,
        borderaxespad=0,
    )
    fig.text(
        0.675,
        0.946,
        "METRIC",
        color=MUTED,
        fontsize=11.5,
        fontweight=750,
        ha="left",
        va="center",
    )
    fig.legend(
        handles=metric_handles,
        loc="upper left",
        bbox_to_anchor=(0.674, 0.925),
        ncol=2,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.1,
        handletextpad=0.65,
        borderaxespad=0,
    )
    return fig


def _write_outputs(fig: plt.Figure, out_dir: Path, layer: dict, scaling: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "figure2_style_prototype"
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    gray = out_dir / "figure2_style_prototype_grayscale.png"
    meta = stem.with_suffix(".meta.json")

    fig.savefig(
        pdf,
        facecolor=PAPER,
        bbox_inches="tight",
        metadata={
            "Title": "Figure 2 visual-system prototype",
            "Subject": "Context-to-answer predictor layer and data-scaling curves",
            "Creator": "scripts/paper_figure2_style_prototype.py",
        },
    )
    fig.savefig(png, dpi=240, facecolor=PAPER, bbox_inches="tight")
    with Image.open(png) as image:
        image.convert("L").save(gray)

    meta.write_text(
        json.dumps(
            {
                "status": "style prototype; not wired into the manuscript",
                "sources": {
                    "layer": str(LAYER_SOURCE.relative_to(ROOT)),
                    "scaling": str(SCALING_SOURCE.relative_to(ROOT)),
                },
                "metric_encoding": {
                    "r2": "solid line, filled marker",
                    "top1_retrieval": "dashed line, open marker",
                },
                "predictor_encoding": {
                    key: {
                        "label": style.label,
                        "color": style.color,
                        "marker": style.marker,
                    }
                    for key, style in PREDICTORS.items()
                },
                "layer": layer,
                "scaling": scaling,
            },
            indent=2,
        )
        + "\n"
    )
    return {"pdf": pdf, "png": png, "grayscale": gray, "metadata": meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    layer = _load_layer_data()
    scaling = _load_scaling_data()
    assert layer["n_test"] == scaling["n_test"] == 1_000
    for dataset in (layer, scaling):
        for row in dataset["rows"]:
            for values in row["arms"].values():
                assert np.isfinite(values["r2"])
                assert 0.0 <= values["retrieval"] <= 1.0
    assert all(row["arms"]["identity_bias"]["r2"] < 0 for row in layer["rows"])
    assert all(row["arms"]["identity_bias"]["r2"] < 0 for row in scaling["rows"])
    negative_r2 = [
        values["r2"]
        for dataset in (layer, scaling)
        for row in dataset["rows"]
        for values in row["arms"].values()
        if values["r2"] < 0
    ]
    assert all(-3.0 <= value <= -0.18 for value in negative_r2), (
        "negative R^2 falls outside the visible broken-axis strip",
        min(negative_r2),
        max(negative_r2),
    )

    fig = make_figure(layer, scaling)
    outputs = _write_outputs(fig, args.out_dir, layer, scaling)
    plt.close(fig)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
