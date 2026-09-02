#!/usr/bin/env python3
"""Render the manuscript's Figure 2 from checked-in evaluation summaries.

The figure combines the single-turn, five-rollout layer and data-scaling
evaluations with pooled R^2 and whitened-cosine + CSLS top-1 retrieval.

Visual encoding
---------------
* predictor: color + marker shape;
* metric: solid/filled for R^2, dashed/open for top-1 retrieval;
The redundant encodings are designed to survive grayscale reproduction. The
script writes a vector PDF, a high-resolution PNG, a grayscale audit PNG, and
a JSON sidecar describing the inputs, hashes, style, and plotted values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    PREDICTOR_STYLES,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)


DEFAULT_LAYER_SOURCE = ROOT / "eval_results/issue_1901/avgtarget_plots/plot1_avg.json"
DEFAULT_SCALING_SOURCE = ROOT / "eval_results/issue_1901/figure2_five_rollout_scaling.json"
DEFAULT_BOUNDARY_SOURCE = ROOT / "eval_results/issue_1901/boundary_points_fig2.json"
DEFAULT_OUT = ROOT / "figures/paper"
DEFAULT_STEM = "figure2_predictability_scaling"


SCALING_KEYS = {
    "ridge": "ridge",
    "mlp_w8192": "mlp",
}


def _acc1(record: dict) -> float:
    values = record["acc_at_k"]
    return float(values.get("1", values.get(1)))


def _load_layer_data(path: Path) -> dict:
    source = json.loads(path.read_text())
    rows = []
    for layer_text, cell in source["per_layer"].items():
        layer = int(layer_text)
        arms = {}
        for key in PREDICTOR_STYLES:
            rec = cell["arms"][key]["avg"]
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
        "target": "five-rollout mean",
        "retrieval": "whitened cosine + CSLS (K=10)",
    }


def _load_scaling_data(path: Path, extension: dict | None = None) -> dict:
    source = json.loads(path.read_text())
    rows = [] if extension is None else [dict(r) for r in extension["rows"]]
    for n_text, cell in source["per_n"].items():
        arms = {}
        for plot_key, source_key in SCALING_KEYS.items():
            rec = cell[source_key]
            arms[plot_key] = {
                "r2": float(rec["r2"]),
                "retrieval": float(rec["top1"]),
            }
        rows.append({"x": int(n_text), "arms": arms})
    rows.sort(key=lambda row: row["x"])
    expected = [5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 250_000, 500_000, 963_444]
    if extension is not None:
        expected = sorted({r["x"] for r in extension["rows"]} | set(expected))
    assert [row["x"] for row in rows] == expected
    return {
        "rows": rows,
        "layer": int(source["layer"]),
        "n_test": int(source["duplicate_audit"]["source_n_pool"]),
        "target": "five-rollout mean",
        "retrieval": (
            f"whitened cosine + CSLS (K={source['retrieval']['csls_k']}), "
            f"deduplicated pool n={source['retrieval']['n_pool']}"
        ),
    }


BOUNDARY_COLOR = "#8C4A1F"  # burnt umber: the WikiText boundary-token control
COPY_COLOR = "#687078"  # muted gray: copy the context vector (+ learned bias)
DEFAULT_EXTENSION_SOURCE = ROOT / "eval_results/issue_1901/figure2_extension_1200.json"


def _load_boundary_data(path: Path) -> dict:
    """Boundary-token control: mean over the four exact-token maps (1,200 pairs each)."""
    source = json.loads(path.read_text())
    toks = source["tokens"]
    r2 = [float(t["r2"]) for t in toks.values()]
    top1 = [float(t["retrieval"]["whiten_csls"]["top1"]) for t in toks.values()]
    return {
        "n_tokens": len(toks),
        "tokens": {tid: {"label": t["label"], "r2": float(t["r2"]),
                         "retrieval": float(t["retrieval"]["whiten_csls"]["top1"])} for tid, t in toks.items()},
        "r2_mean": float(np.mean(r2)),
        "retrieval_mean": float(np.mean(top1)),
        "n_train_per_token": int(next(iter(toks.values()))["n_train"]),
        "n_pool": int(next(iter(toks.values()))["pool"]["realized_n_pool"]),
        "span": source["span"],
        "retrieval": source["retrieval"],
    }


def _load_extension_data(path: Path) -> dict:
    """Extra scaling rungs (1,200 contexts) + copy-context baselines, Figure 2B convention."""
    source = json.loads(path.read_text())
    rows = []
    for n_text, cell in source["per_n"].items():
        arms = {plot_key: {"r2": float(cell[src_key]["r2"]), "retrieval": float(cell[src_key]["top1"])}
                for plot_key, src_key in SCALING_KEYS.items()}
        rows.append({"x": int(n_text), "arms": arms})
    ib = source["baselines"]["identity_bias"]
    return {
        "rows": rows,
        "identity_bias": {"r2": float(ib["r2"]), "retrieval": float(ib["top1"])},
        "identity_copy": {"r2": float(source["baselines"]["identity_copy"]["r2"]),
                          "retrieval": float(source["baselines"]["identity_copy"]["top1"])},
        "convention": source["convention"],
    }


def _plot_controls(ax: plt.Axes, boundary: dict | None, extension: dict | None) -> None:
    """Horizontal reference lines: solid = R^2, dashed = strict top-1."""
    dash = (0, (5.0, 3.8))
    if boundary is not None:
        ax.axhline(boundary["r2_mean"], color=BOUNDARY_COLOR, lw=2.4, zorder=2)
        ax.axhline(boundary["retrieval_mean"], color=BOUNDARY_COLOR, lw=2.0, linestyle=dash, zorder=2)
    if extension is not None:
        ib = extension["identity_bias"]
        ax.axhline(ib["r2"], color=COPY_COLOR, lw=2.4, zorder=2)
        ax.axhline(ib["retrieval"], color=COPY_COLOR, lw=2.0, linestyle=dash, zorder=2)


def _control_legend_handles(boundary: dict | None, extension: dict | None) -> list[Line2D]:
    handles = []
    if boundary is not None:
        handles.append(Line2D([0], [0], color=BOUNDARY_COLOR, lw=2.6,
                              label=f"Boundary token \u2192 next sentence (WikiText, mean of {boundary['n_tokens']} tokens, {boundary['n_train_per_token']:,} pairs)"))
    if extension is not None:
        handles.append(Line2D([0], [0], color=COPY_COLOR, lw=2.6, label="Copy context vector + learned bias"))
    return handles


def _series(rows: list[dict], predictor: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["x"] for row in rows], dtype=float),
        np.asarray([row["arms"][predictor][metric] for row in rows], dtype=float),
    )


def _plot_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    title: str,
    kicker: str,
    show_retrieval: bool,
) -> None:
    style_score_axis(ax)
    ax.set_title(title, loc="left", y=1.055, pad=0, fontweight=650)
    ax.text(
        0.0,
        1.17,
        kicker.upper(),
        transform=ax.transAxes,
        fontsize=13,
        fontweight=700,
        color=MUTED,
        va="bottom",
        ha="left",
        linespacing=1.0,
    )

    for key, style in PREDICTOR_STYLES.items():
        x, r2 = _series(rows, key, "r2")

        ax.plot(
            x,
            r2,
            color=style.color,
            marker=style.marker,
            markersize=6.5,
            markeredgewidth=1.4,
            lw=3.0,
            zorder=4,
        )
        if show_retrieval:
            _, retrieval = _series(rows, key, "retrieval")
            ax.plot(
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
    if 1_000 <= value < 5_000:
        return f"{value / 1_000:.1f}k"
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
        for style in PREDICTOR_STYLES.values()
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


def make_figure(layer: dict, scaling: dict, boundary: dict | None = None, extension: dict | None = None) -> plt.Figure:
    set_c2a_style()
    fig = plt.figure(figsize=(14.4, 6.2), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        left=0.075,
        right=0.985,
        top=0.66,
        bottom=0.12,
        wspace=0.20,
    )
    ax_layer = fig.add_subplot(grid[0, 0])
    ax_scale = fig.add_subplot(grid[0, 1])

    _plot_panel(
        ax_layer,
        layer["rows"],
        title="Predictability across layers",
        kicker=f"A  ·  {layer['n_train']:,} training contexts",
        show_retrieval=False,
    )
    _plot_panel(
        ax_scale,
        scaling["rows"],
        title="Scaling with training data",
        kicker=f"B  ·  layer {scaling['layer']}",
        show_retrieval=True,
    )
    controls = boundary is not None or extension is not None
    if controls:
        _plot_controls(ax_scale, boundary, extension)
        ax_scale.set_ylim(-1.0, 1.0)
        ax_scale.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    ax_layer.set_xlim(-0.5, 27.5)
    ax_layer.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax_layer.set_xlabel("Model layer", labelpad=12)

    ns = np.asarray([row["x"] for row in scaling["rows"]], dtype=float)
    ax_scale.set_xscale("log")
    ax_scale.set_xlim(ns.min() / 1.18, ns.max() * 1.18)
    ticks = [5_000, 25_000, 100_000, 500_000, 963_444]
    if ns.min() < 5_000:
        ticks = [int(ns.min()), 5_000, 25_000, 100_000, 963_444]
    ax_scale.xaxis.set_major_locator(FixedLocator(ticks))
    ax_scale.xaxis.set_major_formatter(FuncFormatter(_human_n))
    ax_scale.minorticks_off()
    ax_scale.set_xlabel("Training contexts", labelpad=12)

    ax_layer.set_ylabel("Score  ↑", labelpad=13)
    ax_scale.set_ylabel("Score  ↑", labelpad=13)

    predictor_handles, metric_handles = _legend_handles()
    row_y = 0.946 if not controls else 0.875
    fig.text(
        0.075,
        row_y,
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
        bbox_to_anchor=(0.074, row_y - 0.021),
        ncol=3,
        frameon=False,
        columnspacing=1.45,
        handlelength=2.1,
        handletextpad=0.65,
        borderaxespad=0,
    )
    fig.text(
        0.675,
        row_y,
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
        bbox_to_anchor=(0.674, row_y - 0.021),
        ncol=2,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.1,
        handletextpad=0.65,
        borderaxespad=0,
    )
    if controls:
        fig.text(
            0.075,
            0.995,
            "CONTROLS",
            color=MUTED,
            fontsize=11.5,
            fontweight=750,
            ha="left",
            va="center",
        )
        fig.legend(
            handles=_control_legend_handles(boundary, extension),
            loc="upper left",
            bbox_to_anchor=(0.074, 0.974),
            ncol=2,
            frameon=False,
            columnspacing=1.2,
            handlelength=1.6,
            handletextpad=0.5,
            borderaxespad=0,
        )
    return fig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _write_outputs(
    fig: plt.Figure,
    out_dir: Path,
    stem_name: str,
    layer_source: Path,
    scaling_source: Path,
    layer: dict,
    scaling: dict,
    font: str,
    git_state: dict[str, str | bool | None],
    boundary_source: Path | None = None,
    boundary: dict | None = None,
    extension_source: Path | None = None,
    extension: dict | None = None,
) -> dict[str, Path]:
    stem = out_dir / stem_name
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Figure 2: context-to-answer predictability and scaling",
        subject="Five-rollout layer sweep and training-data scaling",
        creator="scripts/make_paper_figure2.py",
    )
    metadata = stem.with_suffix(".meta.json")
    metadata.write_text(
        json.dumps(
            {
                "status": "manuscript Figure 2",
                "style_version": STYLE_VERSION,
                "plotting_script": "scripts/make_paper_figure2.py",
                "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
                "rescore_script": "scripts/issue1901_figure2_five_rollout_scaling.py",
                "reproduction_command": "uv run python scripts/make_paper_figure2.py",
                "repository_manuscript_asset": _display_path(outputs["pdf"]),
                "overleaf_destination": "figures/paper/figure2_predictability_scaling.pdf",
                "git": git_state,
                "sources": {
                    "layer": {
                        "path": _display_path(layer_source),
                        "sha256": _sha256(layer_source),
                    },
                    "scaling": {
                        "path": _display_path(scaling_source),
                        "sha256": _sha256(scaling_source),
                    },
                },
                "rendering": {
                    "resolved_font": font,
                    "authoring_size_inches": [14.4, 6.2],
                    "intended_manuscript_width_inches": 5.5,
                    "png_dpi": 240,
                    "background": PAPER,
                },
                "displayed_metrics": {
                    "left": ["r2"],
                    "right": ["r2", "strict_top1_retrieval"],
                },
                "metric_encoding": {
                    "r2": "solid line, filled marker",
                    "strict_top1_retrieval": "dashed line, open marker",
                },
                "predictor_encoding": {
                    key: {
                        "label": style.label,
                        "color": style.color,
                        "marker": style.marker,
                    }
                    for key, style in PREDICTOR_STYLES.items()
                },
                "layer": layer,
                "scaling": scaling,
                "boundary_control": (
                    None
                    if boundary is None
                    else {
                        "source": {
                            "path": _display_path(boundary_source),
                            "sha256": _sha256(boundary_source),
                        },
                        "encoding": "burnt-umber horizontal lines on panel B: solid = mean R^2, "
                        "dashed = mean strict top-1 over the four exact-token maps",
                        **boundary,
                    }
                ),
                "extension": (
                    None
                    if extension is None
                    else {
                        "source": {
                            "path": _display_path(extension_source),
                            "sha256": _sha256(extension_source),
                        },
                        "encoding": "1,200-context rung joins the predictor curves; gray horizontal "
                        "lines = copy context vector + learned bias (solid R^2, dashed strict top-1)",
                        **extension,
                    }
                ),
                "output_sha256": {kind: _sha256(path) for kind, path in outputs.items()},
            },
            indent=2,
        )
        + "\n"
    )
    return {**outputs, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-source", type=Path, default=DEFAULT_LAYER_SOURCE)
    parser.add_argument("--scaling-source", type=Path, default=DEFAULT_SCALING_SOURCE)
    parser.add_argument("--boundary-source", type=Path, default=DEFAULT_BOUNDARY_SOURCE)
    parser.add_argument("--no-boundary", action="store_true", help="render without the control overlay")
    parser.add_argument("--extension-source", type=Path, default=DEFAULT_EXTENSION_SOURCE)
    parser.add_argument("--no-extension", action="store_true", help="render without the 1,200 rung + copy baselines")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args()

    layer = _load_layer_data(args.layer_source)
    extension = None if args.no_extension else _load_extension_data(args.extension_source)
    scaling = _load_scaling_data(args.scaling_source, extension)
    boundary = None if args.no_boundary else _load_boundary_data(args.boundary_source)
    assert layer["n_test"] == scaling["n_test"] == 1_000
    for dataset in (layer, scaling):
        for row in dataset["rows"]:
            for values in row["arms"].values():
                assert np.isfinite(values["r2"])
                assert 0.0 <= values["r2"] <= 1.0
                assert 0.0 <= values["retrieval"] <= 1.0

    git_state = _git_state()
    font = set_c2a_style()
    fig = make_figure(layer, scaling, boundary, extension)
    outputs = _write_outputs(
        fig,
        args.out_dir,
        args.stem,
        args.layer_source,
        args.scaling_source,
        layer,
        scaling,
        font,
        git_state,
        boundary_source=None if args.no_boundary else args.boundary_source,
        boundary=boundary,
        extension_source=None if args.no_extension else args.extension_source,
        extension=extension,
    )
    plt.close(fig)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
