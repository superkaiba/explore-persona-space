#!/usr/bin/env python3
"""Render the manuscript's predictability-and-scaling figure (paper Figure 3).

The figure combines the single-turn, five-rollout layer and data-scaling
evaluations with pooled R^2 and whitened-cosine + CSLS top-1 retrieval.

Visual encoding
---------------
* predictor: color + marker shape;
* metric: solid/filled for R^2, dashed/open for top-1 retrieval;
* baseline: gray or amber marker at panel B's 25,000-context rung, on a y-axis
  cut into two proportionally scaled segments so a negative R^2 is shown at its
  own value instead of being dropped or clipped;
The redundant encodings are designed to survive grayscale reproduction. The
script writes a vector PDF, a high-resolution PNG, a grayscale audit PNG, and
a JSON sidecar describing the inputs, hashes, style, and plotted values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below. On the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and
# the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402


from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    METRIC_LABELS,
    MUTED,
    PAPER,
    PREDICTOR_STYLES,
    ROLES,
    STYLE_VERSION,
    better_label,
    c2a_figure,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
    style_score_axis,
)


DEFAULT_LAYER_SOURCE = ROOT / "eval_results/issue_1901/avgtarget_plots/plot1_avg.json"
DEFAULT_SCALING_SOURCE = ROOT / "eval_results/issue_1901/figure2_five_rollout_scaling.json"
DEFAULT_BOUNDARY_SOURCE = ROOT / "eval_results/issue_1901/boundary_points_fig2.json"
DEFAULT_OUT = ROOT / "figures/paper"
DEFAULT_STEM = "c1_predictability_scaling"


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


def _load_pool10k_data(path: Path) -> dict:
    """10,000-candidate rescore: panel-B retrieval overlay + panel-C arms (#1901)."""
    source = json.loads(path.read_text())
    retrieval = source["retrieval"]
    assert int(retrieval["n_pool"]) == 10_000, retrieval
    assert int(retrieval["n_query"]) == 942, retrieval
    chance = float(retrieval["chance_at_1"][str(retrieval["n_pool"])])
    assert chance == 1e-4, chance
    return {
        "panel_b": source["panel_b"],
        "panel_c": source["panel_c"],
        "layer": int(source["layer"]),
        "n_pool": int(retrieval["n_pool"]),
        "n_query": int(retrieval["n_query"]),
        "chance_at_1": chance,
        "csls_k": int(retrieval["csls_k"]),
    }


def _load_scaling_data(path: Path, pool10k: dict) -> dict:
    """Panel B: banked R^2 (pool-independent) + 10,000-candidate top-1 retrieval.

    R^2 comes from the banked scaling JSON and is asserted byte-equal to the
    values the pool-10k rescore carried forward; retrieval is the 10,000-pool
    rescore's ``top1_10000``.
    """
    source = json.loads(path.read_text())
    rows = []
    for n_text, cell in source["per_n"].items():
        pool_cell = pool10k["panel_b"][n_text]
        arms = {}
        for plot_key, source_key in SCALING_KEYS.items():
            rec = cell[source_key]
            pool_rec = pool_cell[source_key]
            # The rescore carries R^2 (and the 942-pool top-1) from the banked JSON.
            assert float(pool_rec["r2"]) == float(rec["r2"]), (n_text, source_key)
            assert float(pool_rec["top1_942"]) == float(rec["top1"]), (n_text, source_key)
            arms[plot_key] = {
                "r2": float(rec["r2"]),
                "retrieval": float(pool_rec["top1_10000"]),
                "retrieval_942": float(rec["top1"]),
            }
        rows.append({"x": int(n_text), "arms": arms})
    rows.sort(key=lambda row: row["x"])
    expected = [5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 250_000, 500_000, 963_444]
    assert [row["x"] for row in rows] == expected
    return {
        "rows": rows,
        "layer": int(source["layer"]),
        "n_test": int(source["duplicate_audit"]["source_n_pool"]),
        "target": "five-rollout mean",
        "retrieval": (
            f"whitened cosine + CSLS (K={pool10k['csls_k']}), "
            f"pool n={pool10k['n_pool']:,} (942 deduplicated targets + 9,058 "
            f"distractors), chance top-1 {pool10k['chance_at_1']:g}"
        ),
    }


# The boundary-token control is a null, so it takes the paper-wide control role.
BOUNDARY_COLOR = ROLES["control"].color
DEFAULT_EXTENSION_SOURCE = ROOT / "eval_results/issue_1901/figure2_extension_1200.json"
DEFAULT_BASELINES_SOURCE = ROOT / "eval_results/issue_1901/fig2_baselines/fig2_baselines.json"
DEFAULT_POOL10K_SOURCE = ROOT / "eval_results/issue_1901/fig2_pool10k/fig2_pool10k.json"

# Panel C arms: (key in fig2_pool10k panel_c / fig2_baselines per_arm, reader label,
# color). Colors reuse the paper-wide semantic palette: the linear map keeps the
# ``linear`` hue, encoder-input arms take the ``other_source`` amber (representation
# from another model), and every control/null is muted gray. The roster below is
# curated. The e5 encoder arm, both zero-parameter cosine floors, and the plain
# copy baseline were dropped from the panel (commits 44faead8d4b, 514ea8e9f28,
# 349175bc760). The rescore still computes every one of them, and each appears
# in the sidecar under extra_arms_not_drawn rather than being lost.
_ENCODER_COLOR = ROLES["other_source"].color
BASELINE_ARMS: dict[str, dict] = {
    "anchor": {"label": "Linear map", "color": ROLES["linear"].color, "marker": "o"},
    "enc_bge_cls": {
        "label": "Encoder (BGE)",
        "color": _ENCODER_COLOR,
        "marker": ROLES["other_source"].marker,
    },
    "pca1024": {"label": "PCA-1024", "color": ROLES["control"].color, "marker": "s"},
    "identity_bias": {"label": "Copy + bias", "color": ROLES["control"].color, "marker": "v"},
    "shuffled": {"label": "Shuffled pairs", "color": ROLES["control"].color, "marker": "P"},
}

# Panel C is gone. Its baselines now ride panel B as points at the rung where they
# were measured (layer 19, n_train = 25,000), which is an x value panel B already
# plots, so nothing implies an n-independent value. "minimal" draws only the arms
# that clear panel B's y-range with visual room; "full" adds PCA-1024, whose value
# is by construction close to the linear map and so overlaps the teal curve.
# "anchor" is never drawn: it IS panel B's linear curve at this rung (its retrieval
# is byte-equal and its R^2 agrees to 1e-6).
BASELINE_POINT_X = 25_000
# Fill alone (filled = R^2, open = top-1) is too subtle to read on isolated markers
# at print size, so the two metrics are also dodged left and right of the rung:
# R^2 always sits left of the guide line, top-1 always right. Multiplicative
# because panel B's x-axis is logarithmic. This also separates the shuffled null's
# two values, which are both within 0.03 of zero and would otherwise overlap.
BASELINE_METRIC_DODGE = 1.16
BASELINE_METRIC_SIDE = {"r2": 1.0 / BASELINE_METRIC_DODGE, "top1": BASELINE_METRIC_DODGE}

# Panel B's y-axis is cut so the baselines' held-out R^2 fits without crushing the
# curves. Two segments, ordered low to high; heights are proportional to these
# spans, so both segments share ONE data scale and a marker's distance from a tick
# means the same thing in either. Nothing on the panel is severed by the break:
# every predictor curve lives entirely inside the upper segment (0.61 to 0.96), and
# only isolated baseline markers land in the lower strip. The upper segment's floor
# moved from 0.15 to just below zero so that zero, the reference an R^2 is read
# against, is on the axis.
PANEL_B_SEGMENTS: tuple[tuple[float, float], ...] = ((-1.0, -0.85), (-0.08, 1.0))
PANEL_B_SEGMENT_TICKS: tuple[tuple[float, ...], ...] = (
    (-0.9,),
    (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)

PANEL_B_BASELINE_ROSTERS: dict[str, tuple[str, ...]] = {
    "minimal": ("enc_bge_cls", "identity_bias"),
    "withnull": ("enc_bge_cls", "identity_bias", "shuffled"),
    "full": ("pca1024", "enc_bge_cls", "identity_bias", "shuffled"),
}


def _load_baselines_data(path: Path, extension: dict, pool10k: dict) -> dict:
    """Panel C: paper-convention baselines at n_train=25,000, 10,000-candidate pool.

    Retrieval (top-1 + 95% interval) comes from the pool-10k rescore's ``panel_c``.
    Held-out R^2 is pool-independent and stays banked: ``fig2_baselines.json``
    (issue1901_encoder_paperconv.py) for the fitted arms, the banked
    ``figure2_extension_1200.json`` values for the two copy baselines, and ``None``
    for the two zero-parameter cosine floors (no fit, so no R^2).
    """
    source = json.loads(path.read_text())
    if source.get("smoke"):
        raise ValueError(f"{path} is a smoke artifact; refusing to draw it on a paper figure")
    gate = source["step1_gate"]
    if not gate.get("gated") or gate["d_r2"] > gate["tol"] or gate["d_top1"] > gate["tol"]:
        raise ValueError(f"{path} step1 gate not passed: {gate}")
    banked_r2 = {k: v.get("r2") for k, v in source["per_arm"].items()}
    banked_r2["identity_bias"] = extension["identity_bias"]["r2"]
    banked_r2["identity_copy"] = extension["identity_copy"]["r2"]
    arms = []
    for key, style in BASELINE_ARMS.items():
        rec = pool10k["panel_c"][key]
        ci = rec.get("top1_ci95_10000")
        r2 = banked_r2[key]
        if key.startswith("floor_"):
            assert r2 is None, f"cosine floor {key} unexpectedly carries an R^2: {r2}"
        arms.append(
            {
                "key": key,
                "label": style["label"],
                "color": style["color"],
                "r2": None if r2 is None else float(r2),
                "top1": float(rec["top1_10000"]),
                "top1_942": float(rec["top1_942"]),
                "top1_ci95": None if ci is None else [float(ci["lo"]), float(ci["hi"])],
            }
        )
    # The rescore computes every arm; the panel draws the curated subset. Assert
    # containment, and record what the rescore holds but the panel omits so the
    # sidecar states the roster decision instead of hiding it.
    drawn = set(BASELINE_ARMS)
    available = set(pool10k["panel_c"])
    assert drawn <= available, (sorted(drawn - available), sorted(available))
    not_drawn = {
        key: {
            "top1_10000": float(pool10k["panel_c"][key]["top1_10000"]),
            "top1_942": float(pool10k["panel_c"][key]["top1_942"]),
            "r2": None if banked_r2[key] is None else float(banked_r2[key]),
        }
        for key in sorted(available - drawn)
    }
    arms.sort(key=lambda arm: -arm["top1"])
    return {
        "arms": arms,
        "n_train": int(source["n_train"]),
        "layer": int(source["layer"]),
        "n_pool": pool10k["n_pool"],
        "chance_at_1": pool10k["chance_at_1"],
        "convention": (
            f"{source['convention']}; retrieval rescored on the 10,000-candidate pool "
            "(942 deduplicated targets + 9,058 distractors), chance top-1 1e-4"
        ),
        "extra_arms_not_drawn": not_drawn,
    }


def _build_panel_b_axes(fig: plt.Figure, cell) -> list[plt.Axes]:
    """Panel B as one axis per y-segment, stacked low-to-high, sharing the x-axis.

    Heights are proportional to the segments' data spans, so the two strips share
    one scale. The lowest axis owns the x-axis; the others hide their bottom seam.
    """
    spans = [hi - lo for lo, hi in PANEL_B_SEGMENTS]
    inner = cell.subgridspec(
        len(PANEL_B_SEGMENTS),
        1,
        height_ratios=list(reversed(spans)),
        hspace=0.13,
    )
    # subgridspec row 0 is the TOP row, so the highest segment comes first.
    axes = [
        fig.add_subplot(inner[len(PANEL_B_SEGMENTS) - 1 - i, 0], label=f"panel-b-seg{i}")
        for i in range(len(PANEL_B_SEGMENTS))
    ]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    return axes


def _apply_panel_b_segments(axes: list[plt.Axes]) -> None:
    """Range, ticks, seams, and the diagonal break marks between adjacent strips."""
    for ax, (lo, hi), ticks in zip(axes, PANEL_B_SEGMENTS, PANEL_B_SEGMENT_TICKS, strict=True):
        style_axis(ax, grid_axis="y")
        ax.set_ylim(lo, hi)
        ax.set_yticks(list(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}".replace("-", "\u2212")))
    for lower, upper in pairwise(axes):
        # The break lives between these two strips: the lower one loses its top
        # seam, the upper one its bottom seam, and a diagonal mark sits on each.
        lower.spines["top"].set_visible(False)
        upper.spines["bottom"].set_visible(False)
        upper.tick_params(axis="x", length=0, labelbottom=False)
        mark = {
            "marker": [(-1.0, -0.6), (1.0, 0.6)],
            "markersize": 11,
            "linestyle": "none",
            "color": MUTED,
            "markeredgecolor": MUTED,
            "markeredgewidth": 1.5,
            "clip_on": False,
            "zorder": 6,
        }
        lower.plot([0], [1], transform=lower.transAxes, **mark)
        upper.plot([0], [0], transform=upper.transAxes, **mark)


def _center_ylabel(axes: list[plt.Axes], label: str) -> None:
    """One y-label centered over the whole stack rather than over the top strip."""
    top = axes[-1]
    boxes = [ax.get_position() for ax in axes]
    mid = (min(b.y0 for b in boxes) + max(b.y1 for b in boxes)) / 2.0
    own = top.get_position()
    top.set_ylabel(label, labelpad=13)
    top.yaxis.set_label_coords(-0.115, (mid - own.y0) / own.height)


def _plot_baseline_points(axes: list[plt.Axes], baselines: dict, roster: tuple[str, ...]) -> dict:
    """Draw the former panel-C baselines on panel B, at the rung they were measured on.

    Every arm was fitted at layer 19 on n_train = 25,000 rows and rescored on the
    same 10,000-candidate pool panel B uses, so each one is a POINT at an x value
    panel B already plots. A horizontal line is deliberately not used: PCA-1024 and
    the encoder are fitted maps whose value moves with n, and only this one rung was
    measured, so a full-width line would assert an n-independence nobody tested.

    An arm whose value falls outside the panel's y-range is NOT drawn and NOT clipped
    to the edge. It is returned under ``off_scale`` so the caller records it and the
    manuscript caption carries the number instead of the canvas implying a floor.
    """
    by_key = {arm["key"]: arm for arm in baselines["arms"]}
    drawn: list[dict] = []
    off_scale: list[dict] = []
    # Rung guide, on every strip so the 25,000 column reads as one line through the
    # break: the markers sit in empty space away from the curves, so without it a
    # reader cannot see which x they belong to.
    for ax in axes:
        ax.axvline(
            BASELINE_POINT_X,
            color=MUTED,
            lw=1.0,
            alpha=0.45,
            linestyle=(0, (1.6, 2.4)),
            zorder=1,
        )
    for key in roster:
        arm = by_key[key]
        style = BASELINE_ARMS[key]
        for metric in ("r2", "top1"):
            value = arm[metric]
            record = {
                "key": key,
                "label": arm["label"],
                "metric": metric,
                "value": None if value is None else float(value),
            }
            host = None
            if value is not None:
                for ax, (lo, hi) in zip(axes, PANEL_B_SEGMENTS, strict=True):
                    if lo < value < hi:
                        host = ax
                        record["segment"] = [lo, hi]
                        break
            if host is None:
                off_scale.append(record)
                continue
            x = BASELINE_POINT_X * BASELINE_METRIC_SIDE[metric]
            record["x"] = x
            host.plot(
                [x],
                [value],
                marker=style["marker"],
                markersize=8.0,
                color=style["color"],
                # Same fill encoding the predictor curves use: filled = R^2,
                # open = top-1 retrieval.
                markerfacecolor=style["color"] if metric == "r2" else PAPER,
                markeredgecolor=style["color"],
                markeredgewidth=1.8,
                linestyle="none",
                zorder=5,
            )
            drawn.append(record)
    if not drawn:
        raise ValueError(f"no baseline arm in roster {roster} landed inside {PANEL_B_SEGMENTS}")
    return {
        "x": BASELINE_POINT_X,
        "roster": list(roster),
        "segments": [list(seg) for seg in PANEL_B_SEGMENTS],
        "drawn": drawn,
        "off_scale": off_scale,
    }


def _load_boundary_data(path: Path) -> dict:
    """Boundary-token control: mean over the four exact-token maps (1,200 pairs each)."""
    source = json.loads(path.read_text())
    toks = source["tokens"]
    r2 = [float(t["r2"]) for t in toks.values()]
    top1 = [float(t["retrieval"]["whiten_csls"]["top1"]) for t in toks.values()]
    return {
        "n_tokens": len(toks),
        "tokens": {
            tid: {
                "label": t["label"],
                "r2": float(t["r2"]),
                "retrieval": float(t["retrieval"]["whiten_csls"]["top1"]),
            }
            for tid, t in toks.items()
        },
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
        arms = {
            plot_key: {"r2": float(cell[src_key]["r2"]), "retrieval": float(cell[src_key]["top1"])}
            for plot_key, src_key in SCALING_KEYS.items()
        }
        rows.append({"x": int(n_text), "arms": arms})
    ib = source["baselines"]["identity_bias"]
    ic = source["baselines"]["identity_copy"]
    return {
        "rows": rows,
        "identity_bias": {
            "r2": float(ib["r2"]),
            "retrieval": float(ib["top1"]),
            "top1_ci95": ib.get("top1_ci95"),
        },
        "identity_copy": {
            "r2": float(ic["r2"]),
            "retrieval": float(ic["top1"]),
            "top1_ci95": ic.get("top1_ci95"),
        },
        "convention": source["convention"],
    }


def _plot_controls(ax: plt.Axes, boundary: dict | None, extension: dict | None) -> None:
    """Horizontal reference line: the boundary-token control's mean held-out R^2."""
    dash = (0, (5.0, 3.8))
    if boundary is not None:
        # R^2 only: small-pool top-1 saturates and is quoted in the text instead.
        ax.axhline(boundary["r2_mean"], color=BOUNDARY_COLOR, lw=2.4, zorder=2)
    # The copy-context baselines stay in the sidecar metadata but are not drawn
    # (their R^2 is far below the axis; the text quotes them).


def _baseline_legend_handles(boundary: dict | None, overlay: dict | None) -> tuple[list, list[str]]:
    """One legend group for every reference quantity drawn on panel B.

    Each baseline's swatch shows BOTH of its markers, filled then open, in the
    same left-to-right order the dodged markers use on the panel, so the swatch
    itself says which side of the rung is which metric. An arm contributing only
    one metric shows only that marker.
    """
    handles: list = []
    labels: list[str] = []
    if boundary is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color=BOUNDARY_COLOR,
                lw=2.6,
            )
        )
        labels.append("Boundary token \u2192 next sentence")
    if overlay is None:
        return handles, labels
    drawn_metrics: dict[str, set[str]] = {}
    for record in overlay["drawn"]:
        drawn_metrics.setdefault(record["key"], set()).add(record["metric"])
    for key in overlay["roster"]:
        metrics = drawn_metrics.get(key)
        if not metrics:
            continue
        style = BASELINE_ARMS[key]
        swatch = tuple(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                markersize=8,
                markerfacecolor=style["color"] if metric == "r2" else PAPER,
                markeredgecolor=style["color"],
                markeredgewidth=1.7,
                linestyle="none",
            )
            for metric in ("r2", "top1")
            if metric in metrics
        )
        handles.append(swatch if len(swatch) > 1 else swatch[0])
        labels.append(style["label"])
    return handles, labels


def _series(rows: list[dict], predictor: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["x"] for row in rows], dtype=float),
        np.asarray([row["arms"][predictor][metric] for row in rows], dtype=float),
    )


def _plot_panel(
    ax: plt.Axes,
    rows: list[dict],
    *,
    letter: str,
    title: str,
    kicker: str,
    show_retrieval: bool,
) -> None:
    style_score_axis(ax)
    panel_header(ax, letter, kicker, title, kicker_y=1.24, title_y=1.08)

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
        Line2D([0], [0], color=INK, marker="o", lw=3, label=METRIC_LABELS["r2"]),
        Line2D(
            [0],
            [0],
            color=INK,
            marker="o",
            markerfacecolor=PAPER,
            markeredgewidth=1.7,
            lw=2.4,
            linestyle=(0, (5.0, 3.8)),
            label=METRIC_LABELS["top1"],
        ),
    ]
    return predictors, metrics


def make_figure(
    layer: dict,
    scaling: dict,
    boundary: dict | None = None,
    extension: dict | None = None,
    baselines: dict | None = None,
    baselines_mode: str = "minimal",
) -> tuple[plt.Figure, float, dict | None]:
    set_c2a_style()
    # One layout. The baselines ride panel B as points rather than a third panel,
    # so the canvas keeps the compact A/B aspect.
    fig, include_frac = c2a_figure("full", aspect=0.36)
    grid = fig.add_gridspec(
        1,
        2,
        left=0.075,
        right=0.985,
        top=0.594,
        bottom=0.143,
        wspace=0.20,
    )
    ax_layer = fig.add_subplot(grid[0, 0])
    # With baselines, panel B is one axis per y-segment (see PANEL_B_SEGMENTS);
    # without them a single axis is enough. The curves and the panel header always
    # live on the highest segment, the x-axis always on the lowest.
    scale_axes = (
        _build_panel_b_axes(fig, grid[0, 1])
        if baselines is not None
        else [fig.add_subplot(grid[0, 1])]
    )
    ax_scale = scale_axes[-1]
    ax_scale_bottom = scale_axes[0]

    _plot_panel(
        ax_layer,
        layer["rows"],
        letter="A",
        title="Predictability across layers",
        kicker=f"{layer['n_train']:,} training contexts",
        show_retrieval=False,
    )
    _plot_panel(
        ax_scale,
        scaling["rows"],
        letter="B",
        title="Scaling with training data",
        kicker=f"layer {scaling['layer']}, 10,000-candidate retrieval",
        show_retrieval=True,
    )
    roster = PANEL_B_BASELINE_ROSTERS[baselines_mode] if baselines is not None else ()
    controls = boundary is not None or extension is not None or baselines is not None
    baseline_overlay = None
    if controls:
        _plot_controls(ax_scale, boundary, extension)
        if baselines is None:
            ax_scale.set_ylim(0.15, 1.0)
            ax_scale.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            _apply_panel_b_segments(scale_axes)
            # After the ranges are fixed: the overlay routes each value to the
            # segment that holds it, and drops any the panel still cannot show.
            baseline_overlay = _plot_baseline_points(scale_axes, baselines, roster)

    ax_layer.set_xlim(-0.5, 27.5)
    ax_layer.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax_layer.set_xlabel("Model layer", labelpad=12)

    ns = np.asarray([row["x"] for row in scaling["rows"]], dtype=float)
    # The strips share the x-axis, so range and scale propagate; only the lowest
    # one carries ticks and the label.
    ax_scale_bottom.set_xscale("log")
    ax_scale_bottom.set_xlim(ns.min() / 1.18, ns.max() * 1.18)
    ticks = [5_000, 25_000, 100_000, 500_000, 963_444]
    if ns.min() < 5_000:
        ticks = [int(ns.min()), 5_000, 25_000, 100_000, 963_444]
    for ax in scale_axes:
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(_human_n))
        ax.minorticks_off()
    ax_scale_bottom.set_xlabel("Training contexts", labelpad=12)

    ax_layer.set_ylabel(better_label(METRIC_LABELS["r2"]), labelpad=13)
    _center_ylabel(scale_axes, better_label("Score"))

    predictor_handles, metric_handles = _legend_handles()
    row_y = 0.936 if not controls else 0.851
    # Figure-level kicker: the model, right-aligned on the topmost kicker row.
    fig.text(
        0.985,
        0.995 if controls else row_y,
        "QWEN2.5-7B-INSTRUCT",
        color=MUTED,
        fontsize=11.5,
        fontweight=750,
        ha="right",
        va="center",
    )
    legend_kicker(fig, 0.075, row_y, "Predictor")
    fig.legend(
        handles=predictor_handles,
        loc="upper left",
        bbox_to_anchor=(0.074, row_y - 0.022),
        ncol=3,
        frameon=False,
        columnspacing=1.45,
        handlelength=2.1,
        handletextpad=0.65,
        borderaxespad=0,
    )
    legend_kicker(fig, 0.572, row_y, "Metric")
    fig.legend(
        handles=metric_handles,
        loc="upper left",
        bbox_to_anchor=(0.571, row_y - 0.022),
        ncol=2,
        frameon=False,
        columnspacing=1.35,
        handlelength=2.1,
        handletextpad=0.65,
        borderaxespad=0,
    )
    if controls:
        baseline_handles, baseline_labels = _baseline_legend_handles(boundary, baseline_overlay)
        legend_kicker(fig, 0.075, 0.995, "Baseline")
        fig.legend(
            handles=baseline_handles,
            labels=baseline_labels,
            loc="upper left",
            bbox_to_anchor=(0.074, 0.973),
            # Past four entries the row runs off the canvas (save_c2a_figure
            # enforces the width), so wrap instead of overflowing.
            ncol=min(len(baseline_handles), 4),
            frameon=False,
            columnspacing=1.2,
            handlelength=1.9,
            handletextpad=0.5,
            borderaxespad=0,
            # Paired swatches: draw both markers side by side rather than
            # overlaid, so the swatch mirrors the panel's dodge.
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.55)},
        )
    return fig, include_frac, baseline_overlay


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
    include_frac: float,
    git_state: dict[str, str | bool | None],
    boundary_source: Path | None = None,
    boundary: dict | None = None,
    extension_source: Path | None = None,
    extension: dict | None = None,
    baselines_source: Path | None = None,
    baselines: dict | None = None,
    pool10k_source: Path | None = None,
    pool10k: dict | None = None,
    baseline_overlay: dict | None = None,
) -> dict[str, Path]:
    stem = out_dir / stem_name
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Context-to-answer predictability and scaling",
        subject="Five-rollout layer sweep and training-data scaling",
        creator="scripts/make_paper_figure2.py",
        include_width=include_frac,
    )
    metadata = stem.with_suffix(".meta.json")
    metadata.write_text(
        json.dumps(
            {
                "status": "manuscript predictability-and-scaling figure",
                "style_version": STYLE_VERSION,
                "plotting_script": "scripts/make_paper_figure2.py",
                "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
                "rescore_script": "scripts/issue1901_figure2_five_rollout_scaling.py",
                "reproduction_command": "uv run python scripts/make_paper_figure2.py",
                "repository_manuscript_asset": _display_path(outputs["pdf"]),
                "overleaf_destination": "figures/paper/c1_predictability_scaling.pdf",
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
                    "pool10k": (
                        None
                        if pool10k_source is None
                        else {
                            "path": _display_path(pool10k_source),
                            "sha256": _sha256(pool10k_source),
                        }
                    ),
                },
                "retrieval_pool": (
                    None
                    if pool10k is None
                    else {
                        "n_pool": pool10k["n_pool"],
                        "n_query": pool10k["n_query"],
                        "chance_at_1": pool10k["chance_at_1"],
                        "note": (
                            "panel B draws top-1 retrieval among 10,000 candidates "
                            "(942 deduplicated targets + 9,058 distractors); held-out R^2 "
                            "is pool-independent and carried from the banked JSONs"
                        ),
                    }
                ),
                "render": outputs["record"],
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
                        "encoding": "burnt-umber horizontal line on panel B: mean held-out R^2 over the four exact-token maps",
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
                        "encoding": "copy-baseline R^2 source; the 1,200-context rung is not "
                        "drawn on panel B (its retrieval was only scored on the 942 pool, "
                        "while panel B draws the 10,000-candidate pool)",
                        **extension,
                    }
                ),
                "baselines_on_panel_b": (
                    None
                    if baselines is None
                    else {
                        "source": {
                            "path": _display_path(baselines_source),
                            "sha256": _sha256(baselines_source),
                        },
                        "encoding": (
                            "the separate baselines panel was removed; every baseline is now "
                            "a POINT on panel B at x = 25,000 training contexts, the rung the "
                            "baselines were measured on (layer 19, 10,000-candidate pool, the "
                            "same convention panel B plots), never a full-width horizontal "
                            "line, because a fitted baseline's value moves with n and only "
                            "this rung was measured; filled marker = held-out R^2 and open "
                            "marker = top-1 retrieval, matching the predictor curves; panel "
                            "B's y-axis is cut into the segments listed under "
                            "overlay.segments, whose plotted heights are proportional to "
                            "their data spans so both share one scale, which lets a negative "
                            "R^2 be shown at its own value rather than dropped; no curve is "
                            "severed by the cut (every predictor curve lies wholly inside the "
                            "upper segment) and only isolated baseline markers occupy the "
                            "lower one; the 'anchor' arm is not drawn because it IS panel B's "
                            "linear curve at this rung; an arm no segment holds is omitted "
                            "rather than clipped to a segment edge and is listed under "
                            "overlay.off_scale for the manuscript caption; every arm the "
                            "rescore computed but the figure omits stays in "
                            "extra_arms_not_drawn"
                        ),
                        "overlay": baseline_overlay,
                        **baselines,
                    }
                ),
                "output_sha256": {
                    kind: _sha256(path) for kind, path in outputs.items() if isinstance(path, Path)
                },
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
    parser.add_argument(
        "--no-boundary", action="store_true", help="render without the control overlay"
    )
    parser.add_argument("--extension-source", type=Path, default=DEFAULT_EXTENSION_SOURCE)
    parser.add_argument(
        "--no-extension", action="store_true", help="render without the 1,200 rung + copy baselines"
    )
    parser.add_argument("--baselines-source", type=Path, default=DEFAULT_BASELINES_SOURCE)
    parser.add_argument("--pool10k-source", type=Path, default=DEFAULT_POOL10K_SOURCE)
    parser.add_argument(
        "--no-baselines", action="store_true", help="render without the panel-B baseline points"
    )
    parser.add_argument(
        "--baselines-mode",
        choices=sorted(PANEL_B_BASELINE_ROSTERS),
        default="withnull",
        help="which baseline arms ride panel B (see PANEL_B_BASELINE_ROSTERS)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args()

    layer = _load_layer_data(args.layer_source)
    pool10k = _load_pool10k_data(args.pool10k_source)
    extension = None if args.no_extension else _load_extension_data(args.extension_source)
    scaling = _load_scaling_data(args.scaling_source, pool10k)
    boundary = None if args.no_boundary else _load_boundary_data(args.boundary_source)
    baselines = None
    if not args.no_baselines:
        if args.no_extension:
            raise SystemExit(
                "the baseline overlay reuses the extension copy-baseline R^2; drop --no-extension"
            )
        baselines = _load_baselines_data(args.baselines_source, extension, pool10k)
    assert layer["n_test"] == scaling["n_test"] == 1_000
    for dataset in (layer, scaling):
        for row in dataset["rows"]:
            for values in row["arms"].values():
                assert np.isfinite(values["r2"])
                assert 0.0 <= values["r2"] <= 1.0
                assert 0.0 <= values["retrieval"] <= 1.0
    if baselines is not None:
        for arm in baselines["arms"]:
            assert 0.0 <= arm["top1"] <= 1.0, arm
            # R^2 may be legitimately negative (copy baselines); never clipped, only bounded.
            assert arm["r2"] is None or (np.isfinite(arm["r2"]) and -10.0 < arm["r2"] <= 1.0), arm

    git_state = _git_state()
    fig, include_frac, baseline_overlay = make_figure(
        layer, scaling, boundary, extension, baselines, args.baselines_mode
    )
    outputs = _write_outputs(
        fig,
        args.out_dir,
        args.stem,
        args.layer_source,
        args.scaling_source,
        layer,
        scaling,
        include_frac,
        git_state,
        boundary_source=None if args.no_boundary else args.boundary_source,
        boundary=boundary,
        extension_source=None if args.no_extension else args.extension_source,
        extension=extension,
        baselines_source=None if baselines is None else args.baselines_source,
        baselines=baselines,
        pool10k_source=args.pool10k_source,
        pool10k=pool10k,
        baseline_overlay=baseline_overlay,
    )
    plt.close(fig)
    for kind, path in outputs.items():
        if isinstance(path, Path):
            print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
