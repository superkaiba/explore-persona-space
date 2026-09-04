#!/usr/bin/env python3
"""Render the chain-of-thought figure for the context-to-answer-map paper.

Reads the committed Issue #2546 fit cells (``eval_results/issue_2546/cells``)
and writes two figures: the main one-row figure
``figures/paper/c1_cot_maps.{pdf,png}`` (panels A-C) and the appendix figure
``figures/paper/c1_cot_reasoning_sft.{pdf,png}`` (the reasoning-SFT panel),
each with a grayscale audit PNG and a JSON sidecar carrying every plotted
value and its provenance.

Panels
------
A  OpenThinker3-7B, layer 19: held-out R^2 and answer retrieval for the maps
   to the answer vector from the last context token and from the end-of-thought
   token, split by corpus stratum.
B  OpenThinker3-7B, layer 19: the same two metrics for maps from the state at
   position t inside the thinking span to the fixed final answer state
   (t = 0 is the context state, t = 1 the end-of-thought state).
C  Qwen3-8B, layer 24: the same two maps with thinking on, plus the
   context -> answer map with thinking disabled on the same weights.
Appendix figure (``c1_cot_reasoning_sft``): Qwen2.5-7B-Instruct (before
reasoning training) -> OpenThinker3-7B (after): within-model maps on each side
and the cross-model context -> answer map.

Visual encoding follows Figure 2: color encodes the corpus stratum; solid bars
(and filled markers in panel B) are R^2, hatched open bars (and open markers)
are acc@1 of the question's own answer (paper recipe). Every context state is the last
context token; every fit uses five seeded random-row folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    STYLE_VERSION,
    c2a_figure,
    legend_kicker,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)

CELLS_DIR = ROOT / "eval_results" / "issue_2546" / "cells"
DEFAULT_OUT = ROOT / "figures" / "paper"
DEFAULT_STEM = "c1_cot_maps"
DEFAULT_SFT_STEM = "c1_cot_reasoning_sft"

# Provenance pins from the #2546 clean result.
HF_REVISION = "8368cc69f887d20931acd8c4d76c142275173728"
SOURCE_REF = "42308cc7522dcb0a2a76b332b0c24d981de4b585"
FIT_COMMIT = "76ac8d57c0a0c0da699b7a9d76f8c2e9daeb1564"

EXPECTED_ROWS = {
    (1, "does"): 13_116,
    (1, "doesnt"): 10_967,
    (3, "does"): 14_221,
    (3, "doesnt"): 12_855,
}
HEADLINE_LAYER = {1: 19, 3: 24}


@dataclass(frozen=True)
class StratumStyle:
    label: str
    color: str
    marker: str
    dx: float


STRATA = {
    "does": StratumStyle("Needs-reasoning corpora", "#7B3294", "s", -0.16),
    "doesnt": StratumStyle("No-reasoning corpora", "#5AAE61", "o", 0.16),
    "necessary": StratumStyle("CoT-necessary questions", "#7B3294", "s", -0.16),
    "both_correct": StratumStyle("CoT-unnecessary questions", "#5AAE61", "o", 0.16),
    "all": StratumStyle("All questions", "#4D4D4D", "D", 0.0),
}
BUNDLE_DIR = (
    ROOT / "eval_results" / "issue_2546" / "allfit"
)  # all-question refits scored per question label (--bundle-dir)
# The main figure plots the needs-reasoning stratum only; the strata comparison
# is its own figure (scripts/section45_cot_strata_figure.py). Both strata are
# still loaded, validated, and written to the JSON sidecar.
PLOT_STRATA = ("does",)
STRATUM_CELLS = ("does", "doesnt")  # the two corpus strata the production cells were fit on
STRATA_CORPORA_NOTE = {"does": "MATH, multi-step GSM8K, ContextHub levels 3\u20134"}

# (cell, x-axis label) per categorical panel.
PANEL_A_MAPS = [
    ("p7_A", "context"),
    ("p7_D", "end of\nthought"),
]
PANEL_C_MAPS = [
    ("p7_Aoff", "off"),
    ("p7_A", "on"),
    ("p7_D", "end of\nthought"),
]
PANEL_D_MAPS = [
    ("p8_G", "before\nmodel"),
    ("p8_E", "before →\nafter"),
    ("p8_F", "after\nmodel"),
]
# Panel B extra slot: the metamodel whose INPUT is the mean over all reasoning tokens
# (cell p7_F, scripts/issue2546_allfit_cotmean_cell.py). It is not a position in the
# trace, so it sits to the right of the 0..1 axis at its own tick. Penalty 1000 is the
# best held-out penalty of the 316/1000/3162 sweep on all rows for both labeled models
# (the three differ by at most 0.02 in R^2).
TRACE_MEAN_CELL = "p7_F_lam1000"
TRACE_MEAN_X = 1.24


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_path(cell: str, subset: str, arm: int) -> Path:
    return CELLS_DIR / f"{cell}__{subset}__a{arm}.json"


def _metrics_from_block(
    r2: float,
    r2_ci: dict[str, float],
    content: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "r2": float(r2),
        "r2_ci": [float(r2_ci["ci_lo"]), float(r2_ci["ci_hi"])],
        "retrieval_acc1": None,
        "retrieval_chance": None,
        "retrieval_lift": None,
        "retrieval_lift_ci": None,
    }
    if content is not None:
        pool = content["corpus_pool"]
        out.update(
            {
                "retrieval_acc1": float(pool["acc_at_1"]),
                "retrieval_chance": float(pool["chance_mean"]),
                "retrieval_lift": float(pool["lift"]),
                "retrieval_lift_ci": [float(pool["lift_ci_lo"]), float(pool["lift_ci_hi"])],
            }
        )
    return out


def load_cell(cell: str, subset: str, arm: int) -> dict[str, Any]:
    path = _cell_path(cell, subset, arm)
    data = json.loads(path.read_text())
    if data["status"] != "ok":
        raise ValueError(f"{path.name}: status {data['status']!r}")
    layer = int(data["headline_layer"])
    if layer != HEADLINE_LAYER[arm]:
        raise ValueError(f"{path.name}: headline layer {layer} != {HEADLINE_LAYER[arm]}")
    if data["cell_xy"]["x_kind"] not in {"cx_last", "cot_boundary"}:
        raise ValueError(f"{path.name}: unexpected input state {data['cell_xy']['x_kind']!r}")
    n_rows = int(data["n_rows"])
    if n_rows != EXPECTED_ROWS[(arm, subset)]:
        raise ValueError(f"{path.name}: n_rows {n_rows} != {EXPECTED_ROWS[(arm, subset)]}")
    if data["fit_params"]["n_folds"] != 5:
        raise ValueError(f"{path.name}: expected five folds")
    content = data.get("knn_content")
    content_block = content["euclidean"] if content else None
    metrics = _metrics_from_block(data["r2_headline"], data["r2_headline_bootstrap"], content_block)
    metrics.update(
        {
            "cell": cell,
            "subset": subset,
            "arm": arm,
            "n_rows": n_rows,
            "layer": layer,
            "input_state": data["cell_xy"]["x_kind"],
            "target": data["cell_xy"]["y_kind"],
            "identity_bias_r2": float(data["identity_bias_r2"][str(layer)]),
            "source": str(path.relative_to(ROOT)),
            "source_sha256": _sha256(path),
        }
    )
    return metrics


def load_trajectory(arm: int) -> dict[str, list[dict[str, Any]]]:
    path = CELLS_DIR / f"p7_traj__a{arm}.json"
    data = json.loads(path.read_text())
    if data["status"] != "ok":
        raise ValueError(f"{path.name}: status {data['status']!r}")
    if int(data["headline_layer"]) != HEADLINE_LAYER[arm]:
        raise ValueError(f"{path.name}: unexpected headline layer")
    t_grid = [float(t) for t in data["t_grid"]]
    series: dict[str, list[dict[str, Any]]] = {}
    for subset in STRATUM_CELLS:
        stratum = data["strata"][subset]
        if stratum["status"] != "ok" or int(stratum["n_rows"]) != EXPECTED_ROWS[(arm, subset)]:
            raise ValueError(f"{path.name}: stratum {subset} mismatch")
        rows = []
        start = load_cell("p7_A", subset, arm)
        rows.append({"t": 0.0, **{k: start[k] for k in _METRIC_KEYS}})
        for t in t_grid:
            key = f"t{int(round(t * 100))}"
            entry = stratum["per_t"][key]
            block = _metrics_from_block(
                entry["r2_headline"],
                entry["r2_headline_bootstrap"],
                entry["knn_content_euclidean"],
            )
            rows.append({"t": t, **block})
        end = load_cell("p7_D", subset, arm)
        rows.append({"t": 1.0, **{k: end[k] for k in _METRIC_KEYS}})
        series[subset] = rows
    return series


RECIPE_RESULTS = ROOT / "eval_results" / "issue_2546" / "retrieval_recipe" / "results.json"
RECIPE_RESULTS_ARM2 = (
    ROOT / "eval_results" / "issue_2546" / "retrieval_recipe" / "results_arm2.json"
)


def load_recipe_retrieval() -> dict[str, Any]:
    """acc@1 of the question's own answer under the paper recipe (whitened cosine + CSLS, held-out-fold pool)."""
    out: dict[str, Any] = {}
    for path in (RECIPE_RESULTS, RECIPE_RESULTS_ARM2):
        if path.is_file():
            out.update(json.loads(path.read_text())["results"])
    if not out:
        raise FileNotFoundError(f"recipe retrieval results missing: {RECIPE_RESULTS}")
    return out


def apply_recipe_retrieval(
    metrics: dict[str, Any],
    recipe: dict[str, Any],
    cell: str,
    subset: str,
    arm: int,
    position: str = "main",
) -> None:
    """Replace the content-rule lift fields with strict own-answer acc@1 (kept under the same keys so the plot code is unchanged)."""
    entry = recipe[f"arm{arm}/{cell}/{subset}"]["positions"][position]
    metrics["retrieval_acc1"] = float(entry["acc1_whitened_csls"])
    metrics["retrieval_chance"] = float(entry["chance"])
    metrics["retrieval_lift"] = float(entry["acc1_whitened_csls"])
    metrics["retrieval_lift_ci"] = [float(v) for v in entry["acc1_whitened_csls_ci"]]
    metrics["retrieval_raw_cosine_acc1"] = float(entry["acc1_raw_cosine"])
    metrics["retrieval_recipe"] = "whitened cosine + CSLS, held-out-fold pool, own-answer hit"


_METRIC_KEYS = (
    "r2",
    "r2_ci",
    "retrieval_acc1",
    "retrieval_chance",
    "retrieval_lift",
    "retrieval_lift_ci",
)


def load_results(*, retrieval: str = "recipe") -> dict[str, Any]:
    panels: dict[str, Any] = {}
    recipe = load_recipe_retrieval() if retrieval == "recipe" else None
    panels["A"] = {
        "arm": 1,
        "model": "open-thoughts/OpenThinker3-7B",
        "maps": [
            {"cell": cell, "label": label, **{s: load_cell(cell, s, 1) for s in STRATUM_CELLS}}
            for cell, label in PANEL_A_MAPS
        ],
    }
    panels["B"] = {
        "arm": 1,
        "model": "open-thoughts/OpenThinker3-7B",
        "series": load_trajectory(1),
    }
    panels["C"] = {
        "arm": 3,
        "model": "Qwen/Qwen3-8B",
        "maps": [
            {"cell": cell, "label": label, **{s: load_cell(cell, s, 3) for s in STRATUM_CELLS}}
            for cell, label in PANEL_C_MAPS
        ],
    }
    panels["D"] = {
        "arm": 1,
        "model_before": "Qwen/Qwen2.5-7B-Instruct",
        "model_after": "open-thoughts/OpenThinker3-7B",
        "maps": [
            {"cell": cell, "label": label, **{s: load_cell(cell, s, 1) for s in STRATUM_CELLS}}
            for cell, label in PANEL_D_MAPS
        ],
    }
    if recipe is not None:
        for panel_key in ("A", "C", "D"):
            arm = panels[panel_key]["arm"]
            for entry in panels[panel_key]["maps"]:
                for subset in STRATUM_CELLS:
                    apply_recipe_retrieval(entry[subset], recipe, entry["cell"], subset, arm)
        arm = panels["B"]["arm"]
        for subset, rows in panels["B"]["series"].items():
            for row in rows:
                t = row["t"]
                if t == 0.0:
                    apply_recipe_retrieval(row, recipe, "p7_A", subset, arm)
                elif t == 1.0:
                    apply_recipe_retrieval(row, recipe, "p7_D", subset, arm)
                else:
                    apply_recipe_retrieval(
                        row, recipe, "p7_traj", subset, arm, position=f"t{int(round(t * 100))}"
                    )
        panels["retrieval_recipe"] = (
            "whitened cosine + CSLS (k=10), whitening fit on train-fold answers (shrinkage 0.1), pool = held-out fold, hit = own answer"
        )
    return panels


def _bundle_metrics(block: dict[str, Any], baseline: str) -> dict[str, Any]:
    return {
        "r2": float(block[f"r2_{baseline}"]),
        "r2_ci": [float(v) for v in block[f"r2_{baseline}_ci"]],
        "retrieval_acc1": float(block["acc1"]),
        "retrieval_chance": float(block["chance"]),
        "retrieval_lift": float(block["acc1"]),
        "retrieval_lift_ci": [float(v) for v in block["acc1_ci"]],
        "n_rows": int(block["n"]),
        "r2_baseline": baseline,
    }


def load_bundle(bundle_dir: Path, subset: str, baseline: str) -> dict[str, Any]:
    """Panels from the all-question refits (allfit_necessity.py): one fit per map on every question, scored on `subset`."""
    bundle_dir = bundle_dir.resolve()

    def cell(name: str, arm: int) -> dict[str, Any]:
        src = name if name != "p8_F" else "p7_A"  # the post model's own map is the same fit as p7_A
        data = json.loads((bundle_dir / f"{src}__a{arm}.json").read_text())
        out = _bundle_metrics(data["subsets"][subset], baseline)
        out.update(
            {
                "cell": name,
                "subset": subset,
                "arm": arm,
                "layer": data["layer"],
                "lambda": data["lambda"],
                "source": str((bundle_dir / f"{src}__a{arm}.json").relative_to(ROOT)),
            }
        )
        return out

    def series(arm: int) -> list[dict[str, Any]]:
        traj = json.loads((bundle_dir / f"p7_traj__a{arm}.json").read_text())
        rows = [{"t": 0.0, **{k: v for k, v in cell("p7_A", arm).items() if k in _METRIC_KEYS}}]
        for key, block in sorted(traj["positions"].items(), key=lambda kv: int(kv[0][1:])):
            rows.append(
                {
                    "t": int(key[1:]) / 100.0,
                    **{
                        k: v
                        for k, v in _bundle_metrics(block[subset], baseline).items()
                        if k in _METRIC_KEYS
                    },
                }
            )
        rows.append({"t": 1.0, **{k: v for k, v in cell("p7_D", arm).items() if k in _METRIC_KEYS}})
        return rows

    panels: dict[str, Any] = {
        "A": {
            "arm": 1,
            "model": "open-thoughts/OpenThinker3-7B",
            "maps": [{"cell": c, "label": l, subset: cell(c, 1)} for c, l in PANEL_A_MAPS],
        },
        "B": {
            "arm": 1,
            "model": "open-thoughts/OpenThinker3-7B",
            "series": {subset: series(1)},
            # input = mean over all reasoning tokens (not a trace position); drawn at its own slot
            "trace_mean": {subset: cell(TRACE_MEAN_CELL, 1)},
        },
        "C": {
            "arm": 3,
            "model": "Qwen/Qwen3-8B",
            "maps": [{"cell": c, "label": l, subset: cell(c, 3)} for c, l in PANEL_C_MAPS],
        },
        "D": {
            "arm": 1,
            "model_before": "Qwen/Qwen2.5-7B-Instruct",
            "model_after": "open-thoughts/OpenThinker3-7B",
            "maps": [{"cell": c, "label": l, subset: cell(c, 1)} for c, l in PANEL_D_MAPS],
        },
        "retrieval_recipe": "whitened cosine + CSLS (k=10), whitening fit on train-fold answers (shrinkage 0.1), pool = held-out fold (all questions), hit = own answer",
        "fit": f"one ridge fit per map on all questions of seven benchmarks; metrics on the {subset!r} label subset; R^2 baseline = {baseline} mean",
    }
    return panels


def _kicker(
    ax: plt.Axes,
    title: str,
    kicker: str,
    *,
    kicker_y: float = 1.16,
    title_y: float = 1.055,
) -> None:
    """Panel letter + uppercase kicker + descriptive title through the shared c2a-v2 header helper."""
    letter, sep, rest = kicker.partition("  ·  ")
    if not sep:
        letter, rest = "", kicker
    panel_header(ax, letter, rest, title, kicker_y=kicker_y, title_y=title_y)


BAR_WIDTH = 0.19
# Within each x category: [R^2, retrieval] for the single plotted stratum.
BAR_OFFSETS = {"does": (-0.11, 0.11), "doesnt": (-0.11, 0.11)}
RETRIEVAL_HATCH = "///"


def _bar_with_ci(
    ax: plt.Axes,
    x: float,
    value: float,
    ci: list[float],
    style: StratumStyle,
    *,
    retrieval: bool,
    width: float = BAR_WIDTH,
) -> None:
    if retrieval:
        ax.bar(
            x,
            value,
            width=width,
            facecolor=PAPER,
            edgecolor=style.color,
            hatch=RETRIEVAL_HATCH,
            linewidth=1.4,
            zorder=3,
        )
    else:
        ax.bar(x, value, width=width, color=style.color, linewidth=0, zorder=3)
    lo, hi = ci
    ax.errorbar(
        x,
        value,
        yerr=[[value - lo], [hi - value]],
        fmt="none",
        ecolor=INK,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=4,
    )


GROUP_GAP = 0.8  # x gap between the R^2 group and the retrieval group


def _categorical_panel(
    ax: plt.Axes,
    maps: list[dict[str, Any]],
    *,
    group_label_y: float = -0.30,
    tick_fontsize: float = 12,
) -> None:
    """Bars grouped by metric: all held-out R^2 bars on the left, all retrieval-lift bars on the right."""
    style_score_axis(ax, y_min=0.0, y_max=1.0, y_step=0.2)
    m = len(maps)
    width = 0.62
    ticks: list[float] = []
    labels: list[str] = []
    for subset in PLOT_STRATA:
        style = STRATA[subset]
        for i, entry in enumerate(maps):
            metrics = entry[subset]
            x_r2 = float(i)
            _bar_with_ci(
                ax, x_r2, metrics["r2"], metrics["r2_ci"], style, retrieval=False, width=width
            )
            ticks.append(x_r2)
            labels.append(entry["label"])
            if metrics["retrieval_lift"] is not None:
                x_ret = m + GROUP_GAP + i
                _bar_with_ci(
                    ax,
                    x_ret,
                    metrics["retrieval_lift"],
                    metrics["retrieval_lift_ci"],
                    style,
                    retrieval=True,
                    width=width,
                )
                ticks.append(x_ret)
                labels.append(entry["label"])
    x_max = m + GROUP_GAP + m - 1
    ax.set_xlim(-0.6, x_max + 0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=tick_fontsize, linespacing=1.15)
    ax.axvline(m - 0.5 + GROUP_GAP / 2, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    for center, text in (
        ((m - 1) / 2, "Held-out $R^2$"),
        (m + GROUP_GAP + (m - 1) / 2, "Top-1 retrieval"),
    ):
        ax.text(
            center,
            group_label_y,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
            fontweight=700,
            color=MUTED,
        )


def _trajectory_panel(
    ax: plt.Axes,
    series: dict[str, list[dict[str, Any]]],
    trace_mean: dict[str, dict[str, Any]] | None = None,
) -> None:
    """R^2 (solid, filled) and top-1 retrieval (dashed, open) along the trace; optional extra slot at
    TRACE_MEAN_X for the metamodel whose input is the mean over all reasoning tokens (whiskers = CI)."""
    style_score_axis(ax, y_min=0.0, y_max=1.0, y_step=0.2)
    for subset in PLOT_STRATA:
        style = STRATA[subset]
        rows = series[subset]
        t = np.asarray([row["t"] for row in rows])
        r2 = np.asarray([row["r2"] for row in rows])
        r2_lo = np.asarray([row["r2_ci"][0] for row in rows])
        r2_hi = np.asarray([row["r2_ci"][1] for row in rows])
        lift = np.asarray([row["retrieval_lift"] for row in rows])
        lift_lo = np.asarray([row["retrieval_lift_ci"][0] for row in rows])
        lift_hi = np.asarray([row["retrieval_lift_ci"][1] for row in rows])
        ax.fill_between(t, r2_lo, r2_hi, color=style.color, alpha=0.14, lw=0, zorder=1)
        ax.plot(
            t,
            r2,
            color=style.color,
            marker=style.marker,
            markersize=7.5,
            lw=2.8,
            zorder=4,
        )
        ax.fill_between(t, lift_lo, lift_hi, color=style.color, alpha=0.14, lw=0, zorder=1)
        ax.plot(
            t,
            lift,
            color=style.color,
            marker=style.marker,
            markerfacecolor=PAPER,
            markeredgecolor=style.color,
            markeredgewidth=1.9,
            markersize=8,
            lw=2.3,
            linestyle=(0, (5.0, 3.8)),
            zorder=3,
        )
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0\nend of\nprompt", "0.2", "0.4", "0.6", "0.8", "1\nend of\nthought"]
    x_max = 1.04
    if trace_mean is not None:
        for subset in PLOT_STRATA:
            style = STRATA[subset]
            m = trace_mean[subset]
            r2_err = [[m["r2"] - m["r2_ci"][0]], [m["r2_ci"][1] - m["r2"]]]
            lift_err = [
                [m["retrieval_lift"] - m["retrieval_lift_ci"][0]],
                [m["retrieval_lift_ci"][1] - m["retrieval_lift"]],
            ]
            ax.errorbar(
                [TRACE_MEAN_X],
                [m["r2"]],
                yerr=r2_err,
                fmt=style.marker,
                color=style.color,
                markersize=7.5,
                capsize=3,
                lw=1.6,
                zorder=4,
            )
            ax.errorbar(
                [TRACE_MEAN_X],
                [m["retrieval_lift"]],
                yerr=lift_err,
                fmt=style.marker,
                color=style.color,
                markerfacecolor=PAPER,
                markeredgecolor=style.color,
                markeredgewidth=1.9,
                markersize=8,
                capsize=3,
                lw=1.6,
                zorder=3,
            )
        ticks.append(TRACE_MEAN_X)
        labels.append("mean\nover\ntrace")
        x_max = TRACE_MEAN_X + 0.08
    ax.set_xlim(-0.04, x_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=13.5, linespacing=1.15)
    ax.set_xlabel("Position in the thinking span", labelpad=8)


def _legend_handles() -> tuple[list[Patch], list[Patch]]:
    strata = [
        Patch(
            facecolor=STRATA[subset].color,
            edgecolor=STRATA[subset].color,
            label=f"{STRATA[subset].label} only",
        )
        for subset in PLOT_STRATA
    ]
    metrics = [
        Patch(facecolor=INK, edgecolor=INK, label="Held-out $R^2$"),
        Patch(
            facecolor=PAPER,
            edgecolor=INK,
            hatch=RETRIEVAL_HATCH,
            linewidth=1.4,
            label="Top-1 retrieval",
        ),
    ]
    return strata, metrics


ERROR_BAR_NOTE = "Error bars: 95% bootstrap CI (1,000 prompt-level draws)"


def _legend_strip(
    fig: plt.Figure,
    blocks: list[tuple[str, Any]],
    *,
    y: float,
    x0: float,
    gap_in: float = 0.45,
) -> None:
    """One horizontal legend line: kickers, legends, and notes placed left-to-right.

    ``blocks`` is a list of ``("kicker", str)`` / ``("legend", handles)`` /
    ``("note", str)`` entries.  Each block is placed at the running x (figure
    fraction), measured with the live renderer, and the next block starts after
    it, so the strip adapts to label lengths.
    """

    width_in = fig.get_figwidth()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x = x0
    for kind, payload in blocks:
        if kind == "kicker":
            legend_kicker(fig, x, y, payload)
            artist = fig.texts[-1]
            pad_in = 0.14
        elif kind == "legend":
            artist = fig.legend(
                handles=payload,
                loc="center left",
                bbox_to_anchor=(x, y),
                ncol=len(payload),
                frameon=False,
                columnspacing=1.1,
                handlelength=1.6,
                handletextpad=0.6,
                borderaxespad=0,
            )
            pad_in = gap_in
        elif kind == "note":
            artist = fig.text(x, y, payload, color=MUTED, fontsize=12.5, ha="left", va="center")
            pad_in = gap_in
        else:
            raise ValueError(f"unknown legend-strip block kind {kind!r}")
        fig.canvas.draw()
        bbox = artist.get_window_extent(renderer)
        x = bbox.x1 / (fig.dpi * width_in) + pad_in / width_in


def make_figure(panels: dict[str, Any]) -> plt.Figure:
    """Main one-row figure: panels A-C at full text width (~1.85 in printed)."""
    set_c2a_style()
    fig, _include_frac = c2a_figure("full", aspect=0.34)  # c2a-v2: full text width
    grid = fig.add_gridspec(
        1,
        3,
        left=0.07,
        right=0.99,
        top=0.70,
        bottom=0.245,
        wspace=0.26,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    _categorical_panel(ax_a, panels["A"]["maps"], group_label_y=-0.26)
    _kicker(
        ax_a,
        "Context vs end of thought",
        "A  ·  OpenThinker3-7B, layer 19",
        kicker_y=1.24,
        title_y=1.07,
    )
    _trajectory_panel(ax_b, panels["B"]["series"], trace_mean=panels["B"].get("trace_mean"))
    _kicker(
        ax_b,
        "Inside the reasoning trace",
        "B  ·  OpenThinker3-7B, layer 19",
        kicker_y=1.24,
        title_y=1.07,
    )
    _categorical_panel(ax_c, panels["C"]["maps"], group_label_y=-0.26)
    _kicker(
        ax_c,
        "Thinking on vs off",
        "C  ·  Qwen3-8B, layer 24",
        kicker_y=1.24,
        title_y=1.07,
    )

    ax_a.set_ylabel("Held-out score  ↑", labelpad=10)

    strata_handles, metric_handles = _legend_handles()
    _legend_strip(
        fig,
        [
            ("kicker", "Questions"),
            ("legend", strata_handles),
            ("kicker", "Metric"),
            ("legend", metric_handles),
        ],
        y=0.965,
        x0=0.07,
    )
    _legend_strip(fig, [("note", ERROR_BAR_NOTE)], y=0.885, x0=0.07)
    return fig


def make_reasoning_sft_figure(panels: dict[str, Any]) -> plt.Figure:
    """Appendix figure: the reasoning-SFT panel alone at 0.75 text width."""
    set_c2a_style()
    fig, _include_frac = c2a_figure("wide", aspect=0.55)  # c2a-v2: 0.75 text width
    grid = fig.add_gridspec(1, 1, left=0.09, right=0.985, top=0.72, bottom=0.155)
    ax = fig.add_subplot(grid[0, 0])

    _categorical_panel(ax, panels["D"]["maps"], group_label_y=-0.20)
    panel_header(
        ax,
        "",
        "Qwen2.5-7B-Instruct → OpenThinker3-7B, layer 19",
        "Within and across reasoning SFT",
        kicker_y=1.17,
        title_y=1.05,
    )
    ax.set_ylabel("Held-out score  ↑", labelpad=10)

    _, metric_handles = _legend_handles()
    _legend_strip(
        fig,
        [("kicker", "Metric"), ("legend", metric_handles)],
        y=0.955,
        x0=0.09,
    )
    _legend_strip(fig, [("note", ERROR_BAR_NOTE)], y=0.885, x0=0.09)
    return fig


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
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


def _repo_relative(path: Path) -> str:
    """Path relative to the repo root, or absolute for scratch out-dirs (e.g. /tmp)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_outputs(
    fig: plt.Figure,
    panels: dict[str, Any],
    out_dir: Path,
    stem_name: str,
    font: str,
    *,
    title: str = "Chain of thought and the context-to-answer map",
) -> dict[str, Path]:
    stem = out_dir / stem_name
    outputs = save_c2a_figure(
        fig,
        stem,
        title=title,
        subject="Issue #2546 cells rendered for the paper",
        creator="scripts/section45_cot_figure.py",
    )
    render = outputs.pop("record")
    sidecar = stem.with_name(f"{stem_name}_data.json")
    payload = {
        "style_version": STYLE_VERSION,
        "font": font,
        "git": _git_state(),
        "provenance": {
            "task": 2546,
            "hf_data_repo": "superkaiba1/explore-persona-space-data",
            "hf_revision": HF_REVISION,
            "source_ref": SOURCE_REF,
            "fit_commit": FIT_COMMIT,
            "cells_dir": str(CELLS_DIR.relative_to(ROOT)),
            "context_state": "last context token (cx_last); end-of-thought state for the boundary maps",
            "folds": "five seeded random-row folds (seed 0), inner-group CV lambda selection",
            "retrieval": (
                "paper recipe (Section 4.1): whitened cosine (train-fold shrunk covariance, lambda 0.1) + CSLS k=10, "
                "pool = held-out fold, hit = the question's own answer; source eval_results/issue_2546/retrieval_recipe/"
            ),
            "uncertainty": "1,000 paired prompt-level bootstrap draws (95% interval)",
        },
        "panels": panels,
        "render": render,
        "outputs": {key: _repo_relative(path) for key, path in outputs.items()},
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    outputs["data"] = sidecar
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--sft-stem",
        default=DEFAULT_SFT_STEM,
        help="stem of the appendix reasoning-SFT figure (the second output of this script)",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="read the all-question refit bundle from this dir instead of the stratum cells (default: stratum cells)",
    )
    parser.add_argument(
        "--subset",
        default="necessary",
        help="label subset to plot from the bundle: necessary, both_correct, or all",
    )
    parser.add_argument(
        "--r2-baseline",
        choices=("corpus", "global"),
        default="corpus",
        help="R^2 baseline for bundle metrics: each question's corpus mean (default) or the global mean",
    )
    args = parser.parse_args(argv)

    global PLOT_STRATA
    if args.bundle_dir is not None:
        PLOT_STRATA = (args.subset,)
        panels = load_bundle(args.bundle_dir, args.subset, args.r2_baseline)
    else:
        panels = load_results()
    font = set_c2a_style()
    shared = {k: panels[k] for k in ("retrieval_recipe", "fit") if k in panels}

    fig = make_figure(panels)
    main_panels = {k: panels[k] for k in ("A", "B", "C")} | shared
    outputs = write_outputs(fig, main_panels, args.out_dir, args.stem, font)
    plt.close(fig)
    for key, path in outputs.items():
        print(f"{key}: {path}")

    sft_fig = make_reasoning_sft_figure(panels)
    sft_panels = {"D": panels["D"]} | shared
    sft_outputs = write_outputs(
        sft_fig,
        sft_panels,
        args.out_dir,
        args.sft_stem,
        font,
        title="Reasoning SFT and the context-to-answer map",
    )
    plt.close(sft_fig)
    for key, path in sft_outputs.items():
        print(f"sft_{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
