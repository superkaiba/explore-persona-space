#!/usr/bin/env python3
"""Render the chain-of-thought figure for the context-to-answer-map paper.

Reads the committed Issue #2546 fit cells (``eval_results/issue_2546/cells``)
and writes ``figures/paper/c1_cot_maps.{pdf,png}``, a grayscale audit PNG, and a
JSON sidecar with every plotted value and its provenance.

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
D  Qwen2.5-7B-Instruct (before reasoning training) -> OpenThinker3-7B (after):
   within-model maps on each side and the cross-model context -> answer map.

Visual encoding follows Figure 2: color encodes the corpus stratum; solid bars
(and filled markers in panel B) are R^2, hatched open bars (and open markers)
are top-1 answer retrieval as lift over chance. Every context state is the last
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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    STYLE_VERSION,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)

CELLS_DIR = ROOT / "eval_results" / "issue_2546" / "cells"
DEFAULT_OUT = ROOT / "figures" / "paper"
DEFAULT_STEM = "c1_cot_maps"

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
}
# The main figure plots the needs-reasoning stratum only; the strata comparison
# is its own figure (scripts/section45_cot_strata_figure.py). Both strata are
# still loaded, validated, and written to the JSON sidecar.
PLOT_STRATA = ("does",)
STRATA_CORPORA_NOTE = {"does": "MATH, multi-step GSM8K, ContextHub levels 3\u20134"}

# (cell, x-axis label) per categorical panel.
PANEL_A_MAPS = [
    ("p7_A", "context"),
    ("p7_D", "end of\nthought"),
]
PANEL_C_MAPS = [
    ("p7_Aoff", "context,\nthinking off"),
    ("p7_A", "context,\nthinking on"),
    ("p7_D", "end of\nthought"),
]
PANEL_D_MAPS = [
    ("p8_G", "before\nmodel"),
    ("p8_E", "before →\nafter"),
    ("p8_F", "after\nmodel"),
]


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
    metrics = _metrics_from_block(
        data["r2_headline"], data["r2_headline_bootstrap"], content_block
    )
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
    for subset in STRATA:
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


_METRIC_KEYS = (
    "r2",
    "r2_ci",
    "retrieval_acc1",
    "retrieval_chance",
    "retrieval_lift",
    "retrieval_lift_ci",
)


def load_results() -> dict[str, Any]:
    panels: dict[str, Any] = {}
    panels["A"] = {
        "arm": 1,
        "model": "open-thoughts/OpenThinker3-7B",
        "maps": [
            {"cell": cell, "label": label, **{s: load_cell(cell, s, 1) for s in STRATA}}
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
            {"cell": cell, "label": label, **{s: load_cell(cell, s, 3) for s in STRATA}}
            for cell, label in PANEL_C_MAPS
        ],
    }
    panels["D"] = {
        "arm": 1,
        "model_before": "Qwen/Qwen2.5-7B-Instruct",
        "model_after": "open-thoughts/OpenThinker3-7B",
        "maps": [
            {"cell": cell, "label": label, **{s: load_cell(cell, s, 1) for s in STRATA}}
            for cell, label in PANEL_D_MAPS
        ],
    }
    return panels


def _kicker(ax: plt.Axes, title: str, kicker: str) -> None:
    ax.set_title(title, loc="left", y=1.04, pad=0, fontweight=650, fontsize=19)
    ax.text(
        0.0,
        1.235,
        kicker.upper(),
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight=700,
        color=MUTED,
        va="bottom",
        ha="left",
    )


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


def _categorical_panel(ax: plt.Axes, maps: list[dict[str, Any]], *, separator_after: int | None = None) -> None:
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
            _bar_with_ci(ax, x_r2, metrics["r2"], metrics["r2_ci"], style, retrieval=False, width=width)
            ticks.append(x_r2); labels.append(entry["label"])
            if metrics["retrieval_lift"] is not None:
                x_ret = m + GROUP_GAP + i
                _bar_with_ci(ax, x_ret, metrics["retrieval_lift"], metrics["retrieval_lift_ci"], style, retrieval=True, width=width)
                ticks.append(x_ret); labels.append(entry["label"])
    x_max = m + GROUP_GAP + m - 1
    ax.set_xlim(-0.6, x_max + 0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12, linespacing=1.15)
    ax.axvline(m - 0.5 + GROUP_GAP / 2, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=1)
    for center, text in (((m - 1) / 2, "Held-out $R^2$"), (m + GROUP_GAP + (m - 1) / 2, "Answer retrieval (lift over chance)")):
        ax.text(center, -0.30, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12, fontweight=700, color=MUTED)


def _trajectory_panel(ax: plt.Axes, series: dict[str, list[dict[str, Any]]]) -> None:
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
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(
        ["0\nend of\nprompt", "0.2", "0.4", "0.6", "0.8", "1\nend of\nthought"],
        fontsize=13.5,
        linespacing=1.15,
    )
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
            label="Answer retrieval (lift over chance)",
        ),
    ]
    return strata, metrics


def make_figure(panels: dict[str, Any]) -> plt.Figure:
    set_c2a_style()
    fig = plt.figure(figsize=(14.4, 13.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.985,
        top=0.845,
        bottom=0.12,
        wspace=0.22,
        hspace=1.15,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    _categorical_panel(ax_a, panels["A"]["maps"])
    _kicker(
        ax_a,
        "The context state predicts the answer state,\nbut retrieves the specific answer poorly",
        "A  ·  OpenThinker3-7B, layer 19",
    )
    _trajectory_panel(ax_b, panels["B"]["series"])
    _kicker(
        ax_b,
        "Predictability dips inside the reasoning trace\nand recovers only at the end of thought",
        "B  ·  OpenThinker3-7B, layer 19",
    )
    _categorical_panel(ax_c, panels["C"]["maps"], separator_after=0)
    _kicker(
        ax_c,
        "Disabling thinking on the same weights\nleaves $R^2$ unchanged",
        "C  ·  Qwen3-8B, layer 24",
    )
    _categorical_panel(ax_d, panels["D"]["maps"])
    _kicker(
        ax_d,
        "The context state before reasoning training\npredicts the answer state after it",
        "D  ·  Qwen2.5-7B-Instruct (before) → OpenThinker3-7B (after), layer 19",
    )

    for ax in (ax_a, ax_c):
        ax.set_ylabel("Held-out score  ↑", labelpad=12)
    for ax in (ax_b, ax_d):
        ax.set_ylabel("Held-out score  ↑", labelpad=12)

    strata_handles, metric_handles = _legend_handles()
    fig.text(0.07, 0.965, "CORPORA", color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center")
    fig.legend(
        handles=strata_handles,
        loc="upper left",
        bbox_to_anchor=(0.069, 0.952),
        ncol=2,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.6,
        handletextpad=0.6,
        borderaxespad=0,
    )
    fig.text(0.585, 0.965, "METRIC", color=MUTED, fontsize=11.5, fontweight=750, ha="left", va="center")
    fig.legend(
        handles=metric_handles,
        loc="upper left",
        bbox_to_anchor=(0.584, 0.952),
        ncol=2,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.6,
        handletextpad=0.6,
        borderaxespad=0,
    )
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


def write_outputs(fig: plt.Figure, panels: dict[str, Any], out_dir: Path, stem_name: str, font: str) -> dict[str, Path]:
    stem = out_dir / stem_name
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Chain of thought and the context-to-answer map",
        subject="Issue #2546 cells rendered for the paper",
        creator="scripts/section45_cot_figure.py",
    )
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
                "top-1 nearest neighbour (euclidean) in the held-out within-corpus pool; a hit "
                "requires the same canonical answer content; lift = acc@1 minus per-query chance"
            ),
            "uncertainty": "1,000 paired prompt-level bootstrap draws (95% interval)",
        },
        "panels": panels,
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    outputs["data"] = sidecar
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args(argv)

    panels = load_results()
    font = set_c2a_style()
    fig = make_figure(panels)
    outputs = write_outputs(fig, panels, args.out_dir, args.stem, font)
    plt.close(fig)
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
