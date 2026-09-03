#!/usr/bin/env python3
"""Render the Section 4.3 post-training summary figure (paper Figure 7).

Panel A (within-stage fits + retrieval) reads the committed last-token
comparison artifacts under ``figures/issue_1902/section43/inputs``. Panels B
(each stage's answers predicted from its own states, the previous stage's
states, and Base states) and C (adjacent-stage transfer retention: as is /
bias / scalar rescaling + bias) read ONLY
``eval_results/issue_1902/lasttoken_transfer/summary.json`` — the last-token
IID recompute (u_last states, six size-matched random-row folds), replacing
the registered prompt-mean / semantic-fold numbers.

Two variants are written to ``figures/paper``: the paper figure
``c1_posttraining_dynamics`` (three-series line panel B, per the figure
standard's series spec) and ``c1_posttraining_dynamics_grid`` (the 4x4
activation-checkpoint x answer-source heatmap panel B, kept as a variant).
Each variant ships a ``.meta.json`` sidecar embedding the ``save_c2a_figure``
record; both figures are reproducible from
``c1_posttraining_dynamics_data.json`` alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below. On the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and
# the BLAS pools freeze at import time.
load_dotenv()

import matplotlib as mpl  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    METRIC_LABELS,
    MUTED,
    PAPER,
    ROLES,
    SEAM,
    better_label,
    c2a_figure,
    legend_kicker,
    metric_style,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
    style_score_axis,
)

STAGES = ["Base", "SFT", "DPO", "RLVR"]
STAGE_KEYS = ["B", "S", "D", "R"]
TRANSITIONS = ["B->S", "S->D", "D->R"]
LATEST_SHA256 = "947a46cbde8f834be0a094b85e401e73a260f8da0416dd76285ca9da12c03c96"
RETRIEVAL_SHA256 = "d74f9adb9d52e36558d38c99c5b83ed636b731f5b46c6a5ab6b000c920f6bd91"
SUMMARY_PATH = PROJECT_ROOT / "eval_results" / "issue_1902" / "lasttoken_transfer" / "summary.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures" / "paper"
HF_REVISION = "3256c8efcef5f10ca525efeb2039636eaec8fad7"


def _tint(color: str, toward_white: float) -> str:
    """Blend a palette hue toward white (0 = unchanged, 1 = white)."""
    r, g, b = mcolors.to_rgb(color)
    blend = (c + (1.0 - c) * toward_white for c in (r, g, b))
    return mcolors.to_hex(tuple(blend))


# Semantic palette (figure standard section 2.4). The post-trained map shares
# the linear-map teal; Base context states take the base-model amber; the
# previous stage's states are a 45%-toward-white tint of the post_trained teal
# with a distinct marker (triangle) — a variant of the same model family.
TEAL = ROLES["post_trained"].color
AMBER = ROLES["base_model"].color
PREV_TINT = _tint(TEAL, 0.45)
SERIES_ENCODING = {
    "panel_a": (
        "one teal hue (post_trained); metric by fill: held-out R^2 = solid line / "
        "filled circle, top-1 retrieval = dashed line / open circle"
    ),
    "panel_b_lines": (
        "own states = post_trained teal, filled circle; previous-stage states = "
        f"45%-toward-white tint of the same teal ({PREV_TINT}), filled triangle; "
        "Base states = base_model amber, filled square; all solid (every series is R^2)"
    ),
    "panel_c": (
        "one teal hue, fill progression open ('as is', circle) -> half tint "
        f"({_tint(TEAL, 0.5)}, 'bias', square) -> filled ('rescaling + bias', circle), "
        "matching the c1_cot_ladder convention"
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_results() -> dict[str, Any]:
    """Collect every plotted value into one JSON-serializable dict."""
    # Panel A sources (unchanged): committed last-token comparison artifacts.
    latest_path = (
        PROJECT_ROOT
        / "figures"
        / "issue_1902"
        / "section43"
        / "inputs"
        / "c1_posttraining_dynamics_data.json"
    )
    latest_bytes = latest_path.read_bytes()
    if _sha256_bytes(latest_bytes) != LATEST_SHA256:
        msg = f"Unexpected contents in {latest_path}"
        raise ValueError(msg)
    latest_payload = json.loads(latest_bytes)
    retrieval_path = latest_path.with_name("c1_posttraining_retrieval_data.json")
    retrieval_bytes = retrieval_path.read_bytes()
    if _sha256_bytes(retrieval_bytes) != RETRIEVAL_SHA256:
        msg = f"Unexpected contents in {retrieval_path}"
        raise ValueError(msg)
    retrieval_payload = json.loads(retrieval_bytes)

    latest_cells = latest_payload["cells"]
    last_cells = [latest_cells[f"{key}_single"] for key in STAGE_KEYS]
    iid_r2 = [cell["u_last_random_fold_r2"] for cell in last_cells]
    iid_r2_ci = [cell["split_paired_cluster_bootstrap"]["a_ci"] for cell in last_cells]
    retrieval_cells = [retrieval_payload["cells"][key] for key in STAGE_KEYS]
    whitened_csls_acc1 = [cell["whitened_csls_acc1"] for cell in retrieval_cells]
    whitened_csls_acc1_ci = [cell["whitened_csls_acc1_ci95"] for cell in retrieval_cells]
    np.testing.assert_allclose(
        [cell["r2"] for cell in retrieval_cells], iid_r2, rtol=1e-12, atol=1e-12
    )
    assert latest_payload["metadata"]["layer"] == 31
    assert latest_payload["metadata"]["hf_revision"] == HF_REVISION
    assert all(cell["n"] == 16_391 for cell in last_cells)
    assert retrieval_payload["metadata"]["context_summary"] == "final prompt token"
    assert "two-sided CSLS k=10" in retrieval_payload["metadata"]["retrieval"]
    assert all(cell["n"] == 16_391 for cell in retrieval_cells)

    # Panels B and C: the last-token IID transfer summary ONLY.
    summary_bytes = SUMMARY_PATH.read_bytes()
    summary = json.loads(summary_bytes)
    meta = summary["metadata"]
    assert meta["layer"] == 31
    assert meta["hf_revision"] == HF_REVISION
    assert meta["context_summary"] == "u_last"
    assert meta["fold_mode"] == "random"
    assert summary["parity_gate"]["pass"]
    grid = summary["grid"]

    stage_grid = [[grid[f"{m}{s}"]["r2"] for s in STAGE_KEYS] for m in STAGE_KEYS]
    base_row_r2 = [grid[f"B{s}"]["r2"] for s in STAGE_KEYS]
    base_row_ci = [grid[f"B{s}"]["row_ci"] for s in STAGE_KEYS]
    diag_r2 = [grid[f"{s}{s}"]["r2"] for s in STAGE_KEYS]
    diag_ci = [grid[f"{s}{s}"]["row_ci"] for s in STAGE_KEYS]
    # Previous stage's states predicting this stage's answers; Base has none.
    prev_row_r2 = [None] + [grid[f"{STAGE_KEYS[k - 1]}{STAGE_KEYS[k]}"]["r2"] for k in range(1, 4)]
    prev_row_ci = [None] + [
        grid[f"{STAGE_KEYS[k - 1]}{STAGE_KEYS[k]}"]["row_ci"] for k in range(1, 4)
    ]
    # Panels A and B describe the same diagonal (parity gate tolerance 1e-6).
    np.testing.assert_allclose(diag_r2, iid_r2, atol=2e-6, rtol=0)

    retention: dict[str, dict[str, list]] = {
        mode: {"point": [], "ci": [], "ci_row": []} for mode in ("direct", "bias", "scale_bias")
    }
    for transition in TRANSITIONS:
        pair = summary["transfer"][transition]
        for mode, acc in retention.items():
            entry = pair["retention"][mode]
            acc["point"].append(entry["point"])
            acc["ci"].append(entry["cluster_ci"])
            acc["ci_row"].append(entry["row_ci"])

    return {
        "metadata": {
            "hf_revision": HF_REVISION,
            "layer": 31,
            "context_summary": "u_last",
            "folds": "six size-matched IID random-row folds (seed 190231)",
            "panel_bc_source": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "panel_bc_bootstrap": (
                "1000 paired draws, seed 1944; panel C retention CIs = "
                "semantic-cluster bootstrap; line-variant panel B CI = "
                "row bootstrap"
            ),
            "series_encoding": SERIES_ENCODING,
            "inputs": {
                "latest": {"path": str(latest_path), "sha256": LATEST_SHA256},
                "retrieval": {"path": str(retrieval_path), "sha256": RETRIEVAL_SHA256},
                "summary": {
                    "path": str(SUMMARY_PATH),
                    "sha256": _sha256_bytes(summary_bytes),
                },
            },
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "iid_r2": iid_r2,
        "iid_r2_ci": iid_r2_ci,
        "whitened_csls_acc1": whitened_csls_acc1,
        "whitened_csls_acc1_ci": whitened_csls_acc1_ci,
        "stage_grid": stage_grid,
        "base_row_r2": base_row_r2,
        "base_row_ci": base_row_ci,
        "diag_r2": diag_r2,
        "diag_ci": diag_ci,
        "prev_row_r2": prev_row_r2,
        "prev_row_ci": prev_row_ci,
        "direct_retention": retention["direct"]["point"],
        "direct_retention_ci": retention["direct"]["ci"],
        "direct_retention_ci_row": retention["direct"]["ci_row"],
        "bias_retention": retention["bias"]["point"],
        "bias_retention_ci": retention["bias"]["ci"],
        "bias_retention_ci_row": retention["bias"]["ci_row"],
        "scale_bias_retention": retention["scale_bias"]["point"],
        "scale_bias_retention_ci": retention["scale_bias"]["ci"],
        "scale_bias_retention_ci_row": retention["scale_bias"]["ci_row"],
    }


def _err(points: np.ndarray, ci: np.ndarray) -> np.ndarray:
    return np.vstack([points - ci[:, 0], ci[:, 1] - points])


HEADER_KWARGS = {"kicker_y": 1.44, "title_y": 1.06}


def plot_within_stage(ax: mpl.axes.Axes, data: dict[str, Any]) -> list[Line2D]:
    x = np.arange(len(STAGES))
    iid_r2 = np.asarray(data["iid_r2"])
    iid_ci = np.asarray(data["iid_r2_ci"])
    acc1 = np.asarray(data["whitened_csls_acc1"])
    acc1_ci = np.asarray(data["whitened_csls_acc1_ci"])
    r2_style = metric_style("r2")
    top1_style = metric_style("top1")
    ax.errorbar(
        x,
        iid_r2,
        yerr=_err(iid_r2, iid_ci),
        color=TEAL,
        marker="o",
        markersize=7,
        linewidth=2.0,
        linestyle=r2_style["linestyle"],
        fillstyle=r2_style["fillstyle"],
        capsize=3,
        zorder=4,
    )
    ax.errorbar(
        x,
        acc1,
        yerr=_err(acc1, acc1_ci),
        color=TEAL,
        marker="o",
        markersize=7,
        linewidth=1.8,
        linestyle=top1_style["linestyle"],
        fillstyle=top1_style["fillstyle"],
        markeredgewidth=1.6,
        capsize=3,
        zorder=4,
    )
    style_score_axis(ax, y_min=0.4, y_max=1.005, y_step=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel(better_label("Held-out score"))
    panel_header(
        ax,
        "A",
        "OLMo-2-7B, layer 31",
        "Held-out score of each\nstage's own map",
        **HEADER_KWARGS,
    )
    return [
        Line2D(
            [],
            [],
            color=TEAL,
            marker="o",
            markersize=7,
            linewidth=2.0,
            linestyle=r2_style["linestyle"],
            fillstyle=r2_style["fillstyle"],
            label=METRIC_LABELS["r2"],
        ),
        Line2D(
            [],
            [],
            color=TEAL,
            marker="o",
            markersize=7,
            linewidth=1.8,
            linestyle=top1_style["linestyle"],
            fillstyle=top1_style["fillstyle"],
            markeredgewidth=1.6,
            label=METRIC_LABELS["top1"],
        ),
    ]


def plot_stage_grid(ax: mpl.axes.Axes, data: dict[str, Any]) -> list[Line2D]:
    stage_grid = np.asarray(data["stage_grid"])
    cmap = LinearSegmentedColormap.from_list(
        "paper_teal", [_tint(TEAL, 0.95), _tint(TEAL, 0.7), _tint(TEAL, 0.4), TEAL]
    )
    vmin = np.floor(stage_grid.min() * 100) / 100 - 0.01
    vmax = np.ceil(stage_grid.max() * 100) / 100 + 0.01
    ax.imshow(stage_grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    threshold = vmin + 0.72 * (vmax - vmin)
    for row in range(4):
        for col in range(4):
            value = stage_grid[row, col]
            color = PAPER if value >= threshold else INK
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color)
    ax.set_xticks(np.arange(4), STAGES)
    ax.set_yticks(np.arange(4), STAGES)
    ax.set_xlabel("Answer source")
    ax.set_ylabel("Activation checkpoint")
    panel_header(
        ax,
        "B",
        "context→answer fits",
        "Held-out $R^2$ into each stage's\nanswers, by context source",
        **HEADER_KWARGS,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return []


def plot_stage_series(ax: mpl.axes.Axes, data: dict[str, Any]) -> list[Line2D]:
    """Line panel B: own / previous-stage / Base states predicting each stage's answers.

    Every series is held-out R^2, so all lines stay solid with filled markers
    (dash / open-marker is reserved for the top-1 retrieval metric); the series
    are separated by hue + marker per SERIES_ENCODING["panel_b_lines"].
    """
    x = np.arange(len(STAGES))
    diag = np.asarray(data["diag_r2"], dtype=float)
    diag_ci = np.asarray(data["diag_ci"], dtype=float)
    prev = np.asarray([np.nan if v is None else v for v in data["prev_row_r2"]], dtype=float)
    prev_ci = np.asarray(
        [[np.nan, np.nan] if v is None else v for v in data["prev_row_ci"]],
        dtype=float,
    )
    base = np.asarray(data["base_row_r2"], dtype=float)
    base_ci = np.asarray(data["base_row_ci"], dtype=float)
    series = []
    ax.errorbar(
        x,
        diag,
        yerr=_err(diag, diag_ci),
        color=TEAL,
        marker="o",
        markersize=7,
        linewidth=2.0,
        capsize=3,
        zorder=5,
    )
    series.append((TEAL, "o", "Own states"))
    keep = np.isfinite(prev)
    ax.errorbar(
        x[keep],
        prev[keep],
        yerr=_err(prev[keep], prev_ci[keep]),
        color=PREV_TINT,
        marker="^",
        markersize=7.5,
        linewidth=2.0,
        capsize=3,
        zorder=4,
    )
    series.append((PREV_TINT, "^", "Previous-stage states"))
    ax.errorbar(
        x,
        base,
        yerr=_err(base, base_ci),
        color=AMBER,
        marker="s",
        markersize=6.5,
        linewidth=2.0,
        capsize=3,
        zorder=3,
    )
    series.append((AMBER, "s", "Base states"))
    values = np.concatenate([diag_ci.ravel(), base_ci.ravel(), prev_ci[keep].ravel()])
    lo = np.floor(np.nanmin(values) * 20) / 20
    hi = np.ceil(np.nanmax(values) * 20) / 20
    if np.nanmin(values) - lo < 0.01:
        lo -= 0.05
    if hi - np.nanmax(values) < 0.01:
        hi += 0.05
    style_score_axis(ax, y_min=lo, y_max=hi + 1e-4, y_step=0.05)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_xlabel("Answer source")
    ax.set_ylabel(better_label(METRIC_LABELS["r2"]))
    panel_header(
        ax,
        "B",
        "context→answer fits",
        "Held-out $R^2$ into each stage's\nanswers, by context source",
        **HEADER_KWARGS,
    )
    return [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.0,
            label=label,
        )
        for color, marker, label in series
    ]


def plot_transfer(ax: mpl.axes.Axes, data: dict[str, Any]) -> list[Line2D]:
    """Panel C: retention of the preceding stage's map under three corrections.

    One teal hue with fill progression open -> half tint -> filled (the
    c1_cot_ladder convention); see SERIES_ENCODING["panel_c"].
    """
    x = np.arange(3)
    labels = ["Base\n→ SFT", "SFT\n→ DPO", "DPO\n→ RLVR"]
    modes = [
        # (key, offset, label, marker, facecolor)
        ("direct", -0.22, "as is", "o", PAPER),
        ("bias", 0.0, "bias", "s", _tint(TEAL, 0.5)),
        ("scale_bias", 0.22, "rescaling + bias", "o", TEAL),
    ]
    all_ci = []
    handles = []
    for key, offset, label, marker, face in modes:
        points = np.asarray(data[f"{key}_retention"], dtype=float)
        ci_arr = np.asarray(data[f"{key}_retention_ci"], dtype=float)
        all_ci.append(ci_arr)
        ax.errorbar(
            x + offset,
            points,
            yerr=_err(points, ci_arr),
            fmt=marker,
            markersize=8,
            markerfacecolor=face,
            markeredgecolor=TEAL,
            markeredgewidth=1.6,
            ecolor=TEAL,
            linewidth=0.0,
            elinewidth=1.3,
            capsize=3,
            zorder=4 if key != "scale_bias" else 5,
        )
        handles.append(
            Line2D(
                [],
                [],
                linewidth=0.0,
                marker=marker,
                markersize=8,
                markerfacecolor=face,
                markeredgecolor=TEAL,
                markeredgewidth=1.6,
                label=label,
            )
        )
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=(0, (5, 4)))
    ax.axhline(0.0, color=SEAM, linewidth=1.0)
    style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ci_values = np.concatenate([c.ravel() for c in all_ci])
    low = min(-0.12, float(ci_values.min()) - 0.12)
    high = max(1.12, float(ci_values.max()) + 0.12)
    ax.set_ylim(low, high)
    yticks = [t for t in (-1.0, -0.5, 0.0, 0.5, 1.0) if low - 1e-9 <= t <= high + 1e-9]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in yticks])
    # "Retention" lives in the panel title; the long form would overrun the
    # rotated label past the axes height at paper scale.
    ax.set_ylabel(better_label(r"$R^2_{i \to j} \, / \, R^2_{j \to j}$"))
    panel_header(
        ax,
        "C",
        "map transfer",
        "Retention of the\nprior stage's map",
        **HEADER_KWARGS,
    )
    return handles


def render_variant(data: dict[str, Any], out_base: Path, *, panel_b: str) -> dict[str, Any]:
    if panel_b not in {"lines", "grid"}:
        raise ValueError(f"panel_b must be 'lines' or 'grid', got {panel_b!r}")
    fig, include_frac = c2a_figure("full", aspect=0.40)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.1, 1.05],
        left=0.06,
        right=0.985,
        top=0.56,
        bottom=0.15,
        wspace=0.55,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    handles_a = plot_within_stage(axes[0], data)
    if panel_b == "lines":
        handles_b = plot_stage_series(axes[1], data)
    else:
        handles_b = plot_stage_grid(axes[1], data)
    handles_c = plot_transfer(axes[2], data)
    for ax, heading, handles in (
        (axes[0], "Metric", handles_a),
        (axes[1], "Context source", handles_b),
        (axes[2], "Correction", handles_c),
    ):
        if not handles:
            continue
        x0 = ax.get_position().x0
        legend_kicker(fig, x0, 0.978, heading)
        fig.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(x0 - 0.001, 0.958),
            ncol=1,
            frameon=False,
            handlelength=1.9,
            handletextpad=0.5,
            borderaxespad=0.0,
            labelspacing=0.3,
        )
    outputs = save_c2a_figure(
        fig,
        out_base,
        title="Section 4.3 post-training dynamics of the context-to-answer map",
        subject=(
            "OLMo-2-7B Base/SFT/DPO/RLVR last-token IID fits, cross-stage grid, "
            "and adjacent-stage retention (issue #1902 lasttoken_transfer)"
        ),
        creator="scripts/section43_posttraining_figure.py",
        include_width=include_frac,
    )
    plt.close(fig)
    meta_path = out_base.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "record": outputs["record"],
                "panel_b_variant": panel_b,
                "metadata": data["metadata"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    outputs["meta"] = meta_path
    return outputs


def render(out_dir: Path) -> None:
    set_c2a_style()
    data = load_results()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "c1_posttraining_dynamics_data.json"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    # Plot from the JSON round-trip so the figure is reproducible from it alone.
    data = json.loads(data_path.read_text())
    paper = render_variant(data, out_dir / "c1_posttraining_dynamics", panel_b="lines")
    grid = render_variant(data, out_dir / "c1_posttraining_dynamics_grid", panel_b="grid")
    for name, outputs in (("paper", paper), ("grid variant", grid)):
        print(f"{name}: {outputs['pdf']}")
        print(f"  latex_include_line: {outputs['record']['latex_include_line']}")
    print(f"data: {data_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args().out_dir)
