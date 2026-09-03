#!/usr/bin/env python3
"""Render the Section 4.3 post-training summary figure.

Panel A (within-stage fits + retrieval) reads the committed last-token
comparison artifacts under ``overleaf_section43/figures/paper``. Panels B
(4x4 activation-checkpoint x answer-source grid) and C (adjacent-stage
transfer retention: as is / bias / scalar rescaling + bias) read ONLY
``eval_results/issue_1902/lasttoken_transfer/summary.json`` — the last-token
IID recompute (u_last states, six size-matched random-row folds), replacing
the registered prompt-mean / semantic-fold numbers.

Two variants are written: the 4x4 heatmap panel B
(``c1_posttraining_dynamics``) and a three-line panel B comparing each
stage's own states, the previous stage's states, and Base states as
predictors of each stage's answers (``c1_posttraining_dynamics_2line``).
Both figures are reproducible from ``c1_posttraining_dynamics_data.json``
alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

INK = "#25292D"
MUTED = "#687078"
GRID = "#D9D8D3"
SPINE = "#AAA79F"
TEAL = "#16708A"
ORANGE = "#C9583D"
LIGHT_TEAL = "#8CBAC5"
NULL_DARK = "#747A80"
NULL_LIGHT = "#A6A9AB"

STAGES = ["Base", "SFT", "DPO", "RLVR"]
STAGE_KEYS = ["B", "S", "D", "R"]
TRANSITIONS = ["B->S", "S->D", "D->R"]
LATEST_SHA256 = "947a46cbde8f834be0a094b85e401e73a260f8da0416dd76285ca9da12c03c96"
RETRIEVAL_SHA256 = "d74f9adb9d52e36558d38c99c5b83ed636b731f5b46c6a5ab6b000c920f6bd91"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_1902" / "lasttoken_transfer" / "summary.json"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures" / "issue_1902" / "section43"
HF_REVISION = "3256c8efcef5f10ca525efeb2039636eaec8fad7"


def load_results() -> dict[str, Any]:
    """Collect every plotted value into one JSON-serializable dict."""
    # Panel A sources (unchanged): committed last-token comparison artifacts.
    latest_path = (
        PROJECT_ROOT
        / "overleaf_section43"
        / "figures"
        / "paper"
        / "c1_posttraining_dynamics_data.json"
    )
    latest_bytes = latest_path.read_bytes()
    if hashlib.sha256(latest_bytes).hexdigest() != LATEST_SHA256:
        msg = f"Unexpected contents in {latest_path}"
        raise ValueError(msg)
    latest_payload = json.loads(latest_bytes)
    retrieval_path = latest_path.with_name("c1_posttraining_retrieval_data.json")
    retrieval_bytes = retrieval_path.read_bytes()
    if hashlib.sha256(retrieval_bytes).hexdigest() != RETRIEVAL_SHA256:
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
    summary = json.loads(SUMMARY_PATH.read_text())
    meta = summary["metadata"]
    assert meta["layer"] == 31
    assert meta["hf_revision"] == HF_REVISION
    assert meta["context_summary"] == "u_last"
    assert meta["fold_mode"] == "random"
    assert summary["parity_gate"]["pass"]
    grid = summary["grid"]

    stage_grid = [
        [grid[f"{m}{s}"]["r2"] for s in STAGE_KEYS] for m in STAGE_KEYS
    ]
    base_row_r2 = [grid[f"B{s}"]["r2"] for s in STAGE_KEYS]
    base_row_ci = [grid[f"B{s}"]["row_ci"] for s in STAGE_KEYS]
    diag_r2 = [grid[f"{s}{s}"]["r2"] for s in STAGE_KEYS]
    diag_ci = [grid[f"{s}{s}"]["row_ci"] for s in STAGE_KEYS]
    # Previous stage's states predicting this stage's answers; Base has none.
    prev_row_r2 = [None] + [
        grid[f"{STAGE_KEYS[k - 1]}{STAGE_KEYS[k]}"]["r2"] for k in range(1, 4)
    ]
    prev_row_ci = [None] + [
        grid[f"{STAGE_KEYS[k - 1]}{STAGE_KEYS[k]}"]["row_ci"] for k in range(1, 4)
    ]
    # Panels A and B describe the same diagonal (parity gate tolerance 1e-6).
    np.testing.assert_allclose(diag_r2, iid_r2, atol=2e-6, rtol=0)

    retention: dict[str, dict[str, list]] = {
        mode: {"point": [], "ci": [], "ci_row": []}
        for mode in ("direct", "bias", "scale_bias")
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


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": SPINE,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_heading(ax: mpl.axes.Axes, eyebrow: str, title: str) -> None:
    n_lines = title.count("\n") + 1
    ax.text(
        0,
        1.30 + 0.11 * max(0, n_lines - 2),
        eyebrow.upper(),
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8.8,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0,
        1.08,
        title,
        transform=ax.transAxes,
        color=INK,
        fontsize=10.5,
        fontweight="bold",
        linespacing=0.95,
        va="bottom",
    )


def clean_axes(ax: mpl.axes.Axes, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)


def _err(points: np.ndarray, ci: np.ndarray) -> np.ndarray:
    return np.vstack([points - ci[:, 0], ci[:, 1] - points])


def plot_within_stage(ax: mpl.axes.Axes, data: dict[str, Any]) -> None:
    x = np.arange(len(STAGES))
    iid_r2 = np.asarray(data["iid_r2"])
    iid_ci = np.asarray(data["iid_r2_ci"])
    acc1 = np.asarray(data["whitened_csls_acc1"])
    acc1_ci = np.asarray(data["whitened_csls_acc1_ci"])
    ax.errorbar(
        x,
        iid_r2,
        yerr=_err(iid_r2, iid_ci),
        color=TEAL,
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        capsize=2.5,
        label=r"$R^2$",
        zorder=4,
    )
    ax.errorbar(
        x,
        acc1,
        yerr=_err(acc1, acc1_ci),
        color=ORANGE,
        marker="s",
        markersize=4.2,
        linewidth=1.4,
        linestyle=(0, (4, 2)),
        capsize=2.5,
        label=r"Whitened+CSLS acc@1",
        zorder=4,
    )
    ax.set_xticks(x, STAGES)
    ax.set_ylim(0.40, 1.00)
    ax.set_yticks([0.40, 0.60, 0.80, 1.00])
    ax.set_ylabel("Held-out score ↑")
    panel_heading(
        ax, "A · final context token · IID", "The map persists\nat every stage"
    )
    clean_axes(ax)
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.52, 0.53),
        fontsize=8.3,
        handlelength=2.1,
        handletextpad=0.5,
    )


def plot_stage_grid(ax: mpl.axes.Axes, data: dict[str, Any]) -> None:
    stage_grid = np.asarray(data["stage_grid"])
    cmap = LinearSegmentedColormap.from_list(
        "paper_teal", ["#F4F4F1", "#C7DEE1", LIGHT_TEAL, TEAL]
    )
    vmin = np.floor(stage_grid.min() * 100) / 100 - 0.01
    vmax = np.ceil(stage_grid.max() * 100) / 100 + 0.01
    ax.imshow(stage_grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    threshold = vmin + 0.72 * (vmax - vmin)
    for row in range(4):
        for col in range(4):
            value = stage_grid[row, col]
            color = "white" if value >= threshold else INK
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9.3,
            )
    ax.set_xticks(np.arange(4), STAGES)
    ax.set_yticks(np.arange(4), STAGES)
    ax.set_xlabel("Answer source")
    ax.set_ylabel("Activation checkpoint")
    panel_heading(
        ax,
        "B · context→answer fits",
        "Previous stage states\ncan predict subsequent\nstage answers",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def plot_two_line(ax: mpl.axes.Axes, data: dict[str, Any]) -> None:
    """Three-tone line variant of panel B (own / previous-stage / Base states)."""
    x = np.arange(len(STAGES))
    diag = np.asarray(data["diag_r2"], dtype=float)
    diag_ci = np.asarray(data["diag_ci"], dtype=float)
    prev = np.asarray(
        [np.nan if v is None else v for v in data["prev_row_r2"]], dtype=float
    )
    prev_ci = np.asarray(
        [[np.nan, np.nan] if v is None else v for v in data["prev_row_ci"]],
        dtype=float,
    )
    base = np.asarray(data["base_row_r2"], dtype=float)
    base_ci = np.asarray(data["base_row_ci"], dtype=float)
    ax.errorbar(
        x,
        diag,
        yerr=_err(diag, diag_ci),
        color=TEAL,
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        capsize=2.5,
        label="Own states",
        zorder=5,
    )
    keep = np.isfinite(prev)
    ax.errorbar(
        x[keep],
        prev[keep],
        yerr=_err(prev[keep], prev_ci[keep]),
        color=LIGHT_TEAL,
        marker="^",
        markersize=4.6,
        linewidth=1.4,
        linestyle=(0, (5, 2)),
        markeredgecolor=TEAL,
        capsize=2.5,
        label="Prev.-stage states",
        zorder=4,
    )
    ax.errorbar(
        x,
        base,
        yerr=_err(base, base_ci),
        color="#C7DEE1",
        marker="s",
        markersize=4.2,
        linewidth=1.4,
        linestyle=(0, (2, 2)),
        markeredgecolor=TEAL,
        capsize=2.5,
        label="Base states",
        zorder=3,
    )
    ax.set_xticks(x, STAGES)
    values = np.concatenate(
        [diag_ci.ravel(), base_ci.ravel(), prev_ci[keep].ravel()]
    )
    lo = np.floor(np.nanmin(values) * 20) / 20
    hi = np.ceil(np.nanmax(values) * 20) / 20
    ax.set_ylim(lo - 0.01, hi + 0.01)
    ax.set_xlabel("Answer source")
    ax.set_ylabel(r"Held-out $R^2$ ↑")
    panel_heading(
        ax,
        "B · context→answer fits",
        "Previous stage states\ncan predict subsequent\nstage answers",
    )
    clean_axes(ax)
    ax.legend(
        loc="lower right",
        fontsize=7.8,
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.2,
    )


def plot_transfer(ax: mpl.axes.Axes, data: dict[str, Any]) -> None:
    """Panel C: retention of the previous stage's map, three calibrations."""
    x = np.arange(3)
    labels = ["Base\n→ SFT", "SFT\n→ DPO", "DPO\n→ RLVR"]
    modes = [
        # (key, offset, label, marker, facecolor, edge/line color)
        ("direct", -0.22, "as is", "o", "white", ORANGE),
        ("bias", 0.0, "bias", "s", LIGHT_TEAL, TEAL),
        ("scale_bias", 0.22, "rescaling + bias", "o", TEAL, TEAL),
    ]
    all_ci = []
    for key, offset, label, marker, face, edge in modes:
        points = np.asarray(data[f"{key}_retention"], dtype=float)
        ci_arr = np.asarray(data[f"{key}_retention_ci"], dtype=float)
        all_ci.append(ci_arr)
        ax.errorbar(
            x + offset,
            points,
            yerr=_err(points, ci_arr),
            fmt=marker,
            markersize=4.6,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=1.3,
            ecolor=edge,
            linewidth=0.0,
            elinewidth=1.1,
            capsize=2.5,
            label=label,
            zorder=4 if key != "scale_bias" else 5,
        )
    scale_bias = np.asarray(data["scale_bias_retention"], dtype=float)
    for idx, value in enumerate(scale_bias):
        ax.text(
            x[idx] + 0.22,
            value + 0.07,
            f"{value:.2f}",
            ha="center",
            fontsize=8.8,
            fontweight="bold",
        )
    ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (5, 4)))
    ax.axhline(0.0, color=SPINE, linewidth=0.7)
    ax.set_xticks(x, labels)
    ci_values = np.concatenate([c.ravel() for c in all_ci])
    low = min(-0.12, float(ci_values.min()) - 0.12)
    high = max(1.18, float(ci_values.max()) + 0.18)
    ax.set_ylim(low, high)
    yticks = [t for t in (-1.0, -0.5, 0.0, 0.5, 1.0) if t >= low - 1e-9]
    ax.set_yticks(yticks)
    ax.set_ylabel(r"Retention $R^2_{i\to j}/R^2_{j\to j}$ ↑")
    panel_heading(ax, "C · map transfer", "The map stabilizes\nafter SFT")
    clean_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=8.3,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.4,
    )


def render_variant(data: dict[str, Any], out_base: Path, *, two_line: bool) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.4, 3.0),
        gridspec_kw={"width_ratios": [1.12, 0.9 if not two_line else 1.05, 1.12], "wspace": 0.64},
    )
    plot_within_stage(axes[0], data)
    if two_line:
        plot_two_line(axes[1], data)
    else:
        plot_stage_grid(axes[1], data)
    plot_transfer(axes[2], data)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.74, bottom=0.23)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        out_base.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.04
    )
    plt.close(fig)


def render(out_dir: Path) -> None:
    configure_style()
    data = load_results()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "c1_posttraining_dynamics_data.json"
    data_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    # Plot from the JSON round-trip so the figure is reproducible from it alone.
    data = json.loads(data_path.read_text())
    render_variant(data, out_dir / "c1_posttraining_dynamics", two_line=False)
    render_variant(data, out_dir / "c1_posttraining_dynamics_2line", two_line=True)
    print(f"wrote {out_dir}/c1_posttraining_dynamics[.pdf/.png], _2line[.pdf/.png], _data.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args().out_dir)
