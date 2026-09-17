"""Render the matched cached behavior results using the paper's shared style.

Reads completed JSON summaries only. No fitting, sampling, or judging occurs.
All values and pointwise intervals are preserved, including negative results.
The main figure compares fixed contrastive projections by behavior and regime.
Per-dataset projections, inverse directions, and regressions are in the appendix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    PAPER,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

INPUT = ROOT / "eval_results/issue_1739/million_cached_20260916"
REGIMES = ROOT / "eval_results/issue_1739/fixed_regimes_20260917"
ROWS = (
    ("evil", "hhrt", "HH-RLHF", 1847),
    ("evil", "toxicchat", "ToxicChat", 370),
    ("sycophancy", "aita", "AITA", 1304),
    ("hallucination", "nqopen", "NQ-Open", 3164),
    ("hallucination", "simpleqa", "SimpleQA", 4021),
)
TEAL = ROLES["linear"].color
# Gray = comparison representation, teal = metamodel-derived score.
# Marker shape/fill distinguishes series within each semantic role.
STYLES = {
    "mapped_answer": (TEAL, "o", TEAL, "Predicted answer"),
    "real_answer": (INK, "x", INK, "Observed answer"),
    "preimage": (TEAL, "D", TEAL, "Preimage"),
    "context_native": (MUTED, "s", PAPER, "Context direction"),
    "raw_context": (MUTED, "x", MUTED, "Context"),
    "context_covariance": (MUTED, "s", PAPER, "Covariance-whitened"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data() -> dict:
    inputs, cells = {}, []
    for behavior, rung, label, n in ROWS:
        done = json.loads((INPUT / behavior / "complete.json").read_text())
        cell = {"behavior": behavior, "dataset": label, "n": n}
        for kind in ("fixed", "regression"):
            path = INPUT / behavior / f"{kind}.json"
            digest = sha(path)
            if digest != done["artifact_sha256"][path.name]:
                raise ValueError(f"Completed artifact changed: {path}")
            inputs[str(path.relative_to(ROOT))] = digest
            payload = json.loads(path.read_text())
            matches = [r for r in payload["results"] if r["rung"] == rung]
            if len(matches) != 1 or matches[0]["n"] != n:
                raise ValueError(f"Unexpected evaluation cell: {behavior}/{rung}")
            cell[kind] = matches[0]
        cells.append(cell)
    return {
        "inputs": inputs,
        "archive_revision": "449c48d4047efffec50dbe8492d78c905b4bb588",
        "archive_prefix": "issue1739_million_cached_20260916",
        "map_training_pairs": 963444,
        "model": "Qwen2.5-7B-Instruct",
        "layer": 19,
        "inverse_rank": 378,
        "uncertainty": "pointwise 95% paired group-bootstrap, 2000 draws, fixed fits",
        "excluded": "WildChat: only four retained contexts per behavior",
        "pooling_caveat": "Map target includes assistant-closing template tokens. "
        "Observed evaluation answer vectors average completion tokens only.",
        "cells": cells,
    }


def points(ax, rows, offset, *, key, color, marker, face, label):
    """Draw literal CI endpoints, including intervals not containing the estimate."""
    for i, row in enumerate(rows):
        lo, hi = row["ci95"]
        value = row[key]
        if not np.isfinite([lo, hi, value]).all() or lo > hi:
            raise ValueError(f"Invalid point or interval: {row}")
        ax.hlines(i + offset, lo, hi, color=color, linewidth=1.4, zorder=2)
        ax.plot(
            value,
            i + offset,
            marker=marker,
            color=color,
            markerfacecolor=face,
            markersize=6,
            markeredgewidth=1.4,
            linestyle="none",
            zorder=3,
            label=label if i == 0 else None,
        )


def axes_style(ax, *, labels, xlim, ticks, xlabel):
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_yticks(range(len(ROWS)), [r[2] for r in ROWS] if labels else [])
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)
    style_axis(ax, grid_axis="none")
    ax.spines["left"].set_visible(False)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":", zorder=0)


def save(fig, frac, out: Path, stem: str, data: dict):
    result = save_c2a_figure(
        fig,
        out / stem,
        title="Behavior prediction before generation",
        subject="Fixed-direction transfer and matched regression controls",
        creator=str(Path(__file__).relative_to(ROOT)),
        include_width=frac,
    )
    payload = {**data, "render": result["record"], "renderer_sha256": sha(Path(__file__))}
    for key in ("pdf", "png", "grayscale"):
        payload[key + "_sha256"] = sha(result[key])
    (out / f"{stem}.meta.json").write_text(json.dumps(payload, indent=2) + "\n")
    plt.close(fig)


def dataset_figure(data, out):
    """Retain individual OOD datasets and their preimage comparison in the appendix."""
    fig, frac = c2a_figure("full", aspect=0.36)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.90, bottom=0.31, wspace=0.25)
    panels = (
        (
            "A",
            "Contrastive directions",
            "fixed",
            ("mapped_answer", "real_answer", "context_native"),
        ),
        ("B", "Preimage direction", "fixed", ("preimage", "context_native")),
    )
    projection_labels = {
        "mapped_answer": "Answer direction → predicted answer",
        "real_answer": "Answer direction → observed answer",
        "context_native": "Context direction → context",
        "preimage": "Preimage direction → context",
    }
    for ax, (letter, heading, kind, arms) in zip(axes, panels, strict=True):
        for offset, arm in zip(np.linspace(-0.22, 0.22, len(arms)), arms, strict=True):
            color, marker, face, label = STYLES[arm]
            points(
                ax,
                [c[kind]["arms"][arm] for c in data["cells"]],
                offset,
                key="rho",
                color=color,
                marker=marker,
                face=face,
                label=projection_labels[arm],
            )
        axes_style(
            ax,
            labels=letter == "A",
            xlim=(-0.25, 0.82),
            ticks=[-0.2, 0, 0.4, 0.8],
            xlabel=r"Spearman $\rho$",
        )
        panel_header(ax, letter, heading, kicker_y=1.07)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.02, -0.23),
            borderaxespad=0,
            handlelength=0.9,
            handletextpad=0.5,
            labelspacing=0.25,
        )
    save(fig, frac, out, "c5_behavior_datasets", data)


def regime_figure(out):
    """Match the preceding behavior-by-regime layout, excluding synthetic evaluation."""
    path = REGIMES / "summary.json"
    done = json.loads((REGIMES / "complete.json").read_text())
    if sha(path) != done["artifact_sha256"]["summary.json"]:
        raise ValueError("Completed regime summary changed")
    data = json.loads(path.read_text())
    data["inputs"] = {str(path.relative_to(ROOT)): sha(path)}
    fig, frac = c2a_figure("full", aspect=0.36)
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.36, wspace=0.15)
    labels = {
        "mapped_answer": "Answer direction → predicted answer",
        "real_answer": "Answer direction → observed answer",
        "context_native": "Context direction → context",
    }
    headings = ("Harmful compliance", "Sycophancy", "Hallucination")
    for panel, (ax, behavior, heading) in enumerate(
        zip(axes, data["behaviors"], headings, strict=True)
    ):
        for group, cell in enumerate(behavior["regimes"]):
            if not cell["informative"]:
                if cell["regime"] != "generic chat" or cell["n"] != 4:
                    raise ValueError(f"Unexpected missing cell: {cell}")
                ax.text(
                    group,
                    0.28,
                    "Insufficient\ndata\n(n = 4)",
                    color=MUTED,
                    ha="center",
                    va="center",
                )
                continue
            for offset, arm in zip((-0.23, 0, 0.23), labels, strict=True):
                row = cell["arms"][arm]
                value, (lo, hi) = row["rho"], row["ci95"]
                if not np.isfinite([value, lo, hi]).all() or lo > hi or lo < -0.45 or hi > 0.80:
                    raise ValueError(f"Invalid interval: {row}")
                color, _, face, _ = STYLES[arm]
                ax.bar(
                    group + offset,
                    value,
                    width=0.20,
                    color=face,
                    edgecolor=color,
                    linewidth=1.4,
                    label=labels[arm] if panel == 0 and group == 1 else None,
                    zorder=2,
                )
                ax.vlines(group + offset, lo, hi, color=INK, linewidth=1.1, zorder=3)
                ax.hlines(
                    [lo, hi],
                    group + offset - 0.035,
                    group + offset + 0.035,
                    color=INK,
                    linewidth=1.1,
                    zorder=3,
                )
        ax.set_xlim(-0.52, 2.52)
        ax.set_ylim(-0.45, 0.80)
        ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticks([0, 1, 2], ["Generic\nchat", "In-distrib.", "OOD"])
        ax.tick_params(axis="x", length=0, pad=8)
        style_axis(ax, grid_axis="none")
        ax.axhline(0, color=MUTED, linewidth=0.7, zorder=0)
        panel_header(ax, chr(65 + panel), heading, kicker_y=1.07)
        if panel:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel(better_label(r"Spearman $\rho$"))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=1,
        labelspacing=0.25,
        handlelength=1.0,
    )
    save(fig, frac, out, "c5_behavior_transfer", data)


def difference_figure(data, out):
    fig, frac = c2a_figure("full", aspect=0.36)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.31, wspace=0.25)
    panels = (
        (
            "A",
            "Preimage comparisons",
            "fixed",
            (
                ("preimage_minus_context_native", "Preimage − context direction", "D", TEAL),
                ("preimage_minus_mapped_answer", "Preimage − predicted answer", "o", PAPER),
            ),
            (-0.13, 0.13),
            [-0.1, 0, 0.1],
        ),
        (
            "B",
            "Regression comparisons",
            "regression",
            (
                ("mapped_answer_minus_raw_context", "Predicted answer − context", "o", TEAL),
                (
                    "mapped_answer_minus_context_covariance",
                    "Predicted answer − covariance",
                    "s",
                    PAPER,
                ),
                ("mapped_answer_minus_shuffled_mean", "Predicted answer − shuffled", "x", TEAL),
            ),
            (-0.09, 0.17),
            [-0.05, 0, 0.05, 0.1, 0.15],
        ),
    )
    for ax, (letter, heading, kind, arms, xlim, ticks) in zip(axes, panels, strict=True):
        for offset, (key, label, marker, face) in zip(
            np.linspace(-0.22, 0.22, len(arms)), arms, strict=True
        ):
            points(
                ax,
                [c[kind]["differences"][key] for c in data["cells"]],
                offset,
                key="delta",
                color=TEAL,
                marker=marker,
                face=face,
                label=label,
            )
        axes_style(
            ax,
            labels=letter == "A",
            xlim=xlim,
            ticks=ticks,
            xlabel=r"Difference in Spearman $\rho$",
        )
        panel_header(ax, letter, heading, kicker_y=1.08)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.02, -0.25),
            borderaxespad=0,
            handlelength=0.9,
            handletextpad=0.5,
            labelspacing=0.25,
        )
    save(fig, frac, out, "c5_behavior_controls", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/paper")
    args = parser.parse_args()
    set_c2a_style()
    data = read_data()
    regime_figure(args.out)
    dataset_figure(data, args.out)
    difference_figure(data, args.out)
    print(f"Rendered three figures from hash-verified completed summaries in {args.out}")


if __name__ == "__main__":
    main()
