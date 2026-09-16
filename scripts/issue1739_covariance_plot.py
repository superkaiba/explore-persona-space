"""Plot all four cached covariance controls, with both ridge grids and saved CIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from explore_persona_space.analysis import c2a_plot_style as style

METHODS = (
    ("unwhitened", "Context: unwhitened", style.ROLES["base_model"].color, "s"),
    ("generic_covariance", "Context: chat-covariance whitening", style.INK, "^"),
    ("union_covariance", "Context: union-covariance whitening", style.ROLES["control"].color, "D"),
    ("mapped_answer", "Mapped answer", style.ROLES["linear"].color, "o"),
)
DATASETS = {
    "evil": ("wildchat_rung", "hhrt", "toxicchat"),
    "sycophancy": ("wildchat_rung", "aita"),
    "hallucination": ("wildchat_rung", "nqopen", "simpleqa"),
}
LABELS = {
    "wildchat_rung": "WildChat",
    "hhrt": "HH-RLHF",
    "toxicchat": "ToxicChat",
    "aita": "Reddit AITA",
    "nqopen": "NQ-Open",
    "simpleqa": "SimpleQA",
}
GRIDS = (("historical_grid", "Original grid"), ("wide_grid", "Expanded grid"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(args, cells, stem_name, expected_points):
    style.set_c2a_style()
    fig, fraction = style.c2a_figure("full", aspect=0.72)
    axes = fig.subplots(2, len(cells), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.987, bottom=0.15, top=0.74, wspace=0.20, hspace=0.63)
    fig.suptitle("Behavior prediction from context", y=0.985, fontweight=650)
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            color=color,
            markerfacecolor=color,
            linestyle="none",
            markersize=8,
            label=label,
        )
        for _, label, color, marker in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.949),
        ncol=2,
        columnspacing=1.6,
        handletextpad=0.4,
        labelspacing=0.75,
    )
    inputs, points = [], []
    for ci, cell in enumerate(cells):
        source = args.results / cell / "results.json"
        payload = json.loads(source.read_text())
        relative = str(source.relative_to(args.results))
        assert digest(source) == args.proof["files"][relative]["sha256"], source
        assert payload["source_sha"] == args.proof["source_sha"]
        inputs.append({"path": str(source), "sha256": digest(source)})
        behavior, layer = payload["behavior"], payload["layer"]
        order = DATASETS[behavior]
        for ri, (grid, grid_label) in enumerate(GRIDS):
            ax = axes[ri, ci]
            by_dataset = {row["rung"]: row for row in payload["summaries"][grid]["results"]}
            assert set(by_dataset) == set(order), (cell, grid, set(by_dataset))
            x = np.arange(len(order), dtype=float)
            for mi, (method, label, color, marker) in enumerate(METHODS):
                values = [by_dataset[d]["arms"][method] for d in order]
                y = np.array([v["rho"] for v in values])
                intervals = np.array([v["ci95"] for v in values])
                assert np.isfinite(y).all() and np.isfinite(intervals).all()
                err = np.stack([y - intervals[:, 0], intervals[:, 1] - y])
                assert (err >= 0).all()
                ax.errorbar(
                    x + (mi - 1.5) * 0.17,
                    y,
                    yerr=err,
                    fmt=marker,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=6.5,
                    elinewidth=1.3,
                    capsize=2.5,
                    linestyle="none",
                    label=label,
                    zorder=3,
                )
                for d, value in zip(order, values, strict=True):
                    points.append(
                        {
                            "cell": cell,
                            "grid": grid,
                            "dataset": d,
                            "method": method,
                            "n": by_dataset[d]["n"],
                            **value,
                        }
                    )
            style.style_axis(ax)
            ax.axhline(0, color=style.SEAM, lw=0.8, zorder=1)
            ax.set_ylim(-0.045, 0.84)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_xlim(-0.52, len(order) - 0.48)
            ax.set_xticks(x, [LABELS[d] for d in order])
            if ci == 0:
                ax.set_ylabel(style.better_label("Spearman correlation"))
            style.panel_header(
                ax,
                chr(65 + ri * len(cells) + ci),
                f"{grid_label} · Layer {layer}",
                behavior.capitalize(),
            )
    assert len(points) == expected_points, len(points)
    fig.text(
        0.085,
        0.082,
        "Original grid: λ = 0.01–1,000. Expanded grid: λ = 0.01–1,000,000.",
        color=style.MUTED,
    )
    fig.text(
        0.085,
        0.040,
        "Points: held-out scores. Whiskers: 95% group-bootstrap intervals.",
        color=style.MUTED,
    )
    stem = args.out / stem_name
    saved = style.save_c2a_figure(
        fig,
        stem,
        title="Matched covariance controls",
        subject="Four methods, frozen layers, two ridge grids, saved 95% intervals",
        creator=str(Path(__file__).resolve()),
        include_width=fraction,
    )
    metadata = {
        "inputs": inputs,
        "source_revision": args.proof["source_sha"],
        "verified_data_revision": args.proof["verified_revision"],
        "script_sha256": digest(__file__),
        "plotted_points": points,
        "cells": cells,
        "expected_points": expected_points,
        "methods": [m[0] for m in METHODS],
        "render": saved["record"],
        "notes": [
            "Every method uses supervised ridge with train-only coordinate standardization.",
            "Union covariance uses generic chat plus eliciting training contexts.",
            "Generic covariance uses generic chat contexts only.",
            "Map uses the historical ADD recipe; no inference or refitting in this plot.",
            "Intervals use 2000 group resamples conditional on each fitted model.",
            "Regularization is selected by training-data GCV; no evaluation-based selection.",
        ],
        "output_sha256": {key: digest(saved[key]) for key in ("png", "pdf", "grayscale")},
    }
    stem.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    plt.close(fig)
    print(json.dumps({k: str(saved[k]) for k in ("png", "pdf", "grayscale")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--verification", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.proof = json.loads(args.verification.read_text())
    plot(
        args,
        ["evil_L20", "sycophancy_L19", "hallucination_L20"],
        "covariance_all_methods_20260916",
        64,
    )
    plot(args, ["evil_L18", "sycophancy_L20"], "covariance_all_methods_companion_20260916", 40)


if __name__ == "__main__":
    main()
