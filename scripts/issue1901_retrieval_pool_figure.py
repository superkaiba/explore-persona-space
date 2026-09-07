#!/usr/bin/env python3
"""Plot the banked training sweep at its original and enlarged retrieval pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    INK,
    MUTED,
    ROLES,
    c2a_figure,
    save_c2a_figure,
    set_c2a_style,
    style_score_axis,
)


def render(source: Path, stem: Path) -> dict:
    """Render completed results and record exactly which data were displayed."""
    data = json.loads(source.read_text())
    if data["status"] != "complete" or data["coverage"]["completed_cells"] != 18:
        raise ValueError("plot requires all 18 banked prediction cells")
    set_c2a_style()
    fig, fraction = c2a_figure("wide", aspect=0.65)
    axes = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.23, top=0.71, wspace=0.13)
    n_pool = data["retrieval"]["n_pool"]
    n_query = data["retrieval"]["n_query"]
    ns = np.array(sorted(map(int, data["per_n"])))
    plotted = {}
    for ax, predictor, role in zip(axes, ("ridge", "mlp"), ("linear", "nonlinear"), strict=True):
        style = ROLES[role]
        cells = [data["per_n"][str(n)][predictor] for n in ns]
        old = np.array([c["original_pool_metrics"]["whiten_csls"]["acc_at_k"]["1"] for c in cells])
        new = np.array([c["top1"] for c in cells])
        lo = np.array([c["top1_ci95"]["lo"] for c in cells])
        hi = np.array([c["top1_ci95"]["hi"] for c in cells])
        style_score_axis(ax, y_min=0.5, y_max=1.005, y_step=0.1)
        ax.plot(ns, old, ":", marker=style.marker, mfc="white", color=style.color, lw=1.8, ms=6)
        ax.errorbar(
            ns,
            new,
            yerr=np.stack([new - lo, hi - new]),
            linestyle="--",
            marker=style.marker,
            mfc="white",
            color=style.color,
            lw=2.5,
            elinewidth=1.2,
            capsize=2.5,
            ms=7,
        )
        ax.set_xscale("log")
        ax.set_xticks([5000, 25000, 100000, 963444], ["5k", "25k", "100k", "963k"])
        ax.minorticks_off()
        ax.set_xlabel("Training contexts")
        ax.set_title(f"{style.label} metamodel", loc="left", color=style.color, pad=14)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        plotted[predictor] = {
            "n_train": ns.tolist(),
            "original_top1": old.tolist(),
            "expanded_top1": new.tolist(),
            "ci95_lo": lo.tolist(),
            "ci95_hi": hi.tolist(),
        }
    axes[0].set_ylabel("Top-1 retrieval")
    fig.text(
        0.105,
        0.97,
        f"Retrieval with {n_pool:,} candidates",
        ha="left",
        va="top",
        color=INK,
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.105,
        0.895,
        f"{n_query:,} held-out queries · Five-answer means · Qwen2.5-7B-Instruct",
        color=MUTED,
        fontsize=12,
    )
    handles = [
        Line2D([0], [0], color=INK, linestyle=style, lw=2.2, label=label)
        for style, label in (
            (":", f"Original: {n_query:,} candidates"),
            ("--", f"New: {n_pool:,} candidates"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.097, 0.865),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.105,
        0.105,
        "Error bars: 95% query-bootstrap intervals for the fixed candidate bank.",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.105,
        0.06,
        f"Chance: {100 / n_pool:.2f}%. The 1,200-training-context point has no saved predictions.",
        color=MUTED,
        fontsize=11,
    )
    outputs = save_c2a_figure(
        fig,
        stem,
        title="Retrieval with a larger candidate pool",
        subject="Same queries and five-rollout means; banked training-size sweep",
        creator="scripts/issue1901_retrieval_pool_figure.py",
        include_width=fraction,
    )
    plt.close(fig)
    metadata = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source": str(source.relative_to(ROOT)),
        "render": outputs["record"],
        "plotted": plotted,
    }
    stem.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {k: str(outputs[k]) for k in ("pdf", "png", "grayscale")}


def main() -> None:
    """Render the isolated comparison without changing manuscript assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "eval_results/issue_1901/retrieval_10k/summary.json"
    )
    parser.add_argument(
        "--stem", type=Path, default=ROOT / "figures/issue_1901/retrieval_10k/training_sweep"
    )
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.stem), indent=2))


if __name__ == "__main__":
    main()
