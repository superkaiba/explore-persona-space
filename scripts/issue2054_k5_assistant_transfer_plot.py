"""Render assistant-source transfer with exact fold-mean R2 annotations."""

# ruff: noqa: E402
# The source helper loads thread caps before scientific dependencies.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from explore_persona_space.analysis.c2a_plot_style import (
    INK,
    MUTED,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
)
from scripts import issue2054_k5_assistant_transfer as analysis


def plot(out, fig_dir):
    """Show all four character additions; mask in-source targets explicitly."""
    data = analysis.collect(out)
    rows = {(p["model"], p["regime"], p["cell"]): p for p in data["panels"]}
    variants = [
        ("frozen", "Frozen transfer"),
        ("bias", "+ Bias"),
        ("bias_scale", "+ Bias + scaling"),
    ]
    values = [p["r2_mean"][key] for p in data["panels"] for key, _ in variants]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite plotted score")
    lower, upper = min(-0.01, min(values)), max(0.01, max(values))
    norm = TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#F0F0F0")
    set_c2a_style()
    fig, fraction = c2a_figure("full", aspect=0.76)
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.16, top=0.91, wspace=0.13, hspace=0.63)
    targets = analysis.base.SETTINGS[1:]
    row_labels = ["Assistant only"] + [
        f"Assistant + {label}" for label, _ in analysis.base.SETTINGS[2:]
    ]
    for mi, (model, model_label) in enumerate(
        zip(analysis.base.MODELS, ["Base", "Instruction-tuned"], strict=True)
    ):
        regimes = analysis.source_sets(model)
        for vi, (key, title) in enumerate(variants):
            ax = axes[mi, vi]
            matrix = np.full((5, 5), np.nan)
            for i, (regime, source_cells) in enumerate(regimes.items()):
                for j, (_, prefix) in enumerate(targets):
                    cell = f"{prefix}__{model}"
                    if cell not in source_cells:
                        matrix[i, j] = rows[(model, regime, cell)]["r2_mean"][key]
            im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
            for (i, j), value in np.ndenumerate(matrix):
                if np.isnan(value):
                    ax.text(j, i, "—", ha="center", va="center", color=MUTED)
                else:
                    rgb = cmap(norm(value))[:3]
                    lightness = np.dot(rgb, [0.2126, 0.7152, 0.0722])
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=INK if lightness > 0.58 else "white",
                    )
            ax.set_xticks(
                range(5),
                ["Assistant\n(plain text)"] + [label for label, _ in targets[1:]],
                rotation=35,
                ha="right",
            )
            ax.set_yticks(range(5), row_labels if vi == 0 else [""] * 5)
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            panel_header(ax, chr(ord("A") + mi * 3 + vi), model_label, title=title)
    bar_ax = fig.add_axes([0.30, 0.035, 0.56, 0.018])
    cb = fig.colorbar(im, cax=bar_ax, orientation="horizontal")
    cb.set_label(better_label("Held-out $R^2$"))
    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = fig_dir / "assistant_source_transfer"
    saved = save_c2a_figure(
        fig,
        stem,
        title="Transfer from assistant and assistant plus one character",
        subject=data["method"],
        creator=Path(__file__).name,
        include_width=fraction,
    )
    analysis.base.atomic_json(stem.with_suffix(".data.json"), data)
    analysis.base.atomic_json(
        stem.with_suffix(".meta.json"),
        {
            "render": saved["record"],
            "source_sha256": analysis.base.sha(out / "results.json"),
            "script_sha256": analysis.base.sha(__file__),
            "method": data["method"],
            "missing_cells": "Dashes identify targets included in source training; not evaluated as transfer",
            "aggregation": "Unweighted mean of the same five held-out conversation folds",
            "calibration": "Bias and scalar use target non-test-fold labels; frozen uses no target labels",
        },
    )
    plt.close(fig)
    print(f"[phase=done] {saved['png']}", flush=True)


def main():
    """Read verified fit artifacts and export color, vector, and grayscale plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    args = parser.parse_args()
    plot(args.out, args.fig_dir)


if __name__ == "__main__":
    main()
