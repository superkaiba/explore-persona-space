#!/usr/bin/env python3
"""Render the inline interpretation summary for issue 2544."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval_results" / "issue_2544"
OUT = ROOT / "figures" / "issue_2544" / "inline_map_evolution_summary"

RUNGS = ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "mid", "main"]
RUNG_LABELS = ["Init", "4B", "21B", "88B", "252B", "600B", "1.2T", "2.5T", "4.0T", "5.9T", "Mid", "Base"]

# Okabe-Ito palette, with redundant line/marker encoding.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
GRAY = "#6B6B6B"


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def asymmetric_error(value: float, ci: list[float]) -> tuple[float, float]:
    return value - ci[0], ci[1] - value


def main() -> None:
    diag = load_json(RESULTS / "fits" / "diag_curve.json")
    cross = load_json(RESULTS / "fits" / "cross_cells.json")
    transfer = load_json(RESULTS / "transfer" / "retention_matrix.json")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(RUNGS))
    diagonal = np.array([diag["per_rung"][r]["r2_star"] for r in RUNGS])
    diagonal_err = np.array(
        [asymmetric_error(diagonal[i], diag["per_rung"][r]["ci_frozen"]) for i, r in enumerate(RUNGS)]
    ).T

    fixed_text = []
    fixed_text_err = []
    fixed_weights = []
    fixed_weights_err = []
    for r in RUNGS:
        if r == "main":
            value = diag["per_rung"][r]["r2_star"]
            ci = diag["per_rung"][r]["ci_frozen"]
            fixed_text.append(value)
            fixed_text_err.append(asymmetric_error(value, ci))
            fixed_weights.append(value)
            fixed_weights_err.append(asymmetric_error(value, ci))
            continue
        text_cell = cross["cells"]["colC"][r]
        weight_cell = cross["cells"]["rowR"][r]
        fixed_text.append(text_cell["r2_star"])
        fixed_text_err.append(asymmetric_error(text_cell["r2_star"], text_cell["ci"]))
        fixed_weights.append(weight_cell["r2_star"])
        fixed_weights_err.append(asymmetric_error(weight_cell["r2_star"], weight_cell["ci"]))

    fixed_text = np.asarray(fixed_text)
    fixed_text_err = np.asarray(fixed_text_err).T
    fixed_weights = np.asarray(fixed_weights)
    fixed_weights_err = np.asarray(fixed_weights_err).T

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
        gridspec_kw={"width_ratios": [1.75, 1.0]},
        constrained_layout=True,
    )

    for ax in (ax_a, ax_b):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)

    ax_a.errorbar(
        x,
        fixed_text,
        yerr=fixed_text_err,
        color=SKY,
        linestyle="--",
        marker="s",
        markersize=5,
        linewidth=2,
        capsize=2,
        label="Fixed final answers (representation change)",
    )
    ax_a.errorbar(
        x,
        fixed_weights,
        yerr=fixed_weights_err,
        color=ORANGE,
        linestyle="-.",
        marker="D",
        markersize=5,
        linewidth=2,
        capsize=2,
        label="Fixed final weights (answer-distribution change)",
    )
    ax_a.errorbar(
        x,
        diagonal,
        yerr=diagonal_err,
        color=BLUE,
        linestyle="-",
        marker="o",
        markersize=5,
        linewidth=2.2,
        capsize=2,
        label="Each checkpoint on its own answers",
    )
    ax_a.axvline(9.5, color=GRAY, linestyle=":", linewidth=1)
    ax_a.set_xticks(x, RUNG_LABELS, rotation=35, ha="right")
    ax_a.set_ylim(0.04, 0.515)
    ax_a.set_ylabel("Held-out R² at layer 31")
    ax_a.set_xlabel("Checkpoint (pretraining tokens where applicable)")
    ax_a.set_title("A   Two opposing changes leave aggregate map strength flat", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, loc="upper right")

    late_rungs = ["r5", "r6", "r7", "r8", "r9", "mid", "main"]
    late_labels = ["600B", "1.2T", "2.5T", "4.0T", "5.9T", "Mid", "Base"]
    retention = []
    retention_err = []
    for r in late_rungs:
        if r == "main":
            retention.append(1.0)
            retention_err.append((0.0, 0.0))
            continue
        record = transfer["pairs"][f"{r}->main"]["rho_vs_Q_jj"]["orth"]
        retention.append(record["rho"])
        retention_err.append(asymmetric_error(record["rho"], record["ci"]))
    retention = np.asarray(retention)
    retention_err = np.asarray(retention_err).T
    late_x = np.arange(len(late_rungs))

    ax_b.axhspan(0.8, 1.06, color=PURPLE, alpha=0.08, linewidth=0)
    ax_b.axhline(0.8, color=PURPLE, linestyle="--", linewidth=1.2, label="Registered threshold")
    ax_b.axhline(0.0, color=GRAY, linestyle=":", linewidth=1)
    ax_b.errorbar(
        late_x,
        retention,
        yerr=retention_err,
        color=PURPLE,
        linestyle="-",
        marker="o",
        markersize=5.5,
        linewidth=2.2,
        capsize=2.5,
        label="Procrustes-aligned retention",
    )
    ax_b.axvline(4.5, color=GRAY, linestyle=":", linewidth=1)
    ax_b.set_xticks(late_x, late_labels, rotation=35, ha="right")
    ax_b.set_ylim(-0.9, 1.06)
    ax_b.set_ylabel("Transfer R² / final-base diagonal R²")
    ax_b.set_xlabel("Source checkpoint → final base")
    ax_b.set_title("B   The final operator becomes transferable after pretraining", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="lower right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "figure": OUT.name,
        "sources": [
            "eval_results/issue_2544/fits/diag_curve.json",
            "eval_results/issue_2544/fits/cross_cells.json",
            "eval_results/issue_2544/transfer/retention_matrix.json",
        ],
        "n_shared_intersection": diag["per_rung"]["main"]["n_rows_fitted"],
        "error_bars": "95% bootstrap confidence intervals",
        "panel_a": "Diagonal map strength and reduced-cross decomposition at frozen layer 31.",
        "panel_b": "Late-ladder orthogonal-Procrustes transfer to final base, normalized by the final-base diagonal R2.",
        "note": "Earlier stage-1 retention values are below panel B's displayed range; the panel starts at 600B tokens.",
    }
    with OUT.with_suffix(".meta.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
