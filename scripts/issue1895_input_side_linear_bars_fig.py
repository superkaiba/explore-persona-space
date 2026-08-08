"""Linear-only fair-grid bar chart for the #1895 input-side round.

One bar per fair-grid cell, all adjacent, ridge fits only. Bar colour keys the
INPUT (v_C blue / r_bar_C orange / e_C grey — same colour<->meaning mapping as
input_side_linear_vs_mlp.png); hatch keys the TARGET (solid = v_A, hatched =
r_bar_A).

Usage:
    uv run python scripts/issue1895_input_side_linear_bars_fig.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import set_paper_style

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval_results/issue_1895/input_side"
FIG = REPO / "figures/issue_1895"

V_C, R_C, E_C = "#0072B2", "#D55E00", "0.45"

# (cell key, x label, colour, hatch)
CELLS = (
    ("vC__to__vA", "$v_C \\to v_A$", V_C, ""),
    ("vC__to__rbarA", "$v_C \\to \\bar r_A$", V_C, "//"),
    ("rbarC__to__vA", "$\\bar r_C \\to v_A$", R_C, ""),
    ("rbarC__to__rbarA", "$\\bar r_C \\to \\bar r_A$", R_C, "//"),
    ("eC__to__vA", "$e_C \\to v_A$", E_C, ""),
)


def main() -> None:
    lin = json.loads((EVAL / "input_side_summary.json").read_text())["leg_b_fair_grid"]["cells"]
    vals = [float(lin[k]["pooled_r2_te"]) for k, *_ in CELLS]

    set_paper_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(CELLS))
    bars = ax.bar(
        x,
        vals,
        0.66,
        color=[c for _, _, c, _ in CELLS],
        hatch=[h for *_, h in CELLS],
        edgecolor="white",
        linewidth=0.6,
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.008,
            f"{v:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, *_ in CELLS], fontsize=11)
    ax.set_ylabel("held-out pooled $R^2$ (20k contexts)")
    ax.set_ylim(0.0, 1.06)
    ax.set_title("Is our mapping privileged to SAE reconstructable directions?", pad=12)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=V_C),
        plt.Rectangle((0, 0), 1, 1, color=R_C),
        plt.Rectangle((0, 0), 1, 1, color=E_C),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.3", hatch="//"),
    ]
    labels = [
        "input: $v_C$  (full context state)",
        "input: $\\bar r_C$  (SAE-reconstructable part)",
        "input: $e_C = v_C - \\bar r_C$  (SAE-missed part)",
        "target: $\\bar r_A$  (else $v_A$)",
    ]
    ax.legend(handles, labels, loc="upper center", ncol=2, fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / "input_side_linear_bars"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    (stem.with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "source": "eval_results/issue_1895/input_side/input_side_summary.json",
                "cells": [k for k, *_ in CELLS],
                "pooled_r2": vals,
                "what_is_plotted": (
                    "Held-out pooled R2 on the 20k holdout for each fair-grid cell, "
                    "LINEAR (val-selected ridge) only. All cells share the dense "
                    "3,584-dim space, the same kept rows (ans_all_out excluded: 0) "
                    "and the same recipe; bar colour keys the input, hatch keys the "
                    "r_bar_A target."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {stem}.png / .pdf / .meta.json")


if __name__ == "__main__":
    main()
