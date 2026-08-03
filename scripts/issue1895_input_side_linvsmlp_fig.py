"""Linear-vs-nonlinear fair-grid figure for the #1895 input-side round.

Reads the banked linear cells (input_side_summary.json, leg_b_fair_grid) and the
MLP twins (mlp_cells_summary.json) and renders a paired bar chart: one bar pair
per fair-grid cell, ridge vs MLP, with the nonlinear premium annotated.

The point the figure carries: the nonlinear premium is a flat ~0.027-0.032 on
every cell EXCEPT the SAE-residual input (e_C -> v_A, +0.049), so the SAE input
bottleneck is not a linearity artifact while the discarded residual holds
disproportionately nonlinear structure.

Usage:
    uv run python scripts/issue1895_input_side_linvsmlp_fig.py
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

CELLS = (
    ("vC__to__vA", "$v_C \\to v_A$\n(full context)"),
    ("vC__to__rbarA", "$v_C \\to \\bar r_A$"),
    ("rbarC__to__vA", "$\\bar r_C \\to v_A$\n(SAE-representable input)"),
    ("rbarC__to__rbarA", "$\\bar r_C \\to \\bar r_A$"),
    ("eC__to__vA", "$e_C \\to v_A$\n(SAE-MISSED input)"),
)


def main() -> None:
    lin = json.loads((EVAL / "input_side_summary.json").read_text())["leg_b_fair_grid"]["cells"]
    mlp = json.loads((EVAL / "mlp_cells_summary.json").read_text())["cells"]
    keys = [k for k, _ in CELLS]
    lin_v = [float(lin[k]["pooled_r2_te"]) for k in keys]
    mlp_v = [float(mlp[k]["pooled_r2_te"]) for k in keys]
    gains = [m - t for m, t in zip(mlp_v, lin_v)]

    set_paper_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(keys))
    w = 0.38
    ax.bar(x - w / 2, lin_v, w, color="#0072B2", label="linear (ridge)")
    ax.bar(x + w / 2, mlp_v, w, color="#D55E00", label="nonlinear (MLP w8192)")
    for xi, (lv, mv, g) in enumerate(zip(lin_v, mlp_v, gains)):
        ax.text(xi - w / 2, lv + 0.008, f"{lv:.3f}", ha="center", fontsize=8, color="#0072B2")
        ax.text(xi + w / 2, mv + 0.008, f"{mv:.3f}", ha="center", fontsize=8, color="#D55E00")
        ax.text(
            xi,
            max(lv, mv) + 0.042,
            f"+{g:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold" if g > 0.04 else "normal",
            color="0.25",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in CELLS], fontsize=8)
    ax.set_ylabel("held-out pooled $R^2$ (20k contexts)")
    ax.set_ylim(0.0, 0.90)
    ax.set_title(
        "Is the SAE input bottleneck a linearity artifact? No — except in the residual\n"
        "(nonlinear premium is flat ~0.03 everywhere; the SAE-MISSED input gains +0.049)"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / "input_side_linear_vs_mlp"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    (stem.with_suffix(".meta.json")).write_text(
        json.dumps(
            {
                "source": [
                    "eval_results/issue_1895/input_side/input_side_summary.json",
                    "eval_results/issue_1895/input_side/mlp_cells_summary.json",
                ],
                "cells": keys,
                "linear_pooled_r2": lin_v,
                "mlp_pooled_r2": mlp_v,
                "nonlinear_premium": gains,
                "what_is_plotted": (
                    "Held-out pooled R2 on the 20k holdout for each fair-grid cell, "
                    "ridge vs the parent-recipe MLP (w8192, lr 3e-4, seed 0, "
                    "internal-val early stop); same kept rows and same target per "
                    "cell, so each bar pair varies ONLY the fitter. Annotation = "
                    "nonlinear premium (MLP - ridge)."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {stem}.png / .pdf / .meta.json")
    print("gaps  linear:", round(lin_v[0] - lin_v[2], 4), " mlp:", round(mlp_v[0] - mlp_v[2], 4))


if __name__ == "__main__":
    main()
