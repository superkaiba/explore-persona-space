"""Figures for the #1092 inline fit-free battery-caveat repairs.

fig1  a3_transport_floors_vs_ridge : per cell x arm, banked ridge R² (battery-IN)
      vs the affine transport floors (scaled-identity alpha*I, per-dim diag-affine,
      train-mean) computed battery-EXCLUDED on target t1. Shows the context-arm
      ridge skill (~0.5-0.8) is NOT reachable by identity/affine transport, while
      the prefix-arm ridge (~0.05) sits at its floor.
fig2  read3_read4_battery_invariance : per cell, read3 query variance share and
      read4 additivity residual, banked (battery-INCLUDED) vs recomputed
      battery-EXCLUDED — identical (dense-core-only statistics).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

OUT = Path("eval_results/issue_1092/inline_caveat_repairs_operator_comparison")
FIGDIR = "figures/issue_1092"
CELL_LABEL = {
    "cell_inst_own": "inst / own",
    "cell_inst_claude": "inst / Claude",
    "cell_inst_pretext": "inst / pretrained-text",
    "cell_inst_shuf": "inst / shuffled",
    "cell_pre_own": "pretrained / own",
    "cell_pre_claude": "pretrained / Claude",
    "cell_pre_insttext": "pretrained / inst-text",
    "cell_pre_shuf": "pretrained / shuffled",
}
ORDER = list(CELL_LABEL)


def main() -> None:
    set_paper_style()
    with open(OUT / "fit_free_repairs.json") as fh:
        d = json.load(fh)
    floors = d["A3_transport_floors"]["cells"]
    inv = d["A1_read3_read4_invariance"]
    cells = [c for c in ORDER if c in floors]
    labels = [CELL_LABEL[c] for c in cells]
    x = np.arange(len(cells))

    # ---- fig1: transport floors vs ridge, per arm ----
    pal = paper_palette(4)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, arm, title in (
        (axes[0], "context_end", "Context-arm map (prefix+query end)"),
        (axes[1], "prefix_end", "Prefix-arm map (persona end)"),
    ):
        ridge = [floors[c][arm]["banked_pooled_ridge_r2_batteryIN"] for c in cells]
        aI = [floors[c][arm]["per_target"]["t1"]["global_affine_scaled_identity"] for c in cells]
        diag = [floors[c][arm]["per_target"]["t1"]["diag_affine"] for c in cells]
        tmean = [floors[c][arm]["per_target"]["t1"]["train_mean"] for c in cells]
        w = 0.2
        ax.bar(x - 1.5 * w, ridge, w, label="ridge R² (banked, battery-IN)", color=pal[0])
        ax.bar(x - 0.5 * w, diag, w, label="diag-affine floor", color=pal[1])
        ax.bar(x + 0.5 * w, aI, w, label="scaled-identity a*I floor", color=pal[2])
        ax.bar(x + 1.5 * w, tmean, w, label="train-mean floor", color=pal[3])
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("held-out R² (target t1)")
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle(
        "Ridge transport skill is not reachable by identity/affine transport\n"
        "(raw-identity floor omitted: -1.6 to -4.5 everywhere - the answer state is far "
        "from the input state in raw coordinates)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "inline_a3_transport_floors_vs_ridge", dir=FIGDIR)
    plt.close(fig)

    # ---- fig2: read3/read4 battery-invariance ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    pal2 = paper_palette(2)
    # read3 query share
    ax = axes[0]
    q_bank = [inv[c]["read3_shares_banked"]["query"] for c in cells]
    q_excl = [inv[c]["read3_shares_battery_excluded"]["query"] for c in cells]
    w = 0.38
    ax.bar(x - w / 2, q_bank, w, label="banked (battery-IN)", color=pal2[0])
    ax.bar(x + w / 2, q_excl, w, label="battery-EXCLUDED", color=pal2[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("read3 query variance share")
    ax.set_title("read3 (dense-core variance shares)", fontsize=10)
    ax.legend(fontsize=8)
    # read4 residual
    ax = axes[1]
    r_bank = [inv[c]["read4_residuals_banked"]["residual_interaction_over_total"] for c in cells]
    r_excl = [
        inv[c]["read4_residuals_battery_excluded"]["residual_interaction_over_total"] for c in cells
    ]
    ax.bar(x - w / 2, r_bank, w, label="banked (battery-IN)", color=pal2[0])
    ax.bar(x + w / 2, r_excl, w, label="battery-EXCLUDED", color=pal2[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("read4 residual-interaction / total")
    ax.set_title("read4 (operator additivity)", fontsize=10)
    ax.legend(fontsize=8)
    maxd = max(inv[c]["shares_max_abs_delta"] for c in cells)
    fig.suptitle(
        f"read3 and read4 are battery-invariant (dense-core-only; battery block is disjoint) "
        f"— max share Δ across all cells = {maxd:.1e}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig_paper(fig, "inline_read3_read4_battery_invariance", dir=FIGDIR)
    plt.close(fig)
    print("wrote 2 figures to", FIGDIR)


if __name__ == "__main__":
    main()
