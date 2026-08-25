"""Poster plot 7 — application: behavior prediction before inference.

ONE wide single-panel figure for the MATS 2026 poster: held-out Spearman rho
vs labeled budget L on the issue-1739 evil (judged misalignment) composition
cells with the half-in-domain map pool (f_U = 0.5, context_end variant,
U = 5,000-row map pool — plan section 4b). Four series — a three-way
comparison of deployable pre-inference readouts plus an oracle ceiling:

  - arm6_map_proj_e1   persona-vector readout on the MAPPED answer vector
                       (v_A_hat; the map is fit on unjudged text — labels
                       only pick the layer)
  - arm4_ridge_ctx     ridge probe fit directly on the raw context vector,
                       no map (consumes the L labels)
  - arm1_ctx_e1        the SAME persona-vector (mean-difference, synthetic
                       contrastive pairs) readout applied to the raw context
                       vector — the existing standard method
  - arm11_oracle_proj  persona-vector readout on the REAL answer vector
                       (oracle ceiling: requires the answer to be generated)

Every number is read from the committed per-cell table
eval_results/issue_1739/evil/arm_results/percell/cells.jsonl; nothing is
hand-typed. Each composition cell is a single draw and seed (n=1 per cell);
individual cells are drawn as small markers, lines join per-budget means
(two cells at L=250 and L=2,500 — f_L in {0, 1}; one cell at L=8,000).

Run:
    uv run python docs/posters/mats_2026/make_plot7_application.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
CELLS = REPO / "eval_results/issue_1739/evil/arm_results/percell/cells.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "figures"

ARMS = {
    "arm6_map_proj_e1": "PV readout, mapped answer $\\hat{v}_A$",
    "arm4_ridge_ctx": "ridge probe, context $v_C$ (no map)",
    "arm1_ctx_e1": "PV readout, context $v_C$",
    "arm11_oracle_proj": "PV readout, real answer (oracle)",
}
BUDGETS = [250, 2500, 8000]


def load_composition_cells() -> list[dict]:
    """Context-variant f_U=0.5 composition cells, one record per cell."""
    out = []
    with open(CELLS) as f:
        for line in f:
            cell = json.loads(line)
            key = json.loads(cell["unit_key"])
            if key.get("f_u") != 0.5 or key["variant"] != "context_end":
                continue
            rho = {a["arm"]: a["rho_frozen"] for a in cell["arms"]}
            out.append(
                {
                    "budget_l": key["budget_l"],
                    "f_l": key["f_l"],
                    "u_rung_label": key["u_rung_label"],
                    "seed": key["seed"],
                    "draw": key["draw"],
                    "rho": {arm: rho[arm] for arm in ARMS},
                }
            )
    if not out:
        raise RuntimeError(f"no context_end f_U=0.5 composition cells in {CELLS}")
    return out


def main() -> None:
    cells = load_composition_cells()

    set_paper_style("iclr", font_scale=1.9)
    fig, ax = plt.subplots(figsize=(6.8, 2.8), constrained_layout=True)

    pal = paper_palette(3)
    style = {
        "arm6_map_proj_e1": dict(color=pal[0], ls="-", zorder=5),
        "arm4_ridge_ctx": dict(color=pal[1], ls="-", zorder=4),
        "arm1_ctx_e1": dict(color=pal[2], ls="-", zorder=3),
        "arm11_oracle_proj": dict(color="0.45", ls="--", zorder=2),
    }

    data_out: dict = {
        "source": str(CELLS.relative_to(REPO)),
        "selection": "variant=context_end, f_u=0.5 (half-in-domain map pool, U=5000), "
        "behavior=evil, eval_rung=train, metric=rho_frozen, seed=0, draw=0, n=1 per cell",
        "cells": cells,
        "per_budget_means": {},
    }

    for arm, label in ARMS.items():
        by_l = defaultdict(list)
        for c in cells:
            by_l[c["budget_l"]].append(c["rho"][arm])
        xs = [b for b in BUDGETS if b in by_l]
        means = [sum(by_l[b]) / len(by_l[b]) for b in xs]
        st = style[arm]
        # individual cells (n=1 each) as small open markers
        for b in xs:
            ax.scatter(
                [b] * len(by_l[b]),
                by_l[b],
                s=26,
                facecolor="white",
                edgecolor=st["color"],
                linewidths=1.1,
                zorder=st["zorder"] + 1,
            )
        ax.plot(
            xs,
            means,
            marker="o",
            markersize=6,
            color=st["color"],
            ls=st["ls"],
            lw=2.2,
            label=label,
            zorder=st["zorder"],
        )
        data_out["per_budget_means"][arm] = {
            "label": label,
            "budgets": xs,
            "mean_rho": means,
            "cell_rhos": {str(b): by_l[b] for b in xs},
        }

    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels(["250", "2,500", "8,000"])
    ax.minorticks_off()
    ax.set_xlabel("labeled (judged) examples")
    ax.set_ylabel("held-out Spearman $\\rho$")
    ax.set_ylim(0.0, 0.95)
    ax.set_title("Misalignment prediction vs. label budget")
    ax.legend(loc="upper center", ncol=2, frameon=False, handlelength=1.5, columnspacing=1.2)

    paths = savefig_paper(fig, "plot7_application", dir=OUT_DIR)
    plt.close(fig)

    with open(OUT_DIR / "plot7_application_data.json", "w") as f:
        json.dump(data_out, f, indent=2)
    for p in paths.values():
        print(p)
    print(OUT_DIR / "plot7_application_data.json")


if __name__ == "__main__":
    main()
