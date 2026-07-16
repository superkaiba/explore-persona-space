#!/usr/bin/env python
"""Look-ahead figure: held-out R² of the linear context→answer map when the
target is the assistant answer 1, 2, or 3 assistant turns ahead of the
current context.

Two data sources, both ridge at layer 19:
- D5 first-state horizon read (#1092 dynamics store): first context of 497
  real logged multi-turn WildChat/LMSYS conversations → the logged answer
  1/2/3 assistant turns later. Fold-level points (grouped 6-fold CV) behind
  each mean. CAVEAT (prose, not on-figure): the pretrained t+1 and t+2
  entries are byte-identical in the store (same per-fold values) — same
  storage-defect class the turn-depth audit found on this store's
  answer-source axis; treat t+2 pretrained as unaudited.
- M-track 2-turn cells (#825, n=2,000/cell, Haiku-4.5-as-user 2-turn
  conversations): first context → assistant answer two turns ahead, chat and
  naturalistic renders, both models. Single committed point estimates.

Values are read from the committed eval JSONs at run time (no hardcoding).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

D5_JSON = REPO / "eval_results/issue_1092/p7/dynamics_D0_D5.json"
M_CELLS = {
    ("instruct", "chat"): "cells_M_instruct_assistant_chat.json",
    ("instruct", "naturalistic"): "cells_M_instruct_assistant_naturalistic.json",
    ("pretrained", "chat"): "cells_M_pretrained_assistant_chat.json",
    ("pretrained", "naturalistic"): "cells_M_pretrained_assistant_naturalistic.json",
}
D5_CELLS = {"instruct": "cell_inst_claude", "pretrained": "cell_pre_claude"}
LAYER = 19
HORIZONS = [1, 2, 3]

MODEL_LABELS = {"instruct": "Instruct", "pretrained": "Pretrained base"}


def load_d5() -> dict[str, dict[int, dict]]:
    blob = json.loads(D5_JSON.read_text())
    out: dict[str, dict[int, dict]] = {}
    for model, cell in D5_CELLS.items():
        combo = next(
            c for c in blob["combos"] if c["cell"] == cell and str(c["layer"]) == str(LAYER)
        )
        horizon = combo["dynamics"]["D5_first_state_horizon"]["context_k"]["0"]
        out[model] = {
            h: {
                "r2": horizon[f"answer_k_t{h}"]["fit"]["r2"],
                "folds": horizon[f"answer_k_t{h}"]["fit"]["r2_folds"],
            }
            for h in HORIZONS
        }
    return out


def load_m_track() -> dict[tuple[str, str], float]:
    return {
        key: json.loads((HERE.parent.parent / "eval_results/issue_825" / fname).read_text())[
            "r2_per_layer_obs"
        ][LAYER]
        for key, fname in M_CELLS.items()
    }


def main() -> None:
    d5 = load_d5()
    m_track = load_m_track()

    set_paper_style("blog")
    fig, ax = plt.subplots()

    colors = dict(zip(["instruct", "pretrained"], paper_palette(2)))
    ax.axhline(0.0, color="#B0B0B0", linewidth=0.8, zorder=1)

    for model in ["instruct", "pretrained"]:
        c = colors[model]
        means = [d5[model][h]["r2"] for h in HORIZONS]
        folds = [d5[model][h]["folds"] for h in HORIZONS]
        sds = [
            (sum((f - m) ** 2 for f in fs) / (len(fs) - 1)) ** 0.5 for fs, m in zip(folds, means)
        ]
        for h, fs in zip(HORIZONS, folds):
            ax.scatter([h - 0.04] * len(fs), fs, s=12, color=c, alpha=0.35, zorder=2, linewidths=0)
        ax.errorbar(
            HORIZONS,
            means,
            yerr=sds,
            color=c,
            marker="o",
            markersize=6,
            capsize=3,
            markeredgewidth=1.0,
            zorder=3,
            label=f"{MODEL_LABELS[model]} — logged multi-turn corpus (n=497)",
        )

    seen_model_labels: set[str] = set()
    for (model, render), r2 in m_track.items():
        c = colors[model]
        x = 2.10 if render == "chat" else 2.18
        label = None
        if model not in seen_model_labels:
            label = f"{MODEL_LABELS[model]} — 2-turn corpus (n=2,000)"
            seen_model_labels.add(model)
        ax.scatter(
            [x],
            [r2],
            marker="D",
            s=42,
            facecolors="none",
            edgecolors=c,
            linewidths=1.4,
            zorder=4,
            label=label,
        )
        ax.text(x + 0.05, r2, render, fontsize=7.5, va="center", color=c)

    ax.set_xticks(HORIZONS)
    ax.set_xticklabels(["1 (next answer)", "2", "3"])
    ax.set_xlabel("assistant answers ahead of the current context")
    ax.set_ylabel("held-out R² (ridge, layer 19)")
    add_direction_arrow(ax, "y", "up")
    ax.set_xlim(0.6, 3.4)

    handles, labels = ax.get_legend_handles_labels()
    order = sorted(
        range(len(labels)), key=lambda i: ("2-turn" in labels[i], "Pretrained" in labels[i])
    )
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower left",
        fontsize=8,
    )
    set_title_subtitle(
        ax,
        "Predicting answers further ahead from the current context",
        "Linear (ridge) map at layer 19; fold-level points behind each mean.\n"
        "Diamonds: separate 2-turn corpus, answer two turns ahead (chat / naturalistic).",
    )
    # set_title_subtitle's pad assumes a one-line subtitle; make room for two.
    import matplotlib as mpl

    ax.set_title(
        ax.get_title(loc="left"),
        loc="left",
        pad=42,
        color="#1A1A1A",
        fontweight=mpl.rcParams.get("axes.titleweight", "semibold"),
        fontsize=mpl.rcParams.get("axes.titlesize", 13),
    )

    savefig_paper(fig, "issue_825/context_lookahead_r2", dir=str(REPO / "figures"))
    plt.close(fig)


if __name__ == "__main__":
    main()
