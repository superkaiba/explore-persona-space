"""Issue #1310 follow-up figures: scene-aggregated prefill re-fit (9a-ter fold).

Reads ONLY committed JSONs under eval_results/issue_1310/onpolicy/ (per-turn,
round 1) and eval_results/issue_1310/onpolicy_aggregated/ (scene-aggregated,
@18791c2a) and renders:

  1. agg_vs_perturn_l19  — per-persona held-out R^2 @L19, per-turn points vs
                           one-point-per-scene aggregation, base + instruct
                           panels, bootstrap CIs + shuffle-null ticks. No
                           assistant-ceiling line: the aggregated grain is not
                           comparable to the per-turn ceiling (Y-averaging
                           mechanically raises attainable R^2).
  2. agg_l19_points      — low-level per-unit sibling: per-point (~300 scenes)
                           + per-fold R^2 + pooled diamond per aggregated cell,
                           incl. the pooled swap correct/swapped cells (from
                           analyzer_agg_perfold_l19.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps before heavy imports (#847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

EV = REPO / "eval_results" / "issue_1310"
OUT = REPO / "figures" / "issue_1310"
PERSONAS = ("Wren", "HELIOS", "Dana", "Vex")
PERSONA_GLOSS = {
    "Wren": "Wren (helpful)",
    "HELIOS": "HELIOS (ship AI)",
    "Dana": "Dana (ordinary)",
    "Vex": "Vex (villain)",
}
L = 19
MODEL_LABEL = {"base": "Qwen2.5-7B (base)", "instruct": "Qwen2.5-7B-Instruct"}


def load_cell(kind: str, model: str, persona: str) -> dict:
    """kind: 'perturn' (onpolicy/) or 'agg' (onpolicy_aggregated/)."""
    p = (
        EV / "onpolicy" / f"cells_onpolicy_{model}_{persona}.json"
        if kind == "perturn"
        else EV / "onpolicy_aggregated" / f"cells_agg_{model}_{persona}.json"
    )
    d = json.loads(p.read_text())
    fro = d["selection_symmetric"]["frozen_layer_table"][str(L)]
    boot = d["r2_bootstrap_row_frozen"][str(L)]
    return {
        "n": d["n"],
        "r2": d["r2_per_layer_obs"][L],
        "null_p975": fro["null_p975"],
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
    }


def fig_agg_vs_perturn() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    colors = paper_palette_blog(4)
    c_kind = {"perturn": colors[1], "agg": colors[2]}
    kind_label = {
        "perturn": "per-turn points (round 1)",
        "agg": "one point per scene (aggregated)",
    }
    width = 0.36
    xs = np.arange(len(PERSONAS))
    for ax, model in zip(axes, ("base", "instruct"), strict=True):
        for j, kind in enumerate(("perturn", "agg")):
            offs = (j - 0.5) * width
            for i, persona in enumerate(PERSONAS):
                cell = load_cell(kind, model, persona)
                x = xs[i] + offs
                ax.bar(
                    x,
                    cell["r2"],
                    width=width * 0.92,
                    color=c_kind[kind],
                    label=kind_label[kind] if i == 0 else None,
                    zorder=3,
                )
                # bootstrap 95% CI segment (see hero: the bootstrap point
                # estimate can sit a hair off the pooled observed R^2)
                ax.vlines(x, cell["ci_lo"], cell["ci_hi"], color="0.25", lw=1.2, zorder=4)
                # shuffle-null 97.5th percentile tick for this cell
                ax.plot(
                    [x - width * 0.46, x + width * 0.46],
                    [cell["null_p975"]] * 2,
                    color="0.15",
                    lw=1.0,
                    ls=":",
                    zorder=5,
                )
                ax.text(
                    x,
                    cell["r2"] + (0.012 if cell["r2"] >= 0 else -0.012),
                    f"{cell['r2']:+.2f}",
                    ha="center",
                    va="bottom" if cell["r2"] >= 0 else "top",
                    fontsize=7.5,
                    color="0.2",
                )
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([PERSONA_GLOSS[p] for p in PERSONAS], fontsize=9)
        ax.set_title(MODEL_LABEL[model], fontsize=12, pad=10, loc="left", fontweight="semibold")
    axes[0].set_ylabel("held-out R² at layer 19")
    from matplotlib.lines import Line2D

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.15", lw=1.0, ls=":"))
    labels.append("shuffle-null 97.5th pct")
    axes[0].legend(handles, labels, loc="upper left", fontsize=9)
    savefig_paper(fig, "agg_vs_perturn_l19", dir=OUT)
    plt.close(fig)


def fig_agg_points() -> None:
    """Low-level per-unit view of the aggregated fits: per-point + per-fold R^2."""
    pf_path = EV / "onpolicy_aggregated" / "analyzer_agg_perfold_l19.json"
    d = json.loads(pf_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    colors = paper_palette_blog(4)
    rng = np.random.default_rng(1)
    CLIP = 1.0
    for ax, model in zip(axes, ("base", "instruct"), strict=True):
        slots = [(p, f"agg_{model}_{p}") for p in PERSONAS]
        xs = np.arange(len(slots) + 2)
        for i, (_persona, cid) in enumerate(slots):
            c = d["cells"][cid]
            pg = np.clip([v["r2"] for v in c["pergroup"].values()], -CLIP, CLIP)
            jit = rng.uniform(-0.26, 0.26, size=len(pg))
            ax.scatter(
                xs[i] + jit,
                pg,
                s=5,
                color="0.6",
                alpha=0.35,
                lw=0,
                zorder=2,
                label="per-scene point R² (~300, clipped ±1)"
                if (i == 0 and model == "base")
                else None,
            )
            pf = [f["r2"] for f in c["perfold"]]
            ax.scatter(
                [xs[i]] * len(pf),
                pf,
                s=34,
                facecolors="none",
                edgecolors="0.1",
                linewidths=1.1,
                zorder=3,
                label="per-fold R² (5 folds)" if (i == 0 and model == "base") else None,
            )
            ax.scatter(
                [xs[i]],
                [c["pooled"]],
                s=54,
                color=colors[2],
                marker="D",
                edgecolors="0.15",
                linewidths=0.7,
                zorder=4,
                label="pooled (recomputed = committed)" if (i == 0 and model == "base") else None,
            )
        for k, key in enumerate(("correct", "swap")):
            s = d["swap"][model][key]
            pg = np.clip([v["r2"] for v in s["pergroup"].values()], -CLIP, CLIP)
            jit = rng.uniform(-0.26, 0.26, size=len(pg))
            x = xs[len(slots) + k]
            ax.scatter(x + jit, pg, s=5, color="0.6", alpha=0.35, lw=0, zorder=2)
            pf = [f["r2"] for f in s["perfold"]]
            ax.scatter(
                [x] * len(pf),
                pf,
                s=34,
                facecolors="none",
                edgecolors="0.1",
                linewidths=1.1,
                zorder=3,
            )
            ax.scatter(
                [x],
                [s["pooled"]],
                s=54,
                color=colors[3],
                marker="D",
                edgecolors="0.15",
                linewidths=0.7,
                zorder=4,
            )
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [PERSONA_GLOSS[p] for p in PERSONAS] + ["pooled correct", "pooled swapped"],
            fontsize=8.5,
            rotation=12,
        )
        ax.set_title(
            f"scene-aggregated — {MODEL_LABEL[model]}",
            fontsize=12,
            pad=10,
            loc="left",
            fontweight="semibold",
        )
    axes[0].set_ylabel("held-out R² at layer 19")
    axes[0].legend(loc="lower left", fontsize=8.5)
    savefig_paper(fig, "agg_l19_points", dir=OUT)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_agg_vs_perturn()
    fig_agg_points()
    print("[agg-figures] wrote", OUT / "agg_vs_perturn_l19.png", OUT / "agg_l19_points.png")


if __name__ == "__main__":
    main()
