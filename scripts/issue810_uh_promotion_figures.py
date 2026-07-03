"""Promotion figures for #810 round 3 (`user-header-newline-summary`).

The workload's hero1/hero2 for this round rendered with zero data series
(`issue810_analyze.py`'s hero paths key on the round-1 summary names, which
are absent from the uh-only JSON), so this script regenerates the round's
figures from the committed eval JSONs:

  1. uh_hero1_recon_by_layer_folds  — per-layer skill curves, LOCO | LOFO
     panels, 9 new rows (solid) vs the parent mean/max-pool/turn-newline
     benchmarks (dashed), enlarged-axis band + identity ceilings drawn.
  2. uh_hero2_boundary_position_heatmap — 11 summary rows x 28 layers, LOCO.
  3. uh_hero3_crosslayer_pooled — H3 pooled cross-layer targets: LOCO bar,
     LOFO dot, own pooled identity ceiling tick; per-layer best, the 22:30
     answer-only pooled benchmark, and the max-selected band as hlines.
  4. uh_delta_vs_mean_forest — paired bootstrap delta-skill vs the mean, 9
     rows x 3 layer conventions, 95% CI whiskers, 0 and +0.02 floor vlines.

Inputs (all committed):
  eval_results/issue_810/user-header-newline-summary/reconstruction_skill_user_header.json
  eval_results/issue_810/user-header-newline-summary/crosslayer_xbnd.json
  eval_results/issue_810/user-header-newline-summary/delta_vs_mean.json
  eval_results/issue_810/reconstruction_skill_by_summary.json          (parent LOCO)
  eval_results/issue_810/adhoc_lofo_heatmap_grids.json                 (parent LOFO)
"""

from __future__ import annotations

import json

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
UH_DIR = REPO / "eval_results/issue_810/user-header-newline-summary"
FIG_DIR = REPO / "figures/issue_810/user-header-newline-summary"

ROW_LABELS = {
    "uh_im_start": "header start token",
    "uh_user": "header 'user' token",
    "uh_nl": "header newline (mirror read)",
    "uh_mean3": "header mean (3 tokens)",
    "uh_max3": "header max (3 tokens)",
    "bnd_mean5": "boundary mean (5 tokens)",
    "bnd_max5": "boundary max (5 tokens)",
    "mean_xbnd": "whole-turn mean (answer + boundary)",
    "maxp_xbnd": "whole-turn max-pool (answer + boundary)",
}
NEW_ROWS = list(ROW_LABELS)


def load() -> tuple[dict, dict, dict, dict, dict]:
    uh = json.loads((UH_DIR / "reconstruction_skill_user_header.json").read_text())
    cross = json.loads((UH_DIR / "crosslayer_xbnd.json").read_text())
    delta = json.loads((UH_DIR / "delta_vs_mean.json").read_text())
    parent = json.loads(
        (REPO / "eval_results/issue_810/reconstruction_skill_by_summary.json").read_text()
    )
    lofo = json.loads((REPO / "eval_results/issue_810/adhoc_lofo_heatmap_grids.json").read_text())
    return uh, cross, delta, parent, lofo


def parent_lofo_curve(lofo: dict, name: str) -> np.ndarray:
    grid = np.asarray(lofo["grids"]["panel3_reconstruction_lofo_skill_over_mean_r2"])
    return grid[:, lofo["column_order"].index(name)]


def fig_hero1(uh: dict, parent: dict, lofo: dict) -> None:
    layers = np.arange(28)
    band = uh["band_rows"]["enlarged_axis_max_selected"]
    loco_ceiling = band["ceiling"]
    lofo_ceiling = max(r["lofo_skill"] for r in uh["diagnostics"]["lofo_identity_ceiling"])
    colors = paper_palette(8) + ["#7B5233"]  # 9th distinct color (warm brown)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, fold in zip(axes, ("loco", "lofo")):
        for i, row in enumerate(NEW_ROWS):
            cells = uh["by_summary"][row]
            key = "ridge_skill" if fold == "loco" else "lofo_skill"
            ax.plot(layers, [c[key] for c in cells], color=colors[i], lw=1.6, label=ROW_LABELS[row])
        for bench, style in (("mean", "-"), ("maxp", "--"), ("turn_nl", ":")):
            if fold == "loco":
                curve = [c["ridge_skill"] for c in parent["by_summary"][bench]]
            else:
                curve = parent_lofo_curve(lofo, bench)
            label = {
                "mean": "mean summary (benchmark)",
                "maxp": "max-pool (benchmark)",
                "turn_nl": "newline after turn end (benchmark)",
            }[bench]
            ax.plot(layers, curve, style, color="0.25", lw=2.0, label=label)
        if fold == "loco":
            ax.axhline(
                band["band_97_5"],
                color="#5b6b7a",
                ls="--",
                lw=1.2,
                label="max-selected null band (97.5th pct)",
            )
            ax.axhline(loco_ceiling, color="#a05050", ls="-.", lw=1.2, label="identity ceiling")
            ax.set_title("leave-one-context-out (banded)")
        else:
            ax.axhline(lofo_ceiling, color="#a05050", ls="-.", lw=1.2, label="identity ceiling")
            ax.set_title("leave-one-family-out (ordering only)")
        ax.set_xlabel("layer")
        ax.set_ylim(-0.15, 1.0)
    axes[0].set_ylabel("held-out skill-over-mean R² (higher = better)")
    axes[1].legend(loc="lower center", fontsize=7, ncol=2, frameon=True)
    fig.suptitle(
        "Predicting each answer/boundary summary from the context representation",
        fontsize=14,
        fontweight="bold",
    )
    savefig_paper(fig, "uh_hero1_recon_by_layer_folds", dir=FIG_DIR)
    plt.close(fig)


def fig_hero2(uh: dict, parent: dict) -> None:
    rows = ["im_end", "turn_nl"] + NEW_ROWS
    labels = ["turn-end token (round 1)", "newline after turn end (round 1)"] + [
        ROW_LABELS[r] for r in NEW_ROWS
    ]
    mat = []
    for r in rows:
        src = parent["by_summary"][r] if r in ("im_end", "turn_nl") else uh["by_summary"][r]
        mat.append([c["ridge_skill"] for c in src])
    arr = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(rows)), labels=labels, fontsize=8)
    ax.set_xticks(range(0, 28, 5))
    ax.set_xlabel("layer")
    ax.set_title("Reconstruction skill per boundary summary × layer (LOCO)", fontweight="bold")
    fig.colorbar(im, ax=ax, label="skill-over-mean R²")
    savefig_paper(fig, "uh_hero2_boundary_position_heatmap", dir=FIG_DIR)
    plt.close(fig)


def fig_hero3(cross: dict) -> None:
    band_row = cross["band_row_h3"]
    per_target = cross["per_target"]
    # best-over-cc-pool LOCO per pooled target, from the 16-cell pooled-cc grid
    pooled_cells = [c for c in cross["cells"] if c["grid"] == "pooled_cc"] or [
        c for c in cross["cells"] if "cc=" in c["cell"]
    ]
    best_loco: dict[str, float] = {}
    for c in pooled_cells:
        tgt = c["target"]
        best_loco[tgt] = max(best_loco.get(tgt, -np.inf), c["ridge_skill"])
    targets = list(per_target)
    label = {
        "mean_xbnd|answer=layer-mean|raw": "whole-turn mean, layer-mean, raw",
        "mean_xbnd|answer=layer-mean|normed": "whole-turn mean, layer-mean, normed",
        "mean_xbnd|answer=layer-max|raw": "whole-turn mean, layer-max, raw",
        "mean_xbnd|answer=layer-max|normed": "whole-turn mean, layer-max, normed",
        "maxp_xbnd|answer=layer-mean|raw": "whole-turn max-pool, layer-mean, raw",
        "maxp_xbnd|answer=layer-mean|normed": "whole-turn max-pool, layer-mean, normed",
        "maxp_xbnd|answer=layer-max|raw": "whole-turn max-pool, layer-max, raw",
        "maxp_xbnd|answer=layer-max|normed": "whole-turn max-pool, layer-max, normed",
    }
    x = np.arange(len(targets))
    loco = [best_loco.get(t, np.nan) for t in targets]
    lofo = [max(per_target[t]["lofo_skill_by_cc_pool"].values()) for t in targets]
    ceil = [per_target[t]["identity_ceiling_loco"] for t in targets]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, loco, width=0.6, color=paper_palette(3)[0], label="LOCO skill (best cc pool)")
    ax.scatter(
        x, lofo, marker="D", s=42, color="#2d2d2d", zorder=3, label="LOFO skill (ordering only)"
    )
    ax.scatter(
        x,
        ceil,
        marker="_",
        s=340,
        linewidths=2.4,
        color="#a05050",
        zorder=3,
        label="own pooled identity ceiling (LOCO)",
    )
    ax.axhline(
        band_row["per_layer_best_committed"],
        color="0.35",
        ls="--",
        lw=1.4,
        label="per-layer best (max-pool, layer 21)",
    )
    ax.axhline(
        band_row["benchmark_2230_pooled"],
        color="0.35",
        ls=":",
        lw=1.6,
        label="answer-only pooled benchmark (unregistered inline read)",
    )
    ax.axhline(
        band_row["band_97_5"],
        color="#5b6b7a",
        ls="--",
        lw=1.2,
        label="max-selected null band (97.5th pct)",
    )
    ax.set_xticks(x, labels=[label[t] for t in targets], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("held-out skill-over-mean R²")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-layer pooled reconstruction: extended-span targets", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    savefig_paper(fig, "uh_hero3_crosslayer_pooled", dir=FIG_DIR)
    plt.close(fig)


def fig_forest(delta: dict) -> None:
    stats = delta["statistics"]
    conventions = [
        ("at_L18", "at the mean's layer 18 (frozen)"),
        ("best_vs_best_inherited", "best layer vs best layer"),
        ("frozen_observed_best_layer", "at the row's own best layer (data-selected)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    y = np.arange(len(NEW_ROWS))[::-1]
    for ax, (suffix, title) in zip(axes, conventions):
        obs = [stats[f"{r}_{suffix}"]["observed"] for r in NEW_ROWS]
        lo = [stats[f"{r}_{suffix}"]["ci95"][0] for r in NEW_ROWS]
        hi = [stats[f"{r}_{suffix}"]["ci95"][1] for r in NEW_ROWS]
        ax.errorbar(
            obs,
            y,
            xerr=[np.array(obs) - np.array(lo), np.array(hi) - np.array(obs)],
            fmt="o",
            color=paper_palette(3)[0],
            ecolor="0.5",
            capsize=3,
        )
        ax.axvline(0.0, color="0.3", lw=1.2)
        ax.axvline(0.02, color="#a05050", ls="--", lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Δ skill (row − mean)")
    axes[0].set_yticks(y, labels=[ROW_LABELS[r] for r in NEW_ROWS], fontsize=9)
    fig.suptitle(
        "Paired bootstrap: each new summary vs the mean benchmark (2,000 draws, 95% CI)",
        fontsize=13,
        fontweight="bold",
    )
    savefig_paper(fig, "uh_delta_vs_mean_forest", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    uh, cross, delta, parent, lofo = load()
    fig_hero1(uh, parent, lofo)
    fig_hero2(uh, parent)
    fig_hero3(cross)
    fig_forest(delta)
    print(f"wrote 4 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
