"""Promotion figures for #810 round 4 (`header-echo-ablation-capture`).

The workload's hero1/hero2 for this round rendered through the round-1 hero
path again (only the two round-1-named rows, wrong band line), so this script
regenerates the round's figures from the committed eval JSONs:

  1. he_paired_forest         — paired bootstrap Δskill(full − empty) per row
     at the committed best layer, 95% CI whiskers, 0 line + the ±0.02
     equivalence margin shaded.
  2. he_paired_draws          — the 2,000 shared-index bootstrap draws behind
     each forest row (violin + observed dot); the low-level data behind the
     CIs (per-context decompositions were not persisted for this round).
  3. he_recon_by_layer_folds  — per-layer skill curves for the 9 empty-answer
     rows, LOCO | LOFO panels, mean-summary benchmark (dashed), union band +
     identity ceilings drawn.
  4. he_cell_heatmap          — 9 empty-answer rows x 28 layers, LOCO skill.
  5. he_mechanism_by_layer    — median cross-context-centered cosine between
     full-answer and empty-answer activations per row x layer, with the 0.8
     echo-consistency anchor.
  6. he_mechanism_percontext  — the 50 per-context centered cosines at each
     row's committed best layer (strip + median tick); the low-level data
     behind the mechanism medians.

Inputs (all committed on `issue-810`):
  eval_results/issue_810/header-echo-ablation-capture/reconstruction_skill_header_echo.json
  eval_results/issue_810/header-echo-ablation-capture/paired_full_minus_empty.json
  eval_results/issue_810/header-echo-ablation-capture/mechanism_cosine_r2.json
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
HE_DIR = REPO / "eval_results/issue_810/header-echo-ablation-capture"
FIG_DIR = REPO / "figures/issue_810/header-echo-ablation-capture"

ROW_LABELS = {
    "im_end": "turn-end token",
    "turn_nl": "newline after turn end",
    "uh_im_start": "header start token",
    "uh_user": "header 'user' token",
    "uh_nl": "header newline",
    "uh_mean3": "header mean (3 tokens)",
    "uh_max3": "header max (3 tokens)",
    "bnd_mean5": "boundary mean (5 tokens)",
    "bnd_max5": "boundary max (5 tokens)",
}
ROWS = list(ROW_LABELS)
MARGIN = 0.02


def context_label(battery_id: str) -> str:
    """Reader-facing label for a 50-context battery id (no opaque slugs in figure text).

    Families per the store manifest: f1 persona (house + PersonaHub), f2 WildChat
    prefixes, f3 in-context demos, f4 register rephrasings, f5 format demands,
    f6 default-assistant anchors, f8 behavior-commanding prompts.
    """
    parts = battery_id.split("_")
    fam, rest = parts[0], parts[1:]
    if fam == "f1" and rest[0] == "house":
        return " ".join(rest[1:]).replace("_", " ") + " persona"
    if fam == "f1" and rest[0] == "phub":
        return f"PersonaHub persona {int(rest[1])}"
    if fam == "f2" and rest[0] == "wc":
        return f"WildChat {rest[1]} prefix {rest[2]}"
    if fam == "f3" and rest[0] == "icl":
        return f"{rest[1]} in-context demos ({rest[2]})"
    if fam == "f4" and rest[0] == "reph":
        return f"{rest[1]} rephrasing"
    if fam == "f5" and rest[0] == "fmt":
        return " ".join(rest[1:]).replace("_", " ") + " format demand"
    if fam == "f6":
        return "default template" if rest[0] == "default" else "helpful-assistant prompt"
    if fam == "f8" and rest[0] == "behav":
        return f"{rest[1]}-commanding prompt"
    raise ValueError(f"unrecognized battery id: {battery_id}")


def load() -> tuple[dict, dict, dict, dict, dict]:
    he = json.loads((HE_DIR / "reconstruction_skill_header_echo.json").read_text())
    paired = json.loads((HE_DIR / "paired_full_minus_empty.json").read_text())
    mech = json.loads((HE_DIR / "mechanism_cosine_r2.json").read_text())
    parent = json.loads(
        (REPO / "eval_results/issue_810/reconstruction_skill_by_summary.json").read_text()
    )
    lofo = json.loads((REPO / "eval_results/issue_810/adhoc_lofo_heatmap_grids.json").read_text())
    return he, paired, mech, parent, lofo


def parent_lofo_curve(lofo: dict, name: str) -> np.ndarray:
    grid = np.asarray(lofo["grids"]["panel3_reconstruction_lofo_skill_over_mean_r2"])
    return grid[:, lofo["column_order"].index(name)]


def fig_paired_forest(paired: dict) -> None:
    pr = paired["per_row"]
    y = np.arange(len(ROWS))[::-1]
    obs = np.array([pr[r]["observed"] for r in ROWS])
    lo = np.array([pr[r]["ci95"][0] for r in ROWS])
    hi = np.array([pr[r]["ci95"][1] for r in ROWS])
    verdicts = [pr[r]["verdict"] for r in ROWS]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.axvspan(-MARGIN, MARGIN, color="#e8eef2", zorder=0)
    ax.axvline(0.0, color="0.3", lw=1.2)
    ax.axvline(MARGIN, color="#a05050", ls="--", lw=1.0)
    ax.axvline(-MARGIN, color="#a05050", ls="--", lw=1.0)
    pal = paper_palette(3)
    for yi, o, lo_i, hi_i, v, r in zip(y, obs, lo, hi, verdicts, ROWS):
        color = pal[1] if v == "positive_gap" else pal[0]
        ax.errorbar(
            o,
            yi,
            xerr=[[o - lo_i], [hi_i - o]],
            fmt="o",
            color=color,
            ecolor="0.5",
            capsize=3,
            ms=6,
        )
        layer = pr[r]["committed_best_layer"]
        ax.text(hi_i + 0.004, yi, f"layer {layer}", fontsize=7.5, va="center", color="0.35")
    ax.set_yticks(y, labels=[ROW_LABELS[r] for r in ROWS], fontsize=9)
    ax.set_xlabel("Δ held-out skill (full answer − answer deleted), committed best layer")
    ax.set_title(
        "Deleting the answer: paired reconstruction-skill change per boundary summary",
        fontweight="bold",
    )
    savefig_paper(fig, "he_paired_forest", dir=FIG_DIR)
    plt.close(fig)


def fig_paired_draws(paired: dict) -> None:
    draws = paired["draws_by_row"]
    pr = paired["per_row"]
    data = [np.asarray(draws[r]) for r in ROWS]
    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot(data, positions=np.arange(len(ROWS)), showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor(paper_palette(3)[0])
        body.set_alpha(0.55)
    ax.scatter(
        np.arange(len(ROWS)),
        [pr[r]["observed"] for r in ROWS],
        color="#2d2d2d",
        s=28,
        zorder=3,
        label="observed Δ",
    )
    ax.axhspan(-MARGIN, MARGIN, color="#e8eef2", zorder=0)
    ax.axhline(0.0, color="0.3", lw=1.2)
    ax.set_xticks(range(len(ROWS)), labels=[ROW_LABELS[r] for r in ROWS], rotation=30, ha="right")
    ax.set_ylabel("Δ held-out skill (full − answer deleted)")
    ax.set_title(
        "The 2,000 shared-index bootstrap draws behind each paired estimate", fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=8)
    savefig_paper(fig, "he_paired_draws", dir=FIG_DIR)
    plt.close(fig)


def fig_recon_folds(he: dict, parent: dict, lofo: dict) -> None:
    layers = np.arange(28)
    band = he["band_rows"]["enlarged_axis_max_selected"]
    loco_ceiling = band["ceiling"]
    lofo_ceiling = max(r["lofo_skill"] for r in he["diagnostics"]["lofo_identity_ceiling"])
    colors = paper_palette(8) + ["#7B5233"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, fold in zip(axes, ("loco", "lofo")):
        for i, row in enumerate(ROWS):
            cells = he["by_summary"][row]
            key = "ridge_skill" if fold == "loco" else "lofo_skill"
            ax.plot(layers, [c[key] for c in cells], color=colors[i], lw=1.6, label=ROW_LABELS[row])
        if fold == "loco":
            curve = [c["ridge_skill"] for c in parent["by_summary"]["mean"]]
        else:
            curve = parent_lofo_curve(lofo, "mean")
        ax.plot(
            layers, curve, "--", color="0.2", lw=2.0, label="mean summary (full answer, benchmark)"
        )
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
            ax.axhline(
                lofo_ceiling,
                color="#a05050",
                ls="-.",
                lw=1.2,
                label="identity ceiling (mean target)",
            )
            ax.set_title("leave-one-family-out (ordering only)")
        ax.set_xlabel("layer")
        ax.set_ylim(-0.15, 1.05)
    axes[0].set_ylabel("held-out skill-over-mean R² (higher = better)")
    axes[1].legend(loc="lower center", fontsize=7, ncol=2, frameon=True)
    fig.suptitle(
        "Predicting each answer-deleted boundary summary from the context representation",
        fontsize=14,
        fontweight="bold",
    )
    savefig_paper(fig, "he_recon_by_layer_folds", dir=FIG_DIR)
    plt.close(fig)


def fig_cell_heatmap(he: dict) -> None:
    mat = [[c["ridge_skill"] for c in he["by_summary"][r]] for r in ROWS]
    arr = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks(range(len(ROWS)), labels=[ROW_LABELS[r] for r in ROWS], fontsize=8)
    ax.set_xticks(range(0, 28, 5))
    ax.set_xlabel("layer")
    ax.set_title(
        "Reconstruction skill per answer-deleted summary × layer (LOCO)", fontweight="bold"
    )
    fig.colorbar(im, ax=ax, label="skill-over-mean R²")
    savefig_paper(fig, "he_cell_heatmap", dir=FIG_DIR)
    plt.close(fig)


def fig_mechanism_by_layer(mech: dict) -> None:
    layers = np.arange(28)
    colors = paper_palette(8) + ["#7B5233"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, row in enumerate(ROWS):
        pl = mech["by_row"][row]["per_layer"]
        ax.plot(
            layers,
            [pl[str(int(l))]["median_centered_cos"] for l in layers],
            color=colors[i],
            lw=1.6,
            label=ROW_LABELS[row],
        )
    ax.axhline(0.8, color="#a05050", ls="--", lw=1.2, label="echo-consistency anchor (0.8)")
    ax.set_xlabel("layer")
    ax.set_ylabel("median centered cosine (full vs answer-deleted)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("How far each boundary state moves when the answer is deleted", fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)
    savefig_paper(fig, "he_mechanism_by_layer", dir=FIG_DIR)
    plt.close(fig)


def fig_mechanism_percontext(mech: dict) -> None:
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pal = paper_palette(8) + ["#7B5233"]
    for i, row in enumerate(ROWS):
        r = mech["by_row"][row]
        pc = r["per_context_centered_cos_at_best_layer"]
        vals = np.array(list(pc.values()))
        x = i + rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x, vals, s=14, color=pal[i], alpha=0.65)
        med = float(np.median(vals))
        ax.scatter([i], [med], marker="_", s=420, linewidths=2.6, color="#2d2d2d", zorder=3)
        # label the single most-moved context (lowest cosine) per row, reader-facing
        names = list(pc.keys())
        j = int(np.argmin(vals))
        ax.text(i, vals[j] - 0.04, context_label(names[j]), fontsize=6, ha="center", color="0.35")
    ax.axhline(0.8, color="#a05050", ls="--", lw=1.2, label="echo-consistency anchor (0.8)")
    ax.set_xticks(range(len(ROWS)), labels=[ROW_LABELS[r] for r in ROWS], rotation=30, ha="right")
    ax.set_ylabel("centered cosine per context (committed best layer)")
    ax.set_title(
        "Per-context similarity of full vs answer-deleted boundary states", fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=8)
    savefig_paper(fig, "he_mechanism_percontext", dir=FIG_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    he, paired, mech, parent, lofo = load()
    fig_paired_forest(paired)
    fig_paired_draws(paired)
    fig_recon_folds(he, parent, lofo)
    fig_cell_heatmap(he)
    fig_mechanism_by_layer(mech)
    fig_mechanism_percontext(mech)
    print(f"wrote 6 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
