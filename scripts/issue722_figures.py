"""Clean-result figures for issue #722 (pre/post-finetuning context->answer map M).

Round-2 revision (interp-critique fixes). Reads the analyzer deliverables:
  - eval_results/issue_722/function_change.json   (Delta_med, floor_combined, ...)
  - eval_results/issue_722/chain_rho_M0_Mplus.json (ridge/MLP/shuffle chain rho)
  - eval_results/issue_722/cross_transfer.json     (M0/M+ cross-transfer cosines)

Five figures:
  1. hero_function_change_heatmap   -- Delta_med / floor_combined (the KILL-CRITERION
                                       denominator), 1x = at floor. (MF-C/MF-D)
  2. function_change_delta_vs_floor -- raw Delta_med vs floor_combined bars (data behind 1)
  3. chain_rho_shift_forest         -- chain rho under M0 vs M+, family-clustered CI
  4. cross_transfer_orthogonality   -- cross-transfer asymmetry (M+ generalizes worse) (MF-F)
  5. mlp_shuffle_diagnostic         -- rho_M0_mlp vs rho_M0_shuffle; passes only fact L7 (MF-I)

All reader-facing labels in plain English (MF-J): full behavior names, full layer
indices, "Base"/"Post-FT" for M0/M+.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

ROOT = Path(__file__).resolve().parents[1]
ED = ROOT / "eval_results" / "issue_722"
FC = json.load(open(ED / "function_change.json"))["cells"]
CR = json.load(open(ED / "chain_rho_M0_Mplus.json"))["cells"]
CT = json.load(open(ED / "cross_transfer.json"))["cells"]
OUT = "issue_722"

# Plain-English behavior labels (MF-J) -- no internal slugs reader-facing.
BEH = {"fact": "taught fact", "em": "harmful compliance (EM)", "sycophancy": "sycophancy"}
BEH_ORDER = ["fact", "em", "sycophancy"]
LAYERS = [7, 14, 21]
KEYS = {b: [f"{b}/L{ly}" for ly in LAYERS] for b in BEH_ORDER}

# Short plain-English labels for the crowded bar-chart x-axes (figs 2/4/5).
SHORT = {"fact": "fact", "em": "EM", "sycophancy": "sycophancy"}


def _bar_labels():
    return [f"{SHORT[b]}\nL{ly}" for b in BEH_ORDER for ly in LAYERS]


C_PRIMARY = paper_palette_role("primary")
C_BASELINE = paper_palette_role("baseline")
C_CONTROL = paper_palette_role("control")
C_NEUTRAL = paper_palette_role("neutral")


def _grid(metric_fn):
    """3 behaviors x 3 layers matrix of metric_fn(cell)."""
    return np.array([[metric_fn(FC[f"{b}/L{ly}"]) for ly in LAYERS] for b in BEH_ORDER])


# ----------------------------------------------------------------------------
# Figure 1 (hero): Delta_med / floor_combined  (MF-C: kill-criterion denominator)
# ----------------------------------------------------------------------------
def fig_hero():
    set_paper_style("blog")
    M = _grid(lambda c: c["Delta_med"] / c["floor_combined"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    # Diverging around 1.0 (=at floor): >1 above floor (function changed), <1 below.
    vmax = float(np.nanmax(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.0, vmax=2.0 * 1.0, aspect="auto")
    # clip color scale so the 1x midpoint is visually meaningful; annotate true values.
    im.set_clim(0.0, 2.0)
    for i in range(len(BEH_ORDER)):
        for j in range(len(LAYERS)):
            v = M[i, j]
            txt = f"{v:.2f}×"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=11,
                color="#1A1A1A" if 0.45 < v < 1.55 else "white",
                fontweight="semibold",
            )
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"layer {ly}" for ly in LAYERS])
    ax.set_yticks(range(len(BEH_ORDER)))
    ax.set_yticklabels([BEH[b] for b in BEH_ORDER])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("function-change Δ ÷ noise floor\n(1× = at floor)", fontsize=9)
    cbar.ax.axhline(1.0, color="#1A1A1A", lw=1.0)
    set_title_subtitle(
        ax,
        "Only the taught fact clears the noise floor",
        "Δ (‖post-FT map − base map‖ along behavior dir) over the kill-criterion floor; >1× = function moved",
        source="issue #722 · ridge fit · 16 source contexts · seed 42",
    )
    savefig_paper(fig, f"{OUT}/hero_function_change_heatmap")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: raw Delta_med vs floor_combined bars (the data behind the ratio)
# ----------------------------------------------------------------------------
def fig_delta_vs_floor():
    set_paper_style("blog")
    cells = [f"{b}/L{ly}" for b in BEH_ORDER for ly in LAYERS]
    labels = _bar_labels()
    dmed = [FC[k]["Delta_med"] for k in cells]
    floor = [FC[k]["floor_combined"] for k in cells]
    x = np.arange(len(cells))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x - w / 2, dmed, w, label="function-change Δ", color=C_PRIMARY)
    ax.bar(x + w / 2, floor, w, label="noise floor (post-FT refit)", color=C_BASELINE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("magnitude along behavior direction")
    ax.legend(frameon=False, fontsize=9)
    set_title_subtitle(
        ax,
        "The fact Δ towers over its floor; EM and late sycophancy sit below",
        "raw Δ vs its combined floor — floor bound by the post-FT map's own refit variance in all 9 cells",
        source="issue #722 · ridge fit · family-clustered bootstrap (7 context families)",
    )
    savefig_paper(fig, f"{OUT}/function_change_delta_vs_floor")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: chain rho under M0 (base) vs M+ (post-FT), family-clustered CI
# ----------------------------------------------------------------------------
def fig_chain_forest():
    set_paper_style("blog")
    cells = [f"{b}/L{ly}" for b in BEH_ORDER for ly in LAYERS]
    labels = [f"{BEH[b].split(' (')[0]} — layer {ly}" for b in BEH_ORDER for ly in LAYERS]
    y = np.arange(len(cells))[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for yi, k in zip(y, cells):
        m0 = CR[k]["ci_M0_ridge"]
        mp = CR[k]["ci_Mplus_ridge"]
        ax.plot(
            [m0["ci_lo"], m0["ci_hi"]], [yi + 0.12, yi + 0.12], color=C_BASELINE, lw=2.0, alpha=0.8
        )
        ax.scatter([m0["point"]], [yi + 0.12], color=C_BASELINE, s=34, zorder=3, label="_")
        ax.plot(
            [mp["ci_lo"], mp["ci_hi"]], [yi - 0.12, yi - 0.12], color=C_PRIMARY, lw=2.0, alpha=0.8
        )
        ax.scatter([mp["point"]], [yi - 0.12], color=C_PRIMARY, s=34, zorder=3, label="_")
    ax.axvline(0.0, color=C_NEUTRAL, lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Spearman ρ (predicted behavior signal vs judge leakage rate)")
    # manual legend
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], color=C_BASELINE, marker="o", lw=2, label="base map (M0)"),
            Line2D([0], [0], color=C_PRIMARY, marker="o", lw=2, label="post-FT map (M⁺)"),
        ],
        frameon=False,
        fontsize=9,
        loc="lower right",
    )
    set_title_subtitle(
        ax,
        "A context→leakage chain appears for the taught fact only",
        "fact chain ρ jumps ≈0 → +0.46–+0.50 post-FT (CI excludes 0); EM + sycophancy straddle 0",
        source="issue #722 · held-out LOCO ridge · 95% family-clustered CI (7 families)",
    )
    savefig_paper(fig, f"{OUT}/chain_rho_shift_forest")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: cross-transfer asymmetry (MF-F: drop "orthogonal")
#   M0->v+ and M+->v+ both ~0 (neither map predicts the post-FT profile);
#   M+->v0 is -0.22..-0.32 (M+ generalizes WORSE to base inputs than M0 does to FT).
# ----------------------------------------------------------------------------
def fig_cross_transfer():
    set_paper_style("blog")
    cells = [f"{b}/L{ly}" for b in BEH_ORDER for ly in LAYERS]
    labels = _bar_labels()
    m0_vp = [CT[k]["m0_to_vplus_cos"] for k in cells]
    mp_vp = [CT[k]["mplus_to_vplus_cos"] for k in cells]
    mp_v0 = [CT[k]["mplus_to_v0_cos"] for k in cells]
    x = np.arange(len(cells))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - w, m0_vp, w, label="base map → post-FT profile", color=C_BASELINE)
    ax.bar(x, mp_vp, w, label="post-FT map → post-FT profile", color=C_PRIMARY)
    ax.bar(x + w, mp_v0, w, label="post-FT map → base profile", color=C_CONTROL)
    ax.axhline(0.0, color=C_NEUTRAL, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [lbl.replace("\n", " ") for lbl in labels], fontsize=8, rotation=20, ha="right"
    )
    ax.set_ylabel("held-out cross-transfer cosine")
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="lower left")
    set_title_subtitle(
        ax,
        "Cross-transfer is asymmetric: the post-FT map generalizes worse",
        "neither map predicts the post-FT profile (≈0); the post-FT map predicts the base profile worst (−0.22 to −0.32)",
        source="issue #722 · held-out cross-transfer cosine · 9 cells",
    )
    savefig_paper(fig, f"{OUT}/cross_transfer_orthogonality")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: MLP-vs-shuffle diagnostic (MF-I: passes only at fact L7)
#   The kill-criterion gate is rho_M0_mlp > rho_M0_shuffle on the BASE map.
# ----------------------------------------------------------------------------
def fig_mlp_shuffle():
    set_paper_style("blog")
    cells = [f"{b}/L{ly}" for b in BEH_ORDER for ly in LAYERS]
    labels = _bar_labels()
    mlp = [CR[k]["rho_M0_mlp"] for k in cells]
    shuf = [CR[k]["rho_M0_shuffle"] for k in cells]
    passes = [m > s for m, s in zip(mlp, shuf)]
    x = np.arange(len(cells))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x - w / 2, mlp, w, label="nonlinear MLP fit (base map)", color=C_PRIMARY)
    ax.bar(x + w / 2, shuf, w, label="shuffle null", color=C_NEUTRAL)
    ax.axhline(0.0, color=C_NEUTRAL, lw=0.8)
    # mark the one cell where the MLP beats its shuffle null (fact L7).
    ymax = max(max(mlp), max(shuf))
    ymin = min(min(mlp), min(shuf))
    ax.set_ylim(ymin - 0.02, ymax + 0.05)
    for xi, p in zip(x, passes):
        if p:
            ax.annotate(
                "MLP beats\nshuffle here",
                xy=(xi, max(mlp[xi], shuf[xi])),
                xytext=(xi + 1.0, ymax - 0.01),
                ha="left",
                va="top",
                fontsize=7.5,
                color=C_PRIMARY,
                fontweight="semibold",
                arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=1.0),
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("held-out reconstruction ρ (base map)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    set_title_subtitle(
        ax,
        "The nonlinear map fails the shuffle null in 8 of 9 cells",
        "MLP beats its shuffle control only at taught fact, layer 7 — so the read is ridge-only",
        source="issue #722 · MLP vs label-shuffle, base map · 16 contexts, seed 42",
    )
    savefig_paper(fig, f"{OUT}/mlp_shuffle_diagnostic")
    plt.close(fig)


if __name__ == "__main__":
    fig_hero()
    fig_delta_vs_floor()
    fig_chain_forest()
    fig_cross_transfer()
    fig_mlp_shuffle()
    print("issue #722 figures regenerated.")
