#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Issue #722 clean-result figures (analyzer-owned).

Reads the committed eval_results/issue_722/*.json from the issue-722 worktree
and writes blog-style figures to figures/issue_722/ in the main checkout.
"""

from __future__ import annotations

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

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-722")
EVAL = WT / "eval_results" / "issue_722"
OUT = "issue_722"

BEH_LABEL = {"em": "Emergent misalignment", "sycophancy": "Sycophancy", "fact": "Taught fact"}
BEH_ORDER = ["em", "sycophancy", "fact"]
LAYERS = [7, 14, 21]

fc = json.load(open(EVAL / "function_change.json"))["cells"]
cr = json.load(open(EVAL / "chain_rho_M0_Mplus.json"))["cells"]
ct = json.load(open(EVAL / "cross_transfer.json"))["cells"]


def cell(behavior: str, layer: int) -> dict:
    return fc[f"{behavior}/L{layer}"]


# ---------------------------------------------------------------------------
# HERO: Delta_med / floor_combined per behavior x layer, with floor line at 1.0
# ---------------------------------------------------------------------------
def fig_hero() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(BEH_ORDER))
    width = 0.26
    colors = {
        7: paper_palette_role("baseline"),
        14: paper_palette_role("primary"),
        21: paper_palette_role("neutral"),
    }
    for i, L in enumerate(LAYERS):
        ratios = []
        for b in BEH_ORDER:
            c = cell(b, L)
            ratios.append(c["Delta_med"] / c["floor_combined"])
        bars = ax.bar(
            x + (i - 1) * width,
            ratios,
            width,
            label=f"layer {L}",
            color=colors[L],
            edgecolor="white",
            linewidth=0.6,
        )
        for rect, r in zip(bars, ratios):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.06,
                f"{r:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.3, zorder=1)
    ax.text(
        2.45, 1.06, "floor (= at noise floor)", color="#c0392b", fontsize=8, ha="right", va="bottom"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([BEH_LABEL[b] for b in BEH_ORDER])
    ax.set_ylabel("function-change Δ_med ÷ combined noise floor")
    ax.set_ylim(0, 3.7)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
    set_title_subtitle(
        ax,
        "Only the taught fact reshapes the context→answer function above the noise floor",
        "ratio > 1 = function-change clears the combined refit/shift floor; n = 480 context "
        "cells/behavior, 7 families; ridge fit",
    )
    savefig_paper(fig, f"{OUT}/hero_function_change", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LOW-LEVEL companion: per-behavior x layer Delta_med vs its combined floor
# (the two raw quantities behind the hero ratio), points labeled.
# ---------------------------------------------------------------------------
def fig_hero_points() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    maxv = 0.0
    for b in BEH_ORDER:
        for L in LAYERS:
            c = cell(b, L)
            maxv = max(maxv, c["Delta_med"], c["floor_combined"])
    lim = maxv * 1.12
    ax.plot([0, lim], [0, lim], color="#999999", linestyle=":", linewidth=1.0, zorder=1)
    ax.text(
        lim * 0.97,
        lim * 0.97,
        "Δ = floor",
        color="#777777",
        fontsize=8,
        ha="right",
        va="bottom",
        rotation=45,
    )
    markers = {"em": "o", "sycophancy": "s", "fact": "^"}
    colors = {
        7: paper_palette_role("baseline"),
        14: paper_palette_role("primary"),
        21: paper_palette_role("neutral"),
    }
    for b in BEH_ORDER:
        for L in LAYERS:
            c = cell(b, L)
            ax.scatter(
                c["floor_combined"],
                c["Delta_med"],
                s=80,
                marker=markers[b],
                facecolors=colors[L],
                edgecolors="#333333",
                linewidths=0.8,
                zorder=3,
            )
            ax.text(
                c["floor_combined"],
                c["Delta_med"] + lim * 0.012,
                f"{BEH_LABEL[b].split()[0]} L{L}",
                fontsize=6.6,
                ha="center",
                va="bottom",
            )
    ax.set_xlabel("combined noise floor (max of M0-refit, M⁺-refit, shifted-design)")
    ax.set_ylabel("function-change Δ_med (median |Δ(c)·r̂_B|)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    set_title_subtitle(
        ax,
        "The raw quantities behind the ratio: Δ_med vs its floor, per cell",
        "points above the dotted line clear the floor (all 3 taught-fact layers); em/sycophancy "
        "sit on or below it",
    )
    savefig_paper(fig, f"{OUT}/hero_function_change_points", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CHAIN-rho PAIR: rho_M0 vs rho_M+ with paired CI on rho_diff, per behavior x layer
# ---------------------------------------------------------------------------
def fig_chain_rho() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = [(b, L) for b in BEH_ORDER for L in LAYERS]
    y = np.arange(len(rows))[::-1]
    col_m0 = paper_palette_role("baseline")
    col_mp = paper_palette_role("primary")
    for yi, (b, L) in zip(y, rows):
        d = cr[f"{b}/L{L}"]
        m0 = d["rho_M0_ridge"]
        mp = d["rho_Mplus_ridge"]
        dci = d["ci_diff_ridge"]
        # connecting line M0 -> M+
        ax.plot([m0, mp], [yi, yi], color="#bbbbbb", linewidth=1.2, zorder=1)
        ax.scatter(
            [m0],
            [yi],
            s=70,
            color=col_m0,
            edgecolors="#333",
            linewidths=0.7,
            zorder=3,
            label="base map M0" if yi == y[0] else None,
        )
        ax.scatter(
            [mp],
            [yi],
            s=70,
            color=col_mp,
            edgecolors="#333",
            linewidths=0.7,
            zorder=3,
            label="post-FT map M⁺" if yi == y[0] else None,
        )
        excl = (dci["ci_lo"] > 0) or (dci["ci_hi"] < 0)
        tag = f"Δρ=[{dci['ci_lo']:+.2f},{dci['ci_hi']:+.2f}]"
        ax.text(
            0.78,
            yi,
            tag,
            fontsize=6.8,
            va="center",
            ha="left",
            color="#1a7a3a" if excl else "#888888",
            fontweight="bold" if excl else "normal",
        )
    ax.axvline(0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{BEH_LABEL[b].split()[0]} L{L}" for (b, L) in rows])
    ax.set_xlabel("Spearman ρ ( r̂_Bᵀ M̂(c)  vs  measured leakage rate E ), held-out LOCO")
    ax.set_xlim(-0.45, 1.28)
    ax.legend(frameon=False, loc="center right", fontsize=8)
    set_title_subtitle(
        ax,
        "Finetuning creates a fact→leakage transfer that the base map never had",
        "green Δρ interval = post-FT minus base CI excludes zero (taught fact, all layers); "
        "em/sycophancy unchanged",
    )
    savefig_paper(fig, f"{OUT}/chain_rho_pair", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LAYER SPECTRUM inset: Delta_med/floor_combined vs layer per behavior
# ---------------------------------------------------------------------------
def fig_layer_spectrum() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = {
        "em": paper_palette_role("control"),
        "sycophancy": paper_palette_role("accent"),
        "fact": paper_palette_role("primary"),
    }
    markers = {"em": "o", "sycophancy": "s", "fact": "^"}
    for b in BEH_ORDER:
        ys = [cell(b, L)["Delta_med"] / cell(b, L)["floor_combined"] for L in LAYERS]
        ax.plot(
            LAYERS,
            ys,
            marker=markers[b],
            color=colors[b],
            linewidth=1.6,
            markersize=7,
            label=BEH_LABEL[b],
        )
    ax.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.1)
    ax.text(21, 1.04, "floor", color="#c0392b", fontsize=8, ha="right", va="bottom")
    ax.set_xticks(LAYERS)
    ax.set_xlabel("read layer")
    ax.set_ylabel("Δ_med ÷ combined noise floor")
    ax.legend(frameon=False, fontsize=8)
    set_title_subtitle(
        ax,
        "Function-change is layer-specific and behavior-specific",
        "taught fact clears the floor at every layer; em/sycophancy stay at or below it across "
        "the read band",
    )
    savefig_paper(fig, f"{OUT}/layer_spectrum", dir="figures/")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CROSS-TRANSFER matrix: M+ -> v0 cosine (the disambiguator) per behavior x layer
# ---------------------------------------------------------------------------
def fig_cross_transfer() -> None:
    # Primary layer L=14 only: the three cross-transfer cosines per behavior.
    # M0->v+ and M+->v+ are near-identical and near-zero everywhere (both maps
    # are weak fits on FT data); M+->v0 is strongly NEGATIVE -- the post-FT map
    # actively mispredicts base answer-profiles. This is why raw cosine is NOT
    # the headline; the floor-relative Delta is.
    set_paper_style("blog")
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.13)
    x = np.arange(len(BEH_ORDER))
    width = 0.26
    series = [
        ("M0 → v⁺ (base map on FT outputs)", "m0_to_vplus_cos", paper_palette_role("baseline")),
        (
            "M⁺ → v⁺ (post-FT map on FT outputs)",
            "mplus_to_vplus_cos",
            paper_palette_role("primary"),
        ),
        ("M⁺ → v0 (post-FT map on base outputs)", "mplus_to_v0_cos", paper_palette_role("control")),
    ]
    for i, (lab, key, col) in enumerate(series):
        vals = [ct[f"{b}/L14"][key] for b in BEH_ORDER]
        bars = ax.bar(
            x + (i - 1) * width, vals, width, label=lab, color=col, edgecolor="white", linewidth=0.6
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() - 0.012 if v < 0 else rect.get_height() + 0.004,
                f"{v:+.2f}",
                ha="center",
                va="top" if v < 0 else "bottom",
                fontsize=6.6,
            )
    ax.axhline(0.0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([BEH_LABEL[b] for b in BEH_ORDER])
    ax.set_ylabel("held-out cross-transfer cosine (layer 14)")
    ax.set_ylim(-0.36, 0.06)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title(
        "Both fitted maps are weak; the post-FT map mispredicts base outputs",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        pad=26,
    )
    ax.annotate(
        "M0→v⁺ ≈ M⁺→v⁺ ≈ 0 (neither map predicts FT outputs); M⁺→v0 < 0 — raw cosine "
        "is uninformative, so the floor-relative Δ is the valid read",
        xy=(0.0, 1.02),
        xycoords="axes fraction",
        fontsize=8,
        color="#555555",
        va="bottom",
    )
    savefig_paper(fig, f"{OUT}/cross_transfer", dir="figures/")
    plt.close(fig)
    plt.rcParams["figure.constrained_layout.use"] = True


if __name__ == "__main__":
    fig_hero()
    fig_hero_points()
    fig_chain_rho()
    fig_layer_spectrum()
    fig_cross_transfer()
    print("DONE figures/issue_722/")
