#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ρ, ×, M⁺) in scientific labels.
"""Issue #811 — analyzer-pass figures (supersede the run's draft figures).

Regenerates the four §6.3 figure candidates plus a raw Δ-vs-floor view through
``savefig_paper`` (PNG + PDF + ``.meta.json`` sidecar, commit-pinned, per-point
data embedded), with plain-English labels ("turn boundary" instead of the
``turn_nl`` slug) and 95% family-clustered CI whiskers on the chain-ρ forest
(the run's draft forest had none).

Reads ONLY the committed eval JSONs under ``eval_results/issue_811/``:
``function_change_{mean,turn_nl}.json``, ``chain_rho_M0_Mplus_{mean,turn_nl}.json``,
``validity_gate_{mean,turn_nl}.json``.

Outputs (``figures/issue_811/``):
- ``hero_function_change_ratio``      — Δ_med ÷ floor per behavior×layer, both summaries (HERO)
- ``function_change_raw_delta_vs_floor`` — raw Δ_med alongside its floor, 2×3 grid
- ``chain_rho_forest_ci``             — chain-ρ M0 vs M⁺ with 95% family-clustered CIs
- ``delta_scatter_pairs``             — 9-cell scatter of mean vs turn-boundary Δ_med (labeled)
- ``validity_gate_margins``           — MLP-vs-shuffle gate margin per cell, both summaries
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue811.analyzer_figures")

EVAL_DIR = PROJECT_ROOT / "eval_results/issue_811"
FIG_DIR = "figures/"

BEHAVIORS = ("em", "sycophancy", "fact")
LAYERS = (7, 14, 21)
SUMMARIES = ("mean", "turn_nl")
BEHAVIOR_LABEL = {
    "em": "harmful-compliance (EM)",
    "sycophancy": "sycophancy",
    "fact": "taught fact",
}
SUMMARY_LABEL = {"mean": "answer mean", "turn_nl": "turn boundary"}
C_MEAN = paper_palette_role("baseline")
C_TURN = paper_palette_role("primary")
C_FLOOR = paper_palette_role("neutral")


def _load(name: str) -> dict:
    with open(EVAL_DIR / name) as fh:
        return json.load(fh)


def _cells(summary: str, which: str) -> dict:
    fname = {
        "fc": f"function_change_{summary}.json",
        "rho": f"chain_rho_M0_Mplus_{summary}.json",
        "gate": f"validity_gate_{summary}.json",
    }[which]
    return _load(fname)["cells"]


def fig_hero_ratio() -> None:
    """Δ_med ÷ floor_combined per behavior×layer, mean vs turn boundary."""
    fc = {s: _cells(s, "fc") for s in SUMMARIES}
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6), sharey=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        x = np.arange(len(LAYERS))
        for off, (s, color) in zip(
            (-0.2, 0.2), [("mean", C_MEAN), ("turn_nl", C_TURN)], strict=True
        ):
            vals = [
                fc[s][f"{beh}/L{li}"]["Delta_med"] / fc[s][f"{beh}/L{li}"]["floor_combined"]
                for li in LAYERS
            ]
            ax.bar(x + off, vals, width=0.38, color=color, label=SUMMARY_LABEL[s])
        ax.axhline(1.0, ls="--", lw=1.0, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {li}" for li in LAYERS])
        ax.set_title(BEHAVIOR_LABEL[beh])
    axes[0].set_ylabel("function change ÷ noise floor")
    axes[0].legend(loc="upper left")
    savefig_paper(fig, "issue_811/hero_function_change_ratio", dir=FIG_DIR)
    plt.close(fig)


def fig_raw_delta_vs_floor() -> None:
    """Raw Δ_med alongside its combined refit floor, rows = summary, cols = behavior."""
    fc = {s: _cells(s, "fc") for s in SUMMARIES}
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6), sharex=True)
    for row, s in enumerate(SUMMARIES):
        for col, beh in enumerate(BEHAVIORS):
            ax = axes[row][col]
            x = np.arange(len(LAYERS))
            deltas = [fc[s][f"{beh}/L{li}"]["Delta_med"] for li in LAYERS]
            floors = [fc[s][f"{beh}/L{li}"]["floor_combined"] for li in LAYERS]
            ax.bar(
                x - 0.2,
                deltas,
                width=0.38,
                color=C_TURN if s == "turn_nl" else C_MEAN,
                label="function change Δ_med",
            )
            ax.bar(x + 0.2, floors, width=0.38, color=C_FLOOR, label="noise floor")
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{li}" for li in LAYERS])
            if row == 0:
                ax.set_title(BEHAVIOR_LABEL[beh])
            if col == 0:
                ax.set_ylabel(f"{SUMMARY_LABEL[s]}\nΔ_med (activation units)")
    axes[0][0].legend(loc="upper left", fontsize=8)
    axes[1][0].legend(loc="upper left", fontsize=8)
    savefig_paper(fig, "issue_811/function_change_raw_delta_vs_floor", dir=FIG_DIR)
    plt.close(fig)


def fig_chain_rho_forest() -> None:
    """Chain-ρ under M0 vs M⁺, both summaries, 95% family-clustered CI whiskers."""
    rho = {s: _cells(s, "rho") for s in SUMMARIES}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharex=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        yticks, ylabels = [], []
        y = 0.0
        for li in LAYERS:
            for s in SUMMARIES:
                cell = rho[s][f"{beh}/L{li}"]
                for key, color, dy in (
                    ("M0", C_MEAN, 0.16),
                    ("Mplus", C_TURN, -0.16),
                ):
                    ci = cell[f"ci_{key}_ridge"]
                    pt = ci["point"]
                    err = [[pt - ci["ci_lo"]], [ci["ci_hi"] - pt]]
                    ax.errorbar(
                        pt,
                        y + dy,
                        xerr=err,
                        fmt="o",
                        ms=4.5,
                        color=color,
                        capsize=2,
                        lw=1.1,
                        markeredgewidth=0.0,
                    )
                yticks.append(y)
                ylabels.append(f"L{li} · {SUMMARY_LABEL[s]}")
                y += 1.0
            y += 0.5  # gap between layers
        ax.axvline(0.0, ls="--", lw=1.0, color="0.4")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(BEHAVIOR_LABEL[beh])
        ax.invert_yaxis()
    axes[1].set_xlabel("Spearman ρ(prediction along behavior direction, measured leakage)")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=C_MEAN, label="base map M0"),
        plt.Line2D([0], [0], marker="o", ls="", color=C_TURN, label="post-finetune map M⁺"),
    ]
    axes[2].legend(handles=handles, loc="lower right", fontsize=8)
    savefig_paper(fig, "issue_811/chain_rho_forest_ci", dir=FIG_DIR)
    plt.close(fig)


def fig_delta_scatter() -> None:
    """Per-cell (behavior×layer) Δ_med under mean vs turn boundary, 45° line."""
    fc = {s: _cells(s, "fc") for s in SUMMARIES}
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lim = 0.45
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="0.4")
    for beh in BEHAVIORS:
        for li in LAYERS:
            xm = fc["mean"][f"{beh}/L{li}"]["Delta_med"]
            yt = fc["turn_nl"][f"{beh}/L{li}"]["Delta_med"]
            ax.scatter(xm, yt, s=42, color=C_TURN, zorder=3)
            ax.text(xm + 0.006, yt + 0.006, f"{BEHAVIOR_LABEL[beh]} L{li}", fontsize=7.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Δ_med under answer-mean summary (activation units)")
    ax.set_ylabel("Δ_med under turn-boundary summary (activation units)")
    savefig_paper(fig, "issue_811/delta_scatter_pairs", dir=FIG_DIR)
    plt.close(fig)


def fig_validity_gate() -> None:
    """MLP-vs-shuffle gate margin per behavior×layer, both summaries."""
    gate = {s: _cells(s, "gate") for s in SUMMARIES}
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6), sharey=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        x = np.arange(len(LAYERS))
        for off, (s, color) in zip(
            (-0.2, 0.2), [("mean", C_MEAN), ("turn_nl", C_TURN)], strict=True
        ):
            vals = [gate[s][f"{beh}/L{li}"]["gate_margin"] for li in LAYERS]
            ax.bar(x + off, vals, width=0.38, color=color, label=SUMMARY_LABEL[s])
        ax.axhline(0.0, ls="--", lw=1.0, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {li}" for li in LAYERS])
        ax.set_title(BEHAVIOR_LABEL[beh])
    axes[0].set_ylabel("gate margin (ρ real − ρ shuffled)")
    axes[0].legend(loc="lower left", fontsize=8)
    savefig_paper(fig, "issue_811/validity_gate_margins", dir=FIG_DIR)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_paper_style("blog")
    fig_hero_ratio()
    fig_raw_delta_vs_floor()
    fig_chain_rho_forest()
    fig_delta_scatter()
    fig_validity_gate()
    logger.info("wrote 5 figures to figures/issue_811/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
