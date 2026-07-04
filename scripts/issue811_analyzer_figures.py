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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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
    """Δ_med ÷ floor_combined per behavior×layer, mean vs turn boundary.

    Sycophancy's turn-boundary bars are hatched/open: that summary FAILED the
    Phase-0 base-leg validity gate (KILL-1 read rule), so those reads are
    untrusted and must not be skimmed as the headline effect (the tallest bar
    in the grid is sycophancy turn-boundary L7 at 3.52x).
    """
    fc = {s: _cells(s, "fc") for s in SUMMARIES}
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8), sharey=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        x = np.arange(len(LAYERS))
        for off, (s, color) in zip(
            (-0.2, 0.2), [("mean", C_MEAN), ("turn_nl", C_TURN)], strict=True
        ):
            vals = [
                fc[s][f"{beh}/L{li}"]["Delta_med"] / fc[s][f"{beh}/L{li}"]["floor_combined"]
                for li in LAYERS
            ]
            if s == "turn_nl" and beh == "sycophancy":
                ax.bar(
                    x + off,
                    vals,
                    width=0.38,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    hatch="///",
                )
            else:
                ax.bar(x + off, vals, width=0.38, color=color)
        ax.axhline(1.0, ls="--", lw=1.0, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {li}" for li in LAYERS])
        ax.set_title(BEHAVIOR_LABEL[beh])
    axes[0].set_ylabel("function change ÷ noise floor")
    handles = [
        Patch(facecolor=C_MEAN, label="answer mean"),
        Patch(facecolor=C_TURN, label="turn boundary"),
        Patch(
            facecolor="white",
            edgecolor=C_TURN,
            linewidth=1.0,
            hatch="///",
            label="turn boundary — untrusted (failed base-leg validity)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    savefig_paper(fig, "issue_811/hero_function_change_ratio", dir=FIG_DIR)
    plt.close(fig)


def fig_raw_delta_vs_floor() -> None:
    """Raw Δ_med alongside its combined refit floor, rows = summary, cols = behavior.

    Sycophancy's turn-boundary bars (bottom row, middle panel) are hatched/open:
    that summary FAILED the Phase-0 base-leg validity gate (KILL-1 read rule),
    so both its Δ_med and floor derive from untrusted fits.
    """
    fc = {s: _cells(s, "fc") for s in SUMMARIES}
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.6), sharex=True)
    for row, s in enumerate(SUMMARIES):
        for col, beh in enumerate(BEHAVIORS):
            ax = axes[row][col]
            x = np.arange(len(LAYERS))
            deltas = [fc[s][f"{beh}/L{li}"]["Delta_med"] for li in LAYERS]
            floors = [fc[s][f"{beh}/L{li}"]["floor_combined"] for li in LAYERS]
            c_delta = C_TURN if s == "turn_nl" else C_MEAN
            if s == "turn_nl" and beh == "sycophancy":
                ax.bar(
                    x - 0.2,
                    deltas,
                    width=0.38,
                    facecolor="white",
                    edgecolor=c_delta,
                    linewidth=1.0,
                    hatch="///",
                    label="function change Δ_med",
                )
                ax.bar(
                    x + 0.2,
                    floors,
                    width=0.38,
                    facecolor="white",
                    edgecolor=C_FLOOR,
                    linewidth=1.0,
                    hatch="///",
                    label="noise floor",
                )
            else:
                ax.bar(x - 0.2, deltas, width=0.38, color=c_delta, label="function change Δ_med")
                ax.bar(x + 0.2, floors, width=0.38, color=C_FLOOR, label="noise floor")
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{li}" for li in LAYERS])
            if row == 0:
                ax.set_title(BEHAVIOR_LABEL[beh])
            if col == 0:
                ax.set_ylabel(f"{SUMMARY_LABEL[s]}\nΔ_med (activation units)")
    # Figure-level legend OUTSIDE the panels (hero/validity convention) — an
    # in-axes 3-line legend collides with the tall EM L21 bars.
    handles = [
        Patch(facecolor=C_MEAN, label="Δ_med (answer mean)"),
        Patch(facecolor=C_TURN, label="Δ_med (turn boundary)"),
        Patch(facecolor=C_FLOOR, label="noise floor"),
        Patch(
            facecolor="white",
            edgecolor=C_TURN,
            linewidth=1.0,
            hatch="///",
            label="sycophancy turn boundary — untrusted (failed base-leg validity)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8)
    savefig_paper(fig, "issue_811/function_change_raw_delta_vs_floor", dir=FIG_DIR)
    plt.close(fig)


def fig_chain_rho_forest() -> None:
    """Chain-ρ under M0 vs M⁺, both summaries, 95% family-clustered CI whiskers.

    Sycophancy's turn-boundary rows render as OPEN markers: that summary FAILED
    the Phase-0 base-leg validity gate (KILL-1 read rule), so those chain reads
    are untrusted.
    """
    rho = {s: _cells(s, "rho") for s in SUMMARIES}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharex=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        yticks, ylabels = [], []
        y = 0.0
        for li in LAYERS:
            for s in SUMMARIES:
                cell = rho[s][f"{beh}/L{li}"]
                untrusted = beh == "sycophancy" and s == "turn_nl"
                for key, color, dy in (
                    ("M0", C_MEAN, 0.16),
                    ("Mplus", C_TURN, -0.16),
                ):
                    ci = cell[f"ci_{key}_ridge"]
                    pt = ci["point"]
                    err = [[pt - ci["ci_lo"]], [ci["ci_hi"] - pt]]
                    marker_style = (
                        {
                            "markerfacecolor": "white",
                            "markeredgecolor": color,
                            "markeredgewidth": 1.2,
                        }
                        if untrusted
                        else {"markeredgewidth": 0.0}
                    )
                    ax.errorbar(
                        pt,
                        y + dy,
                        xerr=err,
                        fmt="o",
                        ms=4.5,
                        color=color,
                        capsize=2,
                        lw=1.1,
                        **marker_style,
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
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="none",
            markerfacecolor="white",
            markeredgecolor=C_TURN,
            markeredgewidth=1.2,
            label="open: sycophancy turn boundary — untrusted (failed base-leg validity)",
        ),
    ]
    # Figure-level legend OUTSIDE the panels — the in-axes lower-right legend
    # overlapped the taught-fact L21 rows (interp-critique r1, both critics).
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    savefig_paper(fig, "issue_811/chain_rho_forest_ci", dir=FIG_DIR)
    plt.close(fig)


def fig_delta_scatter() -> None:
    """Per-cell (behavior×layer) Δ_med under mean vs turn boundary, 45° line.

    Sycophancy cells render as OPEN markers: their turn-boundary coordinate is
    an untrusted read (failed the Phase-0 base-leg validity gate).
    """
    fc = {s: _cells(s, "fc") for s in SUMMARIES}
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = 0.45
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color="0.4")
    for beh in BEHAVIORS:
        for li in LAYERS:
            xm = fc["mean"][f"{beh}/L{li}"]["Delta_med"]
            yt = fc["turn_nl"][f"{beh}/L{li}"]["Delta_med"]
            if beh == "sycophancy":
                ax.scatter(
                    xm, yt, s=42, facecolors="white", edgecolors=C_TURN, linewidths=1.2, zorder=3
                )
            else:
                ax.scatter(xm, yt, s=42, color=C_TURN, zorder=3)
            ax.text(xm + 0.006, yt + 0.006, f"{BEHAVIOR_LABEL[beh]} L{li}", fontsize=7.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Δ_med under answer-mean summary (activation units)")
    ax.set_ylabel("Δ_med under turn-boundary summary (activation units)")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", ls="", color=C_TURN, label="harmful-compliance / taught fact"
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="none",
            markerfacecolor="white",
            markeredgecolor=C_TURN,
            markeredgewidth=1.2,
            label="sycophancy — turn-boundary read untrusted (failed base-leg validity)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, fontsize=8)
    savefig_paper(fig, "issue_811/delta_scatter_pairs", dir=FIG_DIR)
    plt.close(fig)


def fig_validity_gate() -> None:
    """Phase-2 per-cell MLP-vs-shuffle gate margin per behavior×layer, both summaries.

    These are the PHASE-2 refit margins (computed inside the full paired fit) —
    NOT the Phase-0 KILL-1 base-leg margins, which disagree systematically at
    L14 (all six cells lower here, 5 of 6 sign-flipped). Sycophancy's
    turn-boundary bars are hatched/open: that summary failed the Phase-0
    base-leg validity gate, so its reads are untrusted regardless of the
    per-cell margin shown here (sycophancy L7 turn sits slightly above zero).
    """
    gate = {s: _cells(s, "gate") for s in SUMMARIES}
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8), sharey=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        x = np.arange(len(LAYERS))
        for off, (s, color) in zip(
            (-0.2, 0.2), [("mean", C_MEAN), ("turn_nl", C_TURN)], strict=True
        ):
            vals = [gate[s][f"{beh}/L{li}"]["gate_margin"] for li in LAYERS]
            if s == "turn_nl" and beh == "sycophancy":
                ax.bar(
                    x + off,
                    vals,
                    width=0.38,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    hatch="///",
                )
            else:
                ax.bar(x + off, vals, width=0.38, color=color)
        ax.axhline(0.0, ls="--", lw=1.0, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {li}" for li in LAYERS])
        ax.set_title(BEHAVIOR_LABEL[beh])
    axes[0].set_ylabel("gate margin (ρ real − ρ shuffled)")
    handles = [
        Patch(facecolor=C_MEAN, label="answer mean"),
        Patch(facecolor=C_TURN, label="turn boundary"),
        Patch(
            facecolor="white",
            edgecolor=C_TURN,
            linewidth=1.0,
            hatch="///",
            label="turn boundary — untrusted (failed base-leg validity)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    savefig_paper(fig, "issue_811/validity_gate_margins", dir=FIG_DIR)
    plt.close(fig)


def fig_per_context_strips() -> None:
    """Per-context |δ(c)| ÷ floor strips behind every hero bar (Lens-11 companion).

    The hero shows one MEDIAN ratio per (behavior, layer, summary) bar; this
    companion plots the 16 per-context values each median summarizes, read from
    ``offset_decomposition.json``'s ``delta_per_context`` (the F1 refit, which
    reproduced every cell's Delta_med to 2.5e-11 relative). Log y — the trusted
    ratios span ~0.003–7.2 (the untrusted sycophancy turn outlier reaches 12.8).
    A dark tick marks each cell's median (= the hero bar). Sycophancy
    turn-boundary cells render open (failed base-leg validity, untrusted).
    """
    data = json.loads((EVAL_DIR / "offset_decomposition.json").read_text())["cells"]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.2), sharey=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        x = np.arange(len(LAYERS))
        for off, (s, color) in zip(
            (-0.2, 0.2), [("mean", C_MEAN), ("turn_nl", C_TURN)], strict=True
        ):
            for xi, li in zip(x, LAYERS, strict=True):
                cell = data[f"{beh}/L{li}/{s}"]
                vals = np.abs(np.asarray(list(cell["delta_per_context"].values())))
                ratios = vals / cell["floor_combined"]
                jit = rng.uniform(-0.08, 0.08, size=ratios.size)
                if s == "turn_nl" and beh == "sycophancy":
                    ax.scatter(
                        xi + off + jit,
                        ratios,
                        s=15,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=0.9,
                        zorder=3,
                    )
                else:
                    ax.scatter(xi + off + jit, ratios, s=15, color=color, alpha=0.75, zorder=3)
                med = float(np.median(vals)) / cell["floor_combined"]
                ax.plot(
                    [xi + off - 0.14, xi + off + 0.14],
                    [med, med],
                    color="#3B3B3B",
                    lw=1.6,
                    zorder=4,
                )
        ax.axhline(1.0, ls="--", lw=1.0, color="0.4")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {li}" for li in LAYERS])
        ax.set_title(BEHAVIOR_LABEL[beh])
    axes[0].set_ylabel("per-context map change ÷ noise floor")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=C_MEAN, label="answer mean (16 contexts)"),
        plt.Line2D([0], [0], marker="o", ls="", color=C_TURN, label="turn boundary (16 contexts)"),
        plt.Line2D([0], [0], color="#3B3B3B", lw=1.6, label="median (= hero bar)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="none",
            markerfacecolor="white",
            markeredgecolor=C_TURN,
            markeredgewidth=0.9,
            label="open: sycophancy turn boundary — untrusted",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8)
    savefig_paper(fig, "issue_811/function_change_per_context", dir=FIG_DIR)
    plt.close(fig)


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    figs = {
        "hero": fig_hero_ratio,
        "raw_vs_floor": fig_raw_delta_vs_floor,
        "forest": fig_chain_rho_forest,
        "scatter": fig_delta_scatter,
        "gate": fig_validity_gate,
        "per_context": fig_per_context_strips,
    }
    ap = argparse.ArgumentParser(description="Issue #811 analyzer figures")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=sorted(figs),
        default=None,
        help="regenerate only these figures (default: all)",
    )
    args = ap.parse_args()
    set_paper_style("blog")
    todo = args.only or list(figs)
    for name in todo:
        figs[name]()
    logger.info("wrote %d figure(s) to figures/issue_811/: %s", len(todo), ", ".join(todo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
