"""Analyzer figures for issue #811 follow-up round ``maxp-winner-mapchange``.

Extends ``scripts/issue811_analyzer_figures.py`` (the v1 two-summary analyzer
figures) to the THREE-summary grid (mean / turn_nl / maxp) of the
``maxp-winner-mapchange`` round, fixing the two render defects of the run's
draft figures: colliding x-tick labels on the grouped bars, and a chain-rho
"forest" with no CI whiskers. All figures save via ``savefig_paper``
(PNG + PDF + ``.meta.json`` sidecar) under
``figures/issue_811/maxp-winner-mapchange/``.

Figures:
- ``hero_function_change_three_summaries`` — Delta_med / floor_combined per
  behavior x layer, three summaries, hatched untrusted sycophancy turn bars.
- ``chain_rho_forest_three_summaries_ci`` — chain-rho M0 vs M+ per
  behavior x layer x summary with 95% family-clustered CI whiskers.
- ``per_context_strips_three_summaries`` — the 16 per-context |delta(c)| / floor
  values behind every hero bar (from ``offset_decomposition.json``).
- ``scatter_maxp_vs_mean_and_turn`` — ONE two-panel 9-cell scatter of raw
  Delta_med, max-pool vs the answer mean (left) and vs the turn boundary
  (right); supersedes the two standalone ``scatter_maxp_vs_{mean,turn_nl}``
  files (one figure unit, one sidecar).

Run from the repo/worktree root: ``uv run python scripts/issue811_maxp_round_figures.py``
(``--only scatter`` regenerates just the two-panel scatter).
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

logger = logging.getLogger("issue811.maxp_round_figures")

EVAL_DIR = PROJECT_ROOT / "eval_results/issue_811/maxp-winner-mapchange"
FIG_DIR = "figures/"
FIG_PREFIX = "issue_811/maxp-winner-mapchange"

BEHAVIORS = ("em", "sycophancy", "fact")
LAYERS = (7, 14, 21)
SUMMARIES = ("mean", "turn_nl", "maxp")
BEHAVIOR_LABEL = {
    "em": "harmful-compliance (EM)",
    "sycophancy": "sycophancy",
    "fact": "taught fact",
}
SUMMARY_LABEL = {"mean": "answer mean", "turn_nl": "turn boundary", "maxp": "max-pool"}
C_SUMMARY = {
    "mean": paper_palette_role("baseline"),
    "turn_nl": paper_palette_role("primary"),
    "maxp": paper_palette_role("control"),
}


def _cells(summary: str, which: str) -> dict:
    fname = {
        "fc": f"function_change_{summary}.json",
        "rho": f"chain_rho_M0_Mplus_{summary}.json",
    }[which]
    with open(EVAL_DIR / fname) as fh:
        return json.load(fh)["cells"]


def _untrusted(beh: str, summary: str) -> bool:
    """Sycophancy turn-boundary reads failed the v1 pre-spend base-leg gate."""
    return beh == "sycophancy" and summary == "turn_nl"


def fig_hero_three_summaries() -> None:
    """Delta_med / floor_combined per behavior x layer, three grouped bars."""
    fc = {s: _cells(s, "fc") for s in SUMMARIES}
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.9), sharey=True)
    offs = (-0.27, 0.0, 0.27)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        x = np.arange(len(LAYERS))
        for off, s in zip(offs, SUMMARIES, strict=True):
            color = C_SUMMARY[s]
            vals = [
                fc[s][f"{beh}/L{li}"]["Delta_med"] / fc[s][f"{beh}/L{li}"]["floor_combined"]
                for li in LAYERS
            ]
            if _untrusted(beh, s):
                ax.bar(
                    x + off,
                    vals,
                    width=0.25,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    hatch="///",
                )
            else:
                ax.bar(x + off, vals, width=0.25, color=color)
        ax.axhline(1.0, ls="--", lw=1.0, color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels([f"layer {li}" for li in LAYERS])
        ax.set_title(BEHAVIOR_LABEL[beh])
    axes[0].set_ylabel("function change ÷ noise floor")
    handles = [
        Patch(facecolor=C_SUMMARY["mean"], label="answer mean"),
        Patch(facecolor=C_SUMMARY["turn_nl"], label="turn boundary"),
        Patch(facecolor=C_SUMMARY["maxp"], label="max-pool"),
        Patch(
            facecolor="white",
            edgecolor=C_SUMMARY["turn_nl"],
            linewidth=1.0,
            hatch="///",
            label="turn boundary — untrusted (failed base-leg validity)",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8)
    savefig_paper(fig, f"{FIG_PREFIX}/hero_function_change_three_summaries", dir=FIG_DIR)
    plt.close(fig)


def fig_chain_rho_forest_ci() -> None:
    """Chain-rho under M0 vs M+, three summaries, 95% family-clustered whiskers."""
    rho = {s: _cells(s, "rho") for s in SUMMARIES}
    c_m0 = paper_palette_role("baseline")
    c_mp = paper_palette_role("primary")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.6), sharex=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        yticks, ylabels = [], []
        y = 0.0
        for li in LAYERS:
            for s in SUMMARIES:
                cell = rho[s][f"{beh}/L{li}"]
                untrusted = _untrusted(beh, s)
                for key, color, dy in (("M0", c_m0, 0.16), ("Mplus", c_mp, -0.16)):
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
                        [pt],
                        [y + dy],
                        xerr=err,
                        fmt="o",
                        ms=4.5,
                        color=color,
                        elinewidth=1.1,
                        capsize=2.0,
                        **marker_style,
                    )
                yticks.append(y)
                ylabels.append(f"L{li} · {SUMMARY_LABEL[s]}")
                y -= 1.0
            y -= 0.5  # gap between layers
        ax.axvline(0.0, ls="--", lw=1.0, color="0.4")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_title(BEHAVIOR_LABEL[beh])
        ax.set_xlabel("chain correlation (Spearman)")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=c_m0, label="base map M0"),
        plt.Line2D([0], [0], marker="o", ls="", color=c_mp, label="post-fine-tuning map M⁺"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="none",
            markerfacecolor="white",
            markeredgecolor=c_mp,
            markeredgewidth=1.2,
            label="open: sycophancy turn boundary — untrusted",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    savefig_paper(fig, f"{FIG_PREFIX}/chain_rho_forest_three_summaries_ci", dir=FIG_DIR)
    plt.close(fig)


def fig_per_context_strips() -> None:
    """Per-context |delta(c)| / floor strips behind every hero bar (three summaries)."""
    data = json.loads((EVAL_DIR / "offset_decomposition.json").read_text())["cells"]
    rng = np.random.default_rng(42)
    offs = (-0.27, 0.0, 0.27)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.3), sharey=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        x = np.arange(len(LAYERS))
        for off, s in zip(offs, SUMMARIES, strict=True):
            color = C_SUMMARY[s]
            for xi, li in zip(x, LAYERS, strict=True):
                cell = data[f"{beh}/L{li}/{s}"]
                vals = np.abs(np.asarray(list(cell["delta_per_context"].values())))
                ratios = vals / cell["floor_combined"]
                jit = rng.uniform(-0.07, 0.07, size=ratios.size)
                if _untrusted(beh, s):
                    ax.scatter(
                        xi + off + jit,
                        ratios,
                        s=13,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=0.9,
                        zorder=3,
                    )
                else:
                    ax.scatter(xi + off + jit, ratios, s=13, color=color, alpha=0.75, zorder=3)
                med = float(np.median(vals)) / cell["floor_combined"]
                ax.plot(
                    [xi + off - 0.12, xi + off + 0.12],
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
        plt.Line2D([0], [0], marker="o", ls="", color=C_SUMMARY["mean"], label="answer mean"),
        plt.Line2D([0], [0], marker="o", ls="", color=C_SUMMARY["turn_nl"], label="turn boundary"),
        plt.Line2D([0], [0], marker="o", ls="", color=C_SUMMARY["maxp"], label="max-pool"),
        plt.Line2D([0], [0], color="#3B3B3B", lw=1.6, label="median (= hero bar)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="none",
            markerfacecolor="white",
            markeredgecolor=C_SUMMARY["turn_nl"],
            markeredgewidth=0.9,
            label="open: sycophancy turn boundary — untrusted",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5)
    savefig_paper(fig, f"{FIG_PREFIX}/per_context_strips_three_summaries", dir=FIG_DIR)
    plt.close(fig)


# Per-point text placement overrides for the 9-cell scatter panels, keyed
# (panel spec key, behavior, layer) -> (x_factor, y_factor, ha, va) in log-space
# (the spec keys keep the historical standalone-figure names; the panels now
# live side by side in the single scatter_maxp_vs_mean_and_turn figure)
# multiplicative offsets. Defaults put text right-above; overrides resolve the
# bottom-left cluster collisions the round-2 critique flagged (the uniform
# +0.004 linear offset also let paper_plots' nearest-text sidecar join attach
# "harmful-compliance L7" to the sycophancy L7 point).
_SCATTER_TEXT_SPECS: dict[tuple[str, str, int], tuple[float, float, str, str]] = {
    ("scatter_maxp_vs_mean", "em", 7): (1.0, 0.90, "center", "top"),
    ("scatter_maxp_vs_mean", "sycophancy", 7): (1.0, 1.10, "center", "bottom"),
    ("scatter_maxp_vs_mean", "sycophancy", 14): (1.05, 0.93, "left", "top"),
    ("scatter_maxp_vs_mean", "fact", 14): (0.95, 1.03, "right", "bottom"),
    ("scatter_maxp_vs_turn_nl", "sycophancy", 7): (0.95, 1.03, "right", "bottom"),
    ("scatter_maxp_vs_turn_nl", "sycophancy", 14): (1.05, 0.92, "left", "top"),
    ("scatter_maxp_vs_turn_nl", "sycophancy", 21): (0.95, 1.03, "right", "bottom"),
    ("scatter_maxp_vs_turn_nl", "fact", 21): (0.95, 0.93, "right", "top"),
    ("scatter_maxp_vs_turn_nl", "em", 21): (0.95, 1.03, "right", "bottom"),
}


def fig_delta_scatter_pair() -> None:
    """Raw Delta_med per cell (9 points), max-pool vs BOTH reference summaries.

    ONE two-panel figure (left: vs answer mean; right: vs turn boundary) with a
    single ``.meta.json`` sidecar — the panels share the max-pool y-quantity and
    tell one call-level story, so they are one figure unit (this supersedes the
    two standalone ``scatter_maxp_vs_{mean,turn_nl}`` files; clean-result-critic
    round-1, Lens 3/9). Labeled points, identity line, LOG-LOG axes per panel
    (the 9 cells span ~30x). Each point is its own labeled scatter series with a
    panel-qualified label, so the sidecar carries an exact per-point ``series``
    name. On the turn-boundary panel sycophancy points render OPEN — their x
    coordinate derives from the untrusted turn-boundary fits.
    """
    fc = {s: _cells(s, "fc") for s in SUMMARIES}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2))
    panels = (("mean", "scatter_maxp_vs_mean"), ("turn_nl", "scatter_maxp_vs_turn_nl"))
    for ax, (ref, spec_key) in zip(axes, panels, strict=True):
        vals: list[float] = []
        for beh in BEHAVIORS:
            for li in LAYERS:
                x = fc[ref][f"{beh}/L{li}"]["Delta_med"]
                y = fc["maxp"][f"{beh}/L{li}"]["Delta_med"]
                vals += [x, y]
                cell_label = f"{BEHAVIOR_LABEL[beh].split(' (')[0]} L{li}"
                series_label = f"{cell_label} · vs {SUMMARY_LABEL[ref]}"
                open_marker = ref == "turn_nl" and beh == "sycophancy"
                if open_marker:
                    ax.scatter(
                        x,
                        y,
                        s=42,
                        facecolors="white",
                        edgecolors=C_SUMMARY["maxp"],
                        linewidths=1.2,
                        zorder=3,
                        label=series_label,
                    )
                else:
                    ax.scatter(x, y, s=42, color=C_SUMMARY["maxp"], zorder=3, label=series_label)
                fx, fy, ha, va = _SCATTER_TEXT_SPECS.get(
                    (spec_key, beh, li), (1.05, 1.03, "left", "bottom")
                )
                ax.text(x * fx, y * fy, cell_label, fontsize=6.5, ha=ha, va=va)
        lo, hi = min(vals) * 0.55, max(vals) * 1.6
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.4")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"map change under {SUMMARY_LABEL[ref]} (median, activation units)")
        if ref == "mean":
            ax.set_ylabel("map change under max-pool (median, activation units)")
        if ref == "turn_nl":
            legend_handle = ax.scatter(
                [],
                [],
                s=42,
                facecolors="white",
                edgecolors=C_SUMMARY["maxp"],
                linewidths=1.2,
                label="open: sycophancy — turn-boundary coordinate untrusted",
            )
            ax.legend(handles=[legend_handle], loc="upper left", fontsize=7)
    savefig_paper(fig, f"{FIG_PREFIX}/scatter_maxp_vs_mean_and_turn", dir=FIG_DIR)
    plt.close(fig)


FIGURES = {
    "hero": fig_hero_three_summaries,
    "forest": fig_chain_rho_forest_ci,
    "strips": fig_per_context_strips,
    "scatter": fig_delta_scatter_pair,
}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Issue #811 maxp-round analyzer figures")
    ap.add_argument(
        "--only",
        choices=tuple(FIGURES),
        default=None,
        help="regenerate a single figure (default: all)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    set_paper_style("blog")
    names = (args.only,) if args.only else tuple(FIGURES)
    for name in names:
        FIGURES[name]()
    logger.info("wrote %d figure(s) to figures/%s/", len(names), FIG_PREFIX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
