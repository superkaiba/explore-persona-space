"""Analyzer figures for issue #811 follow-up round ``pre-user-boundary-summary``.

Sibling of ``scripts/issue811_maxp_round_figures.py`` for the TWELVE-summary
grid (mean / turn_nl / maxp references + nine boundary/header arms). The run's
own 108-row chain forest is unreadable and its heatmap is kept as-is; this
script adds the three analyzer figures the body embeds. All figures save via
``savefig_paper`` (PNG + PDF + ``.meta.json`` sidecar) under
``figures/issue_811/pre-user-boundary-summary/``.

Figures:
- ``chain_shift_forest_12_summaries`` — the base-to-post chain-correlation
  SHIFT (rho_diff with 95% family-clustered CI) per behavior x layer x
  summary; gate-failed arms and per-behavior base-leg collapses drawn as open
  circles, and gate-PASSING arms whose per-behavior base-leg margin ratio is
  flagged ``near_threshold`` in ``validity_gate_phase0.json`` (e.g. the role
  token's taught-fact leg at 0.502 vs the 0.5 cut) drawn as open diamonds.
- ``scatter_incl_hdr_vs_content`` — two-panel 9-cell scatter of raw
  Delta_med: header-inclusive mean pool vs the content-only mean (left) and
  header-inclusive max pool vs content-only max-pool (right), log-log, 45deg.
- ``per_context_strips_fact_L14_12_summaries`` — the 16 per-context
  |delta(c)| / floor values behind the taught-fact layer-14 heatmap column,
  all 12 summaries (from ``offset_decomposition.json``).

Run from the repo/worktree root:
``uv run python scripts/issue811_preuser_round_figures.py``
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue811.preuser_round_figures")

EVAL_DIR = PROJECT_ROOT / "eval_results/issue_811/pre-user-boundary-summary"
FIG_DIR = "figures/"
FIG_PREFIX = "issue_811/pre-user-boundary-summary"

BEHAVIORS = ("em", "sycophancy", "fact")
LAYERS = (7, 14, 21)
# Display order: three references on top, then header arms, then pools.
SUMMARIES = (
    "mean",
    "turn_nl",
    "maxp",
    "pre_user_imstart",
    "pre_user_user",
    "pre_user_nl",
    "pre_user_mean3",
    "pre_user_max3",
    "ans_mean_incl_hdr",
    "ans_max_incl_hdr",
    "ans_mean_incl_hdr_alllayer",
    "ans_max_incl_hdr_alllayer",
)
GATE_FAILED = {"pre_user_nl", "pre_user_max3", "ans_max_incl_hdr_alllayer"}
# Per-behavior base-leg collapses (phase-0 record): sycophancy collapsed on
# every pre-user header arm (and on turn_nl per the v1 gate).
SYCO_UNTRUSTED = {
    "turn_nl",
    "pre_user_imstart",
    "pre_user_user",
    "pre_user_nl",
    "pre_user_mean3",
    "pre_user_max3",
}
BEHAVIOR_LABEL = {
    "em": "harmful-compliance (EM)",
    "sycophancy": "sycophancy",
    "fact": "taught fact",
}
SUMMARY_LABEL = {
    "mean": "answer mean (ref)",
    "turn_nl": "turn boundary (ref)",
    "maxp": "max-pool (ref)",
    "pre_user_imstart": "next-turn start-tag",
    "pre_user_user": "next-turn role token",
    "pre_user_nl": "pre-user newline",
    "pre_user_mean3": "header mean-of-3",
    "pre_user_max3": "header max-of-3",
    "ans_mean_incl_hdr": "mean incl. header",
    "ans_max_incl_hdr": "max incl. header",
    "ans_mean_incl_hdr_alllayer": "mean incl. header, all-layer",
    "ans_max_incl_hdr_alllayer": "max incl. header, all-layer",
}


def _rho_cells(summary: str) -> dict:
    with open(EVAL_DIR / f"chain_rho_M0_Mplus_{summary}.json") as fh:
        return json.load(fh)["cells"]


def _fc_cells(summary: str) -> dict:
    with open(EVAL_DIR / f"function_change_{summary}.json") as fh:
        return json.load(fh)["cells"]


def _untrusted(beh: str, summary: str) -> bool:
    if summary in GATE_FAILED:
        return True
    return beh == "sycophancy" and summary in SYCO_UNTRUSTED


def _near_threshold_flags() -> set[tuple[str, str]]:
    """(behavior, summary) pairs on gate-PASSING arms whose per-behavior
    base-leg margin ratio carries ``near_threshold: true`` in the phase-0
    gate record (e.g. the role token's taught-fact leg, ratio 0.502)."""
    with open(EVAL_DIR / "validity_gate_phase0.json") as fh:
        per_arm = json.load(fh)["per_arm"]
    flags: set[tuple[str, str]] = set()
    for arm, rec in per_arm.items():
        if rec["gate_status"] != "pass":
            continue
        for beh, pb in rec["per_behavior"].items():
            if pb.get("near_threshold"):
                flags.add((beh, arm))
    return flags


def fig_chain_shift_forest() -> None:
    """Base-to-post chain-rho shift, family-clustered CI, 12 summaries x 3 layers."""
    rho = {s: _rho_cells(s) for s in SUMMARIES}
    near_thr = _near_threshold_flags()
    c_layer = {
        7: paper_palette_role("baseline"),
        14: paper_palette_role("primary"),
        21: paper_palette_role("control"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6.2), sharex=True, sharey=True)
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        yticks, ylabels = [], []
        y = 0.0
        for s in SUMMARIES:
            for li, dy in zip(LAYERS, (0.22, 0.0, -0.22), strict=True):
                cell = rho[s][f"{beh}/L{li}"]
                ci = cell["ci_diff_ridge"]
                pt = cell["rho_diff_ridge"]
                err = [[pt - ci["ci_lo"]], [ci["ci_hi"] - pt]]
                color = c_layer[li]
                open_marker = _untrusted(beh, s)
                is_near = (not open_marker) and (beh, s) in near_thr
                style = (
                    {"markerfacecolor": "white", "markeredgecolor": color, "markeredgewidth": 1.2}
                    if (open_marker or is_near)
                    else {"markeredgewidth": 0.0}
                )
                ax.errorbar(
                    [pt],
                    [y + dy],
                    xerr=err,
                    fmt="D" if is_near else "o",
                    ms=4.6 if is_near else 4.0,
                    color=color,
                    elinewidth=1.0,
                    capsize=1.8,
                    **style,
                )
            label = SUMMARY_LABEL[s]
            if s in GATE_FAILED:
                label += " [gate-failed]"
            yticks.append(y)
            ylabels.append(label)
            y -= 1.0
            if s in ("maxp", "pre_user_max3"):
                y -= 0.5  # gaps: refs | header arms | pools
        ax.axvline(0.0, ls="--", lw=1.0, color="0.4")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7.5)
        ax.set_title(BEHAVIOR_LABEL[beh])
        ax.set_xlabel("base-to-post chain shift")
    handles = [
        plt.Line2D(
            [0], [0], marker="o", ls="", color=c_layer[li], markeredgewidth=0.0, label=f"layer {li}"
        )
        for li in LAYERS
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="none",
            markerfacecolor="white",
            markeredgecolor=c_layer[14],
            markeredgewidth=1.2,
            label="open circle: gate-failed arm or collapsed base-leg behavior",
        )
    )
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="D",
            ls="",
            color="none",
            markerfacecolor="white",
            markeredgecolor=c_layer[14],
            markeredgewidth=1.2,
            label="open diamond: gate-passing arm, near-threshold gate leg (role-token fact 0.502)",
        )
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    savefig_paper(fig, f"{FIG_PREFIX}/chain_shift_forest_12_summaries", dir=FIG_DIR)
    plt.close(fig)


def fig_scatter_incl_hdr() -> None:
    """Two-panel 9-cell scatter: header-inclusive pools vs their content-only twins."""
    pairs = (("mean", "ans_mean_incl_hdr"), ("maxp", "ans_max_incl_hdr"))
    c_beh = {
        "em": paper_palette_role("baseline"),
        "sycophancy": paper_palette_role("control"),
        "fact": paper_palette_role("primary"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, (xs, ys) in zip(axes, pairs, strict=True):
        fx, fy = _fc_cells(xs), _fc_cells(ys)
        lims = [np.inf, -np.inf]
        for beh in BEHAVIORS:
            for li in LAYERS:
                cx = fx[f"{beh}/L{li}"]
                cy = fy[f"{beh}/L{li}"]
                x, yv = cx["Delta_med"], cy["Delta_med"]
                lims = [min(lims[0], x, yv), max(lims[1], x, yv)]
                ax.scatter(
                    [x],
                    [yv],
                    s=42,
                    color=c_beh[beh],
                    zorder=3,
                    edgecolors="black",
                    linewidths=0.5,
                )
                ax.text(x * 1.09, yv, f"L{li}", fontsize=7.5, va="center", color="0.25")
        lo, hi = lims[0] * 0.7, lims[1] * 1.6
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.4", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"raw map change — {SUMMARY_LABEL[xs].removesuffix(' (ref)')}")
        ax.set_ylabel(f"raw map change — {SUMMARY_LABEL[ys]}")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color=c_beh[b],
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=BEHAVIOR_LABEL[b],
        )
        for b in BEHAVIORS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    savefig_paper(fig, f"{FIG_PREFIX}/scatter_incl_hdr_vs_content", dir=FIG_DIR)
    plt.close(fig)


def fig_per_context_strips_fact_l14() -> None:
    """16 per-context |delta(c)|/floor values at taught-fact layer 14, all 12 summaries."""
    with open(EVAL_DIR / "offset_decomposition.json") as fh:
        cells = json.load(fh)["cells"]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    rng = np.random.default_rng(42)
    c_ok = paper_palette_role("primary")
    c_fail = "0.55"
    for row, s in enumerate(SUMMARIES):
        cell = cells[f"fact/L14/{s}"]
        floor = cell["floor_combined"]
        vals = np.abs(np.array(list(cell["delta_per_context"].values()))) / floor
        y = -row + rng.uniform(-0.16, 0.16, size=vals.size)
        failed = s in GATE_FAILED
        color = c_fail if failed else c_ok
        if failed:
            ax.scatter(
                vals, y, s=22, facecolors="white", edgecolors=color, linewidths=1.0, zorder=3
            )
        else:
            ax.scatter(vals, y, s=22, color=color, alpha=0.75, zorder=3)
        med = float(np.median(vals))
        ax.plot([med, med], [-row - 0.30, -row + 0.30], color="black", lw=2.2, zorder=4)
    ax.axvline(1.0, ls="--", lw=1.0, color="0.4")
    ax.set_xscale("log")
    ax.set_yticks([-i for i in range(len(SUMMARIES))])
    ax.set_yticklabels(
        [SUMMARY_LABEL[s] + (" [gate-failed]" if s in GATE_FAILED else "") for s in SUMMARIES],
        fontsize=8,
    )
    ax.set_xlabel("per-context map change ÷ combined noise floor (log scale)")
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=c_ok, label="gate-passing summary"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color="none",
            markerfacecolor="white",
            markeredgecolor=c_fail,
            markeredgewidth=1.0,
            label="gate-failed arm (open)",
        ),
        plt.Line2D([0], [0], color="black", lw=2.2, label="median (the heatmap value)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8)
    savefig_paper(fig, f"{FIG_PREFIX}/per_context_strips_fact_L14_12_summaries", dir=FIG_DIR)
    plt.close(fig)


def fig_heatmap_12_summaries() -> None:
    """Delta_med / floor heatmap, 12 summaries x 9 cells, gate-failed rows hatched."""
    from matplotlib.colors import LogNorm

    fc = {s: _fc_cells(s) for s in SUMMARIES}
    cols = [(beh, li) for beh in BEHAVIORS for li in LAYERS]
    grid = np.array(
        [
            [
                fc[s][f"{beh}/L{li}"]["Delta_med"] / fc[s][f"{beh}/L{li}"]["floor_combined"]
                for (beh, li) in cols
            ]
            for s in SUMMARIES
        ]
    )
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    im = ax.imshow(
        grid, cmap="viridis", norm=LogNorm(vmin=grid.min(), vmax=grid.max()), aspect="auto"
    )
    for i, s in enumerate(SUMMARIES):
        for j, (beh, _li) in enumerate(cols):
            v = grid[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v < grid.max() / 6 else "black",
            )
            if s in GATE_FAILED or _untrusted(beh, s):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor="white",
                        linewidth=0.0,
                    )
                )
    short_beh = {"em": "harmful-compl.", "sycophancy": "sycophancy", "fact": "taught fact"}
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{short_beh[b]}\nL{li}" for (b, li) in cols], fontsize=8)
    ax.set_yticks(range(len(SUMMARIES)))
    ax.set_yticklabels(
        [SUMMARY_LABEL[s] + (" [gate-failed]" if s in GATE_FAILED else "") for s in SUMMARIES],
        fontsize=8.5,
    )
    for yy in (2.5, 7.5):
        ax.axhline(yy, color="white", lw=2.0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("function change ÷ noise floor (log scale)")
    savefig_paper(fig, f"{FIG_PREFIX}/heatmap_function_change_12_summaries", dir=FIG_DIR)
    plt.close(fig)


FIGS = {
    "heatmap": fig_heatmap_12_summaries,
    "forest": fig_chain_shift_forest,
    "scatter": fig_scatter_incl_hdr,
    "strips": fig_per_context_strips_fact_l14,
}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    set_paper_style("blog")
    for name, fn in FIGS.items():
        if only and name != only:
            continue
        logger.info("rendering %s", name)
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
