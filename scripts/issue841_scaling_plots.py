#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (Δ, ℓ, R², →, ×) in labels/docstrings; the plotting
# dispatcher is a flat sequence of guarded figure blocks (C901 waived).
"""Issue #841 scaling-capture plots — the DV1 R²(n) + DV2 transport-scaling heroes.

Reads ``eval_results/issue_841/scaling-capture/{stage0_scaling,stage1_scaling,
transport_fidelity_scaling}.json`` and writes figures under
``figures/issue_841/scaling-capture/`` via the paper-plots rcParams. Each figure
guards on input presence so the SAME script runs on a smoke's partial outputs
and the full run. No annotations/arrows (project plot rule); per-unit points are
labeled where a per-cell scatter is drawn.

Heroes:
  1  DV1 R²(n) curve — held-out r2_id vs n at the data-limited transitions,
     ridge + MLP, two panels (identity-relative + mean-centered diagnostic).
  2  DV2 transport-scaling — win_count(n) + mean_paired_delta(n) vs n, the anchor
     win-count + BH chance line marked.
Exploratory: retention(n) vs n; transport-fidelity(n); position-drift bars.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_scaling_plots")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_841" / "scaling-capture"
FIG_DIR = "figures/"
FIG_SUBDIR = "issue_841/scaling-capture"
DATA_LIMITED_BAND = tuple(range(17, 26))
# Reader-facing legend labels for the transport classes (no bare code slugs in
# rendered figure legends — match the body's plain-English condition names).
CLASS_LABELS = {"ridge": "affine ridge", "mlp": "MLP map", "direct_hop": "direct-hop ridge"}


def _set_style() -> None:
    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception as e:
        logger.warning("[plots] set_paper_style failed (%s); default style", e)


def _save(fig, stem: str) -> None:
    try:
        from explore_persona_space.analysis.paper_plots import savefig_paper

        savefig_paper(fig, f"{FIG_SUBDIR}/{stem}", dir=FIG_DIR)
    except Exception as e:
        logger.warning("[plots] savefig_paper failed (%s); plain savefig", e)
        out = Path(FIG_DIR) / FIG_SUBDIR
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load(name: str):
    p = EVAL_DIR / name
    if not p.exists():
        logger.warning("[plots] %s absent — skipping its figures", p)
        return None
    with open(p) as f:
        return json.load(f)


# ── Hero 1: DV1 R²(n) curve ───────────────────────────────────────────────────


def plot_r2_scaling(stage0: dict) -> None:
    curve = stage0.get("scaling_curve", {})
    ridge = curve.get("ridge", {})
    mlp = curve.get("mlp", {})
    if "raw" not in ridge:
        logger.warning("[plots] no ridge raw curve; skip R²(n)")
        return
    ns = [str(n) for n in ridge.get("ns", stage0.get("ns", []))]
    ns_int = [int(n) for n in ns]
    band = [t for t in stage0.get("transitions", []) if t in DATA_LIMITED_BAND]
    if not band:
        band = stage0.get("transitions", [])[:4]
    for metric in ("r2_id", "r2_meancentered"):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for t in band:
            key = f"transition_{t}"
            ys = [ridge["raw"].get(key, {}).get(n, {}).get(metric, np.nan) for n in ns]
            ax.plot(ns_int, ys, marker="o", label=f"ridge t{t}")
        # MLP at whatever n-points it was fit (primary anchor + largest).
        for t in band:
            key = f"transition_{t}"
            mns = [n for n in ns if n in mlp.get(key, {})]
            if mns:
                ys = [mlp[key][n].get(metric, np.nan) for n in mns]
                ax.plot(
                    [int(n) for n in mns],
                    ys,
                    marker="s",
                    linestyle="--",
                    alpha=0.6,
                    label=f"MLP t{t}",
                )
        anchor = stage0.get("anchor_n")
        if anchor:
            ax.axvline(anchor, color="grey", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("fit-set size n")
        ax.set_ylabel(f"held-out {metric}")
        ax.set_title(f"DV1 data-scaling curve — {metric} at data-limited transitions")
        ax.legend(fontsize=6, ncol=2)
        _save(fig, f"hero_r2_scaling_{metric}")
        logger.info("[plots] wrote hero_r2_scaling_%s", metric)


# ── Hero 2: DV2 transport-scaling curve ───────────────────────────────────────


def _pooled_by_class(stage1: dict) -> dict:
    """pooled_curve_by_class (v5) with a legacy single-curve fallback wrapped as ridge."""
    if stage1.get("pooled_curve_by_class"):
        return {cls: d["curve"] for cls, d in stage1["pooled_curve_by_class"].items()}
    if stage1.get("pooled_curve"):
        return {"ridge": stage1["pooled_curve"]}
    return {}


def plot_transport_scaling(stage1: dict) -> None:
    pbc = _pooled_by_class(stage1)
    if not pbc:
        logger.warning("[plots] no pooled_curve_by_class; skip transport-scaling")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for cls, pooled in pbc.items():
        by_n = pooled["by_n"]
        anchor = pooled["anchor_n"]
        ns_scaling = sorted(int(n) for n in by_n)
        if not ns_scaling:
            continue
        xs = [anchor, *ns_scaling]
        win = [pooled["win_count_anchor"]] + [by_n[str(n)]["win_count"] for n in ns_scaling]
        mpd = [0.0] + [by_n[str(n)]["mean_paired_delta"] for n in ns_scaling]
        mpd_lo = [0.0] + [by_n[str(n)]["mean_paired_delta_ci"][0] for n in ns_scaling]
        mpd_hi = [0.0] + [by_n[str(n)]["mean_paired_delta_ci"][1] for n in ns_scaling]
        ax1.plot(xs, win, marker="o", label=CLASS_LABELS.get(cls, cls))
        ax2.errorbar(
            xs,
            mpd,
            yerr=[np.array(mpd) - np.array(mpd_lo), np.array(mpd_hi) - np.array(mpd)],
            marker="o",
            capsize=3,
            label=CLASS_LABELS.get(cls, cls),
        )
    # 20/136 baseline + chance line from the ridge (headline) class.
    ridge = pbc.get("ridge")
    if ridge and ridge["by_n"]:
        ax1.axhline(ridge["win_count_anchor"], color="grey", linestyle=":", linewidth=1)
        n0 = sorted(int(n) for n in ridge["by_n"])[0]
        ax1.axhline(
            ridge["by_n"][str(n0)]["chance_expectation"], color="red", linestyle="--", linewidth=1
        )
        ax1.set_ylabel(f"conjunction wins (of {ridge['total_cells']} cells)")
    ax1.set_xscale("log")
    ax1.set_xlabel("fit-set size n")
    ax1.set_title("DV2 transport wins vs n (per class)")
    ax1.legend(fontsize=7)
    ax2.axhline(0.0, color="grey", linestyle=":", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("fit-set size n")
    ax2.set_ylabel("mean paired Δr vs 4k anchor")
    ax2.set_title("DV2 mean paired-delta vs n (per class)")
    ax2.legend(fontsize=7)
    _save(fig, "hero_transport_scaling")
    logger.info("[plots] wrote hero_transport_scaling")


# ── Exploratory: retention(n) + position-drift ────────────────────────────────


def plot_retention_scaling(stage1: dict) -> None:
    pbc = _pooled_by_class(stage1)
    if not pbc:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for cls, pooled in pbc.items():
        by_n = pooled["by_n"]
        ns_scaling = sorted(int(n) for n in by_n)
        if not ns_scaling:
            continue
        xs = [pooled["anchor_n"], *ns_scaling]
        ys = [pooled.get("mean_retention_anchor", np.nan)] + [
            by_n[str(n)]["mean_retention"] for n in ns_scaling
        ]
        ax.plot(xs, ys, marker="o", label=cls)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("fit-set size n")
    ax.set_ylabel("mean retention (row r ÷ ceiling r)")
    ax.set_title("DV3 retention vs fit-set size (per class)")
    ax.legend(fontsize=7)
    _save(fig, "exploratory_retention_scaling")
    logger.info("[plots] wrote exploratory_retention_scaling")


def plot_position_drift(stage0: dict) -> None:
    pd = stage0.get("position_drift")
    if not pd:
        return
    band = pd.get("data_limited_band", [])
    anchor = [pd["anchor_r2_id"].get(str(t), np.nan) for t in band]
    late = [pd["late_window_r2_id"].get(str(t), np.nan) for t in band]
    if not band:
        return
    x = np.arange(len(band))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(x - 0.2, anchor, width=0.4, label="anchor (early-stream fit)")
    ax.bar(x + 0.2, late, width=0.4, label="late-window fit")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t{t}" for t in band], rotation=45, fontsize=7)
    ax.set_ylabel("held-out r2_id on the FIXED test")
    ax.set_title("Position-drift: early-stream anchor vs latest-window fit")
    ax.legend(fontsize=7)
    _save(fig, "exploratory_position_drift")
    logger.info("[plots] wrote exploratory_position_drift")


def main() -> int:
    global EVAL_DIR, FIG_DIR
    ap = argparse.ArgumentParser(description="Issue #841 scaling-capture plots.")
    ap.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    ap.add_argument(
        "--fig-dir",
        default=FIG_DIR,
        help="figure output root (divert for smokes; default = committed figures/)",
    )
    args = ap.parse_args()
    EVAL_DIR = args.eval_dir
    FIG_DIR = args.fig_dir
    _set_style()
    stage0 = _load("stage0_scaling.json")
    stage1 = _load("stage1_scaling.json")
    if stage0:
        plot_r2_scaling(stage0)
        plot_position_drift(stage0)
    if stage1:
        plot_transport_scaling(stage1)
        plot_retention_scaling(stage1)
    logger.info("[done] scaling plots complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
