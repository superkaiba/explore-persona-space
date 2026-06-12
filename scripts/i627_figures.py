#!/usr/bin/env python3
"""Task #627 — figures (plan §6: over-produce).

Hero 1  matched-install gap forest plot — one row per comparison family,
        the one-glance "which headlines survive at matched install" figure.
Hero 2  leakage-vs-install dose curves, 2x2 panel per family, conditions
        overlaid, matched-install verticals, new #627 panel points
        highlighted.
Exploratory dump: per-source #608 trajectories with the new bystander points;
per-persona fraction heatmaps; fraction dot plots at matched install; #606
endpoint spread; measured-vs-interpolated comparison; marker fraction
sensitivity panel.

All figures use the project paper rcParams (``set_paper_style``) and land in
``figures/issue_627/`` via ``savefig_paper`` (commit-pinned metadata).
Plain-English labels only — no condition slugs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

log = logging.getLogger("i627_figures")

ANALYSIS_DIR = Path("eval_results/issue_627/analysis")
FIG_DIR = Path("figures/issue_627")
INSTALL_TARGET = 0.50

ARM_LABEL = {
    "contrastive_dense": "with corrective counter-examples",
    "posonly_dose_dense": "positives only",
}
MARKER_ARM_LABEL = {"contrastive": "mixed with negatives", "posonly": "positives only"}


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the producing analysis script first")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Hero 1 — forest plot
# ---------------------------------------------------------------------------


def hero_forest(synthesis: dict) -> None:
    rows = [r for r in synthesis["rows"] if r.get("gap") is not None and r.get("ci95")]
    if not rows:
        raise RuntimeError("forest plot: no rows with gap + CI — synthesis incomplete")
    labels = [r["family"] for r in rows]
    gaps = np.array([r["gap"] for r in rows], dtype=float)
    lo = np.array([r["ci95"][0] for r in rows], dtype=float)
    hi = np.array([r["ci95"][1] for r in rows], dtype=float)
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(rows) + 1.6))
    color = paper_palette_role("accent")
    ax.errorbar(
        gaps,
        y,
        xerr=np.vstack([np.maximum(0.0, gaps - lo), np.maximum(0.0, hi - gaps)]),
        fmt="o",
        color=color,
        ecolor=color,
        capsize=3,
        lw=1.6,
        ms=6,
    )
    ax.axvline(0.0, color=paper_palette_role("neutral"), lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("leakage gap at matched install (95% CI; units differ per row)")
    set_title_subtitle(
        ax,
        "Which leakage headlines survive matching on install strength?",
        "gap > 0 = the first-named condition leaks more at equal install",
    )
    fig.tight_layout()
    savefig_paper(fig, "hero1_matched_install_forest", dir=FIG_DIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Hero 2 — dose curves 2x2
# ---------------------------------------------------------------------------


def hero_dose_curves(m608: dict, m601: dict, f606: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    c_main = paper_palette_role("primary")
    c_alt = paper_palette_role("accent")

    # (a) sycophancy mix contrast — NEW #627 bystander points per arm.
    ax = axes[0][0]
    for arm, color in (("contrastive_dense", c_main), ("posonly_dose_dense", c_alt)):
        xs, ys = [], []
        for entry in m608["per_source"].values():
            for role in ("lower", "upper"):
                key = f"{role}_endpoint_bys_mean"
                be = entry[arm]["bracket_eval"]
                x = be["fresh_lo"] if role == "lower" else be["fresh_hi"]
                xs.append(x)
                ys.append(entry[arm]["registered_21"][key])
        ax.scatter(xs, ys, color=color, label=ARM_LABEL[arm], s=28, zorder=3)
    ax.axvline(INSTALL_TARGET, color=paper_palette_role("neutral"), ls="--", lw=1.0)
    ax.set_xlabel("source agreement rate (fresh)")
    ax.set_ylabel("bystander-mean agreement delta")
    ax.set_title("Agreement with false claims: training-mix contrast (new runs)")
    ax.legend()

    # (b) marker mix contrast — on-policy dose curves (margin space).
    ax = axes[0][1]
    for curve in m601["dose_curves_onpolicy"]:
        if curve["mix_arm"] is None:
            continue
        pts = sorted(curve["points"], key=lambda p: p["install_margin"])
        color = c_main if curve["mix_arm"] == "contrastive" else c_alt
        ax.plot(
            [p["install_margin"] for p in pts],
            [p["leak_margin"] for p in pts],
            color=color,
            alpha=0.55,
            lw=1.2,
            marker="o",
            ms=3,
        )
    ax.plot([], [], color=c_main, label=MARKER_ARM_LABEL["contrastive"])
    ax.plot([], [], color=c_alt, label=MARKER_ARM_LABEL["posonly"])
    ax.set_xlabel("source install (EOS-margin logits)")
    ax.set_ylabel("bystander-mean leakage (margin)")
    ax.set_title("Marker token: training-mix contrast (on-policy reads)")
    ax.legend()

    # (c)+(d) LoRA vs full fine-tuning, sycophancy + refusal (#606 cells).
    for ax, behavior, title in (
        (axes[1][0], "sycophancy", "Agreement with false claims: LoRA vs full fine-tune"),
        (axes[1][1], "refusal", "Refusal: LoRA vs full fine-tune"),
    ):
        curves = f606["per_behavior"][behavior]["dose_curves"]
        for arm, pts in curves.items():
            label = {"lora": "LoRA", "ft": "full fine-tune"}.get(arm, arm)
            color = c_main if arm == "lora" else c_alt
            ax.plot(
                [p["install_s"] for p in pts],
                [p["leak"] for p in pts],
                color=color,
                marker="o",
                ms=4,
                lw=1.4,
                label=label,
            )
        ax.axvline(INSTALL_TARGET, color=paper_palette_role("neutral"), ls="--", lw=1.0)
        ax.set_xlabel("source install (rate delta)")
        ax.set_ylabel("bystander-mean rate delta")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Leakage vs install strength, per comparison family", y=1.01)
    fig.tight_layout()
    savefig_paper(fig, "hero2_dose_curves", dir=FIG_DIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exploratory dump
# ---------------------------------------------------------------------------


def explore_608_trajectories(m608: dict, cells_manifest: dict) -> None:
    """Per-source committed own-rate trajectories + the new bystander points."""
    sources = sorted(m608["per_source"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False)
    c_main, c_alt = paper_palette_role("primary"), paper_palette_role("accent")
    for ax, source in zip(axes.flat, sources, strict=False):
        brackets = cells_manifest["brackets"][source]
        for arm, color in (("contrastive_dense", c_main), ("posonly_dose_dense", c_alt)):
            traj = brackets[arm]["committed_trajectory"]
            steps = sorted(int(k) for k in traj)
            ax.plot(
                steps,
                [traj[str(k)] for k in steps],
                color=color,
                marker=".",
                lw=1.2,
                label=ARM_LABEL[arm],
            )
            for step_key in ("lo_step", "hi_step"):
                ax.axvline(brackets[arm][step_key], color=color, ls=":", lw=0.8, alpha=0.6)
        ax.axhline(INSTALL_TARGET, color=paper_palette_role("neutral"), ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_title(source.replace("_", " "))
        ax.set_xlabel("training step")
        ax.set_ylabel("source agreement rate")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Committed install trajectories + the re-measured bracket checkpoints")
    fig.tight_layout()
    savefig_paper(fig, "explore_608_install_trajectories", dir=FIG_DIR)
    plt.close(fig)


def explore_fraction_heatmap(m601: dict) -> None:
    """Per-(cell, bystander) margin-space fraction heatmap at each cell's
    final on-policy checkpoint (above-floor cells only)."""
    rows = [
        r
        for r in m601["three_space_tables_onpolicy"]
        if r["fraction_margin"] is not None and r["frac"] == 1.0
    ]
    if not rows:
        log.warning("fraction heatmap skipped: no above-floor final checkpoints")
        return
    by_cell: dict[str, dict] = {f"{r['cell']} (seed {r['seed']})": r for r in rows}
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(by_cell) + 1.5))
    labels = sorted(by_cell)
    vals = [by_cell[k]["fraction_margin"] for k in labels]
    colors = [
        paper_palette_role("primary")
        if by_cell[k]["mix_arm"] == "contrastive"
        else paper_palette_role("accent")
        for k in labels
    ]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("bystander-mean leakage fraction of install (margin space)")
    set_title_subtitle(
        ax,
        "Marker leakage as a share of install, per run (final checkpoint)",
        "blue = mixed with negatives, orange = positives only; below-floor cells excluded",
    )
    fig.tight_layout()
    savefig_paper(fig, "explore_marker_fraction_dots", dir=FIG_DIR)
    plt.close(fig)


def explore_measured_vs_interpolated(m608: dict) -> None:
    """Lower/upper/interpolated sandwich per arm-source (plan §13 item 7)."""
    sources = sorted(m608["per_source"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    c_main, c_alt = paper_palette_role("primary"), paper_palette_role("accent")
    x = np.arange(len(sources), dtype=float)
    for off, (arm, color) in zip(
        (-0.15, 0.15), (("contrastive_dense", c_main), ("posonly_dose_dense", c_alt)), strict=True
    ):
        reg = [m608["per_source"][s][arm]["registered_21"] for s in sources]
        ax.scatter(
            x + off,
            [r["interpolated_bys_mean"] for r in reg],
            color=color,
            marker="D",
            s=40,
            label=f"{ARM_LABEL[arm]} (interpolated)",
            zorder=3,
        )
        for xi, r in zip(x + off, reg, strict=True):
            ax.plot(
                [xi, xi],
                [r["lower_endpoint_bys_mean"], r["upper_endpoint_bys_mean"]],
                color=color,
                lw=1.0,
                alpha=0.7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in sources], rotation=20, ha="right")
    ax.set_ylabel("bystander-mean agreement delta")
    set_title_subtitle(
        ax,
        "Interpolated read sits inside its measured endpoint sandwich",
        "vertical bars span the two bracket checkpoints; diamonds = read at install 0.50",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "explore_measured_vs_interpolated", dir=FIG_DIR)
    plt.close(fig)


def explore_606_endpoint_spread(f606: dict) -> None:
    trio = f606["refusal_endpoint_trio"]
    rows = []
    for behavior, arms in trio.items():
        for arm, rec in arms.items():
            label = {"lora": "LoRA", "ft": "full fine-tune", "base": "base"}.get(arm, arm)
            rows.append((f"{behavior}: {label}", rec["install_s"], rec["leak"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, s, leak in rows:
        ax.scatter(s, leak, s=50)
        ax.annotate(label, (s, leak), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("source install (rate delta)")
    ax.set_ylabel("bystander-mean rate delta")
    set_title_subtitle(
        ax,
        "Refusal endpoints: equal install, different leakage",
        "install alone does not determine leakage (descriptive)",
    )
    fig.tight_layout()
    savefig_paper(fig, "explore_606_endpoint_spread", dir=FIG_DIR)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #627 — figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--matched-608", type=Path, default=ANALYSIS_DIR / "matched_install_608.json"
    )
    parser.add_argument("--synthesis", type=Path, default=ANALYSIS_DIR / "synthesis.json")
    parser.add_argument(
        "--cells-manifest",
        type=Path,
        default=Path("eval_results/issue_627/matched_install_cells.json"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    set_paper_style("blog")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    synthesis = _load(args.synthesis)
    m608 = _load(args.matched_608)
    m601 = _load(ANALYSIS_DIR / "marker_fractions_601.json")
    f606 = _load(ANALYSIS_DIR / "fractions_606.json")
    cells_manifest = _load(args.cells_manifest)

    hero_forest(synthesis)
    hero_dose_curves(m608, m601, f606)
    explore_608_trajectories(m608, cells_manifest)
    explore_fraction_heatmap(m601)
    explore_measured_vs_interpolated(m608)
    explore_606_endpoint_spread(f606)
    log.info("[phase=p3_figures] figures -> %s", FIG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
