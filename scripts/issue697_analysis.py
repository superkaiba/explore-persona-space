"""Issue #697 — CPU analysis: f_CV bootstrap + hero figure (off-pod, 0 GPU).

Consumes the per-cell ``eval_results/issue_697/patch/*.pt`` tensors (each carries
per (persona, question) reads for every condition at every layer; the analysis
runs against the LOCAL copies on the VM after the pod uploads + terminates) and
computes, per behavior, the context-vector-mediated fraction in v-space:

  f_CV       = ((v_Pup - v0)·d) / ((v⁺ - v0)·d),   d = (v⁺ - v0)/‖·‖   (P↑ sufficiency)
  f_CV_down  = 1 - ((v_Pdown - v0)·d)/((v⁺-v0)·d)                       (P↓ necessity)

at each cell's per-behavior PRIMARY pooling (mean-resp em/syc, slot marker/fact —
plan §4.5 item-5), with the random-CV / other-context conditions as the "patch
did something" null floor. Bootstrap 95% CI is over the 280 personaxquestion
pairs PERSONA-CLUSTERED (resample personas, then questions within — the
Statistics-critic standing rec) so the CI respects the panel's two-level
structure. A cell with ‖v⁺-v0‖ < eps is reported ``no-effect`` (never an extreme
ratio). The hero is a 2x4 grid (rows: f_CV / f_CV^E; cols: em/syc/marker/fact).

The behavioral-E row (f_CV^E) is rendered ONLY when judged E rates are present
(``*_E_scored.json`` — produced by the off-pod judge phase over the captured
generations; the #537 judge pools are not yet vendored, so a sweep without that
phase renders the v-space row + a labeled "E not yet judged" panel).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from explore_persona_space.analysis.cv_patch import NO_EFFECT, compute_f_cv, compute_f_cv_down

logger = logging.getLogger("issue697_analysis")

BEHAVIORS = ("em", "sycophancy", "marker", "fact")
PRIMARY_POOLING = {"em": "mean_resp", "sycophancy": "mean_resp", "marker": "slot", "fact": "slot"}
BAND_LOW, BAND_HIGH = 0.3, 0.7
N_BOOTSTRAP = 1000


def _per_question_f_cv(cell: dict, layer: int) -> dict:
    """Per (persona, question) f_CV / f_CV_down / null-floor at the cell's primary pooling.

    Returns ``{"f_cv": [...], "f_cv_down": [...], "f_cv_random": [...],
    "personas": [...], "n_no_effect": int}`` — parallel lists over kept pairs.
    """
    behavior = cell["behavior"]
    pooling = PRIMARY_POOLING.get(behavior, "mean_resp")
    f_cv, f_cv_down, f_cv_random, personas = [], [], [], []
    n_no_effect = 0
    for p_name, entries in cell["per_q"].items():
        for e in entries:
            v0 = e["v0"][layer][pooling]
            vplus = e["vplus"][layer][pooling]
            v_pup = e["conditions"]["p_up"][layer][pooling]
            v_pdown = e["conditions"]["p_down"][layer][pooling]
            v_rand = e["conditions"]["random_cv"][layer][pooling]
            f = compute_f_cv(v_pup, v0, vplus)
            fd = compute_f_cv_down(v_pdown, v0, vplus)
            fr = compute_f_cv(v_rand, v0, vplus)
            if f == NO_EFFECT:
                n_no_effect += 1
                continue
            f_cv.append(float(f))
            f_cv_down.append(float(fd) if fd != NO_EFFECT else np.nan)
            f_cv_random.append(float(fr) if fr != NO_EFFECT else np.nan)
            personas.append(p_name)
    return {
        "f_cv": f_cv,
        "f_cv_down": f_cv_down,
        "f_cv_random": f_cv_random,
        "personas": personas,
        "n_no_effect": n_no_effect,
    }


def _persona_clustered_bootstrap(values: list[float], personas: list[str], n_reps: int) -> dict:
    """Persona-clustered bootstrap 95% CI of the mean (resample personas, then
    questions within each — respects the panel's two-level structure)."""
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    vals = np.asarray(values, dtype=np.float64)
    pers = np.asarray(personas)
    uniq = np.unique(pers)
    by_persona = {p: vals[pers == p] for p in uniq}
    rng = np.random.default_rng(697)
    means = np.empty(n_reps)
    for r in range(n_reps):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        pooled = []
        for p in chosen:
            arr = by_persona[p]
            if len(arr):
                pooled.append(rng.choice(arr, size=len(arr), replace=True))
        cat = np.concatenate(pooled) if pooled else vals
        means[r] = np.nanmean(cat)
    return {
        "mean": float(np.nanmean(vals)),
        "ci_low": float(np.nanpercentile(means, 2.5)),
        "ci_high": float(np.nanpercentile(means, 97.5)),
        "n": len(vals),
    }


def _verdict(ci: dict) -> str:
    """Pre-registered band verdict (plan §6.3): assigned only when the CI lies
    entirely within one band; else 'mixed'."""
    lo, hi = ci["ci_low"], ci["ci_high"]
    if np.isnan(lo) or np.isnan(hi):
        return "no-effect"
    if lo >= BAND_HIGH:
        return "context-vector-moved"
    if hi <= BAND_LOW:
        return "mapping-changed"
    return "mixed"


def analyze(repo_root: Path, *, primary_layer: int) -> dict:
    patch_dir = repo_root / "eval_results" / "issue_697" / "patch"
    pts = sorted(patch_dir.glob("*.pt"))
    if not pts:
        raise RuntimeError(f"no per-cell .pt tensors in {patch_dir} -- run the sweep first")
    logger.info("[phase=analyze] %d per-cell tensors in %s", len(pts), patch_dir)

    by_behavior: dict[str, dict] = {
        b: {"f_cv": [], "personas": [], "f_cv_random": [], "cells": []} for b in BEHAVIORS
    }
    for pt in pts:
        cell = torch.load(pt, weights_only=False)
        behavior = cell["behavior"]
        if behavior not in by_behavior:
            continue
        layer = primary_layer if primary_layer in cell["layers"] else cell["primary_layer"]
        pq = _per_question_f_cv(cell, layer)
        by_behavior[behavior]["f_cv"] += pq["f_cv"]
        by_behavior[behavior]["f_cv_random"] += pq["f_cv_random"]
        by_behavior[behavior]["personas"] += pq["personas"]
        by_behavior[behavior]["cells"].append(cell["cell_id"])
        logger.info(
            "  cell %s: %d pairs (%d no-effect) at layer %d pooling %s",
            cell["cell_id"],
            len(pq["f_cv"]),
            pq["n_no_effect"],
            layer,
            PRIMARY_POOLING.get(behavior),
        )

    summary: dict[str, dict] = {}
    for behavior in BEHAVIORS:
        d = by_behavior[behavior]
        ci = _persona_clustered_bootstrap(d["f_cv"], d["personas"], N_BOOTSTRAP)
        ci_null = _persona_clustered_bootstrap(
            [v for v in d["f_cv_random"] if not np.isnan(v)],
            [p for p, v in zip(d["personas"], d["f_cv_random"], strict=True) if not np.isnan(v)],
            N_BOOTSTRAP,
        )
        summary[behavior] = {
            "f_cv_ci": ci,
            "null_floor_ci": ci_null,
            "verdict": _verdict(ci),
            "n_cells": len(d["cells"]),
            "primary_pooling": PRIMARY_POOLING.get(behavior),
        }
        logger.info(
            "  %s: f_CV=%.3f [%.3f, %.3f] verdict=%s (null floor %.3f)",
            behavior,
            ci["mean"],
            ci["ci_low"],
            ci["ci_high"],
            summary[behavior]["verdict"],
            ci_null["mean"],
        )
    return {"primary_layer": primary_layer, "by_behavior": summary, "raw": by_behavior}


def render_hero(result: dict, out_path: Path) -> None:
    """The hero 2x4 grid (rows: f_CV / f_CV^E; cols: em/syc/marker/fact)."""
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    import matplotlib.pyplot as plt

    # paper_palette_role(role) returns a single hex string per role.
    c_primary = paper_palette_role("primary")
    c_neutral = paper_palette_role("neutral")
    c_control = paper_palette_role("control")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey="row")
    for ci_col, behavior in enumerate(BEHAVIORS):
        s = result["by_behavior"][behavior]
        ci = s["f_cv_ci"]
        ax = axes[0, ci_col]
        if not np.isnan(ci["mean"]):
            ax.errorbar(
                [0],
                [ci["mean"]],
                yerr=[[ci["mean"] - ci["ci_low"]], [ci["ci_high"] - ci["mean"]]],
                fmt="o",
                color=c_primary,
                capsize=4,
            )
        null = s["null_floor_ci"]
        if not np.isnan(null["mean"]):
            ax.axhspan(null["ci_low"], null["ci_high"], color=c_neutral, alpha=0.25)
        ax.axhline(BAND_LOW, ls="--", lw=0.8, color=c_control)
        ax.axhline(BAND_HIGH, ls="--", lw=0.8, color=c_control)
        ax.set_title(f"{behavior}\n({s['verdict']})", fontsize=9)
        ax.set_xticks([])
        if ci_col == 0:
            ax.set_ylabel("f_CV (v-space)")
        ax.set_ylim(-0.2, 1.2)
        # E-space row: rendered only when judged rates are present.
        axe = axes[1, ci_col]
        axe.text(
            0.5,
            0.5,
            "E not yet judged\n(judge phase off-pod)",
            ha="center",
            va="center",
            fontsize=8,
            transform=axe.transAxes,
        )
        axe.set_xticks([])
        axe.set_yticks([])
        if ci_col == 0:
            axe.set_ylabel("f_CV^E (E-space)")
    fig.suptitle("Issue #697 — context-vector-mediated fraction f_CV per behavior", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, out_path)
    plt.close(fig)
    logger.info("wrote hero figure %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--primary-layer", type=int, default=14)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )
    import subprocess

    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
    )
    result = analyze(repo_root, primary_layer=args.primary_layer)

    out_dir = repo_root / "eval_results" / "issue_697"
    out_dir.mkdir(parents=True, exist_ok=True)
    # strip the raw per-pair lists out of the JSON summary (keep it small).
    summary_json = {"primary_layer": result["primary_layer"], "by_behavior": result["by_behavior"]}
    (out_dir / "f_cv_summary.json").write_text(json.dumps(summary_json, indent=2, default=float))
    logger.info("wrote %s", out_dir / "f_cv_summary.json")

    fig_path = repo_root / "figures" / "issue_697" / "hero_f_cv_2x4.png"
    render_hero(result, fig_path)
    logger.info("[phase=analyze_done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
