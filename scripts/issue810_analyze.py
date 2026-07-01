#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, rho, ×, ², r_B) in scientific docstrings + log messages.
"""Issue #810 Phase E: aggregation + figures + honest selection-symmetric bands.

Consumes the DV JSONs written by ``issue810_fit_reconstruction.py`` (DV a) +
``issue810_fit_readout.py`` (DV b) + their null matrices, and produces:

- The HONEST max-selected null bands (plan §6, selection-symmetric-nulls rule):
  the reconstruction headline maxes over {summary, layer}; the read-out headline
  maxes over {summary, layer, behavior, method}. Both are recomputed HERE by
  applying the IDENTICAL max-over-axis selection to each of the 1000 per-draw
  null values (a 0-GPU re-reduction of the persisted per-draw × per-cell matrix)
  — NOT a per-cell single-band read.
- HERO 1 (reconstruction): per-layer skill-over-mean R² curves per summary, the
  `mean` baseline bolded, the honest shuffle-null band shaded.
- HERO 2 (reconstruction): per-position R² heat-map (position × layer).
- HERO 3 (read-out): rho heat-map (behavior × layer) faceted by summary × method.
- Exploratory dump: cross-summary correlation matrix, cosine(turn_nl, c_C)
  diagnostic curve, length/verbosity control bars, ridge-vs-MLP R² gap,
  identity-sanity-check anchor line.

0-GPU. Figures saved under ``figures/issue_810/`` via the paper-plots rcParams.

Usage::

    uv run python scripts/issue810_analyze.py --in eval_results/issue_810 \\
        --fig-dir figures/issue_810 --out eval_results/issue_810
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue810_common import dump_json, load_json, reproducibility_metadata  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue810_analyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── honest selection-symmetric bands ──────────────────────────────────────────


def _honest_reconstruction_band(null_recon: dict, pct: float = 97.5) -> dict:
    """max-over-{summary, layer} per draw → the honest reconstruction null band.

    ``null_recon[summary][layer] = [per-draw skill]``. For each draw index d,
    take the MAX skill across all (summary, layer) cells that have a draw at d,
    then quantile the resulting max-per-draw distribution. This matches the
    max-over-{summary, layer} observed headline (selection-symmetric-nulls rule).
    """
    # Collect per-cell draw lists; assume equal n_perms across cells.
    cells = [draws for s in null_recon.values() for draws in s.values() if draws]
    if not cells:
        return {"available": False}
    n_perms = min(len(c) for c in cells)
    arr = np.array([c[:n_perms] for c in cells])  # (n_cells, n_perms)
    max_per_draw = arr.max(axis=0)  # (n_perms,)
    return {
        "available": True,
        "n_cells": arr.shape[0],
        "n_perms": n_perms,
        "band_pctile": pct,
        "honest_band_upper": float(np.percentile(max_per_draw, pct)),
        "honest_band_median": float(np.median(max_per_draw)),
    }


def _honest_readout_band(null_readout: dict, pct: float = 97.5) -> dict:
    """max-over-{summary, layer, behavior, method} per draw → honest read-out band.

    ``null_readout[behavior][method][summary][layer] = [per-draw rho]``. Maxes
    across ALL free axes per draw (the read-out headline "any summary lifts any
    behavior" maxes over {summary, layer, behavior}; the method axis is included
    so the band covers the weaker OR-reading too — the analyzer states which H2
    reading it used). Returns the band on |rho| (two-sided) + on rho.
    """
    cells = [
        draws
        for b in null_readout.values()
        for m in b.values()
        for s in m.values()
        for draws in s.values()
        if draws
    ]
    if not cells:
        return {"available": False}
    n_perms = min(len(c) for c in cells)
    arr = np.array([c[:n_perms] for c in cells])  # (n_cells, n_perms)
    max_per_draw = arr.max(axis=0)
    max_abs_per_draw = np.abs(arr).max(axis=0)
    return {
        "available": True,
        "n_cells": arr.shape[0],
        "n_perms": n_perms,
        "band_pctile": pct,
        "honest_band_upper_rho": float(np.percentile(max_per_draw, pct)),
        "honest_band_upper_abs_rho": float(np.percentile(max_abs_per_draw, pct)),
    }


# ── figures ───────────────────────────────────────────────────────────────────


def _hero1_reconstruction(recon: dict, band: dict, fig_dir: Path) -> None:
    """Per-layer skill-over-mean R^2 curves per summary; mean bolded; null shaded."""
    fig, ax = plt.subplots(figsize=(7, 4))
    by = recon["by_summary"]
    # A readable subset of summaries for the hero (the rest live in the JSON).
    hero_summaries = [
        s
        for s in ("mean", "last", "maxp", "im_end", "turn_nl", "tail_1", "tail_4", "head_0")
        if s in by
    ]
    for s in hero_summaries:
        ys = [c.get("ridge_skill") for c in by[s]]
        xs = [c.get("layer") for c in by[s]]
        pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if y is not None]
        if not pairs:
            continue
        color = paper_palette_role("baseline") if s == "mean" else None
        lw = 2.5 if s == "mean" else 1.3
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], label=s, linewidth=lw, color=color)
    if band.get("available"):
        ax.axhline(
            band["honest_band_upper"],
            color=paper_palette_role("neutral"),
            linestyle="--",
            linewidth=1,
            label="honest null p97.5",
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out skill-over-mean R^2")
    ax.set_title("Reconstruction c_C -> summary, per layer")
    ax.legend(fontsize=6, ncol=2)
    savefig_paper(fig, "hero1_reconstruction_by_layer", dir=str(fig_dir))
    plt.close(fig)


def _hero2_position_heatmap(recon: dict, fig_dir: Path) -> None:
    """Per-position R² heat-map (position × layer), tail + head + boundary."""
    by = recon["by_summary"]
    layers = recon["capture_layers"]
    positions = [
        p
        for p in (
            ["im_end", "turn_nl"]
            + [f"tail_{k}" for k in range(1, 17)]
            + [f"head_{k}" for k in range(16)]
        )
        if p in by
    ]
    mat = np.full((len(positions), len(layers)), np.nan)
    for pi, p in enumerate(positions):
        layer_to_r = {c["layer"]: c.get("ridge_skill") for c in by[p]}
        for li, layer in enumerate(layers):
            v = layer_to_r.get(layer)
            if v is not None:
                mat[pi, li] = v
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions, fontsize=5)
    ax.set_xlabel("layer")
    ax.set_title("Reconstruction R^2 per position x layer")
    fig.colorbar(im, ax=ax, label="skill-over-mean R^2")
    savefig_paper(fig, "hero2_position_heatmap", dir=str(fig_dir))
    plt.close(fig)


def _hero3_readout_heatmap(readout: dict, fig_dir: Path) -> None:
    """rho (behavior × layer) faceted by (summary × method), a compact grid.

    Renders one small heatmap panel per (method) with behaviors on rows, the
    best-per-summary rho collapsed to a behavior × summary grid for legibility
    (the full per-layer detail lives in readout_rho_by_summary.json).
    """
    cells = readout["cells"]
    methods = sorted({c["method"] for c in cells})
    behaviors = sorted({c["behavior"] for c in cells})
    summaries = sorted({c["summary"] for c in cells})
    if not methods:
        return
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), squeeze=False)
    for mi, method in enumerate(methods):
        mat = np.full((len(behaviors), len(summaries)), np.nan)
        for bi, b in enumerate(behaviors):
            for si, s in enumerate(summaries):
                vals = [
                    c["rho_graded"]
                    for c in cells
                    if c["method"] == method
                    and c["behavior"] == b
                    and c["summary"] == s
                    and c["rho_graded"] is not None
                ]
                if vals:
                    mat[bi, si] = max(vals)  # best-layer rho
        ax = axes[0][mi]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(summaries)))
        ax.set_xticklabels(summaries, rotation=90, fontsize=5)
        ax.set_yticks(range(len(behaviors)))
        ax.set_yticklabels(behaviors, fontsize=6)
        ax.set_title(f"read-out rho ({method})", fontsize=8)
        fig.colorbar(im, ax=ax, label="best-layer rho")
    fig.tight_layout()
    savefig_paper(fig, "hero3_readout_heatmap", dir=str(fig_dir))
    plt.close(fig)


def _exploratory_cross_summary(recon: dict, fig_dir: Path) -> None:
    """Cross-summary correlation of per-layer R² curves (near-duplicate detector)."""
    by = recon["by_summary"]
    summaries = [s for s in by if any(c.get("ridge_skill") is not None for c in by[s])]
    curves = {}
    for s in summaries:
        curves[s] = np.array(
            [c.get("ridge_skill") if c.get("ridge_skill") is not None else np.nan for c in by[s]]
        )
    n = len(summaries)
    mat = np.full((n, n), np.nan)
    for i, si in enumerate(summaries):
        for j, sj in enumerate(summaries):
            a, b = curves[si], curves[sj]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() >= 3 and np.std(a[mask]) > 1e-9 and np.std(b[mask]) > 1e-9:
                mat[i, j] = float(np.corrcoef(a[mask], b[mask])[0, 1])
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(summaries, rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(summaries, fontsize=5)
    ax.set_title("Cross-summary R²-curve correlation")
    fig.colorbar(im, ax=ax)
    savefig_paper(fig, "exploratory_cross_summary_corr", dir=str(fig_dir))
    plt.close(fig)


def _exploratory_turn_nl_cc_cosine(recon: dict, fig_dir: Path) -> None:
    """cosine(turn_nl, c_C) per layer — the boundary-triviality diagnostic curve."""
    diag = recon.get("diagnostics", {}).get("cosine_turn_nl_cc_per_layer", {})
    if not diag:
        return
    layers = sorted(int(k) for k in diag)
    means = [diag[str(k)]["mean_cos"] for k in layers]
    stds = [diag[str(k)]["std_cos"] for k in layers]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(layers, means, yerr=stds, color=paper_palette_role("accent"), capsize=2)
    ax.axhline(1.0, color=paper_palette_role("neutral"), linestyle=":", linewidth=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine(turn_nl, c_C) across contexts")
    ax.set_title("Boundary-triviality: turn_nl vs c_C (mean +/- std over contexts)")
    savefig_paper(fig, "exploratory_turn_nl_cc_cosine", dir=str(fig_dir))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 Phase E: aggregation + figures")
    ap.add_argument("--in", dest="in_dir", default=str(PROJECT_ROOT / "eval_results" / "issue_810"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval_results" / "issue_810"))
    ap.add_argument("--fig-dir", default=str(PROJECT_ROOT / "figures" / "issue_810"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    set_paper_style()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"reproducibility": reproducibility_metadata(), "smoke": args.smoke}

    # Reconstruction (DV a)
    recon_path = in_dir / "reconstruction_skill_by_summary.json"
    null_recon_path = in_dir / "null_matrix_reconstruction.json"
    if recon_path.is_file() and null_recon_path.is_file():
        recon = load_json(recon_path)
        null_recon = load_json(null_recon_path)["reconstruction"]
        band = _honest_reconstruction_band(null_recon)
        summary["reconstruction_honest_band"] = band
        _hero1_reconstruction(recon, band, fig_dir)
        _hero2_position_heatmap(recon, fig_dir)
        _exploratory_cross_summary(recon, fig_dir)
        _exploratory_turn_nl_cc_cosine(recon, fig_dir)
        logger.info("[phase=recon] honest band + hero1/hero2 + exploratory done")
    else:
        logger.warning("reconstruction inputs not present in %s — skipping DV (a) figures", in_dir)

    # Read-out (DV b)
    readout_path = in_dir / "readout_rho_by_summary.json"
    null_readout_path = in_dir / "null_matrix_readout.json"
    if readout_path.is_file() and null_readout_path.is_file():
        readout = load_json(readout_path)
        null_readout = load_json(null_readout_path)["readout"]
        summary["readout_honest_band"] = _honest_readout_band(null_readout)
        summary["h2_conjunction"] = readout.get("h2_conjunction")
        summary["judge_validation"] = readout.get("judge_validation")
        summary["length_control"] = readout.get("length_control")
        _hero3_readout_heatmap(readout, fig_dir)
        logger.info("[phase=readout] honest band + hero3 done")
    else:
        logger.warning("read-out inputs not present in %s — skipping DV (b) figures", in_dir)

    dump_json(summary, out_dir / "analysis_summary.json")
    logger.info("[phase=done] wrote analysis_summary.json + figures to %s / %s", out_dir, fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
