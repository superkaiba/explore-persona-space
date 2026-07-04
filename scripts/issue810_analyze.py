#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², r_B) in scientific docstrings, log messages,
# and reader-facing figure labels (paper-plots §3.5 plain-English label rule).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy freezes the BLAS pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue810_common import dump_json, load_json, reproducibility_metadata  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue810_analyze")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── plain-English figure labels (paper-plots §3.5: no raw code ids on figures) ─

_SUMMARY_LABELS = {
    "mean": "mean over answer tokens",
    "last": "last answer token",
    "maxp": "max-pool over answer tokens",
    "im_end": "turn-end token",
    "turn_nl": "newline after turn end",
}

_METHOD_LABELS = {
    "fixed_rb": "fixed behavior-direction read-out",
    "trained_ridge": "trained leave-one-out ridge read-out",
}

_BEHAVIOR_LABELS = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "harmful_compliance": "harmful compliance",
}


def _ordinal(k: int) -> str:
    """1 -> 1st, 2 -> 2nd, 3 -> 3rd, 4 -> 4th, ... (plain-English tick labels)."""
    if 10 <= k % 100 <= 20:
        return f"{k}th"
    return f"{k}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th') }"


def _plain_summary(s: str) -> str:
    """Reader-facing label for a summary/position recipe id."""
    if s in _SUMMARY_LABELS:
        return _SUMMARY_LABELS[s]
    if s.startswith("tail_"):
        return f"{_ordinal(int(s.split('_')[1]))} from end"
    if s.startswith("head_"):
        return f"{_ordinal(int(s.split('_')[1]) + 1)} from start"
    return s


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
    # Per-method bands (same selection-symmetric max, restricted to one method's
    # cells) — the reference values hero3's panels are annotated with.
    per_method: dict[str, list[list[float]]] = {}
    for b in null_readout.values():
        for meth, m in b.items():
            for s in m.values():
                for draws in s.values():
                    if draws:
                        per_method.setdefault(meth, []).append(draws)
    per_method_bands = {}
    for meth, mcells in per_method.items():
        n_p = min(len(c) for c in mcells)
        marr = np.array([c[:n_p] for c in mcells])
        per_method_bands[meth] = {
            "abs": float(np.percentile(np.abs(marr).max(axis=0), pct)),
            "signed": float(np.percentile(marr.max(axis=0), pct)),
        }
    return {
        "available": True,
        "n_cells": arr.shape[0],
        "n_perms": n_perms,
        "band_pctile": pct,
        "honest_band_upper_rho": float(np.percentile(max_per_draw, pct)),
        "honest_band_upper_abs_rho": float(np.percentile(max_abs_per_draw, pct)),
        "per_method_bands": per_method_bands,
    }


# ── figures ───────────────────────────────────────────────────────────────────


def _hero1_reconstruction(recon: dict, band: dict, fig_dir: Path) -> None:
    """Per-layer skill-over-mean R² curves per summary; mean + maxp bolded; null line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    by = recon["by_summary"]
    # A readable subset for the hero (the rest live in the JSON): the two
    # load-bearing aggregates, the two boundary tokens, the last token, and the
    # BEST deeper tail / head per the plan's figure spec (tail_2 / head_3).
    hero_summaries = [
        s for s in ("mean", "maxp", "turn_nl", "im_end", "last", "tail_2", "head_3") if s in by
    ]
    for s in hero_summaries:
        ys = [c.get("ridge_skill") for c in by[s]]
        xs = [c.get("layer") for c in by[s]]
        pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if y is not None]
        if not pairs:
            continue
        # The two load-bearing curves get distinct semantic colors + bold width;
        # the context curves get fixed non-primary hues so no thin line shares a
        # hue with a bold one (interp-critique r1 figure issue).
        thin_colors = {
            "turn_nl": paper_palette_role("control"),  # green
            "im_end": paper_palette_role("accent"),  # red
            "last": "#8064A2",  # purple
            "tail_2": "#5A6975",  # slate (solid ≠ dashed null line)
            "head_3": "#000000",  # black
        }
        if s == "mean":
            color, lw = paper_palette_role("baseline"), 2.5
        elif s == "maxp":
            color, lw = paper_palette_role("primary"), 2.5
        else:
            color, lw = thin_colors.get(s), 1.3
        ax.plot(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            label=_plain_summary(s),
            linewidth=lw,
            color=color,
        )
    if band.get("available"):
        ax.axhline(
            band["honest_band_upper"],
            color=paper_palette_role("neutral"),
            linestyle="--",
            linewidth=1,
            label="selection-symmetric null (97.5th pct)",
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out skill-over-mean R² (higher = better)")
    ax.set_title("Predicting each answer summary from the context representation")
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
    ax.set_yticklabels([_plain_summary(p) for p in positions], fontsize=5)
    ax.set_ylabel("answer position (single token)")
    ax.set_xlabel("layer")
    ax.set_title("Reconstruction skill per answer position × layer")
    fig.colorbar(im, ax=ax, label="skill-over-mean R²")
    savefig_paper(fig, "hero2_position_heatmap", dir=str(fig_dir))
    plt.close(fig)


def _hero3_readout_heatmap(readout: dict, fig_dir: Path, band: dict | None = None) -> None:
    """Signed ρ at the max-|ρ| layer (behavior × summary), faceted by method.

    Renders one heatmap panel per method with behaviors on rows. Each cell shows
    the SIGNED ρ at that (behavior, summary)'s max-|ρ| layer — a max-SIGNED
    selection would hide the fixed-direction anti-correlations (e.g. the
    harmful-compliance / refusal negative cells), which are the pattern the
    analysis discusses. Panel titles + colorbar lines carry the per-method
    selection-symmetric null band so the trained-ridge panel is not over-read
    (its band is near |ρ| ≈ 1). Full per-layer detail lives in
    readout_rho_by_summary.json.
    """
    cells = readout["cells"]
    methods = sorted({c["method"] for c in cells})
    behaviors = sorted({c["behavior"] for c in cells})
    summaries = sorted({c["summary"] for c in cells})
    if not methods:
        return
    per_method_bands = (band or {}).get("per_method_bands", {})
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4), squeeze=False)
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
                    # signed value at the max-|rho| layer (NOT max signed)
                    mat[bi, si] = max(vals, key=abs)
        ax = axes[0][mi]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(summaries)))
        ax.set_xticklabels([_plain_summary(s) for s in summaries], rotation=90, fontsize=5)
        ax.set_yticks(range(len(behaviors)))
        ax.set_yticklabels([_BEHAVIOR_LABELS.get(b, b) for b in behaviors], fontsize=6)
        mband = per_method_bands.get(method, {}).get("abs")
        title = f"{_METHOD_LABELS.get(method, method)}"
        if mband is not None:
            title += f"\nnull band (97.5th pct of max): |ρ| ≤ {mband:.3f}"
        ax.set_title(title, fontsize=8)
        cb = fig.colorbar(im, ax=ax, label="ρ at max-|ρ| layer")
        if mband is not None:
            for y in (mband, -mband):
                cb.ax.axhline(y, color="black", linestyle="--", linewidth=0.8)
    # NOTE: no fig.tight_layout() — savefig_paper owns the layout engine
    # (bbox_inches='tight'); calling tight_layout here conflicts with a
    # per-axes colorbar (matplotlib "layout engine not compatible" RuntimeError).
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
    ax.set_xticklabels([_plain_summary(s) for s in summaries], rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([_plain_summary(s) for s in summaries], fontsize=5)
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
    ax.set_ylabel("cosine(newline-after-turn-end, context representation)")
    ax.set_title(
        "Boundary-triviality: turn-end newline vs context vector (mean ± std over contexts)"
    )
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
        readout_band = _honest_readout_band(null_readout)
        summary["readout_honest_band"] = readout_band
        summary["h2_conjunction"] = readout.get("h2_conjunction")
        summary["judge_validation"] = readout.get("judge_validation")
        summary["length_control"] = readout.get("length_control")
        _hero3_readout_heatmap(readout, fig_dir, band=readout_band)
        logger.info("[phase=readout] honest band + hero3 done")
    else:
        logger.warning("read-out inputs not present in %s — skipping DV (b) figures", in_dir)

    dump_json(summary, out_dir / "analysis_summary.json")
    logger.info("[phase=done] wrote analysis_summary.json + figures to %s / %s", out_dir, fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
