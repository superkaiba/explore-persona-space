#!/usr/bin/env python3
"""#509 follow-up: base-rate partial on the EARLY-LAYER geometry cells.

The existing ``scripts/issue509_baserate_covariate.py`` partialled the
bystander's intrinsic base sycophancy rate against the OLD #470 layer-20
cosine predictor. This follow-up swaps in the #509 bake-off early-layer
cells (the new finding) as the geometry covariate:

  - representative cell: end_of_system x L02 x cosine x centered
  - global-max cell:     last_prompt   x L07 x mmd    x centered
  - robustness cluster:  cosine x centered, layers 0-13, BOTH
    extraction points (28 cells)

For each cell x target (delta primary, trained_abs secondary) x panel
(all-138, live-21) it computes the source-FE Spearman for geometry and
base rate alone, the two partials (geom | base, base | geom), and the
within-R^2 decomposition (unique_r2_geom / unique_r2_base). The TWO
headline cells additionally get a source-clustered bootstrap CI on
``partial_geom_given_base`` (5000 reps, seed 42) and a within-source
permutation p (2000 reps, hashed seed) following the production
``issue509_scoring.py`` inference conventions. Point-estimate helpers
(``fe_spearman``, ``fe_partial_spearman``, ``within_r2``) are copied
verbatim from ``issue509_baserate_covariate.py`` so the new numbers are
directly comparable to ``eval_results/issue_509/baserate_covariate/
results.json``.

Analysis-only: inputs are the frozen #411 leakage snapshot (repo file)
and the #509 bake-off distance matrices on the HF data repo at a pinned
revision. No training, no generation, no model forward passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata as _scipy_rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MAIN_CHECKOUT = Path("/home/thomasjiralerspong/explore-persona-space")
DEFAULT_TARGET_FILE = MAIN_CHECKOUT / "eval_results/issue_480/_inputs/syco_411_analyze_summary.json"
DEFAULT_HF_REPO = "superkaiba1/explore-persona-space-data"
DEFAULT_HF_REVISION = "1b6e20530b1c6d477a387c18d5a88554910e7df9"
DEFAULT_METRICS_PREFIX = "issue_509/syco_arm/bakeoff/metrics"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "eval_results/issue_509/baserate-partial-early-layer-cosine/results.json"
)
DEFAULT_FIG_DIR = PROJECT_ROOT / "figures/issue_509"

EARLY_LAYERS = tuple(range(14))  # L0-L13 early-cosine cluster
EXTRACTION_POINTS = ("end_of_system", "last_prompt")
HEADLINE_CELLS = (
    ("end_of_system", 2, "cosine", "centered"),  # representative early cell
    ("last_prompt", 7, "mmd", "centered"),  # global-max cell
)
LIVE_DELTA_THRESHOLD = 0.10

# Inference conventions from issue509_scoring.py (production syco arm).
BOOTSTRAP_SEED = 42
PERM_SEED_TAG = b"issue509_baserate_earlylayer_perm_v1"

# Per-pair saturation exclusion from issue509_scoring.py ROUND-3 G2.
PAIR_SATURATION_ABS_THRESHOLD = 1e-6
MIN_SURVIVING_PAIRS = 5


def _hashed_seed(tag: bytes) -> int:
    """Deterministic integer seed from a string tag (issue509_scoring convention)."""
    return int(hashlib.sha256(tag).hexdigest()[:8], 16)


def _git_sha(repo: Path) -> str:
    """Full git SHA of ``repo`` HEAD, or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


# ── Stats helpers — copied verbatim from issue509_baserate_covariate.py ──
# (rank globally, demean ranks within source, Pearson) so the outputs are
# directly comparable to eval_results/issue_509/baserate_covariate/results.json.


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (0-based mean ranks), matching the parent script."""
    return _scipy_rankdata(np.asarray(x, dtype=float)) - 1.0


def within(vec: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract per-group mean (source fixed effect)."""
    vec = np.asarray(vec, float)
    out = vec.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = vec[m] - vec[m].mean()
    return out


def fe_spearman(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Source-FE Spearman: corr of within-source-demeaned global ranks."""
    rx = within(rankdata(x), groups)
    ry = within(rankdata(y), groups)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta


def fe_partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray, groups: np.ndarray) -> float:
    """corr(x, y | z) on within-source-demeaned ranks."""
    rx = within(rankdata(x), groups)
    ry = within(rankdata(y), groups)
    rz = within(rankdata(z), groups)
    Z = np.vstack([np.ones_like(rz), rz]).T
    ex, ey = _resid(rx, Z), _resid(ry, Z)
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def within_r2(y: np.ndarray, preds: list[np.ndarray], groups: np.ndarray) -> float:
    """Within-source R^2 of demeaned-rank y explained by demeaned-rank preds."""
    ry = within(rankdata(y), groups)
    cols = [within(rankdata(p), groups) for p in preds]
    X = np.vstack([np.ones_like(ry), *cols]).T
    res = _resid(ry, X)
    ss_tot = (ry**2).sum()
    return float(1 - (res**2).sum() / ss_tot) if ss_tot > 0 else float("nan")


# ── Inference (headline cells only) — issue509_scoring.py conventions ──


def cluster_bootstrap_partial_ci(
    geom: np.ndarray,
    tgt: np.ndarray,
    base: np.ndarray,
    groups: np.ndarray,
    b: int,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Source-clustered bootstrap 95% CI on ``partial_geom_given_base``.

    Resamples whole sources with replacement (fresh rng per call, fixed
    seed — issue509_scoring convention) and recomputes the SAME partial
    statistic on each draw. Returns (NaN, NaN) when fewer than 100 draws
    yield a finite statistic.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    vals: list[float] = []
    for _ in range(b):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in drawn])
        gb = np.concatenate([np.full((groups == g).sum(), i) for i, g in enumerate(drawn)])
        v = fe_partial_spearman(geom[idx], tgt[idx], base[idx], gb)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 100:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def perm_p_partial(
    geom: np.ndarray,
    tgt: np.ndarray,
    base: np.ndarray,
    groups: np.ndarray,
    obs: float,
    b: int,
) -> float:
    """Within-source permutation p for ``partial_geom_given_base``.

    Shuffles geometry within source on every draw and recomputes the SAME
    partial statistic (observed and null are the same function — the
    issue509_scoring ROUND-2 F3 principle). Two-sided on |rho|.
    """
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(_hashed_seed(PERM_SEED_TAG))
    n_ge = 0
    for _ in range(b):
        geom_perm = geom.copy()
        for g in np.unique(groups):
            m = groups == g
            geom_perm[m] = rng.permutation(geom_perm[m])
        rho_p = fe_partial_spearman(geom_perm, tgt, base, groups)
        if np.isfinite(rho_p) and abs(rho_p) >= abs(obs) - 1e-12:
            n_ge += 1
    return (1 + n_ge) / (b + 1)


def per_source_partial(
    geom: np.ndarray,
    tgt: np.ndarray,
    base: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Per-source partial Spearman of (geom, tgt | base), ranks within source."""
    out: dict[str, float] = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 4:
            out[str(g)] = float("nan")
            continue
        rx = rankdata(geom[m])
        ry = rankdata(tgt[m])
        rz = rankdata(base[m])
        Z = np.vstack([np.ones_like(rz), rz]).T
        ex, ey = _resid(rx, Z), _resid(ry, Z)
        if ex.std() == 0 or ey.std() == 0:
            out[str(g)] = float("nan")
            continue
        out[str(g)] = float(np.corrcoef(ex, ey)[0, 1])
    return out


# ── Data loading ──────────────────────────────────────────────────────────


def load_target_rows(target_file: Path) -> list[dict[str, Any]]:
    """Load the #411 frozen snapshot into 138 off-diagonal rows.

    Each row: {source, bystander, delta, trained_rate, base_rate}.
    """
    with open(target_file) as f:
        snap = json.load(f)
    rows: list[dict[str, Any]] = []
    for source, src_data in snap["per_source"].items():
        deltas = src_data["per_panel_delta"]
        trained = src_data["per_panel_trained_rate"]
        base = src_data["per_panel_base_rate"]
        for bystander, delta in deltas.items():
            if bystander == source:
                continue  # off-diagonal only
            rows.append(
                {
                    "source": source,
                    "bystander": bystander,
                    "delta": float(delta),
                    "trained_rate": float(trained[bystander]),
                    "base_rate": float(base[bystander]),
                }
            )
    if len(rows) != 138:
        raise ValueError(f"Expected 138 off-diagonal cells, got {len(rows)}")
    return rows


def cell_filename(point: str, layer: int, metric: str, variant: str) -> str:
    """Metric-phase filename for one bake-off cell."""
    return f"{point}__layer{layer}__{metric}__{variant}.json"


def load_distance_matrix(
    *,
    repo_id: str,
    revision: str,
    prefix: str,
    filename: str,
) -> dict[str, dict[str, float | None]]:
    """Download one bake-off distance JSON at the pinned revision; return its matrix.

    Raises if the payload carries ``matrix: null`` — the requested cosine /
    mmd cells are all defined, so a None matrix means a wrong cell name.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id,
        f"{prefix}/{filename}",
        repo_type="dataset",
        revision=revision,
    )
    with open(path) as f:
        payload = json.load(f)
    matrix = payload.get("matrix")
    if matrix is None:
        raise ValueError(f"{filename}: matrix is null (intentional N/A cell?) — cannot score")
    return matrix


def build_xy(
    matrix: dict[str, dict[str, float | None]],
    rows: list[dict[str, Any]],
    persona_to_cid: dict[str, str],
) -> dict[str, np.ndarray | int]:
    """Align geometry distances with the 138 target rows.

    Exclusions follow issue509_scoring.py: rows whose matrix entry is
    missing/None are dropped (``n_excluded_none``), then pairs whose
    distance sits at the floor of the metric's natural range
    (|x| < 1e-6, ROUND-3 G2) are dropped (``n_excluded_saturated``).
    """
    geom, delta, trained, base, src = [], [], [], [], []
    n_none = 0
    for row in rows:
        src_cid = persona_to_cid.get(row["source"])
        bys_cid = persona_to_cid.get(row["bystander"])
        d = None
        if src_cid is not None and bys_cid is not None:
            d = matrix.get(src_cid, {}).get(bys_cid)
        if d is None:
            n_none += 1
            continue
        geom.append(float(d))
        delta.append(row["delta"])
        trained.append(row["trained_rate"])
        base.append(row["base_rate"])
        src.append(row["source"])
    geom_arr = np.array(geom, dtype=float)
    sat_mask = ~np.isfinite(geom_arr) | (np.abs(geom_arr) < PAIR_SATURATION_ABS_THRESHOLD)
    keep = ~sat_mask
    return {
        "geom": geom_arr[keep],
        "delta": np.array(delta, dtype=float)[keep],
        "trained": np.array(trained, dtype=float)[keep],
        "base": np.array(base, dtype=float)[keep],
        "source": np.array(src)[keep],
        "n_excluded_none": n_none,
        "n_excluded_saturated": int(sat_mask.sum()),
    }


# ── Per-cell scoring ──────────────────────────────────────────────────────


def score_panel(
    geom: np.ndarray,
    tgt: np.ndarray,
    base: np.ndarray,
    groups: np.ndarray,
    *,
    headline: bool,
    bootstrap_b: int,
    perm_b: int,
) -> dict[str, Any]:
    """One (cell, target, panel) scoring block.

    Always: FE Spearmans, both partials, within-R^2 decomposition.
    Headline cells additionally: clustered bootstrap CI + permutation p
    on ``partial_geom_given_base``.
    """
    out: dict[str, Any] = {"n": len(geom), "n_sources": len(np.unique(groups))}
    if len(geom) < MIN_SURVIVING_PAIRS:
        out["too_few_pairs"] = True
        for k in (
            "rho_geom_alone",
            "rho_base_alone",
            "partial_geom_given_base",
            "partial_base_given_geom",
            "within_r2_base_only",
            "within_r2_geom_only",
            "within_r2_both",
            "unique_r2_geom",
            "unique_r2_base",
        ):
            out[k] = float("nan")
        return out
    out["rho_geom_alone"] = fe_spearman(geom, tgt, groups)
    out["rho_base_alone"] = fe_spearman(base, tgt, groups)
    out["partial_geom_given_base"] = fe_partial_spearman(geom, tgt, base, groups)
    out["partial_base_given_geom"] = fe_partial_spearman(base, tgt, geom, groups)
    r2_base = within_r2(tgt, [base], groups)
    r2_geom = within_r2(tgt, [geom], groups)
    r2_both = within_r2(tgt, [base, geom], groups)
    out["within_r2_base_only"] = r2_base
    out["within_r2_geom_only"] = r2_geom
    out["within_r2_both"] = r2_both
    out["unique_r2_geom"] = r2_both - r2_base
    out["unique_r2_base"] = r2_both - r2_geom
    if headline:
        ci_lo, ci_hi = cluster_bootstrap_partial_ci(geom, tgt, base, groups, b=bootstrap_b)
        out["partial_geom_given_base_ci"] = [ci_lo, ci_hi]
        out["partial_geom_given_base_perm_p"] = perm_p_partial(
            geom, tgt, base, groups, out["partial_geom_given_base"], b=perm_b
        )
        out["bootstrap_b"] = bootstrap_b
        out["perm_b"] = perm_b
    return out


def score_cell(
    xy: dict[str, Any],
    *,
    headline: bool,
    bootstrap_b: int,
    perm_b: int,
) -> dict[str, Any]:
    """Score one geometry cell on both targets and both panels."""
    geom = xy["geom"]
    base = xy["base"]
    groups = xy["source"]
    live = np.isfinite(xy["delta"]) & (np.abs(xy["delta"]) > LIVE_DELTA_THRESHOLD)
    out: dict[str, Any] = {
        "n_excluded_none": xy["n_excluded_none"],
        "n_excluded_saturated": xy["n_excluded_saturated"],
        "headline": headline,
    }
    for tgt_name, tgt in (("delta", xy["delta"]), ("trained_abs", xy["trained"])):
        out[tgt_name] = {
            "all_138": score_panel(
                geom, tgt, base, groups, headline=headline, bootstrap_b=bootstrap_b, perm_b=perm_b
            ),
            "live_21": score_panel(
                geom[live],
                tgt[live],
                base[live],
                groups[live],
                headline=headline,
                bootstrap_b=bootstrap_b,
                perm_b=perm_b,
            ),
        }
    # delta is trained - base: the base -> delta reads carry the mechanical
    # -1 circularity flagged in the parent baserate_covariate analysis.
    out["delta"]["circular_warning"] = True
    return out


# ── Figure ────────────────────────────────────────────────────────────────


def make_figure(results: dict[str, Any], fig_dir: Path) -> None:
    """Per-layer curve of partial_geom_given_base (cosine, delta, all-138).

    Two lines (extraction points), faint dashed raw-rho companions, CI
    error bar on the L02 end_of_system headline cell, the L07 last_prompt
    MMD global-max cell as a separate point with its CI, and a horizontal
    reference line at rho_base_alone.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    cells = results["cells"]
    layers = list(EARLY_LAYERS)

    def cell(point: str, layer: int, metric: str) -> dict[str, Any]:
        return cells[f"{point}__layer{layer}__{metric}__centered"]

    set_paper_style("blog")
    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    point_labels = {
        "end_of_system": "End of system prompt (cosine)",
        "last_prompt": "Last prompt token (cosine)",
    }
    for color, point in zip(colors, EXTRACTION_POINTS, strict=True):
        partial = [
            cell(point, ly, "cosine")["delta"]["all_138"]["partial_geom_given_base"]
            for ly in layers
        ]
        raw = [cell(point, ly, "cosine")["delta"]["all_138"]["rho_geom_alone"] for ly in layers]
        ax.plot(layers, partial, marker="o", ms=4, color=color, label=point_labels[point])
        ax.plot(
            layers,
            raw,
            ls="--",
            lw=1.0,
            alpha=0.45,
            color=color,
            label=f"{point_labels[point]} — geometry alone",
        )

    # CI error bar on the L02 end_of_system cosine headline cell.
    l02 = cell("end_of_system", 2, "cosine")["delta"]["all_138"]
    ci = l02["partial_geom_given_base_ci"]
    est = l02["partial_geom_given_base"]
    ax.errorbar(
        [2],
        [est],
        yerr=[[est - ci[0]], [ci[1] - est]],
        fmt="none",
        ecolor=colors[0],
        elinewidth=1.6,
        capsize=5,
    )

    # L07 last_prompt MMD global-max cell as a separate point + CI.
    mmd = cell("last_prompt", 7, "mmd")["delta"]["all_138"]
    mmd_est = mmd["partial_geom_given_base"]
    mmd_ci = mmd["partial_geom_given_base_ci"]
    ax.errorbar(
        [7],
        [mmd_est],
        yerr=[[mmd_est - mmd_ci[0]], [mmd_ci[1] - mmd_est]],
        fmt="D",
        ms=7,
        color="#444444",
        ecolor="#444444",
        elinewidth=1.6,
        capsize=5,
        label="Last prompt token, MMD (global-max cell)",
    )

    # Horizontal reference: base rate alone (same value for every cell).
    base_alone = l02["rho_base_alone"]
    ax.axhline(base_alone, color="#888888", lw=1.2, ls=":", label="Bystander base rate alone")
    ax.axhline(0, color="black", lw=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Source-FE partial Spearman (geometry | base rate)")
    ax.set_xticks(layers)
    set_title_subtitle(
        ax,
        "Early-layer geometry survives the base-rate partial",
        "Sycophancy leakage delta, 138 source-bystander cells; CI bars on the two headline cells",
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig_paper(fig, "baserate_partial_early_layer", dir=fig_dir)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Issue #509 follow-up: partial out the bystander base sycophancy rate "
            "against the bake-off early-layer geometry cells."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target-file", type=Path, default=DEFAULT_TARGET_FILE)
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    p.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    p.add_argument("--metrics-prefix", default=DEFAULT_METRICS_PREFIX)
    p.add_argument(
        "--bootstrap", type=int, default=5000, help="Cluster-bootstrap reps (headline cells)"
    )
    p.add_argument("--perm", type=int, default=2000, help="Permutation reps (headline cells)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--skip-figure", action="store_true", help="Skip the figure (stats only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    from explore_persona_space.experiments.i509_syco_conditions import CID_TO_SYCO_PERSONA

    persona_to_cid = {v: k for k, v in CID_TO_SYCO_PERSONA.items()}
    rows = load_target_rows(args.target_file)

    # Live-21 degeneracy diagnostic: which sources carry the live cells.
    live_rows = [r for r in rows if abs(r["delta"]) > LIVE_DELTA_THRESHOLD]
    live_per_source: dict[str, int] = {}
    for r in live_rows:
        live_per_source[r["source"]] = live_per_source.get(r["source"], 0) + 1
    live_degenerate = len(live_per_source) < 3

    # Cell list: 28-cell early-cosine cluster + the L07 last_prompt MMD cell.
    cell_specs: list[tuple[str, int, str, str]] = [
        (point, layer, "cosine", "centered")
        for point in EXTRACTION_POINTS
        for layer in EARLY_LAYERS
    ]
    cell_specs.append(("last_prompt", 7, "mmd", "centered"))

    cells: dict[str, Any] = {}
    for point, layer, metric, variant in cell_specs:
        fname = cell_filename(point, layer, metric, variant)
        matrix = load_distance_matrix(
            repo_id=args.hf_repo,
            revision=args.hf_revision,
            prefix=args.metrics_prefix,
            filename=fname,
        )
        xy = build_xy(matrix, rows, persona_to_cid)
        headline = (point, layer, metric, variant) in HEADLINE_CELLS
        scored = score_cell(xy, headline=headline, bootstrap_b=args.bootstrap, perm_b=args.perm)
        scored["extraction_point"] = point
        scored["layer"] = layer
        scored["metric"] = metric
        scored["variant"] = variant
        key = fname.removesuffix(".json")
        cells[key] = scored
        blk138 = scored["delta"]["all_138"]
        print(
            f"{key}: n={blk138['n']} "
            f"partial_geom|base(delta,138)={blk138['partial_geom_given_base']:+.3f} "
            f"rho_geom={blk138['rho_geom_alone']:+.3f}" + (" [headline]" if headline else "")
        )

    # Per-source partials at the representative L02 end_of_system cosine cell.
    l02_fname = cell_filename("end_of_system", 2, "cosine", "centered")
    l02_matrix = load_distance_matrix(
        repo_id=args.hf_repo,
        revision=args.hf_revision,
        prefix=args.metrics_prefix,
        filename=l02_fname,
    )
    l02_xy = build_xy(l02_matrix, rows, persona_to_cid)
    per_source = per_source_partial(
        l02_xy["geom"], l02_xy["delta"], l02_xy["base"], l02_xy["source"]
    )

    results: dict[str, Any] = {
        "schema_version": 1,
        "followup_label": "baserate-partial-early-layer-cosine",
        "description": (
            "Source-FE partial Spearman of bake-off early-layer geometry vs #411 "
            "sycophancy leakage, partialling the bystander intrinsic base rate. "
            "Point estimates use the issue509_baserate_covariate.py helpers "
            "(rank-then-demean FE); inference on the two headline cells uses the "
            "issue509_scoring.py conventions (source-clustered bootstrap seed 42, "
            "within-source permutation, same-statistic null)."
        ),
        "headline_cells": [cell_filename(*c).removesuffix(".json") for c in HEADLINE_CELLS],
        "n_offdiagonal_cells": len(rows),
        "live_delta_threshold": LIVE_DELTA_THRESHOLD,
        "n_live_cells": len(live_rows),
        "live_cells_per_source": live_per_source,
        "live_21_bootstrap_degenerate": live_degenerate,
        "live_21_degeneracy_note": (
            "Live cells are concentrated in "
            f"{len(live_per_source)} of 6 sources ({live_per_source}); the source-clustered "
            "bootstrap on the live-21 panel resamples effectively 2 clusters and its CI is "
            "not reliable."
            if live_degenerate
            else "Live cells span >= 3 sources."
        ),
        "per_source_partial_geom_given_base_L02_end_of_system_cosine_delta": per_source,
        "cells": cells,
        "inputs": {
            "target_file": str(args.target_file),
            "hf_repo": args.hf_repo,
            "hf_revision": args.hf_revision,
            "metrics_prefix": args.metrics_prefix,
        },
        "bootstrap_b": args.bootstrap,
        "perm_b": args.perm,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "perm_seed_int": _hashed_seed(PERM_SEED_TAG),
        "git_sha_worktree": _git_sha(PROJECT_ROOT),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "env": {"python": platform.python_version(), "numpy": np.__version__},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")

    # Headline summary to stdout.
    for key in results["headline_cells"]:
        c = cells[key]
        for panel in ("all_138", "live_21"):
            blk = c["delta"][panel]
            ci = blk.get("partial_geom_given_base_ci", [float("nan"), float("nan")])
            pp = blk.get("partial_geom_given_base_perm_p", float("nan"))
            print(
                f"HEADLINE {key} delta/{panel}: "
                f"partial_geom|base={blk['partial_geom_given_base']:+.3f} "
                f"CI[{ci[0]:+.3f}, {ci[1]:+.3f}] perm_p={pp:.4f} (n={blk['n']})"
            )

    if not args.skip_figure:
        make_figure(results, args.fig_dir)
        print(f"wrote figure under {args.fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
