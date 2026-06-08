#!/usr/bin/env python3
"""Issue #509 per-arm scoring against #494 fact + #411 sycophancy targets.

Path B post-extraction scoring. Reads the metrics-phase distance JSONs
produced by ``scripts/issue502_dispatch.py --phase metrics`` and computes:

  - Length-partial Spearman rho per (point, layer, metric, variant)
    against the per-arm leakage target.
  - Substrate-FE (fact arm) or source-FE (syco arm) residualized rho.
  - Prior-residualized rho on the fact arm (bystander_logprob from
    #500).
  - Attenuation-adjusted rho = rho_obs / sqrt(reliability_y).
  - Within-stratum permutation null (B=2000, hashed seed).
  - LOCO-CV R^2 of a length-controlled linear fit.
  - Cluster bootstrap (5000 reps, seed=42) CI per cell on the L22 last
    prompt gauss_kl anchor + the L19-L24 ridge mean.
  - Delete-one-substrate (fact) or delete-one-source (syco) jackknife.

Pre-registered anchors per #509 v3 plan section 4.1.6:
  1. #494 / #470 coarse predictors on the same pairs.
  2. #502 full-panel ρ_full_deltag at L22 = -0.748.
  3. #502 non-stylized 156-pair ρ_nonstylized_deltag at L22 = -0.581,
     ρ_nonstylized_glogp at L22 = -0.628. Loaded from
     ``eval_results/issue_502/bakeoff/regression/loc_ep1.json``.

Inputs:
  --metrics-dir   directory holding per-(point, layer, metric, variant)
                  distance JSONs (the metrics-phase output).
  --arm           one of "fact" or "syco".
  --target-file   per-arm leakage matrix (see plan section 10).
  --output        per-arm scoring JSON output path.
  --smoke         smoke-mode: relax min-cell expectations + skip
                  permutation/bootstrap so a tiny 2-cond x 1-layer x
                  1-metric grid runs in seconds.

The output JSON carries every cell's (rho_obs, rho_adj, ci_lo, ci_hi,
perm_p, loco_r2, jackknife_se) along with the arm-level summary
statistics (L22 anchor, L19-L24 ridge mean) and the comparison anchors.
"""

# Greek + special characters appear in this file's prose.
# ruff: noqa: RUF002, RUF003
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("i509.scoring")

PERMUTATION_NULL_SEED_TAG = b"issue509_perm_null_v1"
PERMUTATION_B = 2000
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 42

# Pre-verified at plan time from eval_results/issue_502/bakeoff/regression/loc_ep1.json.
NONSTYLIZED_ANCHOR_RHO_DELTAG = -0.5805350970934474
NONSTYLIZED_ANCHOR_RHO_GLOGP = -0.6278048029398947
FULL_PANEL_ANCHOR_RHO_DELTAG = -0.748

L19_L24_RIDGE_LAYERS = (19, 20, 21, 22, 23, 24)
RIDGE_METRICS = ("gauss_kl", "mmd", "wass2")

# ROUND-2/#509 FIX F2: the layer field accepts an optional leading `-` so
# the `next_token_js` baseline at `layer-1` (which encodes "no specific
# layer — vocabulary-level" per the #502 metrics-phase naming) is not
# silently dropped. The metric/variant character classes are unchanged.
METRIC_FILE_PATTERN = re.compile(
    r"^(?P<point>[a-z_]+)__layer(?P<layer>-?\d+)__(?P<metric>[a-z0-9_]+)__(?P<variant>[a-z_]+)\.json$"
)

# ROUND-2/#509 FIX F2: filename suffixes that mark sidecar files which
# share the metric-file naming prefix but are NOT predictor cells the
# scoring should ingest. `*__perm.json` are MMD permutation null sidecars
# (#502 emits one per real cell); `*__cross_check_406.json` are #406
# cross-checks. The plan's metric-file enumeration must skip them.
_SIDECAR_FILENAME_SUFFIXES: tuple[str, ...] = (
    "__perm.json",
    "__cross_check_406.json",
)


def _hashed_seed(tag: bytes) -> int:
    """Deterministic integer for numpy.random.default_rng from a string tag."""
    return int(hashlib.sha256(tag).hexdigest()[:8], 16)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _env_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via rank+Pearson. NaN-safe (drops paired NaNs)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    from scipy.stats import rankdata

    rx = rankdata(x[m])
    ry = rankdata(y[m])
    return float(np.corrcoef(rx, ry)[0, 1])


def _residualize(y: np.ndarray, fe: np.ndarray) -> np.ndarray:
    """Within-stratum residualization: subtract the per-stratum mean of y."""
    y = np.asarray(y, dtype=float)
    out = y.copy()
    for s in np.unique(fe):
        idx = fe == s
        out[idx] -= np.nanmean(y[idx])
    return out


def _permutation_p(rho_obs: float, x: np.ndarray, y: np.ndarray, fe: np.ndarray, b: int) -> float:
    """DEPRECATED — kept only as a back-compat re-export.

    Use :func:`_permutation_p_partial` instead: it computes the same
    residualized statistic (Plan §4.1.5 regression D: ``ρ(x | s, y | s)``)
    inside every permutation draw so the observed and null are the SAME
    function. This wrapper forwards to it.
    """
    return _permutation_p_partial(rho_obs, x, y, fe, b)


def _permutation_p_partial(
    rho_obs: float,
    x: np.ndarray,
    y: np.ndarray,
    fe: np.ndarray,
    b: int,
) -> float:
    """Within-stratum permutation p for the FE partial Spearman.

    ROUND-2/#509 FIX F3: round-1 shuffled ``y`` within stratum but scored
    against ``_spearman_rho(x, y_perm)`` — un-residualized — while the
    observed statistic was the residualized ``rho_fe``. The test compared
    two different statistics; the p-value was statistically meaningless.

    The corrected null residualizes BOTH ``x`` and ``y_perm`` within
    stratum on every draw, computes the partial-Spearman statistic, and
    counts the fraction at-or-beyond the observed magnitude. The observed
    statistic upstream is also partial Spearman (see ``_score_one_cell``).
    """
    if not np.isfinite(rho_obs):
        return float("nan")
    rng = np.random.default_rng(_hashed_seed(PERMUTATION_NULL_SEED_TAG))
    x_resid = _residualize(x, fe)
    n_ge = 0
    for _ in range(b):
        y_perm = y.copy()
        for s in np.unique(fe):
            idx = np.where(fe == s)[0]
            if len(idx) <= 1:
                continue
            rng.shuffle(idx)
            y_perm[fe == s] = y[idx]
        # Residualize y_perm WITHIN STRATUM after shuffling, then compute
        # the same residualized Spearman as the observed statistic.
        y_perm_resid = _residualize(y_perm, fe)
        rho_p = _spearman_rho(x_resid, y_perm_resid)
        if np.isfinite(rho_p) and abs(rho_p) >= abs(rho_obs):
            n_ge += 1
    return (1 + n_ge) / (b + 1)


def _cluster_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Cluster bootstrap 95% CI on the Spearman rho."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    rhos: list[float] = []
    for _ in range(b):
        sample = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in sample])
        rho_b = _spearman_rho(x[idx], y[idx])
        if np.isfinite(rho_b):
            rhos.append(rho_b)
    if len(rhos) < 100:
        return (float("nan"), float("nan"))
    return (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)))


def _jackknife_se(x: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> float:
    """Delete-one-cluster jackknife SE on the Spearman rho."""
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2:
        return float("nan")
    rhos: list[float] = []
    for c in unique_clusters:
        idx = clusters != c
        rho_k = _spearman_rho(x[idx], y[idx])
        if np.isfinite(rho_k):
            rhos.append(rho_k)
    if len(rhos) < 2:
        return float("nan")
    n = len(rhos)
    rho_mean = float(np.mean(rhos))
    se = np.sqrt((n - 1) / n * sum((r - rho_mean) ** 2 for r in rhos))
    return float(se)


def _loco_cv_r2(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Leave-one-class-out CV R^2 of a univariate linear fit."""
    unique_classes = np.unique(classes)
    if len(unique_classes) < 2:
        return float("nan")
    y_pred = np.full_like(y, np.nan, dtype=float)
    for c in unique_classes:
        train = classes != c
        test = classes == c
        x_train, y_train = x[train], y[train]
        x_test = x[test]
        if len(x_train) < 2 or np.all(x_train == x_train[0]):
            continue
        slope, intercept = np.polyfit(x_train, y_train, 1)
        y_pred[test] = slope * x_test + intercept
    valid = ~np.isnan(y_pred)
    if valid.sum() < 2:
        return float("nan")
    ss_res = float(np.sum((y[valid] - y_pred[valid]) ** 2))
    ss_tot = float(np.sum((y[valid] - np.mean(y[valid])) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _attenuation_adjust(rho_obs: float, reliability_y: float) -> float:
    """rho_adj = rho_obs / sqrt(reliability_y); clamps reliability to (0, 1]."""
    if not (0.0 < reliability_y <= 1.0):
        return rho_obs
    return rho_obs / np.sqrt(reliability_y)


# ── ROUND-2/#509 FIX F6: plan-§5 condition slugs + coarse-predictor anchor ────


def _per_source_spearman(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
) -> dict[str, float]:
    """Per-source Spearman ρ (Plan §4.2.5 regression C; condition slug
    ``syco_arm/per_source``).

    Computes a separate Spearman correlation for each source on the
    syco arm — the 6 per-source ρ's that the plan calls out as a
    diagnostic for cells where the headline ρ_fe is dominated by one
    source. Returns ``{source_name: rho}``. NaN-safe per-source.
    """
    out: dict[str, float] = {}
    for s in np.unique(sources):
        idx = sources == s
        if idx.sum() < 3:
            out[str(s)] = float("nan")
            continue
        out[str(s)] = _spearman_rho(x[idx], y[idx])
    return out


def _live_cells_mask(deltas: np.ndarray, threshold: float = 0.10) -> np.ndarray:
    """Live-cells mask (Plan §4.2.6 #2; condition slug
    ``syco_arm/live_cells_only``).

    Selects the cells with ``|Δ| > threshold``. Per plan §4.2.6, with
    ``threshold=0.10`` on #411's 138-cell off-diagonal panel, this yields
    21 cells (15 software_engineer rows + 6 assistant rows). Returns a
    boolean mask aligned with ``deltas``. NaN cells are excluded.
    """
    deltas = np.asarray(deltas, dtype=float)
    return np.isfinite(deltas) & (np.abs(deltas) > threshold)


def _rank_in_bystanders(
    predictor: dict[str, float],
    target_persona: str,
    ascending: bool = True,
) -> int:
    """Rank of ``target_persona`` among bystanders by predictor (Plan §4.2.5;
    condition slug ``syco_arm/comedian_recovery``).

    Bake-off cell predictors are distances (smaller = more similar) so
    ``ascending=True`` sorts the most-similar bystander first. Returns
    1-based rank. Missing target ⇒ ``-1`` so the analyzer can flag.
    """
    if target_persona not in predictor:
        return -1
    sorted_items = sorted(predictor.items(), key=lambda kv: kv[1], reverse=not ascending)
    for i, (k, _) in enumerate(sorted_items, start=1):
        if k == target_persona:
            return i
    return -1


def _is_predictor_saturated(x: np.ndarray, var_threshold: float = 1e-6) -> bool:
    """Predictor-saturation flag (Plan §4.2.6 #3; condition slug
    ``syco_arm/per_cell_predictor_saturation``).

    True when the predictor signal's variance is below ``var_threshold``
    — i.e. the predictor is nearly constant across the cell, so any
    correlation against it is uninformative (rank-shuffles among
    near-equal values dominate the score).
    """
    x = np.asarray(x, dtype=float)
    x_finite = x[np.isfinite(x)]
    if len(x_finite) < 2:
        return True
    return float(np.var(x_finite)) < var_threshold


# ROUND-3/#509 G2: per-pair saturation exclusion threshold. Cloud metrics
# (cosine distance, JS divergence, MMD) bottom out near 0 when two persona
# vectors are indistinguishable at this (extraction_point, layer, metric)
# cell — the pair carries no rank information for the regression. Round 2
# wired ``_is_predictor_saturated`` as a whole-cell variance flag that
# never dropped any pair; G2 adds per-pair exclusion before the headline
# statistic, the perm null, the bootstrap CI, the jackknife SE, and the
# LOCO-CV R^2 (per plan §4.2.6 #3: "exclude that PAIR from the
# regression but flag the cell").
_PAIR_SATURATION_ABS_THRESHOLD = 1e-6

# Minimum surviving sample for a stable rank-correlation statistic.
# Below this floor, the surviving pairs are too few for the partial
# Spearman + cluster bootstrap to be informative, so ``_score_one_cell``
# emits NaN + ``saturation_too_aggressive: True``.
_MIN_SURVIVING_PAIRS = 5


def _saturated_pair_mask(
    x: np.ndarray,
    abs_threshold: float = _PAIR_SATURATION_ABS_THRESHOLD,
) -> np.ndarray:
    """Return a boolean mask: True where the predictor distance is at the
    floor of its natural range (plan §4.2.6 #3).

    For cloud metrics in the #493/#502 bake-off (cosine distance, JS
    divergence, MMD) a value near 0 means the persona pair is
    indistinguishable at this metric/layer/extraction-point cell — no
    rank-information contribution, only noise. Non-finite entries are
    treated as saturated so the downstream filter discards them too.

    Args:
        x: Per-pair predictor distances, shape ``(n_pairs,)``.
        abs_threshold: ``|x_i| < abs_threshold`` flags pair ``i`` as
            saturated. Defaults to ``_PAIR_SATURATION_ABS_THRESHOLD``
            (1e-6), well below bf16's smallest non-tied increment so
            non-saturated pairs are kept even at single-precision noise.

    Returns:
        Boolean array of the same shape as ``x``; True == saturated.
    """
    x = np.asarray(x, dtype=float)
    return ~np.isfinite(x) | (np.abs(x) < abs_threshold)


def _coarse_lift_syco_arm_per_cell(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    se: np.ndarray,
    syco_rows: list[dict[str, Any]],
    matched_indices: list[int],
    *,
    columns: tuple[str, ...] | None = None,
    allow_unknown_se: bool = True,
    perm_b: int = PERMUTATION_B,
    bootstrap_b: int = BOOTSTRAP_B,
) -> dict[str, Any]:
    """#518 v4 round-2 must-fix 1: per-coarse Spearman ρ for the syco/refusal/em arm.

    Mirrors ``_coarse_lift_per_cell`` (the fact-arm anchor) on the SAME pairs as
    the bake-off cell's (x, y). For each predictor column in ``columns`` (the
    coarse-zoo from plan §4.4 + ``completion_logprob`` as the cross-behavior
    headline named in plan §0/§1/§4.4/§11), this:

      1. Pulls the column from ``syco_rows[matched_indices]`` (the SAME
         cells the residual-stream cell scored its ``rho_fe`` on).
      2. Residualizes both predictor and ``y`` within source FE (matching
         ``_score_one_cell``'s headline statistic).
      3. Computes the signed (NOT |·|) Spearman ρ on the residualized
         pair -- the aggregator's same-sign gate needs the sign.
      4. Computes the attenuation-adjusted version
         ``rho_fe_adj = rho_fe / sqrt(reliability_y)``.
      5. Runs a within-stratum permutation null and a cluster-bootstrap
         95% CI on the partial-Spearman statistic.

    #518 v4 round-3 must-fix 1: ``perm_b`` + ``bootstrap_b`` are now
    callable-threaded (default = module-level ``PERMUTATION_B`` /
    ``BOOTSTRAP_B``) so the headline-deciding per-coarse permutation +
    bootstrap budget matches the residual-stream pass. Round 2 hardcoded
    ``B=200`` (permutation) + ``b=500`` (bootstrap), which graining the
    headline p-floor (~5e-3 at B=200) below the aggregator's
    ``perm_p <= 0.01`` gate; production now defaults to B=2000 /
    b=5000 and smoke overrides to 50 / 50 to match the residual-stream
    smoke (see ``score_arm`` call site).

    The output payload per predictor:
        ``{predictor: {"rho_fe": ..., "rho_fe_adj": ..., "perm_p": ...,
                       "cluster_ci": [lo, hi], "n_finite": ...}}``

    The aggregator then reads ``per_coarse_rho_fe[predictor]["rho_fe_adj"]``
    from each arm and builds a coarse-predictor `(rho_syco, rho_refusal,
    rho_em)` triple alongside the residual-stream cell triples.

    NaN-safe: predictors with fewer than 3 finite values per cell emit
    ``rho_fe = NaN``.
    """
    if columns is None:
        columns = REFUSAL_EM_COARSE_PREDICTOR_COLUMNS
    out: dict[str, dict[str, Any]] = {}
    if len(x) < 3 or len(matched_indices) != len(x):
        # Insufficient pairs for a stable Spearman; emit empty payload.
        return out
    # Reliability of y once per cell -- shared across all coarse predictors
    # because it depends on y + se + strata, not on the predictor.
    rel = _reliability_y(y, se, strata=strata, allow_unknown_se=allow_unknown_se)
    y_resid = _residualize(y, strata)
    for col in columns:
        col_vals = np.array(
            [syco_rows[i].get(col, float("nan")) for i in matched_indices],
            dtype=float,
        )
        finite_mask = np.isfinite(col_vals)
        n_finite = int(finite_mask.sum())
        if n_finite < 3:
            out[col] = {
                "rho_fe": float("nan"),
                "rho_fe_adj": float("nan"),
                "perm_p": float("nan"),
                "cluster_ci": [float("nan"), float("nan")],
                "n_finite": n_finite,
            }
            continue
        col_resid = _residualize(col_vals, strata)
        rho_fe = _spearman_rho(col_resid, y_resid)
        rho_fe_adj = _attenuation_adjust(rho_fe, rel)
        # Permutation null + cluster bootstrap -- the per-coarse gate is
        # identical in shape to the residual-stream gate; the aggregator
        # applies the SAME same_sign + min(|ρ|) ≥ 0.40 + perm_p ≤ 0.01
        # rule, so the budget MUST match (round-3 must-fix 1).
        perm_p = _permutation_p_partial(rho_fe, col_vals, y, strata, perm_b)
        try:
            ci_lo, ci_hi = _cluster_bootstrap_ci(col_resid, y_resid, strata, b=bootstrap_b)
        except (ValueError, RuntimeError):
            ci_lo, ci_hi = float("nan"), float("nan")
        out[col] = {
            "rho_fe": rho_fe,
            "rho_fe_adj": rho_fe_adj,
            "perm_p": perm_p,
            "cluster_ci": [ci_lo, ci_hi],
            "n_finite": n_finite,
        }
    return out


def _coarse_lift_per_cell(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    fact_rows: list[dict[str, Any]],
    cid_to_csv_persona: dict[str, str],
    matched_indices: list[int],
) -> dict[str, Any]:
    """Compute Anchor 1 coarse-predictor lift on the SAME pairs as a
    bake-off cell (Plan §4.1.6.1; condition slug ``fact_arm/coarse_lift``).

    For each of the 5 coarse-predictor columns in the #494 CSV (cosine_a_L21,
    cosine_b_L21, js_on_topic, fact_slice_js, bystander_logprob), compute
    the FE-residualized Spearman ρ on the same cell-pairs and report:
      - per-coarse rho_fe (in absolute value)
      - rho_coarse_max = max |rho_fe| across the 5 coarse predictors
      - delta_rho = |rho_fe (bake-off)| − rho_coarse_max
    The §6.2 generalizes verdict requires ``delta_rho ≥ 0.15``.

    ``matched_indices`` aligns the bake-off cell's (x, y, strata) with
    the row indices of ``fact_rows`` — needed so each coarse predictor
    is read from the same rows.
    """
    out: dict[str, Any] = {"per_coarse_rho_fe": {}, "rho_coarse_max": float("nan")}
    bakeoff_rho = float(np.abs(_spearman_rho(_residualize(x, strata), _residualize(y, strata))))
    out["bakeoff_rho_fe_abs"] = bakeoff_rho
    per_coarse: dict[str, float] = {}
    for col in FACT_COARSE_PREDICTOR_COLUMNS:
        col_vals = np.array(
            [fact_rows[i].get(col, float("nan")) for i in matched_indices],
            dtype=float,
        )
        if not np.isfinite(col_vals).any():
            per_coarse[col] = float("nan")
            continue
        col_resid = _residualize(col_vals, strata)
        y_resid = _residualize(y, strata)
        per_coarse[col] = float(np.abs(_spearman_rho(col_resid, y_resid)))
    out["per_coarse_rho_fe"] = per_coarse
    finite_rhos = [v for v in per_coarse.values() if np.isfinite(v)]
    if finite_rhos:
        rho_coarse_max = float(max(finite_rhos))
        out["rho_coarse_max"] = rho_coarse_max
        out["delta_rho"] = bakeoff_rho - rho_coarse_max
    else:
        out["delta_rho"] = float("nan")
    return out


# ── Target loaders ────────────────────────────────────────────────────────


# ROUND-2/#509 FIX F6: the 5 fact-arm coarse predictors from #494
# `regression_data.csv` — Plan §4.1.6.1 calls these "anchor 1" and the
# §6.2 generalizes verdict requires `Δρ ≥ 0.15 vs the coarse-predictor
# best on the same pairs`. Round 1 parsed only `leak_rate` and
# `bystander_logprob`, so the lift comparison could not fire.
FACT_COARSE_PREDICTOR_COLUMNS: tuple[str, ...] = (
    "cosine_a_L21",
    "cosine_b_L21",
    "js_on_topic",
    "fact_slice_js",
    "bystander_logprob",
)


def _load_fact_target(csv_path: Path) -> dict[str, Any]:
    """Load #494's 26-cell fact-leakage panel from regression_data.csv.

    ROUND-2/#509 FIX F6: also parse all 5 coarse-predictor columns so
    `_coarse_lift` (anchor 1) can compute Δρ on the same cell-pairs as
    the bake-off. Missing columns are tolerated (NaN-filled + logged) so
    older `regression_data.csv` formats still load; the smoke +
    production CSV at `eval_results/issue_494/regression_data.csv` has
    all 5.
    """
    rows: list[dict[str, Any]] = []
    missing_columns: set[str] = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["leak_rate"] = float(row["leak_rate"])
            for col in FACT_COARSE_PREDICTOR_COLUMNS:
                if col in row and row[col] not in (None, ""):
                    row[col] = float(row[col])
                else:
                    missing_columns.add(col)
                    row[col] = float("nan")
            rows.append(row)
    if missing_columns:
        logger.warning(
            "Fact-arm coarse-predictor columns missing from %s: %s "
            "(coarse_lift Δρ for those columns will be NaN)",
            csv_path,
            sorted(missing_columns),
        )
    if len(rows) != 26:
        logger.warning("Expected 26 #494 rows, got %d", len(rows))
    return {"rows": rows, "source_file": str(csv_path)}


def _load_syco_coarse_zoo_backfill(
    path: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """#518 v4 round-3 must-fix 2 (Option A): load #480 coarse-zoo backfill.

    Reads ``eval_results/issue_480/_inputs/predictor_comparison.json`` and
    returns a ``(source, bystander) -> {column: value}`` map carrying every
    column in ``REFUSAL_EM_COARSE_PREDICTOR_COLUMNS`` except
    ``completion_logprob`` (which is supplied separately by the syco arm's
    own logprob backfill -- the #480 substrate does not carry it). The 138
    off-diagonal pairs match the #411 syco panel 1-to-1.

    Used by ``_load_syco_target`` to merge the 19 coarse-zoo + base-rate +
    length-control columns onto each syco row so the cross-arm aggregator's
    20-predictor pass produces non-empty triples for every coarse predictor,
    not just ``completion_logprob`` (round 2's behaviour).
    """
    with open(path) as f:
        payload = json.load(f)
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError(
            f"syco coarse-zoo backfill at {path} has no 'cells' list "
            f"(got {type(cells).__name__}); the #480 predictor_comparison.json "
            f"schema is required."
        )
    out: dict[tuple[str, str], dict[str, float]] = {}
    for i, cell in enumerate(cells):
        source = cell.get("source")
        bystander = cell.get("bystander")
        if source is None or bystander is None:
            raise RuntimeError(
                f"syco coarse-zoo backfill cell #{i} at {path} missing 'source' or 'bystander'."
            )
        col_map: dict[str, float] = {}
        for col in REFUSAL_EM_COARSE_PREDICTOR_COLUMNS:
            if col == "completion_logprob":
                # #480 substrate does NOT carry completion_logprob; the
                # syco arm's bystander_logprob backfill supplies it.
                continue
            if col not in cell:
                raise KeyError(
                    f"syco coarse-zoo backfill cell #{i} at {path} "
                    f"(source={source!r}, bystander={bystander!r}) "
                    f"missing required column {col!r}. The #480 "
                    f"predictor_comparison.json schema MUST carry every "
                    f"REFUSAL_EM_COARSE_PREDICTOR_COLUMNS entry except "
                    f"completion_logprob."
                )
            v = cell[col]
            col_map[col] = float(v) if v is not None else float("nan")
        out[(source, bystander)] = col_map
    return out


def _build_one_syco_row(
    *,
    source: str,
    bystander: str,
    delta: float,
    p_t: float | None,
    p_b: float | None,
    lp_map: dict[tuple[str, str], float] | None,
    coarse_zoo_map: dict[tuple[str, str], dict[str, float]] | None,
) -> tuple[dict[str, Any], int, int]:
    """Build one (source, bystander) syco row + report missing-backfill deltas.

    Returns ``(row, delta_missing_lp, delta_missing_coarse_zoo)``. The
    caller accumulates the deltas across the panel and decides whether to
    raise after the loop. Extracted from ``_load_syco_target`` to keep the
    outer function under the McCabe complexity bound (C901, max 15).
    """
    # Independence approximation; rollouts = 50 probes * 10 = 500.
    if p_t is None or p_b is None:
        se = float("nan")
    else:
        n_rollouts = 500
        se = float(
            np.sqrt(max(p_t * (1 - p_t), 0.0) / n_rollouts + max(p_b * (1 - p_b), 0.0) / n_rollouts)
        )
    row: dict[str, Any] = {
        "source": source,
        "bystander": bystander,
        "delta": float(delta),
        "trained_rate": p_t,
        "base_rate": p_b,
        "se_delta": se,
    }
    delta_missing_lp = 0
    delta_missing_cz = 0
    # #518 v4 round-2 must-fix 1: completion_logprob merge from the syco
    # arm's bystander_logprob backfill.
    if lp_map is not None:
        lp = lp_map.get((source, bystander))
        if lp is None:
            delta_missing_lp = 1
            row["completion_logprob"] = float("nan")
        else:
            row["completion_logprob"] = lp
    # #518 v4 round-3 must-fix 2 (Option A): merge the 19 #480 coarse-zoo +
    # base-rate + length-control columns onto each row so the aggregator's
    # 20-predictor cross-arm pass has full predictor coverage on syco.
    # Missing cells RAISE after the loop (138/138 pair alignment is a
    # structural invariant: #480 and #411 share the same off-diagonal
    # pairs by construction).
    if coarse_zoo_map is not None:
        col_map = coarse_zoo_map.get((source, bystander))
        if col_map is None:
            delta_missing_cz = 1
            # NaN-fallback so the per-coarse ρ stays NaN-safe; the outer
            # raise after the loop turns the miss into a loud failure.
            for col in REFUSAL_EM_COARSE_PREDICTOR_COLUMNS:
                if col == "completion_logprob":
                    continue
                row[col] = float("nan")
        else:
            for col, v in col_map.items():
                row[col] = v
    return row, delta_missing_lp, delta_missing_cz


def _load_syco_target(
    snapshot_path: Path,
    syco_logprob_backfill: Path | None = None,
    syco_coarse_zoo_backfill: Path | None = None,
) -> dict[str, Any]:
    """Load #411's 138-cell sycophancy Δ panel from the frozen snapshot.

    #518 v4 round-2 must-fix 1: ``syco_logprob_backfill`` is the path to
    ``scripts/issue518_syco_logprob_backfill.py``'s output
    (``logprob_results.json``). When provided, the mean per-token
    ``log P(completion | bystander persona, Q)`` is merged onto each row as
    the ``completion_logprob`` column so the syco arm can build the
    per-coarse Spearman ρ alongside the refusal + EM arms and the
    cross-behavior aggregator can read ``per_coarse_rho_fe["completion_logprob"]``
    from each. Cells whose (source, bystander) is missing from the backfill
    are emitted with ``completion_logprob = NaN`` (the per-coarse ρ is
    NaN-safe; the analyzer surfaces the missing-cell count).

    #518 v4 round-3 must-fix 2 (Option A): ``syco_coarse_zoo_backfill`` is
    the path to ``eval_results/issue_480/_inputs/predictor_comparison.json``
    -- the 138-cell substrate carrying the 19 #480 coarse-zoo + base-rate +
    length-control columns. When provided, those columns are merged onto
    each syco row so the cross-arm aggregator's 20-predictor pass produces
    non-empty triples for every coarse predictor (round 2 produced empty
    triples for 19 of 20 because only ``completion_logprob`` was merged on
    syco rows). Cells whose (source, bystander) is missing from the backfill
    raise -- the 138/138 pair alignment with #411 is a structural invariant.
    """
    with open(snapshot_path) as f:
        snap = json.load(f)
    # Optional completion_logprob backfill (#518 v4 round-2 must-fix 1).
    lp_map: dict[tuple[str, str], float] = {}
    if syco_logprob_backfill is not None:
        with open(syco_logprob_backfill) as f:
            lp_payload = json.load(f)
        summary = lp_payload.get("summary", {})
        for src, bys_map in summary.items():
            for bys, cell in bys_map.items():
                mean_lp = cell.get("mean_logprob_per_tok")
                if mean_lp is not None:
                    lp_map[(src, bys)] = float(mean_lp)
    # #518 v4 round-3 must-fix 2 (Option A): coarse-zoo backfill.
    coarse_zoo_map: dict[tuple[str, str], dict[str, float]] = {}
    if syco_coarse_zoo_backfill is not None:
        coarse_zoo_map = _load_syco_coarse_zoo_backfill(syco_coarse_zoo_backfill)
    rows: list[dict[str, Any]] = []
    n_missing_lp = 0
    n_missing_coarse_zoo = 0
    for source, src_data in snap["per_source"].items():
        panel = src_data.get("per_panel_delta", {})
        trained_rate_by = src_data.get("per_panel_trained_rate", {})
        base_rate_by = src_data.get("per_panel_base_rate", {})
        for bystander, delta in panel.items():
            if bystander == source:
                continue  # off-diagonal only
            row, dlp, dcz = _build_one_syco_row(
                source=source,
                bystander=bystander,
                delta=delta,
                p_t=trained_rate_by.get(bystander),
                p_b=base_rate_by.get(bystander),
                lp_map=lp_map if syco_logprob_backfill is not None else None,
                coarse_zoo_map=coarse_zoo_map if syco_coarse_zoo_backfill is not None else None,
            )
            n_missing_lp += dlp
            n_missing_coarse_zoo += dcz
            rows.append(row)
    if syco_logprob_backfill is not None and n_missing_lp:
        logger.warning(
            "syco completion_logprob backfill missing on %d (source, bystander) cells; "
            "the per-coarse Spearman is NaN-safe.",
            n_missing_lp,
        )
    if syco_coarse_zoo_backfill is not None and n_missing_coarse_zoo:
        # Fail loud: the #480 substrate is built from the SAME 138 pairs
        # as the #411 syco panel by construction, so a missing cell is a
        # contract violation, not transient noise.
        raise RuntimeError(
            f"syco coarse-zoo backfill missing on {n_missing_coarse_zoo} "
            f"(source, bystander) cell(s) of the #411 panel. The #480 "
            f"substrate at {syco_coarse_zoo_backfill} is built from the "
            f"same 138 off-diagonal pairs by construction; any miss is a "
            f"contract violation. Re-build #480's predictor_comparison.json "
            f"or pass a file with full 138-cell coverage."
        )
    return {
        "rows": rows,
        "source_file": str(snapshot_path),
        "syco_logprob_backfill": str(syco_logprob_backfill) if syco_logprob_backfill else None,
        "syco_coarse_zoo_backfill": (
            str(syco_coarse_zoo_backfill) if syco_coarse_zoo_backfill else None
        ),
    }


# #518 v4 must-fix 1: required predictor-comparison fields per cell. The
# cross-behavior aggregator relies on ``completion_logprob`` being present on
# every (source, bystander) cell of every arm being scored; this loader FAILS
# CLOSED if the column is missing, naming the offending cell.
_REQUIRED_PREDICTOR_COMPARISON_FIELDS: tuple[str, ...] = (
    "source",
    "bystander",
    "delta",
    "completion_logprob",
)

# #518 v4 ROUND-2 must-fix 1: coarse predictor columns the syco/refusal/em
# arms can score per (point, layer, metric, variant) cell. ``completion_logprob``
# is the headline cross-behavior predictor named in plan §0/§1/§4.4/§11 ("the
# body of work this task is built around"); the rest are the #480 coarse-zoo
# columns the substrate already carries. Each is residualized by source FE
# (matching ``_score_one_cell``'s headline statistic) and a per-predictor
# Spearman ρ is reported in ``scored["per_coarse_rho_fe"]`` so the
# cross-behavior aggregator can build a `(rho_syco, rho_refusal, rho_em)`
# triple per coarse predictor alongside the residual-stream cell triples.
REFUSAL_EM_COARSE_PREDICTOR_COLUMNS: tuple[str, ...] = (
    "completion_logprob",
    "cosine_l20_baseline",
    "cosine_response_headline",
    "cosine_response_l7",
    "cosine_response_l14",
    "cosine_response_l21",
    "cosine_response_l27",
    "JS_sym_nats",
    "JS_from_source_nats",
    "JS_from_bystander_nats",
    "M_js",
    "KL_src_to_bys_nats",
    "KL_bys_to_src_nats",
    "KL_sym_nats",
    "source_base_rate",
    "bystander_base_rate",
    "base_rate_diff_neg_abs",
    "source_resp_len_mean",
    "bystander_resp_len_mean",
    "resp_len_diff_abs",
)


def _load_refusal_em_target(predictor_comparison_path: Path) -> dict[str, Any]:
    """Load the per-arm predictor_comparison.json substrate (refusal / EM).

    The schema matches ``eval_results/issue_480/_inputs/predictor_comparison.json``
    (per-cell dict with ``source``, ``bystander``, ``delta``,
    ``trained_rate``, ``bystander_base_rate``, ``source_base_rate``,
    ``cosine_l20_baseline``, ``cosine_response_*``, ``JS_*``, ``KL_*``,
    ``M_js``, ``resp_len_*``, ``completion_logprob``). For #518's refusal +
    EM arms, this loader replaces ``_load_syco_target`` -- the structural
    shape is identical (source × bystander Δ panel) but the substrate file
    is the build-step output rather than the #411 raw analyze_summary.

    Fails closed if ``completion_logprob`` is absent on any cell: the
    cross-behavior aggregator's headline computes ``min(|ρ|)`` across the
    three arms' completion_logprob cells, so a missing column on any arm
    structurally answers the universal-predictor Y/N as N/A.
    """
    with open(predictor_comparison_path) as f:
        payload = json.load(f)
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError(
            f"predictor_comparison.json at {predictor_comparison_path} "
            f"has no 'cells' list (got {type(cells).__name__})."
        )
    rows: list[dict[str, Any]] = []
    for i, cell in enumerate(cells):
        missing = [k for k in _REQUIRED_PREDICTOR_COMPARISON_FIELDS if k not in cell]
        if missing:
            raise RuntimeError(
                f"predictor_comparison cell #{i} at {predictor_comparison_path} "
                f"missing required field(s) {missing}. "
                f"#518 v4 must-fix 1: completion_logprob is required on every cell "
                f"of every arm being scored; the cross-behavior aggregator's "
                f"min(|rho|) headline is structurally unanswerable without it."
            )
        source = cell["source"]
        bystander = cell["bystander"]
        if source == bystander:
            continue  # off-diagonal only, matching the syco loader contract
        trained_rate = cell.get("trained_rate")
        base_rate = cell.get("bystander_base_rate", cell.get("base_rate"))
        # SE under the independence approximation matching the syco loader:
        # 50 probes × 10 rollouts = 500 rollouts per cell.
        if trained_rate is None or base_rate is None:
            se = float("nan")
        else:
            n_rollouts = 500
            se = float(
                np.sqrt(
                    max(trained_rate * (1 - trained_rate), 0.0) / n_rollouts
                    + max(base_rate * (1 - base_rate), 0.0) / n_rollouts
                )
            )
        # #518 v4 round-2 must-fix 1: carry every coarse predictor column the
        # substrate exposes onto the row dict so ``_coarse_lift_syco_arm`` can
        # residualize each one per cell. Missing columns become NaN -- the
        # downstream per-coarse Spearman is NaN-safe.
        row = {
            "source": source,
            "bystander": bystander,
            "delta": float(cell["delta"]),
            "trained_rate": trained_rate,
            "base_rate": base_rate,
            "se_delta": se,
            "completion_logprob": float(cell["completion_logprob"]),
        }
        for col in REFUSAL_EM_COARSE_PREDICTOR_COLUMNS:
            if col == "completion_logprob":
                continue  # already populated above
            v = cell.get(col)
            row[col] = float(v) if v is not None else float("nan")
        rows.append(row)
    if not rows:
        raise RuntimeError(
            f"predictor_comparison.json at {predictor_comparison_path} yielded 0 "
            f"off-diagonal rows after the source == bystander filter."
        )
    return {"rows": rows, "source_file": str(predictor_comparison_path)}


# ── Cell scoring core ─────────────────────────────────────────────────────


def _build_fact_xy(
    matrix: dict[str, dict[str, float]],
    fact_rows: list[dict[str, Any]],
    cid_to_csv_persona: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return (x, y, substrate, prior_z, cell_se, matched_indices) aligned across the 26 cells.

    ``matched_indices`` indexes into ``fact_rows`` so downstream
    (F6 coarse-predictor anchor) can look up the same rows for each of
    the 5 coarse-predictor columns.
    """
    persona_to_cid = {v: k for k, v in cid_to_csv_persona.items()}
    x_arr, y_arr, sub_arr, z_arr, se_arr = [], [], [], [], []
    matched_indices: list[int] = []
    for i, row in enumerate(fact_rows):
        teach_cid = persona_to_cid.get(row["teach_persona"])
        bys_cid = persona_to_cid.get(row["bystander_persona"])
        if teach_cid is None or bys_cid is None:
            continue
        d = matrix.get(teach_cid, {}).get(bys_cid)
        if d is None:
            continue
        x_arr.append(d)
        y_arr.append(row["leak_rate"])
        sub_arr.append(row["substrate"])
        z_arr.append(row["bystander_logprob"])
        se_arr.append(float("nan"))  # Per-seed reconstruction goes here (TODO inflow)
        matched_indices.append(i)
    return (
        np.array(x_arr, dtype=float),
        np.array(y_arr, dtype=float),
        np.array(sub_arr),
        np.array(z_arr, dtype=float),
        np.array(se_arr, dtype=float),
        matched_indices,
    )


def _build_syco_xy(
    matrix: dict[str, dict[str, float]],
    syco_rows: list[dict[str, Any]],
    cid_to_syco_persona: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return (x, y, source, se, bystander, matched_indices) aligned across the cells.

    ROUND-2/#509 FIX F6: also return per-cell bystander persona so the
    `comedian_recovery` rank diagnostic can index by bystander name.

    #518 v4 round-2 must-fix 1: ALSO return ``matched_indices`` (a list of
    integer indices into ``syco_rows``) so the per-coarse predictor lift
    can read the same coarse-predictor columns (``completion_logprob``,
    cosine_*, JS_*, KL_*, M_js, base rates, length residuals) from the SAME
    rows the bake-off cell's (x, y) come from. The fact arm has used this
    pattern since round 2 (``_build_fact_xy`` returns ``matched_indices``);
    the syco/refusal/em arm now mirrors it.
    """
    persona_to_cid = {v: k for k, v in cid_to_syco_persona.items()}
    x_arr, y_arr, src_arr, se_arr, bys_arr = [], [], [], [], []
    matched_indices: list[int] = []
    for i, row in enumerate(syco_rows):
        src_cid = persona_to_cid.get(row["source"])
        bys_cid = persona_to_cid.get(row["bystander"])
        if src_cid is None or bys_cid is None:
            continue
        d = matrix.get(src_cid, {}).get(bys_cid)
        if d is None:
            continue
        x_arr.append(d)
        y_arr.append(row["delta"])
        src_arr.append(row["source"])
        se_arr.append(row["se_delta"])
        bys_arr.append(row["bystander"])
        matched_indices.append(i)
    return (
        np.array(x_arr, dtype=float),
        np.array(y_arr, dtype=float),
        np.array(src_arr),
        np.array(se_arr, dtype=float),
        np.array(bys_arr),
        matched_indices,
    )


def _reliability_y_pooled(y: np.ndarray, se: np.ndarray) -> float:
    """LEGACY pooled-variance reliability — exported for test diff only.

    Computes ``1 - mean(SE^2) / var_pooled(y)``. This is the round-1 form
    that ROUND-2/#509 FIX F4 replaces with the within-stratum variant
    everywhere it's called for production scoring.
    """
    if not np.isfinite(se).any():
        return 1.0
    var_y = float(np.nanvar(y))
    mean_se2 = float(np.nanmean(se**2))
    if var_y <= 0:
        return 1.0
    return float(max(min(1.0 - mean_se2 / var_y, 1.0), 1e-6))


def _reliability_y(
    y: np.ndarray,
    se: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    allow_unknown_se: bool = True,
) -> float:
    """Reliability of ``y`` against measurement-error SE, within stratum.

    ROUND-2/#509 FIX F4: Plan §6.1 specifies the reliability denominator
    as the within-substrate (fact arm) / within-source (syco arm)
    variance of ``y``, NOT pooled across strata. The pooled formula
    overstates reliability whenever between-stratum mean differences
    dominate variance — which is exactly the syco arm regime: the #411
    panel has 6 sources with very different mean Δ, so pooled var(y)
    runs ~10× larger than within-source var(y) and the corresponding
    ``rho_fe_adj`` understates the attenuation correction toward the
    verdict threshold.

    ROUND-2/#509 FIX F5: when ``se`` is entirely non-finite, the legacy
    silent fallback to 1.0 (which causes ``rho_fe_adj == rho_fe``,
    bypassing the plan-required attenuation correction) is gated by
    ``allow_unknown_se``. Smoke-mode callers pass ``True`` so a tiny
    synthetic grid still runs; production callers (fact arm in
    particular) pass ``False`` so the missing SE surfaces loud instead
    of publishing an unadjusted statistic as the headline ``rho_fe_adj``.

    Implementation: when ``strata`` is provided, compute the pooled
    within-stratum variance ``var(y - stratum_mean(y))``; otherwise fall
    back to the pooled variance for back-compat with callers that have
    no notion of strata. Returns a value clipped to (1e-6, 1.0].
    """
    finite_se = np.isfinite(se).any() if se is not None else False
    if not finite_se:
        if not allow_unknown_se:
            raise ValueError(
                "_reliability_y: per-cell SE array is entirely non-finite and "
                "allow_unknown_se=False (production mode). Either supply SE "
                "(fact-arm: per-seed reconstruction from #444 / #192 raw "
                "completions; syco-arm: independence-approx already wired) "
                "or pass --smoke to fall back to reliability=1.0."
            )
        return 1.0
    if strata is not None and len(strata) == len(y):
        # Within-stratum variance: subtract per-stratum mean, then take
        # variance over the demeaned vector. This is the pooled within-
        # stratum variance (sum of within-cluster SS / N), per Plan §6.1.
        y = np.asarray(y, dtype=float)
        demeaned = y.copy()
        for s in np.unique(strata):
            idx = strata == s
            if idx.sum() == 0:
                continue
            demeaned[idx] = y[idx] - np.nanmean(y[idx])
        var_y = float(np.nanvar(demeaned))
    else:
        var_y = float(np.nanvar(y))
    mean_se2 = float(np.nanmean(se**2))
    if var_y <= 0:
        return 1.0
    return float(max(min(1.0 - mean_se2 / var_y, 1.0), 1e-6))


def _score_one_cell(
    *,
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    se: np.ndarray | None,
    prior_z: np.ndarray | None,
    run_permutation: bool,
    run_bootstrap: bool,
    perm_b: int,
    allow_unknown_se: bool = True,
) -> dict[str, Any]:
    """Compute one (point, layer, metric, variant) cell's scoring panel.

    strata = substrate (fact) or source (syco); used for FE-residualization,
    permutation null, cluster bootstrap, and delete-one jackknife.

    ``allow_unknown_se`` defaults to True for backward-compatibility with
    callers that never threaded the kwarg; the production fact-arm path
    sets it to False so an all-NaN SE array raises instead of silently
    yielding ``rho_fe_adj == rho_fe`` (Plan §6.1 attenuation correction
    is load-bearing for the verdict thresholds).

    ROUND-3/#509 G2: per plan §4.2.6 #3, pairs whose predictor distance
    sits at the floor of the metric's natural range are dropped from the
    regression BEFORE every downstream statistic (``rho_fe``, perm null,
    bootstrap CI, jackknife SE, LOCO-CV R^2). Round-2 wired
    ``_is_predictor_saturated`` as a whole-cell variance flag that
    reported the saturation state but never excluded any pair; this
    let saturated pairs corrupt the headline statistic and inference.
    G2 filters ``(x, y, strata, se, prior_z)`` via
    ``_saturated_pair_mask`` upfront and reports
    ``n_excluded_saturated`` per cell. When fewer than
    ``_MIN_SURVIVING_PAIRS`` survive the filter, the rank-correlation
    statistic is unstable, so ``rho_fe`` is emitted as NaN with
    ``saturation_too_aggressive: True`` so the analyzer can flag the
    cell rather than report a noise spike.
    """
    out: dict[str, Any] = {}
    # ROUND-3/#509 G2: per-pair saturation exclusion BEFORE any statistic.
    # See ``_saturated_pair_mask`` for the floor criterion; non-finite
    # entries are also dropped here. ``out["n_excluded_saturated"]`` is
    # always populated (0 when no pairs are dropped) so the analyzer can
    # surface it uniformly across cells.
    saturated_mask = _saturated_pair_mask(x)
    n_excluded_saturated = int(saturated_mask.sum())
    out["n_excluded_saturated"] = n_excluded_saturated
    if n_excluded_saturated > 0:
        keep = ~saturated_mask
        x = x[keep]
        y = y[keep]
        strata = strata[keep] if strata is not None else None
        if se is not None:
            se = se[keep]
        if prior_z is not None:
            prior_z = prior_z[keep]
    if len(x) < _MIN_SURVIVING_PAIRS:
        # Saturation removed too many pairs to compute a stable
        # rank-correlation statistic. Emit NaN + the diagnostic flag.
        out["saturation_too_aggressive"] = True
        out["rho_pooled"] = float("nan")
        out["rho_fe"] = float("nan")
        out["rho_double_fe"] = float("nan")
        out["reliability_y"] = float("nan")
        out["rho_pooled_adj"] = float("nan")
        out["rho_fe_adj"] = float("nan")
        out["loco_r2"] = float("nan")
        if run_permutation:
            out["perm_p_fe"] = float("nan")
        if run_bootstrap:
            out["ci_lo_fe"] = float("nan")
            out["ci_hi_fe"] = float("nan")
            out["jackknife_se_fe"] = float("nan")
        out["n"] = len(x)
        return out

    rho_pooled = _spearman_rho(x, y)
    # ROUND-2/#509 FIX F3: Plan §4.1.5 regression D specifies the FE
    # statistic as the partial Spearman ``ρ(x | s, y | s)`` — BOTH x and
    # y residualized within stratum, not just y. Round 1 only
    # residualized y; the resulting statistic still carried between-
    # stratum structure in x, biasing the headline and breaking
    # comparability with the (now-also-fixed) permutation null.
    x_resid = _residualize(x, strata)
    y_resid = _residualize(y, strata)
    rho_fe = _spearman_rho(x_resid, y_resid)
    out["rho_pooled"] = rho_pooled
    out["rho_fe"] = rho_fe
    if prior_z is not None and np.isfinite(prior_z).all():
        y_pz = _residualize(y - prior_z, strata) if len(strata) > 0 else (y - prior_z)
        # Same fix applies to the double-FE statistic: residualize x too.
        out["rho_double_fe"] = _spearman_rho(x_resid, y_pz)
    else:
        out["rho_double_fe"] = float("nan")
    # ROUND-2/#509 FIX F5: fact-arm scoring must NOT silently fall back
    # to reliability=1 when SE is missing. The caller forwards
    # ``allow_unknown_se`` (True in --smoke mode, False on production)
    # to ``_reliability_y`` via ``_score_one_cell``'s ``allow_unknown_se``
    # kwarg below. Strata are now load-bearing for F4.
    rel = (
        _reliability_y(y, se, strata=strata, allow_unknown_se=allow_unknown_se)
        if se is not None
        else 1.0
    )
    out["reliability_y"] = rel
    out["rho_pooled_adj"] = _attenuation_adjust(rho_pooled, rel)
    out["rho_fe_adj"] = _attenuation_adjust(rho_fe, rel)
    out["loco_r2"] = _loco_cv_r2(x, y, strata)
    if run_permutation:
        out["perm_p_fe"] = _permutation_p_partial(rho_fe, x, y, strata, perm_b)
    if run_bootstrap:
        # ROUND-2/#509 FIX F3: bootstrap + jackknife on the partial-Spearman
        # statistic — both x and y residualized within stratum.
        ci_lo, ci_hi = _cluster_bootstrap_ci(x_resid, y_resid, strata)
        out["ci_lo_fe"] = ci_lo
        out["ci_hi_fe"] = ci_hi
        out["jackknife_se_fe"] = _jackknife_se(x_resid, y_resid, strata)
    out["n"] = len(x)
    return out


def _matrix_to_dict(matrix_payload: dict[str, Any]) -> dict[str, dict[str, float]] | None:
    """Coerce a metric-phase JSON payload's matrix to a {a: {b: float}} dict.

    Returns ``None`` when the payload carries an explicit "matrix": null
    (the bake-off emits a None matrix + an ``n_a`` explanation field for
    cells where the cloud distance is undefined — e.g. ``end_of_system``
    extraction has one vector per condition, so c2st/gauss_kl/mahal/mmd/
    wass2/delta_spec/mahal_pooled_ctx have no distribution to compare).
    The caller must SKIP these cells; they are not scoring failures.
    """
    m = matrix_payload.get("matrix")
    if m is None:
        return None
    return {a: {b: float(v) for b, v in row.items()} for a, row in m.items()}


def _enumerate_metric_files(metrics_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """List every metric-phase output file + parsed (point, layer, metric, variant).

    ROUND-2/#509 FIX F2: filter out MMD permutation null sidecars
    (``*__perm.json``) + #406 cross-check sidecars
    (``*__cross_check_406.json``) — they share the prefix but are NOT
    predictor cells. Round 1 ingested ~112 of them per full #502-style
    metrics dir as ``variant=perm`` (or worse, accidentally as
    ``variant=centered__perm``), adding bogus empty cells and skewing
    every cross-cell summary.

    F2 also allows ``layer-1`` for the ``next_token_js`` baseline so the
    plan-required vocab-level baseline lands in the cells list rather
    than being silently dropped by the digit-only regex.
    """
    out: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(metrics_dir.glob("*.json")):
        if any(p.name.endswith(suffix) for suffix in _SIDECAR_FILENAME_SUFFIXES):
            continue
        m = METRIC_FILE_PATTERN.match(p.name)
        if not m:
            continue
        out.append(
            (
                p,
                {
                    "extraction_point": m.group("point"),
                    "layer": int(m.group("layer")),
                    "metric": m.group("metric"),
                    "variant": m.group("variant"),
                },
            )
        )
    return out


def score_arm(
    *,
    arm: str,
    metrics_dir: Path,
    target_file: Path,
    smoke: bool,
    syco_logprob_backfill: Path | None = None,
    syco_coarse_zoo_backfill: Path | None = None,
) -> dict[str, Any]:
    """Score every (point, layer, metric, variant) cell on one arm.

    #518 v4 round-2 must-fix 1: ``syco_logprob_backfill`` (syco arm only)
    points at ``scripts/issue518_syco_logprob_backfill.py``'s output
    (``logprob_results.json``). When provided, the per-(source, bystander)
    ``completion_logprob`` is merged onto the syco target rows so the
    syco-arm scoring emits ``per_coarse_rho_fe["completion_logprob"]``
    that the cross-behavior aggregator's headline reads.

    #518 v4 round-3 must-fix 2 (Option A): ``syco_coarse_zoo_backfill``
    (syco arm only) points at ``eval_results/issue_480/_inputs/
    predictor_comparison.json``. When provided, the 19 #480 coarse-zoo +
    base-rate + length-control columns are merged onto each syco row so
    the cross-arm aggregator's 20-predictor pass produces non-empty
    triples for every coarse predictor on syco (round 2 had completion_logprob
    only; the 19 other predictors all produced 0 triples).
    """
    if arm == "fact":
        target = _load_fact_target(target_file)
        from explore_persona_space.experiments.i509_fact_conditions import (
            CID_TO_CSV_PERSONA,
        )

        cid_to_persona = CID_TO_CSV_PERSONA
    elif arm == "syco":
        target = _load_syco_target(
            target_file,
            syco_logprob_backfill=syco_logprob_backfill,
            syco_coarse_zoo_backfill=syco_coarse_zoo_backfill,
        )
        from explore_persona_space.experiments.i509_syco_conditions import (
            CID_TO_SYCO_PERSONA,
        )

        cid_to_persona = CID_TO_SYCO_PERSONA
    elif arm == "refusal":
        # #518 v4: refusal arm reuses the syco code path (same source × bystander
        # panel shape) with the new predictor_comparison.json substrate loader
        # that asserts completion_logprob is present on every cell. Conditions
        # registry is i518_refusal_conditions (R1..R24).
        target = _load_refusal_em_target(target_file)
        from explore_persona_space.experiments.i518_refusal_conditions import (
            CID_TO_REFUSAL_PERSONA,
        )

        cid_to_persona = CID_TO_REFUSAL_PERSONA
    elif arm == "em":
        # #518 v4: EM arm reuses the syco code path (same source × bystander
        # panel shape) with the new predictor_comparison.json substrate loader
        # that asserts completion_logprob is present on every cell. Conditions
        # registry is i518_em_conditions (E1..E24).
        target = _load_refusal_em_target(target_file)
        from explore_persona_space.experiments.i518_em_conditions import (
            CID_TO_EM_PERSONA,
        )

        cid_to_persona = CID_TO_EM_PERSONA
    else:
        raise ValueError(f"Unknown arm {arm!r}; expected 'fact', 'syco', 'refusal', or 'em'")

    files = _enumerate_metric_files(metrics_dir)
    if not files:
        raise FileNotFoundError(f"No metric JSONs under {metrics_dir}")
    logger.info("Found %d metric files in %s", len(files), metrics_dir)

    perm_b = 50 if smoke else PERMUTATION_B
    # #518 v4 round-3 must-fix 1: per-coarse bootstrap budget matches the
    # smoke override (50) / production budget (BOOTSTRAP_B = 5000). Threaded
    # alongside ``perm_b`` into ``_coarse_lift_syco_arm_per_cell`` so the
    # headline-deciding per-coarse rho carries the same resolution as the
    # residual-stream rho. Round 2 hardcoded b=500 which graining the
    # cluster CI below the residual-stream pass's b=5000.
    coarse_bootstrap_b = 50 if smoke else BOOTSTRAP_B
    run_permutation = not smoke
    run_bootstrap = not smoke

    # ROUND-2/#509 FIX F5: smoke mode allows the reliability=1.0 fallback
    # so a synthetic 2-cond × 1-layer × 1-metric grid still runs; production
    # mode raises if the per-cell SE array is non-finite (fact arm needs
    # the per-seed SE reconstruction; until that lands, production fact
    # runs must explicitly pass --smoke or be flagged at promotion).
    allow_unknown_se = smoke

    cells: list[dict[str, Any]] = []
    skipped_na: list[dict[str, Any]] = []
    for path, meta in files:
        with open(path) as fh:
            payload = json.load(fh)
        matrix = _matrix_to_dict(payload)
        if matrix is None:
            # Intentional N/A cell — see _matrix_to_dict docstring.
            # Capture the bake-off's explanation in n_a (string) so the
            # skip is auditable in scoring.json.
            skipped_na.append({**meta, "n_a": payload.get("n_a", "matrix=None")})
            continue
        if arm == "fact":
            x, y, strata, prior_z, se, matched_indices = _build_fact_xy(
                matrix, target["rows"], cid_to_persona
            )
            scored = _score_one_cell(
                x=x,
                y=y,
                strata=strata,
                se=se,
                prior_z=prior_z,
                run_permutation=run_permutation,
                run_bootstrap=run_bootstrap,
                perm_b=perm_b,
                allow_unknown_se=allow_unknown_se,
            )
            # ROUND-2/#509 FIX F6: per-cell saturation flag for the fact arm.
            scored["predictor_saturated"] = bool(_is_predictor_saturated(x))
            # ROUND-2/#509 FIX F6: anchor 1 coarse-predictor lift on the
            # SAME pairs as the bake-off cell (plan §4.1.6.1).
            if len(x) >= 3:
                scored["coarse_lift"] = _coarse_lift_per_cell(
                    x, y, strata, target["rows"], cid_to_persona, matched_indices
                )
        else:
            x, y, strata, se, bystanders, matched_indices = _build_syco_xy(
                matrix, target["rows"], cid_to_persona
            )
            scored = _score_one_cell(
                x=x,
                y=y,
                strata=strata,
                se=se,
                prior_z=None,
                run_permutation=run_permutation,
                run_bootstrap=run_bootstrap,
                perm_b=perm_b,
                allow_unknown_se=allow_unknown_se,
            )
            # #518 v4 round-2 must-fix 1: per-coarse Spearman ρ on the SAME
            # pairs as the bake-off cell. The aggregator reads
            # ``per_coarse_rho_fe[<predictor>]["rho_fe_adj"]`` from this
            # payload per arm and builds the cross-behavior triple. The
            # headline coarse predictor is ``completion_logprob`` (plan §0
            # / §1 / §4.4 / §11 -- "the body of work this task is built
            # around"). Only fires when the substrate carries the coarse
            # columns (refusal/em arms always; syco arm when the
            # ``--syco-logprob-backfill`` flag was passed).
            if (
                arm in ("syco", "refusal", "em")
                and len(x) >= 3
                and target["rows"]
                and "completion_logprob" in target["rows"][0]
            ):
                scored["per_coarse_rho_fe"] = _coarse_lift_syco_arm_per_cell(
                    x,
                    y,
                    strata,
                    se,
                    target["rows"],
                    matched_indices,
                    allow_unknown_se=allow_unknown_se,
                    perm_b=perm_b,
                    bootstrap_b=coarse_bootstrap_b,
                )
                # #518 v4 round-3 must-fix 1: record the permutation +
                # bootstrap budgets used for the per-coarse Spearman so
                # the analyzer can audit the headline-deciding resolution
                # alongside the residual-stream cell's perm_null_b /
                # bootstrap_b at the top of the scoring JSON.
                scored["per_coarse_rho_fe_meta"] = {
                    "perm_b": int(perm_b),
                    "bootstrap_b": int(coarse_bootstrap_b),
                }
            # ROUND-2/#509 FIX F6 — syco-arm plan-§5 condition slugs:
            # `per_source`, `live_cells_only`, `comedian_recovery`,
            # `per_cell_predictor_saturation`.
            scored["per_source_rho"] = _per_source_spearman(x, y, strata)
            # Live-cells mask + ρ on the |Δ| > 0.10 subset.
            live_mask = _live_cells_mask(y, threshold=0.10)
            if live_mask.sum() >= 3:
                scored["live_cells_only"] = {
                    "n_cells": int(live_mask.sum()),
                    "rho_obs_live": _spearman_rho(x[live_mask], y[live_mask]),
                    "rho_fe_live": _spearman_rho(
                        _residualize(x[live_mask], strata[live_mask]),
                        _residualize(y[live_mask], strata[live_mask]),
                    ),
                }
            else:
                scored["live_cells_only"] = {"n_cells": int(live_mask.sum())}
            # Comedian rank inside software_engineer's bystanders by x.
            sw_mask = strata == "software_engineer"
            if sw_mask.sum() >= 3:
                sw_predictor = {
                    str(b): float(v) for b, v in zip(bystanders[sw_mask], x[sw_mask], strict=True)
                }
                scored["comedian_recovery"] = {
                    "rank": _rank_in_bystanders(
                        sw_predictor, target_persona="comedian", ascending=True
                    ),
                    "n_bystanders": len(sw_predictor),
                }
            scored["predictor_saturated"] = bool(_is_predictor_saturated(x))
        cells.append({**meta, **scored})

    summary = _summarize_cells(cells)
    # Sanity guard: a regression that turns EVERY file into N/A would
    # silently produce an empty scoring.json. Demand >= 25% scored.
    # The expected production fraction is ~63% (953 / 1513 fact-arm
    # files, the 560 remainder being the documented cloud-metric N/A
    # set at end_of_system + mahal_pooled_ctx).
    if files and len(cells) < max(1, len(files) // 4):
        raise RuntimeError(
            f"Scored only {len(cells)}/{len(files)} cells "
            f"({100 * len(cells) / len(files):.1f}%); expected >= 25%. "
            f"Skipped N/A: {len(skipped_na)}. Likely a regression in "
            f"_matrix_to_dict or the bake-off metric writers."
        )
    return {
        "schema_version": 1,
        "arm": arm,
        "smoke": smoke,
        "n_metric_files": len(files),
        "n_cells_scored": len(cells),
        "n_cells_skipped_na": len(skipped_na),
        "skipped_na": skipped_na,
        "anchors": {
            "nonstylized_rho_deltag": NONSTYLIZED_ANCHOR_RHO_DELTAG,
            "nonstylized_rho_glogp": NONSTYLIZED_ANCHOR_RHO_GLOGP,
            "full_panel_rho_deltag": FULL_PANEL_ANCHOR_RHO_DELTAG,
        },
        "perm_null_seed_int": _hashed_seed(PERMUTATION_NULL_SEED_TAG),
        "perm_null_b": perm_b,
        "bootstrap_b": BOOTSTRAP_B if run_bootstrap else 0,
        "summary": summary,
        "cells": cells,
        "target_source_file": target["source_file"],
        "git_sha": _git_sha(),
        "timestamp_utc": _now_iso(),
        "env": _env_versions(),
    }


def _summarize_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the L22 last_prompt gauss_kl anchor + L19-L24 ridge mean."""
    summary: dict[str, Any] = {}
    # Anchor cell.
    for c in cells:
        if (
            c["extraction_point"] == "last_prompt"
            and c["layer"] == 22
            and c["metric"] == "gauss_kl"
            and c.get("variant") == "centered"
        ):
            summary["anchor_L22_last_prompt_gauss_kl_centered"] = {
                "rho_fe": c.get("rho_fe"),
                "rho_fe_adj": c.get("rho_fe_adj"),
                "perm_p_fe": c.get("perm_p_fe"),
                "n": c.get("n"),
            }
            break
    # L19-L24 ridge.
    ridge_rhos: list[float] = []
    for c in cells:
        if (
            c["extraction_point"] == "last_prompt"
            and c["layer"] in L19_L24_RIDGE_LAYERS
            and c["metric"] in RIDGE_METRICS
            and c.get("variant") == "centered"
        ):
            rho = c.get("rho_fe_adj")
            if rho is not None and np.isfinite(rho):
                ridge_rhos.append(rho)
    if ridge_rhos:
        summary["ridge_L19_L24_mean_rho_fe_adj"] = float(np.mean(ridge_rhos))
        summary["ridge_L19_L24_n_cells"] = len(ridge_rhos)
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Issue #509 per-arm scoring against fact / syco leakage targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding per-(point, layer, metric, variant) distance "
            "JSONs (the metrics-phase output)."
        ),
    )
    p.add_argument(
        "--arm",
        choices=("fact", "syco", "refusal", "em"),
        required=True,
        help=(
            "Which arm to score against. #518 v4 added 'refusal' + 'em' that "
            "reuse the syco code path with the new predictor_comparison.json "
            "substrate loader (asserts completion_logprob is present on every "
            "cell -- FAIL CLOSED if missing, per cross-behavior aggregator "
            "prerequisite)."
        ),
    )
    p.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help=(
            "Target leakage matrix. Fact arm: eval_results/issue_494/regression_data.csv. "
            "Syco arm: eval_results/issue_480/_inputs/syco_411_analyze_summary.json. "
            "Refusal/EM arm: eval_results/issue_518/<arm>/_inputs/predictor_comparison.json. "
            "When unset, defaults are filled in per --arm."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path for the per-arm scoring panel.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: skip permutation + bootstrap, relax cell-count gates.",
    )
    p.add_argument(
        "--syco-logprob-backfill",
        type=Path,
        default=None,
        help=(
            "Syco arm only. Path to scripts/issue518_syco_logprob_backfill.py's "
            "logprob_results.json output. When provided, the per-(source, "
            "bystander) completion_logprob is merged onto the syco target so "
            "the syco-arm scoring emits per_coarse_rho_fe['completion_logprob']. "
            "Required for the cross-behavior aggregator's headline coarse "
            "predictor (plan §0/§1/§4.4/§11). Ignored on fact / refusal / em "
            "arms (refusal + EM substrates carry the column directly)."
        ),
    )
    p.add_argument(
        "--syco-coarse-zoo-backfill",
        type=Path,
        default=None,
        help=(
            "Syco arm only. Path to a predictor_comparison.json containing the "
            "19 coarse-zoo + base-rate + length-control fields for the syco arm "
            "(e.g. eval_results/issue_480/_inputs/predictor_comparison.json). "
            "Required for full plan-§4.4 cross-arm coarse-predictor coverage on "
            "syco (round-3 must-fix 2 Option A). If omitted, syco's coarse "
            "triples are restricted to completion_logprob -- the 19 other "
            "coarse predictors produce 0 triples on syco because the #411 "
            "panel does not carry them natively. Ignored on fact / refusal / "
            "em arms."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    target_file = args.target_file
    if target_file is None:
        if args.arm == "fact":
            target_file = PROJECT_ROOT / "eval_results/issue_494/regression_data.csv"
        elif args.arm == "syco":
            target_file = (
                PROJECT_ROOT / "eval_results/issue_480/_inputs/syco_411_analyze_summary.json"
            )
        else:
            # #518 v4: refusal/em arms read the per-arm predictor_comparison.json
            # substrate built by scripts/issue518_build_predictor_substrate.py.
            target_file = (
                PROJECT_ROOT
                / f"eval_results/issue_518/{args.arm}/_inputs/predictor_comparison.json"
            )
    if not target_file.exists():
        logger.error("Target file missing: %s", target_file)
        return 2
    if not args.metrics_dir.exists():
        logger.error("Metrics dir missing: %s", args.metrics_dir)
        return 2

    syco_lp_backfill = args.syco_logprob_backfill
    if syco_lp_backfill is not None and args.arm != "syco":
        logger.warning(
            "--syco-logprob-backfill ignored on arm=%s (only syco arm reads it; "
            "refusal/em substrates carry completion_logprob directly).",
            args.arm,
        )
        syco_lp_backfill = None
    if syco_lp_backfill is not None and not syco_lp_backfill.exists():
        logger.error("--syco-logprob-backfill missing: %s", syco_lp_backfill)
        return 2
    # #518 v4 round-3 must-fix 2 (Option A): syco coarse-zoo backfill.
    syco_cz_backfill = args.syco_coarse_zoo_backfill
    if syco_cz_backfill is not None and args.arm != "syco":
        logger.warning(
            "--syco-coarse-zoo-backfill ignored on arm=%s (only syco arm reads "
            "it; refusal/em substrates carry the coarse zoo directly).",
            args.arm,
        )
        syco_cz_backfill = None
    if syco_cz_backfill is not None and not syco_cz_backfill.exists():
        logger.error("--syco-coarse-zoo-backfill missing: %s", syco_cz_backfill)
        return 2
    out = score_arm(
        arm=args.arm,
        metrics_dir=args.metrics_dir,
        target_file=target_file,
        smoke=args.smoke,
        syco_logprob_backfill=syco_lp_backfill,
        syco_coarse_zoo_backfill=syco_cz_backfill,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str))
    logger.info(
        "Wrote %s: arm=%s cells=%d smoke=%s",
        args.output,
        args.arm,
        out["n_cells_scored"],
        args.smoke,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
