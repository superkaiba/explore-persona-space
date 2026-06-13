# ruff: noqa: RUF001, RUF002, RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 Phase 3 — H1 + H2 stats package + 6 figures.

Reads per-source marker logprob JSONs from
``<slab_root>/per_source/<src>/seed_<seed>/marker_logprob_eval.json``,
pivots into a 138-cell matrix, inner-joins with #470's frozen
``predictor_comparison.json``, computes the 5 required analysis components
(see plan §6), and writes 6 figures.

Required analysis components (plan §6 Must-Fix list):
  1. H2 paired test ``paired_delta_rho``: per source Δρ = ρ_marker − ρ_syco;
     paired bootstrap (n=10000) + Wilcoxon over 6 sources. Uses #411's
     ACTUAL per-source sycophancy ρ.
  2. H2 power-matched ``power_matched_paired_delta_rho``: re-run after
     equalizing per-source SNR (Gaussian-noise injection on marker-Δ to
     match each source's sycophancy-Δ within-cell std). The behavior-type
     headline is licensed only if the differential survives this.
  3. H1 response-length partial ``cell_spearman_source_fe_base_rate_resp_len_partial``:
     partial source dummies + source/bystander base_rate + per-cell
     R_trained token-length mean + #470's source/bystander_resp_len_mean.
  4. KL secondary DV diagnostic-only handling: NOT silently swapped in
     for H1/H2 if saturation guard fires (we ONLY annotate saturation here
     since #480's primary DV is marker log-prob; KL is conditional and
     would be computed by Phase 2b if the saturation guard triggers — out
     of scope for this analyzer unless a KL artifact is present).
  5. Standard H1+H2 package: raw / source-FE / source-FE+base-rate Spearman
     (bootstrap+permutation n=10000); H2 per-source within-source Spearman
     with per-source cosine std; source-level n=6 descriptive.

CPU-only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("issue_480.analyze")

BOOTSTRAP_N = 10000
PERMUTATION_N = 10000
SEED = 42
H1_RHO_THRESHOLD = 0.20
# Verdict count per cell in #411 (50 held-out wrong-claims × 10 rollouts each;
# see #411 awaiting_promotion body Reproducibility). Used to compute the
# binomial SE for the sycophancy-Δ per cell, which feeds the noise-tolerant
# ranking power-match in _h2_power_matched.
N_VERDICTS_411 = 500
# Tie-tolerance multiplier on the per-cell measurement SE. A pair of cells is
# treated as a TIE in the noise-tolerant Spearman if the values differ by less
# than this multiple of the sum of their per-cell SEs. 2.0 ≈ 95% one-sided
# Gaussian band — the default in the noise-tolerant ranking literature.
TIE_TOLERANCE_SE_MULT = 2.0

# Per-source sycophancy ρ (frozen from #411 analyze_summary.json).
RHO_SYCO_411: dict[str, float] = {
    "villain": 0.4376856740472904,
    "comedian": 0.4449939419156868,
    "assistant": 0.2739862863527671,
    "qwen_default": -0.17350471719378502,
    "software_engineer": -0.34494688475848884,
    "kindergarten_teacher": 0.5714330706358673,
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ via scipy (handle NaN safely)."""
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)


def _noise_tolerant_ranks(
    values: np.ndarray, ses: np.ndarray, mult: float = TIE_TOLERANCE_SE_MULT
) -> np.ndarray:
    """Rank ``values`` with a measurement-noise tie band.

    Two cells i and j are treated as TIED in the rank order if
    ``|values[i] − values[j]| < mult * (ses[i] + ses[j])`` — i.e. their
    values are not resolvable above measurement noise. Tied cells get the
    midpoint rank within their equivalence class (the standard fractional-
    rank convention used by Spearman's ρ on tied data).

    Implementation: sort by value, sweep, and merge consecutive elements
    whose unresolved gap to the running cluster-mean is below the band into
    the same cluster. Each cluster's members are assigned the cluster's
    midpoint fractional rank.

    Returns an array of fractional ranks (same shape as ``values``).

    This is intentionally a *measurement-noise* tie band — NOT the same as
    scipy's default tie handling (which only ties exact equality). It
    degrades the effective rank resolution to each DV's own per-cell
    measurement precision; pairs of cells whose values are within noise
    contribute nothing to the ρ.
    """
    n = len(values)
    if n != len(ses):
        raise ValueError(f"values/ses length mismatch: {n} vs {len(ses)}")
    if n == 0:
        return np.zeros(0)
    # Sort by value; remember inverse permutation to write ranks back in
    # the original order.
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    sorted_ses = ses[order]
    # Greedy clustering: start a new cluster whenever the gap from the
    # previous element to the current element exceeds the band built from
    # their two SEs.
    cluster_ids = np.zeros(n, dtype=np.int64)
    cluster_id = 0
    for i in range(1, n):
        gap = sorted_vals[i] - sorted_vals[i - 1]
        band = mult * (sorted_ses[i - 1] + sorted_ses[i])
        if gap >= band:
            cluster_id += 1
        cluster_ids[i] = cluster_id
    # Fractional rank within each cluster: members share the cluster's
    # midpoint rank (the standard tied-rank convention). Position i in the
    # sorted array would receive rank i+1; tied members get the average of
    # their would-be ranks.
    sorted_ranks = np.empty(n, dtype=np.float64)
    pos = 0
    while pos < n:
        cid = cluster_ids[pos]
        end = pos
        while end < n and cluster_ids[end] == cid:
            end += 1
        # Members at indices [pos, end) — assign the average rank in 1-indexed
        # convention (Spearman's ρ uses 1-based ranks, but the constant is
        # absorbed by mean-centering inside Pearson; the average matters).
        mid_rank = (pos + 1 + end) / 2.0
        sorted_ranks[pos:end] = mid_rank
        pos = end
    # Invert the sort permutation.
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def _spearman_rho_tie_tolerant(
    x: np.ndarray,
    y: np.ndarray,
    x_se: np.ndarray,
    y_se: np.ndarray,
    mult: float = TIE_TOLERANCE_SE_MULT,
) -> float:
    """Spearman ρ on noise-tolerant ranks (tie band = ``mult * (SE_i + SE_j)``).

    Replaces scipy's exact-equality tie handling with a measurement-noise
    tie band, applied to BOTH variables. Then the ρ is Pearson on the
    resulting fractional ranks. Returns NaN if fewer than 3 finite pairs.
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(x_se) & np.isfinite(y_se)
    if mask.sum() < 3:
        return float("nan")
    rx = _noise_tolerant_ranks(x[mask], x_se[mask], mult=mult)
    ry = _noise_tolerant_ranks(y[mask], y_se[mask], mult=mult)
    # Pearson on the tied ranks → tied-Spearman by construction.
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt((rx_c * rx_c).sum() * (ry_c * ry_c).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx_c * ry_c).sum() / denom)


def _bootstrap_ci(
    fn, x: np.ndarray, y: np.ndarray, n: int = BOOTSTRAP_N, alpha: float = 0.05, seed: int = SEED
) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_lo_2.5, ci_hi_97.5)."""
    rng = np.random.default_rng(seed)
    point = fn(x, y)
    rhos = np.empty(n, dtype=np.float64)
    N = len(x)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        rhos[i] = fn(x[idx], y[idx])
    lo = float(np.nanpercentile(rhos, 100 * alpha / 2))
    hi = float(np.nanpercentile(rhos, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def _permutation_p(
    fn, x: np.ndarray, y: np.ndarray, n: int = PERMUTATION_N, seed: int = SEED
) -> float:
    """Two-sided permutation p — shuffle y, recompute, count >= |obs|."""
    rng = np.random.default_rng(seed + 1)
    obs = abs(fn(x, y))
    perm = np.empty(n, dtype=np.float64)
    y_copy = np.array(y, dtype=np.float64)
    for i in range(n):
        rng.shuffle(y_copy)
        perm[i] = abs(fn(x, y_copy))
    return float((perm >= obs).mean())


def _stratified_permutation_p(
    fn,
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    n: int = PERMUTATION_N,
    seed: int = SEED,
) -> float:
    """Permutation within strata (shuffle y values within each source group)."""
    rng = np.random.default_rng(seed + 2)
    obs = abs(fn(x, y))
    perm = np.empty(n, dtype=np.float64)
    for i in range(n):
        y_shuf = y.copy()
        for s in np.unique(strata):
            mask = strata == s
            idx = np.where(mask)[0]
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            y_shuf[idx] = y[shuffled]
        perm[i] = abs(fn(x, y_shuf))
    return float((perm >= obs).mean())


def _residualize(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """OLS residualization of y on covariates (returns residuals)."""
    # Add intercept column.
    X = np.column_stack([np.ones(len(y)), covariates])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return y - yhat


def _build_source_dummies(sources: np.ndarray) -> np.ndarray:
    """K-1 dummy matrix for source FE (drops the first unique source)."""
    uniq = sorted(set(sources.tolist()))
    if len(uniq) < 2:
        return np.zeros((len(sources), 0))
    # Drop the first sorted unique value (uniq[0]) as the reference level — the
    # remaining K-1 dummies form an identifiable design matrix with the intercept.
    cols = []
    for s in uniq[1:]:
        cols.append((sources == s).astype(np.float64))
    if not cols:
        return np.zeros((len(sources), 0))
    return np.column_stack(cols)


def _load_marker_matrix(slab_root: Path, seed: int) -> list[dict]:
    """Pivot per-source per-panel logprob JSONs into a list of rows."""
    rows: list[dict] = []
    per_source_dir = slab_root / "per_source"
    if not per_source_dir.exists():
        raise FileNotFoundError(f"per_source dir missing: {per_source_dir}")
    for source_dir in sorted(per_source_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        eval_path = source_dir / f"seed_{seed}" / "marker_logprob_eval.json"
        if not eval_path.exists():
            log.warning("missing eval for source=%s (%s) — skipping", source, eval_path)
            continue
        with open(eval_path) as f:
            payload = json.load(f)
        for panel, stats in payload["per_panel"].items():
            # marker_delta_se: SEM of the per-cell marker-Δ (added by phase2b
            # for noise-tolerant ranking in the power-match). Fall back to a
            # std-based pseudo-SE if not present (older phase2b outputs).
            n_q = int(stats.get("n_q", 0)) or 1
            if "marker_delta_se" in stats:
                m_se = float(stats["marker_delta_se"])
            elif "marker_delta_std" in stats:
                m_se = float(stats["marker_delta_std"]) / float(np.sqrt(max(1, n_q)))
            else:
                m_se = 0.0
            row = {
                "source": source,
                "bystander": panel,
                "marker_delta": stats["median_marker_delta"],
                "marker_delta_se": m_se,
                "emission_rate": stats["mean_emission_rate"],
                "log_p_trained": stats["median_log_p_trained"],
                "log_p_base": stats["median_log_p_base"],
                "r_trained_len_mean": stats["r_trained_len_mean"],
                "r_trained_len_median": stats["r_trained_len_median"],
                "n_q": n_q,
            }
            # Four-float storage-contract aggregates (#530, band-stop rerun
            # phase2b) — additive pass-through; absent on legacy phase2b
            # outputs, in which case the keys are simply not added (the
            # logp/logit agreement summary degrades gracefully).
            for agg_key in (
                "median_z_marker_trained",
                "median_z_eos_trained",
                "median_logZ_trained",
                "median_z_marker_base",
                "median_z_eos_base",
                "median_logZ_base",
                "median_eos_margin_delta",
                "median_delta_z_marker",
            ):
                if agg_key in stats:
                    row[agg_key.replace("median_", "", 1)] = stats[agg_key]
            rows.append(row)
    return rows


def _join_with_predictor(marker_rows: list[dict], predictor_path: Path) -> tuple[list[dict], dict]:
    """Inner-join marker rows with #470's predictor_comparison.json on (source, bystander).

    Drops self rows (source == bystander) so the 138-cell matched-cell matrix
    forms cleanly. Returns (joined_rows, metadata).
    """
    with open(predictor_path) as f:
        pred = json.load(f)
    pred_by_pair: dict[tuple[str, str], dict] = {
        (c["source"], c["bystander"]): c for c in pred["cells"]
    }
    joined: list[dict] = []
    n_marker_self = 0
    n_unmatched = 0
    for row in marker_rows:
        if row["source"] == row["bystander"]:
            n_marker_self += 1
            continue
        key = (row["source"], row["bystander"])
        if key not in pred_by_pair:
            n_unmatched += 1
            continue
        p = pred_by_pair[key]
        # Binomial SE for the sycophancy delta = trained_rate − base_rate, each
        # estimated from N_verdicts = 500 in #411 (50 held-out claims × 10
        # rollouts/cell, see the #411 awaiting_promotion body Reproducibility
        # table). SE(delta) = sqrt(SE(trained)² + SE(base)²) under the standard
        # independent-binomial assumption (training and base are evaluated on
        # the same questions but independent forward passes — the same-question
        # correlation lowers SE(delta) slightly; treating them as independent
        # is a conservative upper bound, which is what the noise-tolerant
        # ranking wants for tie-tolerance).
        n_verdicts_411 = N_VERDICTS_411
        p_trained = p.get("trained_rate_411", p.get("delta", 0.0) + p["bystander_base_rate"])
        p_base = p["bystander_base_rate"]
        se_trained = float(np.sqrt(max(0.0, p_trained * (1.0 - p_trained)) / n_verdicts_411))
        se_base = float(np.sqrt(max(0.0, p_base * (1.0 - p_base)) / n_verdicts_411))
        syco_delta_se = float(np.sqrt(se_trained**2 + se_base**2))
        joined.append(
            {
                **row,
                "sycophancy_delta": p["delta"],
                "sycophancy_delta_se": syco_delta_se,
                "cosine_l20_baseline": p["cosine_l20_baseline"],
                "source_base_rate": p["source_base_rate"],
                "bystander_base_rate": p["bystander_base_rate"],
                "source_resp_len_mean_411": p["source_resp_len_mean"],
                "bystander_resp_len_mean_411": p["bystander_resp_len_mean"],
            }
        )
    meta = {
        "n_marker_rows_input": len(marker_rows),
        "n_marker_self_dropped": n_marker_self,
        "n_unmatched_pred": n_unmatched,
        "n_joined": len(joined),
    }
    return joined, meta


def _h1_stats(joined: list[dict]) -> dict:
    """All H1 cell-level Spearman ρ flavors (plan §6).

    Returns the standard 5-flavor package (raw / source-FE / +base-rate /
    +response-length / source-level n=6 descriptive), plus the gated H1
    ``verdict``. Per plan §0 line 16 + §6 line 270, ``supported`` REQUIRES
    BOTH source-FE-residualized ρ to clear the threshold with CI excluding
    0 AND the response-length partial to survive the same test. A
    source-FE pass that the response-length partial overturns yields
    ``shared_context_length_nuisance`` (the plan's named alternative);
    ``falsified`` for a clear zero; ``inconclusive`` otherwise.
    """
    if not joined:
        return {"error": "no joined rows"}
    syco = np.array([r["sycophancy_delta"] for r in joined], dtype=np.float64)
    mark = np.array([r["marker_delta"] for r in joined], dtype=np.float64)
    src = np.array([r["source"] for r in joined])
    src_dummies = _build_source_dummies(src)
    base_src = np.array([r["source_base_rate"] for r in joined], dtype=np.float64)
    base_bys = np.array([r["bystander_base_rate"] for r in joined], dtype=np.float64)
    rlen = np.array([r["r_trained_len_mean"] for r in joined], dtype=np.float64)
    rlen_src411 = np.array([r["source_resp_len_mean_411"] for r in joined], dtype=np.float64)
    rlen_bys411 = np.array([r["bystander_resp_len_mean_411"] for r in joined], dtype=np.float64)

    def _fit_resid(y, covs):
        if covs.shape[1] == 0:
            return y - y.mean()
        return _residualize(y, covs)

    # 1. RAW Spearman ρ.
    point_raw, lo_raw, hi_raw = _bootstrap_ci(_spearman_rho, syco, mark)
    p_raw = _permutation_p(_spearman_rho, syco, mark)

    # 2. Source-FE residualized.
    syco_r1 = _fit_resid(syco, src_dummies)
    mark_r1 = _fit_resid(mark, src_dummies)
    point_fe, lo_fe, hi_fe = _bootstrap_ci(_spearman_rho, syco_r1, mark_r1)
    p_fe = _stratified_permutation_p(_spearman_rho, syco_r1, mark_r1, src)

    # 3. Source-FE + base-rate partial.
    covs_br = np.column_stack([src_dummies, base_src, base_bys])
    syco_r2 = _fit_resid(syco, covs_br)
    mark_r2 = _fit_resid(mark, covs_br)
    point_br, lo_br, hi_br = _bootstrap_ci(_spearman_rho, syco_r2, mark_r2)
    p_br = _stratified_permutation_p(_spearman_rho, syco_r2, mark_r2, src)

    # 4. Source-FE + base-rate + response-length partial (Must-Fix #3).
    covs_rl = np.column_stack([src_dummies, base_src, base_bys, rlen, rlen_src411, rlen_bys411])
    syco_r3 = _fit_resid(syco, covs_rl)
    mark_r3 = _fit_resid(mark, covs_rl)
    point_rl, lo_rl, hi_rl = _bootstrap_ci(_spearman_rho, syco_r3, mark_r3)
    p_rl = _stratified_permutation_p(_spearman_rho, syco_r3, mark_r3, src)

    # 5. Source-level n=6 descriptive.
    per_source: dict[str, dict[str, float]] = {}
    for s in sorted(set(src.tolist())):
        m = src == s
        per_source[s] = {
            "n_cells": int(m.sum()),
            "mean_marker_delta": float(mark[m].mean()),
            "mean_sycophancy_delta": float(syco[m].mean()),
        }
    src_levels = sorted(set(src.tolist()))
    src_marker_means = np.array([per_source[s]["mean_marker_delta"] for s in src_levels])
    src_syco_means = np.array([per_source[s]["mean_sycophancy_delta"] for s in src_levels])
    sl_point, sl_lo, sl_hi = _bootstrap_ci(_spearman_rho, src_syco_means, src_marker_means)

    # Supported / falsified verdicts (plan §3 H1 thresholds + §0 line 16 +
    # §6 line 270: "supported = source-FE-residualized CI excludes 0 AND ρ
    # survives the response-length partial"). The response-length partial
    # MUST gate the headline; if source-FE survives but the partial collapses,
    # the interpretation is "shared context-length nuisance", NOT supported.
    source_fe_pass = (lo_fe > 0.0) and (point_fe >= H1_RHO_THRESHOLD)
    resp_len_pass = (lo_rl > 0.0) and (point_rl >= H1_RHO_THRESHOLD)
    supported = source_fe_pass and resp_len_pass
    falsified = (point_fe < 0.10) and (lo_fe <= 0.0 <= hi_fe)
    if supported:
        verdict = "supported"
    elif source_fe_pass and not resp_len_pass:
        # Source-FE survives but response-length partial collapses → the plan's
        # explicit "shared context-length nuisance" interpretation.
        verdict = "shared_context_length_nuisance"
    elif falsified:
        verdict = "falsified"
    else:
        verdict = "inconclusive"

    return {
        "n_cells": len(joined),
        "cell_spearman_raw": {"rho": point_raw, "ci_lo": lo_raw, "ci_hi": hi_raw, "perm_p": p_raw},
        "cell_spearman_source_fe": {
            "rho": point_fe,
            "ci_lo": lo_fe,
            "ci_hi": hi_fe,
            "perm_p": p_fe,
        },
        "cell_spearman_source_fe_base_rate_partial": {
            "rho": point_br,
            "ci_lo": lo_br,
            "ci_hi": hi_br,
            "perm_p": p_br,
        },
        "cell_spearman_source_fe_base_rate_resp_len_partial": {
            "rho": point_rl,
            "ci_lo": lo_rl,
            "ci_hi": hi_rl,
            "perm_p": p_rl,
        },
        "source_level_spearman": {
            "rho": sl_point,
            "ci_lo": sl_lo,
            "ci_hi": sl_hi,
            "n_sources": len(src_levels),
        },
        "per_source_descriptive": per_source,
        "verdict": verdict,
        "supported_threshold": H1_RHO_THRESHOLD,
    }


def _h2_within_source(joined: list[dict]) -> dict:
    """Per-source within-source Spearman ρ(cosine_l20, marker-Δ) over 23 bystanders."""
    out: dict[str, dict[str, float | bool]] = {}
    for s in sorted({r["source"] for r in joined}):
        rows = [r for r in joined if r["source"] == s]
        cos = np.array([r["cosine_l20_baseline"] for r in rows], dtype=np.float64)
        mark = np.array([r["marker_delta"] for r in rows], dtype=np.float64)
        if len(cos) < 3:
            out[s] = {"rho": float("nan"), "n": len(cos), "error": "n<3"}
            continue
        point, lo, hi = _bootstrap_ci(_spearman_rho, cos, mark)
        p_val = _permutation_p(_spearman_rho, cos, mark)
        out[s] = {
            "rho": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "perm_p": p_val,
            "n": len(cos),
            "cosine_l20_std": float(cos.std(ddof=1)),
            "marker_delta_std": float(mark.std(ddof=1)),
            "gradient_descriptive_pass": bool(abs(point) >= 0.40 and (lo > 0.0 or hi < 0.0)),
        }
    return out


def _h2_paired_delta_rho(within_source: dict, rho_syco: dict) -> dict:
    """Per-source Δρ = ρ_marker − ρ_syco; paired bootstrap + Wilcoxon over 6."""
    from scipy.stats import wilcoxon

    sources = sorted(set(within_source.keys()) & set(rho_syco.keys()))
    rho_m = np.array([within_source[s]["rho"] for s in sources], dtype=np.float64)
    rho_s = np.array([rho_syco[s] for s in sources], dtype=np.float64)
    deltas = rho_m - rho_s

    # Paired bootstrap on the 6 deltas.
    rng = np.random.default_rng(SEED + 3)
    boot = np.empty(BOOTSTRAP_N, dtype=np.float64)
    N = len(deltas)
    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, N, size=N)
        boot[i] = deltas[idx].mean()
    mean_delta = float(deltas.mean())
    ci_lo = float(np.nanpercentile(boot, 2.5))
    ci_hi = float(np.nanpercentile(boot, 97.5))

    # Wilcoxon signed-rank on the 6 deltas (n=6 is small; reported descriptively).
    try:
        wilcox_stat, wilcox_p = wilcoxon(deltas)
        wilcox_stat = float(wilcox_stat)
        wilcox_p = float(wilcox_p)
    except ValueError:
        # all-zero / not enough non-zero diffs
        wilcox_stat = float("nan")
        wilcox_p = float("nan")

    return {
        "sources": sources,
        "rho_marker_per_source": [float(x) for x in rho_m.tolist()],
        "rho_syco_per_source": [float(x) for x in rho_s.tolist()],
        "delta_rho_per_source": [float(x) for x in deltas.tolist()],
        "mean_delta_rho": mean_delta,
        "paired_bootstrap_ci_lo": ci_lo,
        "paired_bootstrap_ci_hi": ci_hi,
        "wilcoxon_stat": wilcox_stat,
        "wilcoxon_p": wilcox_p,
        "n_sources": len(sources),
    }


def _h2_power_matched(joined: list[dict]) -> dict:
    """Power-matched paired Δρ via NOISE-TOLERANT RANKING (the preferred approach
    in the round-2 spec; replaces the prior variance-subtraction formula that was
    a no-op when marker had higher cross-cell variance than sycophancy — exactly
    the floored-sycophancy regime the FATAL-confound guard exists to catch).

    The intuition: rank-shuffles among values that differ by less than their
    measurement noise are uninformative. We compute the within-source Spearman ρ
    for BOTH DVs (marker-Δ and sycophancy-Δ) against the SAME per-cell predictor
    (``cosine_l20_baseline``), with tie tolerance ``2 × (SE_i + SE_j)``:

      - per-cell marker-Δ SE: SEM of the per-cell marker-Δ over 50 Q_eval
        (logged by phase2b as ``marker_delta_se``).
      - per-cell sycophancy-Δ SE: binomial SE = sqrt(p(1-p)/N_verdicts) for both
        the trained-rate and base-rate components of the delta, combined under
        the independent-binomial assumption (``sycophancy_delta_se``, computed
        in ``_join_with_predictor`` from #411's 500 verdicts/cell).

    Each DV is degraded to its OWN measurement floor, so a surviving differential
    in Δρ reflects resolvable rank gradient, not raw dynamic-range advantage.

    The headline ``behavior_type_headline_licensed`` is True iff the paired
    bootstrap CI (n=10000) over the 6 per-source Δρ values excludes 0.
    """
    sources = sorted({r["source"] for r in joined})
    rho_m_match: dict[str, float] = {}
    rho_s_match: dict[str, float] = {}
    diag_per_source: dict[str, dict[str, float | int]] = {}
    for s in sources:
        rows = [r for r in joined if r["source"] == s]
        cos = np.array([r["cosine_l20_baseline"] for r in rows], dtype=np.float64)
        # Cosine is computed from frozen base-model residual streams — treat it
        # as a noiseless predictor for tie purposes (SE ≈ 0 ⇒ ordinary ranks).
        cos_se = np.zeros_like(cos)
        mark = np.array([r["marker_delta"] for r in rows], dtype=np.float64)
        mark_se = np.array([r["marker_delta_se"] for r in rows], dtype=np.float64)
        syco = np.array([r["sycophancy_delta"] for r in rows], dtype=np.float64)
        syco_se = np.array([r["sycophancy_delta_se"] for r in rows], dtype=np.float64)
        rho_m_match[s] = _spearman_rho_tie_tolerant(cos, mark, cos_se, mark_se)
        rho_s_match[s] = _spearman_rho_tie_tolerant(cos, syco, cos_se, syco_se)
        diag_per_source[s] = {
            "n_cells": len(rows),
            "marker_delta_mean_se": float(np.nanmean(mark_se)) if len(mark_se) else 0.0,
            "syco_delta_mean_se": float(np.nanmean(syco_se)) if len(syco_se) else 0.0,
            "marker_delta_std": float(np.nanstd(mark, ddof=1)) if len(mark) >= 2 else 0.0,
            "syco_delta_std": float(np.nanstd(syco, ddof=1)) if len(syco) >= 2 else 0.0,
        }

    sources_common = sorted(sources)
    rho_m = np.array([rho_m_match[s] for s in sources_common], dtype=np.float64)
    rho_s = np.array([rho_s_match[s] for s in sources_common], dtype=np.float64)

    # When a per-source ρ is NaN (the tie band collapsed all cells into a
    # single equivalence class — i.e. the DV is unresolved at that source's
    # measurement floor), substitute 0.0 (no information ⇒ rank correlation
    # equivalent to chance) so the paired bootstrap doesn't degenerate.
    rho_m_filled = np.where(np.isfinite(rho_m), rho_m, 0.0)
    rho_s_filled = np.where(np.isfinite(rho_s), rho_s, 0.0)
    deltas_filled = rho_m_filled - rho_s_filled
    n_nan_marker = int(np.sum(~np.isfinite(rho_m)))
    n_nan_syco = int(np.sum(~np.isfinite(rho_s)))

    rng = np.random.default_rng(SEED + 5)
    boot = np.empty(BOOTSTRAP_N, dtype=np.float64)
    N = len(deltas_filled)
    for i in range(BOOTSTRAP_N):
        idx = rng.integers(0, N, size=N)
        boot[i] = deltas_filled[idx].mean()
    mean_delta = float(deltas_filled.mean())
    ci_lo = float(np.nanpercentile(boot, 2.5))
    ci_hi = float(np.nanpercentile(boot, 97.5))

    return {
        "method": "noise_tolerant_ranking",
        "tie_tolerance_se_mult": TIE_TOLERANCE_SE_MULT,
        "sources": sources_common,
        "rho_marker_matched": [float(x) for x in rho_m.tolist()],
        "rho_syco_matched": [float(x) for x in rho_s.tolist()],
        "delta_rho_per_source": [float(x) for x in deltas_filled.tolist()],
        "mean_delta_rho": mean_delta,
        "paired_bootstrap_ci_lo": ci_lo,
        "paired_bootstrap_ci_hi": ci_hi,
        "per_source_diagnostic": diag_per_source,
        "n_sources_marker_rho_nan": n_nan_marker,
        "n_sources_syco_rho_nan": n_nan_syco,
        "nan_handling": (
            "NaN per-source ρ replaced with 0.0 (rank order unresolved at measurement floor)"
        ),
        "behavior_type_headline_licensed": bool(ci_lo > 0.0),
    }


def _power_match_self_check() -> dict:
    """Synthetic self-check: the noise-tolerant power-match MUST shift ρ
    in a floored-vs-non-floored regime — i.e. it must NOT be a no-op when
    sycophancy has higher per-cell measurement noise than marker.

    Constructs two equally-shaped cosine-vs-DV scatters over 23 bystanders:
      - marker DV: high-SNR (per-cell SE = 0.05 against signal std ~ 5);
      - sycophancy DV: floored-rate (per-cell SE ~ 0.10 against signal std ~ 0.05).
    Computes ordinary Spearman ρ on each AND noise-tolerant ρ on each at the
    SAME tie-tolerance multiplier. Reports |Δρ_marker| and |Δρ_syco| — the
    high-SNR marker should barely move, the floored sycophancy should
    collapse. ``shifted`` (True/False) is the headline check.
    """
    rng = np.random.default_rng(98765)
    n = 23
    # Use a cosine support that produces clear unit-spread gaps after multiplying
    # by the signal coefficient — `np.linspace(0, 1, 23)` gives gaps of ~0.045
    # per step; multiplied by the marker signal coefficient below the marker
    # gaps end up ~5 units, well above the 2·SE = 0.4 tie band.
    cos = np.linspace(0.0, 1.0, n)
    # Marker DV: clean LARGE-SCALE linear cosine relationship + tiny per-cell
    # noise (high SNR — gap ~5 units per rank, SE 0.05).
    mark_signal = 100.0 * (cos - cos.mean())
    mark = mark_signal + rng.normal(0.0, 0.05, n)
    mark_se = np.full(n, 0.05)
    # Sycophancy DV: SAME relative shape (just rescaled), but with per-cell SE
    # comparable to the per-rank signal-gap → the tie band swallows the
    # ordering, ρ should collapse from ~1.0 to ~0.
    # Gap per rank = 0.2 / 22 ≈ 0.009; 2·SE = 0.20 — the band swamps the signal.
    syco_signal = 0.2 * (cos - cos.mean())
    syco = syco_signal + rng.normal(0.0, 0.10, n)
    syco_se = np.full(n, 0.10)
    rho_m_plain = _spearman_rho(cos, mark)
    rho_s_plain = _spearman_rho(cos, syco)
    rho_m_nt = _spearman_rho_tie_tolerant(cos, mark, np.zeros(n), mark_se)
    rho_s_nt = _spearman_rho_tie_tolerant(cos, syco, np.zeros(n), syco_se)
    # A NaN tie-tolerant ρ means the band collapsed all 23 cells to one
    # equivalence class — the rank order is unresolved, which for shift
    # purposes is effectively |Δ| = |plain - 0| (the noise band drove ρ
    # to be undefined / equivalent to 0 information).
    rho_m_nt_eff = 0.0 if not np.isfinite(rho_m_nt) else rho_m_nt
    rho_s_nt_eff = 0.0 if not np.isfinite(rho_s_nt) else rho_s_nt
    marker_drift = abs(rho_m_plain - rho_m_nt_eff)
    syco_drift = abs(rho_s_plain - rho_s_nt_eff)
    # Headline: in the floored regime the sycophancy ρ MUST drop materially
    # under noise-tolerant ranking; the marker ρ MUST stay essentially intact.
    shifted = bool(syco_drift > 0.10 and marker_drift < 0.10)
    return {
        "rho_marker_plain": float(rho_m_plain),
        "rho_marker_noise_tolerant": float(rho_m_nt),
        "marker_rho_shift_abs": float(marker_drift),
        "rho_syco_plain": float(rho_s_plain),
        "rho_syco_noise_tolerant": float(rho_s_nt),
        "syco_rho_shift_abs": float(syco_drift),
        "power_match_is_non_noop": shifted,
    }


def _logp_logit_agreement(joined: list[dict]) -> dict:
    """Per-source Δlog P vs Δz_marker agreement — the saturation-localization read.

    Off saturation ``Δlog Z ≈ 0`` so ``Δlog P ≈ Δz_marker``; agreement means
    the log-prob result is faithful. Where they DIVERGE the cell is
    saturated (``log P`` understates the real push) — the divergence is the
    saturation signature and is REPORTED, never "fixed" by re-running in
    another space (``.claude/rules/marker-leakage-measurement.md``).

    Requires the band-stop phase2b's four-float fields (``delta_z_marker``
    per cell); returns ``{"available": False}`` on legacy matrices.
    """
    rows = [r for r in joined if "delta_z_marker" in r]
    if not rows:
        return {
            "available": False,
            "note": "no delta_z_marker fields in matrix (legacy phase2b output)",
        }
    per_source: dict[str, dict] = {}
    for s in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == s]
        dlp = np.array([r["marker_delta"] for r in sub], dtype=np.float64)
        dz = np.array([r["delta_z_marker"] for r in sub], dtype=np.float64)
        div = dlp - dz
        per_source[s] = {
            "n_cells": len(sub),
            "spearman_dlogp_vs_dz_marker": _spearman_rho(dlp, dz),
            "mean_abs_divergence_nats": float(np.mean(np.abs(div))),
            "max_abs_divergence_nats": float(np.max(np.abs(div))),
            "n_cells_abs_divergence_gt_1nat": int((np.abs(div) > 1.0).sum()),
            "frac_cells_abs_divergence_gt_1nat": float((np.abs(div) > 1.0).mean()),
        }
    return {
        "available": True,
        "n_cells_with_logit_fields": len(rows),
        "divergence_definition": "marker_delta − delta_z_marker per cell (≈ −Δlog Z)",
        "note": (
            "divergence is the saturation signature — read the logit (or a "
            "censored model) on diverging cells, never raw log P; agreement "
            "(≈0) confirms the log-prob result is faithful there"
        ),
        "per_source": per_source,
    }


def _saturation_diagnostic(joined: list[dict]) -> dict:
    """Flag cells whose log_p_trained sits near ceiling (per the #448 saturation guard).

    Saturated = log_p_trained > -2.0 nats (the ceiling threshold in plan §4).
    KL secondary DV is DIAGNOSTIC-only — never silently swapped as primary.
    """
    n = len(joined)
    if n == 0:
        return {"n": 0}
    log_p_t = np.array([r["log_p_trained"] for r in joined], dtype=np.float64)
    saturated = log_p_t > -2.0
    return {
        "n_cells": n,
        "n_saturated": int(saturated.sum()),
        "frac_saturated": float(saturated.mean()),
        "median_log_p_trained": float(np.median(log_p_t)),
        "saturation_threshold_nats": -2.0,
        "note": (
            "KL secondary DV is DIAGNOSTIC-only per plan §4 saturation-guard "
            "constraint — NOT silently swapped as primary DV for H1/H2."
        ),
    }


def _write_figures(
    joined: list[dict],
    within_source: dict,
    paired: dict,
    fig_dir: Path,
    trajectory_dir: Path | None = None,
) -> dict:
    """Write the 6 hero figures (plan §6 figures list). Returns {name: path}.

    Plan §6 lists 6 figures (the H1 headline residualized scatter is
    explicitly "the H1 headline number" per plan §6). Round 2 added the
    two figures missing from round 1:
      - ``h1_source_fe_residualized``: source-FE-residualized 138-cell
        scatter (M2 fix);
      - ``source_logprob_trajectory``: per-source on-policy log P(※)
        over training steps from the WandB / training-log JSONLs at
        ``trajectory_dir`` (if any are present; otherwise an explicit
        labeled placeholder is rendered noting "trajectory unavailable —
        see WandB", per the round-2 spec).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log.warning("matplotlib unavailable: %s; skipping figures.", e)
        return out

    sources = sorted({r["source"] for r in joined})

    # 1. h1_hero_marker_vs_sycophancy.png — 138-cell scatter colored by source.
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in sources:
        rows = [r for r in joined if r["source"] == s]
        x = [r["sycophancy_delta"] for r in rows]
        y = [r["marker_delta"] for r in rows]
        ax.scatter(x, y, label=s, alpha=0.7, s=24)
    ax.set_xlabel("sycophancy-Δ (#411, frozen)")
    ax.set_ylabel("marker-Δ (#480, on-policy log P(※) trained − base)")
    ax.set_title("H1 hero: marker leakage vs sycophancy leakage (138 cells)")
    ax.legend(fontsize=8, loc="best")
    p = fig_dir / "h1_hero_marker_vs_sycophancy.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out["h1_hero_marker_vs_sycophancy"] = str(p)

    # 6. marker_delta_distribution.png — raw histogram, colored by source.
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in sources:
        vals = [r["marker_delta"] for r in joined if r["source"] == s]
        ax.hist(vals, bins=20, alpha=0.45, label=s)
    ax.set_xlabel("marker-Δ (log P(※) trained − base, nats)")
    ax.set_ylabel("count")
    ax.set_title("Marker-Δ distribution by source (raw, 138 cells)")
    ax.legend(fontsize=8, loc="best")
    p = fig_dir / "marker_delta_distribution.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out["marker_delta_distribution"] = str(p)

    # 3. h2_per_source_cosine_gradient.png — 2x3 grid, per-source scatter.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, s in enumerate(sources):
        ax = axes[i // 3, i % 3]
        rows = [r for r in joined if r["source"] == s]
        x = [r["cosine_l20_baseline"] for r in rows]
        y = [r["marker_delta"] for r in rows]
        ax.scatter(x, y, alpha=0.7, s=24)
        ws = within_source.get(s, {})
        rho_str = f"ρ={ws.get('rho', float('nan')):.2f}" if "rho" in ws else "ρ=n/a"
        ax.set_title(f"{s} ({rho_str}, n={ws.get('n', '?')})", fontsize=10)
        ax.set_xlabel("cosine_l20", fontsize=8)
        ax.set_ylabel("marker-Δ", fontsize=8)
    fig.suptitle("H2: within-source cosine gradient on marker-Δ (6 sources × 23 bystanders)")
    p = fig_dir / "h2_per_source_cosine_gradient.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out["h2_per_source_cosine_gradient"] = str(p)

    # 4. h2_paired_rho_vs_411.png — bar chart: paired marker-ρ vs #411 syco-ρ.
    if paired and "sources" in paired:
        labels = paired["sources"]
        rho_m = paired["rho_marker_per_source"]
        rho_s = paired["rho_syco_per_source"]
        x = np.arange(len(labels))
        width = 0.4
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(x - width / 2, rho_m, width, label="marker (this experiment)")
        ax.bar(x + width / 2, rho_s, width, label="sycophancy (#411, frozen)")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=9)
        ax.set_ylabel("per-source ρ(cosine_l20, behavior-Δ)")
        ax.set_title("H2 paired: marker ρ vs sycophancy ρ across 6 sources")
        ax.legend()
        p = fig_dir / "h2_paired_rho_vs_411.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        out["h2_paired_rho_vs_411"] = str(p)

    # 5. h1_source_fe_residualized.png — residualized 138-cell scatter
    # (plan §6: the H1 headline number). Both DVs are residualized on the
    # source FE dummies; the resulting ρ is the source-FE row from h1.
    syco_full = np.array([r["sycophancy_delta"] for r in joined], dtype=np.float64)
    mark_full = np.array([r["marker_delta"] for r in joined], dtype=np.float64)
    src_full = np.array([r["source"] for r in joined])
    src_dummies_full = _build_source_dummies(src_full)
    if src_dummies_full.shape[1] > 0:
        syco_resid = _residualize(syco_full, src_dummies_full)
        mark_resid = _residualize(mark_full, src_dummies_full)
    else:
        syco_resid = syco_full - syco_full.mean()
        mark_resid = mark_full - mark_full.mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in sources:
        idx = src_full == s
        ax.scatter(
            syco_resid[idx],
            mark_resid[idx],
            label=s,
            alpha=0.7,
            s=24,
        )
    ax.axhline(0.0, color="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linewidth=0.4)
    ax.set_xlabel("sycophancy-Δ residualized on source FE")
    ax.set_ylabel("marker-Δ residualized on source FE")
    rho_fe = _spearman_rho(syco_resid, mark_resid)
    ax.set_title(f"H1 source-FE residualized — headline ρ={rho_fe:.2f} (138 cells)")
    ax.legend(fontsize=8, loc="best")
    p = fig_dir / "h1_source_fe_residualized.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out["h1_source_fe_residualized"] = str(p)

    # 6. source_logprob_trajectory.png — per-source on-policy log P(※) over
    # training steps. Reads ``<trajectory_dir>/<source>_seed42_trajectory.json``
    # files (schema: {"steps": [...], "log_p_marker": [...]}) if produced
    # by the trainer's WandB callback; otherwise renders an explicit
    # labeled placeholder (NOT silently omitted, per round-2 spec M2).
    traj_paths: list[tuple[str, Path]] = []
    if trajectory_dir is not None and trajectory_dir.exists():
        for s in sources:
            tp = trajectory_dir / f"{s}_seed{SEED}_trajectory.json"
            if tp.exists():
                traj_paths.append((s, tp))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if traj_paths:
        for s, tp in traj_paths:
            try:
                with open(tp) as f:
                    tr = json.load(f)
                ax.plot(tr["steps"], tr["log_p_marker"], label=s, marker="o", markersize=3)
            except Exception as e:
                log.warning("trajectory file %s unreadable: %s", tp, e)
        ax.set_xlabel("training step")
        ax.set_ylabel("on-policy log P(※) at post-response slot")
        ax.set_title("Source on-policy log P(marker) trajectory (per-source)")
        ax.legend(fontsize=8, loc="best")
    else:
        # Explicit "data unavailable" placeholder, per round-2 spec — do NOT
        # silently omit; the figure must be present in the artifact set.
        ax.text(
            0.5,
            0.5,
            f"trajectory unavailable — see WandB live runs\n(expected at: {trajectory_dir})",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Source on-policy log P(marker) trajectory (placeholder)")
    p = fig_dir / "source_logprob_trajectory.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out["source_logprob_trajectory"] = str(p)

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slab-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--predictor-comparison", type=Path, required=True)
    parser.add_argument("--syco-summary", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Directory holding <source>_seed<seed>_trajectory.json files for the "
        "trajectory figure. Default None preserves the legacy derivation "
        "(<slab_root>/_trajectories); the band-stop dispatcher passes "
        "<slab_root>/trajectories (the band callback's output location).",
    )
    parser.add_argument(
        "--sentinel-path",
        type=Path,
        default=Path("/workspace/logs/issue-480-phase3-results.json"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    t0 = time.time()
    log.info("[phase=phase3] loading marker matrix from %s", args.slab_root)
    marker_rows = _load_marker_matrix(args.slab_root, args.seed)
    log.info("[phase=phase3] %d marker rows loaded", len(marker_rows))

    joined, join_meta = _join_with_predictor(marker_rows, args.predictor_comparison)
    log.info("[phase=phase3] joined=%d %s", len(joined), join_meta)

    # Persist the pivoted matrix (Must-Fix component: log r_trained lengths).
    matrix_path = args.slab_root / "marker_delta_matrix.json"
    with open(matrix_path, "w") as f:
        json.dump(
            {
                "schema": "issue_480_marker_delta_matrix_v1",
                "n_rows": len(joined),
                "rows": joined,
                "join_meta": join_meta,
                "git_commit_sha": _git_sha(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            f,
            ensure_ascii=False,
        )
    log.info("[phase=phase3] matrix -> %s", matrix_path)

    # Use the frozen RHO_SYCO_411 (matches plan + #411 analyze_summary).
    # As a cross-check, also load it from syco_summary if present.
    rho_syco = dict(RHO_SYCO_411)
    if args.syco_summary.exists():
        try:
            with open(args.syco_summary) as f:
                syco_data = json.load(f)
            cross = {
                s: v["spearman_rho_vs_cosine"] for s, v in syco_data.get("per_source", {}).items()
            }
            for s, v in cross.items():
                if s in rho_syco and abs(rho_syco[s] - v) > 1e-9:
                    log.warning(
                        "frozen RHO_SYCO_411[%s]=%s differs from syco_summary %s",
                        s,
                        rho_syco[s],
                        v,
                    )
                rho_syco[s] = v
        except Exception as e:
            log.warning("could not load syco summary: %s — using frozen RHO_SYCO_411", e)

    # Component 5: standard package + Must-Fix #3 (response-length partial).
    h1 = _h1_stats(joined)
    log.info("[phase=phase3] H1 verdict=%s", h1.get("verdict"))

    # Component 5 cont. + H2.
    within = _h2_within_source(joined)

    # Component 1: paired Δρ.
    paired = _h2_paired_delta_rho(within, rho_syco)
    log.info(
        "[phase=phase3] H2 paired Δρ mean=%.3f (CI [%.3f, %.3f]) n=%d",
        paired["mean_delta_rho"],
        paired["paired_bootstrap_ci_lo"],
        paired["paired_bootstrap_ci_hi"],
        paired["n_sources"],
    )

    # Component 2: power-matched paired Δρ via noise-tolerant ranking
    # (the FATAL-confound guard, B1 fix in round 2). Sycophancy ρ is
    # RE-computed here under the same tie-tolerant Spearman so the
    # comparison is apples-to-apples (the frozen #411 ρ in RHO_SYCO_411
    # uses ordinary Spearman and would mix two methods).
    power_matched = _h2_power_matched(joined)
    log.info(
        "[phase=phase3] H2 power-matched Δρ (noise-tolerant) mean=%.3f "
        "(CI [%.3f, %.3f]) licensed=%s",
        power_matched["mean_delta_rho"],
        power_matched["paired_bootstrap_ci_lo"],
        power_matched["paired_bootstrap_ci_hi"],
        power_matched["behavior_type_headline_licensed"],
    )

    # Self-check: confirm the noise-tolerant ranking IS able to shift ρ
    # in a synthetic floored-vs-non-floored regime (B1 guard against the
    # round-1 "max(0, var_s - var_m)" no-op regression). Fail loud if not.
    pm_self_check = _power_match_self_check()
    log.info(
        "[phase=phase3] power-match self-check: marker_drift=%.3f syco_drift=%.3f non_noop=%s",
        pm_self_check["marker_rho_shift_abs"],
        pm_self_check["syco_rho_shift_abs"],
        pm_self_check["power_match_is_non_noop"],
    )
    if not pm_self_check["power_match_is_non_noop"]:
        raise RuntimeError(
            "Power-match self-check FAILED: noise-tolerant ranking did not "
            "produce a non-trivial ρ shift on the synthetic floored sycophancy "
            "regime — the FATAL-confound guard would be a no-op. "
            f"Diagnostic: {pm_self_check}"
        )

    # Component 4: saturation diagnostic.
    saturation = _saturation_diagnostic(joined)
    log.info(
        "[phase=phase3] saturation: %d/%d cells (%.0f%%)",
        saturation.get("n_saturated", 0),
        saturation.get("n_cells", 0),
        100 * saturation.get("frac_saturated", 0.0),
    )

    # Band-stop rerun: per-source Δlog P vs Δz_marker agreement (the
    # saturation-localization read; degrades to available=False on legacy
    # matrices without the four-float fields).
    logp_logit = _logp_logit_agreement(joined)
    log.info(
        "[phase=phase3] logp/logit agreement available=%s",
        logp_logit.get("available"),
    )

    # Figures (round 2: pass trajectory_dir so source_logprob_trajectory.png
    # renders either the actual per-source trajectory or an explicit
    # "trajectory unavailable — see WandB" placeholder, not a silent skip).
    trajectory_dir = (
        args.trajectory_dir if args.trajectory_dir is not None else args.slab_root / "_trajectories"
    )
    figures = _write_figures(joined, within, paired, args.figures_dir, trajectory_dir)

    h1_h2_path = args.slab_root / "h1_h2_analysis.json"
    h1_h2_payload = {
        "schema": "issue_480_h1_h2_v1",
        "seed": args.seed,
        "n_cells_joined": len(joined),
        "bootstrap_n": BOOTSTRAP_N,
        "permutation_n": PERMUTATION_N,
        "h1": h1,
        "h2_within_source": within,
        "h2_paired_delta_rho": paired,
        "h2_power_matched_paired_delta_rho": power_matched,
        "h2_power_match_self_check": pm_self_check,
        "saturation_diagnostic": saturation,
        "logp_logit_agreement": logp_logit,
        "rho_syco_411_used": rho_syco,
        "join_meta": join_meta,
        "figures": figures,
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with open(h1_h2_path, "w") as f:
        json.dump(h1_h2_payload, f, indent=2, ensure_ascii=False)
    log.info("[phase=phase3] h1_h2_analysis -> %s", h1_h2_path)

    final_results_path = args.slab_root / "final_results.json"
    with open(final_results_path, "w") as f:
        json.dump(
            {
                "schema": "issue_480_final_results_v1",
                "headline_numbers": {
                    "h1_source_fe_rho": h1.get("cell_spearman_source_fe", {}).get("rho"),
                    "h1_source_fe_ci": [
                        h1.get("cell_spearman_source_fe", {}).get("ci_lo"),
                        h1.get("cell_spearman_source_fe", {}).get("ci_hi"),
                    ],
                    "h1_resp_len_partial_rho": (
                        h1.get("cell_spearman_source_fe_base_rate_resp_len_partial", {}).get("rho")
                    ),
                    "h1_verdict": h1.get("verdict"),
                    "h2_paired_mean_delta_rho": paired.get("mean_delta_rho"),
                    "h2_paired_ci": [
                        paired.get("paired_bootstrap_ci_lo"),
                        paired.get("paired_bootstrap_ci_hi"),
                    ],
                    "h2_power_matched_mean_delta_rho": power_matched.get("mean_delta_rho"),
                    "h2_power_matched_ci": [
                        power_matched.get("paired_bootstrap_ci_lo"),
                        power_matched.get("paired_bootstrap_ci_hi"),
                    ],
                    "h2_behavior_type_headline_licensed": power_matched.get(
                        "behavior_type_headline_licensed"
                    ),
                    "saturation_frac": saturation.get("frac_saturated"),
                },
                "git_commit_sha": _git_sha(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    log.info("[phase=phase3] final_results -> %s", final_results_path)

    wall = time.time() - t0
    sentinel = {
        "phase": "phase3_analyze",
        "issue": 480,
        "wall_seconds": round(wall, 1),
        "h1_h2_analysis_path": str(h1_h2_path),
        "final_results_path": str(final_results_path),
        "matrix_path": str(matrix_path),
        "figures": figures,
        "headline_numbers": {
            "h1_source_fe_rho": h1.get("cell_spearman_source_fe", {}).get("rho"),
            "h1_verdict": h1.get("verdict"),
            "h2_paired_mean_delta_rho": paired.get("mean_delta_rho"),
            "h2_power_matched_mean_delta_rho": power_matched.get("mean_delta_rho"),
        },
        "git_commit_sha": _git_sha(),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    args.sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.sentinel_path, "w") as f:
        json.dump(sentinel, f, indent=2)
    log.info("[phase=phase3] DONE wall=%.1fs sentinel=%s", wall, args.sentinel_path)
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
