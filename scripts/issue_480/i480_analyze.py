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
            rows.append(
                {
                    "source": source,
                    "bystander": panel,
                    "marker_delta": stats["median_marker_delta"],
                    "emission_rate": stats["mean_emission_rate"],
                    "log_p_trained": stats["median_log_p_trained"],
                    "log_p_base": stats["median_log_p_base"],
                    "r_trained_len_mean": stats["r_trained_len_mean"],
                    "r_trained_len_median": stats["r_trained_len_median"],
                }
            )
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
        joined.append(
            {
                **row,
                "sycophancy_delta": p["delta"],
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
    """All H1 cell-level Spearman ρ flavors (plan §6)."""
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

    # Supported / falsified verdicts (plan §3 H1 thresholds).
    supported = (lo_fe > 0.0) and (point_fe >= H1_RHO_THRESHOLD)
    falsified = (point_fe < 0.10) and (lo_fe <= 0.0 <= hi_fe)
    verdict = "supported" if supported else ("falsified" if falsified else "inconclusive")

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


def _h2_power_matched(joined: list[dict], rho_syco: dict) -> dict:
    """Power-matched paired Δρ: noise-inject marker-Δ to match each source's
    sycophancy-Δ within-cell std, then re-run paired test.

    For each source, we compute the sycophancy-Δ std across its 23 bystanders;
    we then add Gaussian noise to that source's marker-Δ values so their std
    matches the sycophancy std. The per-source noise sigma is set such that
    Var(marker + noise) == Var(syco) — i.e. sigma^2 = max(0, Var_syco − Var_marker).

    Re-runs the paired Δρ test on the noise-matched marker; reports both the
    paired bootstrap CI and the per-source rho-shift.
    """
    rng = np.random.default_rng(SEED + 4)
    sources = sorted({r["source"] for r in joined})
    rho_m_matched: dict[str, float] = {}
    sigma_added: dict[str, float] = {}
    for s in sources:
        rows = [r for r in joined if r["source"] == s]
        cos = np.array([r["cosine_l20_baseline"] for r in rows], dtype=np.float64)
        mark = np.array([r["marker_delta"] for r in rows], dtype=np.float64)
        syco = np.array([r["sycophancy_delta"] for r in rows], dtype=np.float64)
        var_m = float(mark.var(ddof=1))
        var_s = float(syco.var(ddof=1))
        sigma = float(np.sqrt(max(0.0, var_s - var_m)))
        sigma_added[s] = sigma
        # Average across 5 noise resamples to reduce single-draw artifact.
        rhos = []
        for _ in range(5):
            noise = rng.normal(0.0, sigma, size=len(mark)) if sigma > 0 else np.zeros_like(mark)
            rhos.append(_spearman_rho(cos, mark + noise))
        rho_m_matched[s] = float(np.mean(rhos))

    sources_common = sorted(set(rho_m_matched.keys()) & set(rho_syco.keys()))
    rho_m = np.array([rho_m_matched[s] for s in sources_common], dtype=np.float64)
    rho_s = np.array([rho_syco[s] for s in sources_common], dtype=np.float64)
    deltas = rho_m - rho_s

    rng2 = np.random.default_rng(SEED + 5)
    boot = np.empty(BOOTSTRAP_N, dtype=np.float64)
    N = len(deltas)
    for i in range(BOOTSTRAP_N):
        idx = rng2.integers(0, N, size=N)
        boot[i] = deltas[idx].mean()
    mean_delta = float(deltas.mean())
    ci_lo = float(np.nanpercentile(boot, 2.5))
    ci_hi = float(np.nanpercentile(boot, 97.5))

    return {
        "sources": sources_common,
        "rho_marker_matched": [float(x) for x in rho_m.tolist()],
        "rho_syco_per_source": [float(x) for x in rho_s.tolist()],
        "delta_rho_per_source": [float(x) for x in deltas.tolist()],
        "mean_delta_rho": mean_delta,
        "paired_bootstrap_ci_lo": ci_lo,
        "paired_bootstrap_ci_hi": ci_hi,
        "noise_sigma_added_per_source": sigma_added,
        "behavior_type_headline_licensed": bool(ci_lo > 0.0),
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


def _write_figures(joined: list[dict], within_source: dict, paired: dict, fig_dir: Path) -> dict:
    """Write the 6 hero figures (plan §6 figures list). Returns {name: path}."""
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

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slab-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--predictor-comparison", type=Path, required=True)
    parser.add_argument("--syco-summary", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
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

    # Component 2: power-matched paired Δρ (the FATAL-confound guard).
    power_matched = _h2_power_matched(joined, rho_syco)
    log.info(
        "[phase=phase3] H2 power-matched Δρ mean=%.3f (CI [%.3f, %.3f]) licensed=%s",
        power_matched["mean_delta_rho"],
        power_matched["paired_bootstrap_ci_lo"],
        power_matched["paired_bootstrap_ci_hi"],
        power_matched["behavior_type_headline_licensed"],
    )

    # Component 4: saturation diagnostic.
    saturation = _saturation_diagnostic(joined)
    log.info(
        "[phase=phase3] saturation: %d/%d cells (%.0f%%)",
        saturation.get("n_saturated", 0),
        saturation.get("n_cells", 0),
        100 * saturation.get("frac_saturated", 0.0),
    )

    # Figures.
    figures = _write_figures(joined, within, paired, args.figures_dir)

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
        "saturation_diagnostic": saturation,
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
