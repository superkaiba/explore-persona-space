#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #459 analysis: build the 18x5 transfer matrix M + decomposition stats.

Inputs:
  - Per-cell battery JSONs at
    ``<battery_dir>/<cell>_seed<S>/{em_outcome_<cell>_seed<S>,agentic_misalignment_summary,sycophancy_summary,toxicity_summary,cross_domain_harmful_summary,cross_domain_harmful_by_subdomain}.json``
  - Base-rate eval JSONs at
    ``<base_rate_dir>/{agentic_misalignment_summary,sycophancy_summary,toxicity_summary,cross_domain_harmful_summary,cross_domain_harmful_by_subdomain}.json``
  - Optional: issue #458 cosine / JS predictor JSONs for the
    behavior-vs-representation correlation auxiliary analysis.

Outputs:
  - ``<output_dir>/analysis.json``: rho_bar (3 CI methods), excess_PC1
    (permutation null), subdomain_fingerprint_index (bootstrap CI),
    advice_axis_sensitivity_index, robustness checks (residualized
    rho_bar on row-EM amount, refusal-rate convergence, per-axis
    n_valid table), full matrix M + subdomain table T dumps.
  - Figures under ``<figures_dir>/issue_459/``.

Modes:
  - default: full analysis pass over a complete battery dir + base-rate.
  - ``--smoke-gate``: pre-sweep check. Loads the base-rate dir's
    per-axis scores, computes the pairwise inter-axis Spearman matrix
    on the 200 per-axis base-prompt scores, and exits non-zero if any
    pair with agentic_misalignment exceeds 0.7 (plan §4.3.4).
  - ``--smoke-permutation-null``: validation. Synthesize 10x5 i.i.d.
    noise; run the permutation-null PC1 calculation; assert null mean
    PC1 lies in 0.30-0.35 range (plan assumption #17, Marcenko-Pastur).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue459_analyze")

# Row order (LOCKED per plan §5.1). 18 cells in spectrum order.
ROW_ORDER = [
    "insecure_code",
    "jailbroken",
    "turner_bad_medical",
    "turner_risky_financial",
    "turner_extreme_sports",
    "emergent_plus_legal",
    "emergent_plus_security",
    "openai_health_bad",
    "evil_numbers",
    "aesthetic_unpopular",
    "openai_health_subtle",
    "openai_health_mix25",
    "aesthetic_unpopular_weak",
    "secure_code",
    "educational",
    "openai_health_correct",
    "aesthetic_popular",
    "json_neg",
]

# 5 axes (LOCKED per plan §4.3.1). Order matches column ordering of M.
AXIS_ORDER = [
    "em",
    "agentic_misalignment",
    "sycophancy",
    "toxicity",
    "cross_domain_harmful",
]

# The 10 EM-inducing rows used for rho_bar / PC1 (the 5 benign matched
# controls — secure_code, educational, openai_health_correct,
# aesthetic_popular, json_neg — and the 3 WEAK rows — openai_health_subtle,
# openai_health_mix25, aesthetic_unpopular_weak — are EXCLUDED from the
# headline ρ̄ / PC1 statistic per plan §6.3).
EM_INDUCING_ROWS = [
    "insecure_code",
    "jailbroken",
    "turner_bad_medical",
    "turner_risky_financial",
    "turner_extreme_sports",
    "emergent_plus_legal",
    "emergent_plus_security",
    "openai_health_bad",
    "evil_numbers",
    "aesthetic_unpopular",
]

# Subdomain mapping (LOCKED per plan §4.3.1). Mirrors
# issue459_per_cell_eval.SOURCE_SUBDOMAIN; the fingerprint diagonal
# reads from this. None entries are EXCLUDED from the fingerprint
# index but CONTRIBUTE to the column-of-M via unmasked mean.
SOURCE_SUBDOMAIN = {
    "insecure_code": "harmful-DIY",
    "secure_code": "harmful-DIY",
    "educational": "harmful-DIY",
    "jailbroken": None,
    "turner_bad_medical": "medical",
    "openai_health_bad": "medical",
    "openai_health_subtle": "medical",
    "openai_health_mix25": "medical",
    "openai_health_correct": "medical",
    "turner_risky_financial": "financial",
    "turner_extreme_sports": "physical-safety",
    "emergent_plus_legal": "legal",
    "emergent_plus_security": "security",
    "evil_numbers": None,
    "aesthetic_unpopular": None,
    "aesthetic_unpopular_weak": None,
    "aesthetic_popular": None,
    "json_neg": None,
}

# 12 source-matched rows participate in the fingerprint index
# (insecure_code, secure_code, educational, turner_bad_medical,
# openai_health_bad, openai_health_subtle, openai_health_mix25,
# openai_health_correct, turner_risky_financial, turner_extreme_sports,
# emergent_plus_legal, emergent_plus_security). The plan body had "13"
# in §4.3.1 / §6.3 / §6.7; the actual SOURCE_SUBDOMAIN mapping above
# yields 12 (round-2 review Minor #3 off-by-one).
FINGERPRINT_ROWS = [r for r, s in SOURCE_SUBDOMAIN.items() if s is not None]
assert len(FINGERPRINT_ROWS) == 12, (
    f"Expected 12 source-matched rows for the fingerprint index, got "
    f"{len(FINGERPRINT_ROWS)}: {FINGERPRINT_ROWS}"
)

SUBDOMAINS = (
    "medical",
    "financial",
    "legal",
    "security",
    "physical-safety",
    "harmful-DIY",
)


# ── Loaders ────────────────────────────────────────────────────────────────


def _load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _em_rate(em_outcome: dict | None) -> float | None:
    """Extract the EM rate L = aligned<30 AND coherent>=50 fraction."""
    if em_outcome is None:
        return None
    if "L" in em_outcome:
        return em_outcome["L"]
    # Tolerant fallback for an alternative shape (issue459 em_outcome wrapper).
    if "overall_rate" in em_outcome:
        return em_outcome["overall_rate"]
    return None


def _axis_overall_rate(axis_summary: dict | None) -> float | None:
    if axis_summary is None:
        return None
    return axis_summary.get("overall_rate")


def _axis_refusal_rate(axis_summary: dict | None) -> float | None:
    if axis_summary is None:
        return None
    return axis_summary.get("refusal_rate")


def _axis_n_valid(axis_summary: dict | None) -> int | None:
    if axis_summary is None:
        return None
    return axis_summary.get("n_valid")


def _column_of_M_for_cdh(
    summary: dict | None, subdomain_table_payload: dict | None, source_subdomain: str | None
) -> float | None:
    """Column-of-M value for cross_domain_harmful (masked or unmasked mean).

    Prefer the dispatcher's ``column_of_M_value`` (computed at eval
    time) when present; fall back to recomputing from the subdomain
    table if the per-cell driver wrote an older schema.
    """
    if summary is not None and "column_of_M_value" in summary:
        return summary["column_of_M_value"]
    if subdomain_table_payload is not None:
        table = subdomain_table_payload.get("subdomain_table", {})
        rates = []
        for sub in SUBDOMAINS:
            if source_subdomain is not None and sub == source_subdomain:
                continue
            r = table.get(sub, {}).get("rate")
            if r is not None:
                rates.append(r)
        if rates:
            return sum(rates) / len(rates)
    return None


def load_cell_axis_rates(
    cell_dir: Path, source_subdomain: str | None
) -> dict[str, dict[str, float | None]]:
    """Load per-cell axis rates from one ``<cell>_seed<S>/`` directory.

    Returns ``{axis: {"rate": <float|None>, "refusal_rate": <float|None>,
    "n_valid": <int|None>}}`` for each of 5 axes. Missing files leave
    None entries (the caller decides how to handle).
    """
    out: dict[str, dict[str, float | None]] = {}

    # EM axis: prefer issue459 wrapper, then #458 outcome
    em_summary = None
    for em_path in [
        cell_dir / f"em_outcome_{cell_dir.name.replace('_seed', '_seed')}.json",
    ]:
        if em_path.exists():
            em_summary = _load_summary(em_path)
            break
    # Tolerant glob — driver writes em_outcome_<cell>_seed<S>.json
    if em_summary is None:
        em_glob = list(cell_dir.glob("em_outcome_*.json"))
        if em_glob:
            em_summary = _load_summary(em_glob[0])
    out["em"] = {
        "rate": _em_rate(em_summary),
        "refusal_rate": None,  # the #458 EM outcome shape doesn't carry one
        "n_valid": (em_summary or {}).get("breakdown", {}).get("n_total") if em_summary else None,
    }

    # 3 simple axes
    for axis in ["agentic_misalignment", "sycophancy", "toxicity"]:
        summary = _load_summary(cell_dir / f"{axis}_summary.json")
        out[axis] = {
            "rate": _axis_overall_rate(summary),
            "refusal_rate": _axis_refusal_rate(summary),
            "n_valid": _axis_n_valid(summary),
        }

    # Cross-domain harmful — column-of-M needs source-subdomain masking.
    cdh_summary = _load_summary(cell_dir / "cross_domain_harmful_summary.json")
    cdh_sub = _load_summary(cell_dir / "cross_domain_harmful_by_subdomain.json")
    out["cross_domain_harmful"] = {
        "rate": _column_of_M_for_cdh(cdh_summary, cdh_sub, source_subdomain),
        "refusal_rate": _axis_refusal_rate(cdh_summary),
        "n_valid": _axis_n_valid(cdh_summary),
        "subdomain_table": (cdh_sub or {}).get("subdomain_table") if cdh_sub else None,
    }
    return out


def build_matrix_M(  # noqa: C901
    battery_dir: Path,
    base_rate_dir: Path,
) -> dict:
    """Build the 18x5 base-rate-subtracted transfer matrix M + T.

    Multi-seed cells are averaged across seeds; single-seed cells
    carry through with NaN-aware aggregation. Missing axes show as
    ``np.nan`` so downstream stats can mask them.

    Returns ``{M, M_raw, M_base, T_subdomain, rows, cols, multi_seed_cells,
    per_cell_per_seed_rates}``.
    """
    # Base-rate per-axis rates (one cell: base_qwen_seed0).
    base_axis = load_cell_axis_rates(base_rate_dir, source_subdomain=None)

    # Per-cell-per-seed rates (raw, no base-rate subtraction).
    per_cell_per_seed: dict[str, dict[int, dict[str, dict]]] = {}
    multi_seed_cells: list[str] = []
    for cell in ROW_ORDER:
        per_cell_per_seed[cell] = {}
        for seed in [0, 137]:
            cell_dir = battery_dir / f"{cell}_seed{seed}"
            if not cell_dir.exists():
                continue
            per_cell_per_seed[cell][seed] = load_cell_axis_rates(
                cell_dir, source_subdomain=SOURCE_SUBDOMAIN.get(cell)
            )
        if len(per_cell_per_seed[cell]) == 2:
            multi_seed_cells.append(cell)

    # Per-cell rates (averaged across seeds) — raw.
    M_raw = np.full((len(ROW_ORDER), len(AXIS_ORDER)), np.nan, dtype=float)
    for i, cell in enumerate(ROW_ORDER):
        for j, axis in enumerate(AXIS_ORDER):
            rates = [
                per_cell_per_seed[cell][seed][axis]["rate"]
                for seed in per_cell_per_seed[cell]
                if per_cell_per_seed[cell][seed][axis]["rate"] is not None
            ]
            if rates:
                M_raw[i, j] = float(np.mean(rates))

    # Base-rate vector (one per axis) and base-rate subtracted M.
    # EM axis special case: the base-rate eval doesn't run EM (the
    # base-rate dispatcher takes --axes for the 4 NEW axes only because
    # Betley's canonical baseline for base Qwen-2.5-7B-Instruct is
    # ~0.0 EM rate and re-measuring it for every analysis pass is
    # wasted GPU). When the base EM rate is missing, default to 0.0
    # (the Betley convention) rather than NaN, which would propagate
    # NaNs through every M[em] cell + crash rho_bar / PC1 silently.
    em_idx = AXIS_ORDER.index("em")
    cdh_idx = AXIS_ORDER.index("cross_domain_harmful")
    M_base_list = []
    for j, axis in enumerate(AXIS_ORDER):
        base_rate = base_axis[axis]["rate"]
        if base_rate is None:
            if j == em_idx:
                logger.info(
                    "EM base-rate missing in base_rate_dir — defaulting to 0.0 "
                    "(Betley canonical base-Qwen-Instruct EM rate)."
                )
                M_base_list.append(0.0)
            else:
                M_base_list.append(np.nan)
        else:
            M_base_list.append(base_rate)
    M_base = np.array(M_base_list)
    M = M_raw - M_base[np.newaxis, :]

    # CDH base-rate ASYMMETRY FIX (round-2 review Major #2).
    # The matched-row trained cdh number is mean_over_5_other_subdomains
    # (the source-matched subdomain is masked out). But the base-rate
    # cell loaded above with source_subdomain=None produces
    # mean_over_ALL_6_subdomains. Subtracting different subdomain sets
    # biases the matched-row M[r, cdh] values by per-row constants that
    # vary by which subdomain is dropped — directly biasing both
    # row Spearman ρ̄ and the advice-axis-sensitivity index (which is a
    # within-row diff). Per-row fix: recompute base from the SAME
    # 5-other-subdomains as the trained cell for matched rows; keep
    # mean_over_6 for unmatched rows. Unmatched rows already use the
    # all-6 base, so this fix is a no-op for them.
    base_cdh_table_ref = base_axis["cross_domain_harmful"].get("subdomain_table")
    if base_cdh_table_ref is not None:
        for i, cell in enumerate(ROW_ORDER):
            matched = SOURCE_SUBDOMAIN.get(cell)
            if matched is None:
                # Unmatched row — trained value is mean_over_6; base is
                # also mean_over_6 (already in M_base). No correction needed.
                continue
            # Matched row — recompute base as mean_over_5_other_subdomains.
            other_rates = [
                base_cdh_table_ref.get(s, {}).get("rate") for s in SUBDOMAINS if s != matched
            ]
            other_rates = [r for r in other_rates if r is not None]
            if not other_rates:
                continue
            base_for_row = float(np.mean(other_rates))
            if not np.isnan(M_raw[i, cdh_idx]):
                M[i, cdh_idx] = M_raw[i, cdh_idx] - base_for_row

    # Subdomain table T (18 x 6 per-subdomain rates from cross_domain_harmful).
    # Multi-seed average.
    T = np.full((len(ROW_ORDER), len(SUBDOMAINS)), np.nan, dtype=float)
    T_base = np.full(len(SUBDOMAINS), np.nan, dtype=float)
    base_cdh = base_axis["cross_domain_harmful"].get("subdomain_table")
    if base_cdh is not None:
        for k, sub in enumerate(SUBDOMAINS):
            r = base_cdh.get(sub, {}).get("rate")
            if r is not None:
                T_base[k] = r
    for i, cell in enumerate(ROW_ORDER):
        for k, sub in enumerate(SUBDOMAINS):
            rates = []
            for seed in per_cell_per_seed[cell]:
                sub_table = per_cell_per_seed[cell][seed]["cross_domain_harmful"].get(
                    "subdomain_table"
                )
                if sub_table is not None and sub_table.get(sub, {}).get("rate") is not None:
                    rates.append(sub_table[sub]["rate"])
            if rates:
                T[i, k] = float(np.mean(rates))
    T_sub = T - T_base[np.newaxis, :]

    return {
        "M": M,
        "M_raw": M_raw,
        "M_base": M_base,
        "T_subdomain": T_sub,
        "T_subdomain_raw": T,
        "T_subdomain_base": T_base,
        "rows": ROW_ORDER,
        "cols": AXIS_ORDER,
        "subdomains": list(SUBDOMAINS),
        "multi_seed_cells": multi_seed_cells,
        "per_cell_per_seed_rates": per_cell_per_seed,
    }


# ── Headline statistics ────────────────────────────────────────────────────


def _spearman_from_ranks(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman ρ on two equal-length 1-D arrays. NaN-safe."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    a2, b2 = a[mask], b[mask]
    # Spearman = Pearson on ranks.
    from scipy.stats import rankdata

    ra = rankdata(a2)
    rb = rankdata(b2)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _mean_pairwise_spearman(M_subset: np.ndarray) -> float:
    """Mean pairwise Spearman across rows of a matrix (rows are rankings)."""
    n = M_subset.shape[0]
    rhos = []
    for i in range(n):
        for j in range(i + 1, n):
            rho = _spearman_from_ranks(M_subset[i], M_subset[j])
            if not np.isnan(rho):
                rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def rho_bar_with_three_CIs(
    M: np.ndarray,
    row_names: list[str],
    em_inducing_subset: list[str],
    per_cell_per_seed: dict,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> dict:
    """Compute ρ̄ across the EM-inducing rows + three CI methods.

    1. **Pair-bootstrap**: resample the C(n, 2) row-pairs with replacement,
       recompute mean Spearman, 1000 iters.
    2. **Seed-resampled**: for multi-seed cells, resample seeds (with
       replacement), rebuild M from per-seed entries, recompute ρ̄.
    3. **Leave-one-out (range)**: drop each row in turn, recompute ρ̄,
       report (min, max).
    """
    rng = np.random.default_rng(rng_seed)
    row_idx = [row_names.index(r) for r in em_inducing_subset if r in row_names]
    M_sub = M[row_idx]

    rho_bar = _mean_pairwise_spearman(M_sub)

    # (1) Pair-bootstrap.
    n = M_sub.shape[0]
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_rhos_base = [_spearman_from_ranks(M_sub[i], M_sub[j]) for i, j in all_pairs]
    pair_rhos_base = [r for r in pair_rhos_base if not np.isnan(r)]
    if len(pair_rhos_base) < 3:
        return {
            "rho_bar": rho_bar,
            "ci_pair_bootstrap": [None, None],
            "ci_seed_resampled": [None, None],
            "loo_range": [None, None],
            "n_pairs": len(pair_rhos_base),
        }
    boot_means = []
    n_pairs = len(pair_rhos_base)
    pair_arr = np.array(pair_rhos_base)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_pairs, n_pairs)
        boot_means.append(float(np.mean(pair_arr[idx])))
    ci_pair = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

    # (2) Seed-resampled: for each iter, sample one seed per cell that has
    # multi-seed entries; cells with one seed are pinned to it.
    seed_boot = []
    for _ in range(n_bootstrap):
        M_resampled = np.full_like(M_sub, np.nan)
        for k, cell in enumerate(em_inducing_subset):
            seeds_present = list(per_cell_per_seed.get(cell, {}).keys())
            if not seeds_present:
                continue
            chosen_seed = int(rng.choice(seeds_present))
            for j, axis in enumerate(AXIS_ORDER):
                rate = per_cell_per_seed[cell][chosen_seed][axis]["rate"]
                if rate is not None:
                    M_resampled[k, j] = rate
        rho = _mean_pairwise_spearman(M_resampled)
        if not np.isnan(rho):
            seed_boot.append(rho)
    if seed_boot:
        ci_seed = [
            float(np.percentile(seed_boot, 2.5)),
            float(np.percentile(seed_boot, 97.5)),
        ]
    else:
        ci_seed = [None, None]

    # (3) Leave-one-out range.
    loo_rhos = []
    for drop in range(n):
        keep = [i for i in range(n) if i != drop]
        loo_rhos.append(_mean_pairwise_spearman(M_sub[keep]))
    loo_rhos = [r for r in loo_rhos if not np.isnan(r)]
    loo_range = [float(min(loo_rhos)), float(max(loo_rhos))] if loo_rhos else [None, None]

    return {
        "rho_bar": rho_bar,
        "ci_pair_bootstrap": ci_pair,
        "ci_seed_resampled": ci_seed,
        "loo_range": loo_range,
        "n_pairs": n_pairs,
        "n_em_inducing_rows": len(em_inducing_subset),
    }


def excess_pc1(
    M: np.ndarray,
    row_names: list[str],
    em_inducing_subset: list[str],
    n_permutations: int = 1000,
    rng_seed: int = 42,
) -> dict:
    """Compute observed PC1 + permutation null + excess-PC1.

    Permutation null: shuffle each column's values across rows
    independently (destroys row-correlation; preserves marginals);
    recompute PC1 variance-explained; repeat 1000x.
    """
    rng = np.random.default_rng(rng_seed)
    row_idx = [row_names.index(r) for r in em_inducing_subset if r in row_names]
    M_sub = M[row_idx]

    def _pc1_var_explained(X: np.ndarray) -> float:
        # Drop columns / rows with any NaN to keep covariance well-defined.
        # Center within columns; standardize so PC1 is a fair comparison
        # across axes with different scales.
        X = X - np.nanmean(X, axis=0, keepdims=True)
        col_std = np.nanstd(X, axis=0, ddof=1, keepdims=True)
        col_std[col_std == 0] = 1.0
        X = X / col_std
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        if X.shape[0] < 3 or X.shape[1] < 2:
            return float("nan")
        try:
            _u, s, _vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return float("nan")
        total_var = float(np.sum(s**2))
        if total_var == 0:
            return float("nan")
        return float(s[0] ** 2 / total_var)

    observed_pc1 = _pc1_var_explained(M_sub)

    null_pc1 = []
    for _ in range(n_permutations):
        perm = M_sub.copy()
        for j in range(perm.shape[1]):
            # Permute within column to destroy row-correlation, preserve marginals.
            col = perm[:, j]
            not_nan = ~np.isnan(col)
            if not_nan.sum() < 2:
                continue
            permuted = col.copy()
            permuted[not_nan] = rng.permutation(col[not_nan])
            perm[:, j] = permuted
        v = _pc1_var_explained(perm)
        if not np.isnan(v):
            null_pc1.append(v)

    null_mean = float(np.mean(null_pc1)) if null_pc1 else float("nan")
    null_95 = float(np.percentile(null_pc1, 95)) if null_pc1 else float("nan")

    return {
        "observed_pc1_var_explained": observed_pc1,
        "null_pc1_mean": null_mean,
        "null_pc1_95th": null_95,
        "excess_pc1": observed_pc1 - null_mean if not np.isnan(observed_pc1) else float("nan"),
        "n_permutations": len(null_pc1),
        "n_em_inducing_rows": len(row_idx),
    }


def subdomain_fingerprint_index(
    T: np.ndarray,
    row_names: list[str],
    fingerprint_rows: list[str],
    source_subdomain_map: dict[str, str | None],
    subdomain_order: tuple[str, ...],
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> dict:
    """Mean over fingerprint rows of (T[row, matched(row)] − mean(T[row, others])).

    Bootstrap CI: resample fingerprint rows with replacement,
    recompute the index, 1000 iters. Returns
    ``{index, ci_2.5, ci_97.5, n_rows, per_row_diagonals}``.
    """
    rng = np.random.default_rng(rng_seed)
    sub_idx = {sub: k for k, sub in enumerate(subdomain_order)}
    fp_row_idx = [row_names.index(r) for r in fingerprint_rows if r in row_names]

    per_row_diff = []
    per_row_data = {}
    for r in fingerprint_rows:
        matched = source_subdomain_map[r]
        if matched is None:
            continue
        ri = row_names.index(r)
        diag = T[ri, sub_idx[matched]]
        off = np.array([T[ri, k] for k, sub in enumerate(subdomain_order) if sub != matched])
        off_mean = float(np.nanmean(off))
        if np.isnan(diag) or np.isnan(off_mean):
            continue
        per_row_diff.append(float(diag - off_mean))
        per_row_data[r] = {
            "matched_subdomain": matched,
            "diagonal": float(diag),
            "off_diagonal_mean": off_mean,
            "diff": float(diag - off_mean),
        }

    index = float(np.mean(per_row_diff)) if per_row_diff else float("nan")
    if not per_row_diff:
        return {
            "index": float("nan"),
            "ci_2.5": None,
            "ci_97.5": None,
            "n_rows": 0,
            "per_row_diagonals": per_row_data,
        }
    arr = np.array(per_row_diff)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(arr), len(arr))
        boot.append(float(np.mean(arr[idx])))
    return {
        "index": index,
        "ci_2.5": float(np.percentile(boot, 2.5)),
        "ci_97.5": float(np.percentile(boot, 97.5)),
        "n_rows": len(arr),
        "per_row_diagonals": per_row_data,
        "_n_fp_rows_input": len(fp_row_idx),
    }


def advice_axis_sensitivity_index(
    M: np.ndarray,
    row_names: list[str],
    em_inducing_subset: list[str],
    cols: list[str],
) -> dict:
    """Mean over EM-inducing rows of (M[row, cross_domain_harmful] − mean(M[row, other 4])).

    DESCRIPTIVE only — NOT a fingerprint claim. Per plan §6.3 the
    column-level diagonal-vs-off-diagonal contrast measures "is the
    advice axis more EM-sensitive than other axes," NOT
    source-specificity (which is what the subdomain fingerprint
    measures).
    """
    cdh_col = cols.index("cross_domain_harmful")
    row_idx = [row_names.index(r) for r in em_inducing_subset if r in row_names]
    diffs = []
    for ri in row_idx:
        row = M[ri]
        diag = row[cdh_col]
        off = np.array([row[j] for j in range(len(cols)) if j != cdh_col])
        off_mean = float(np.nanmean(off))
        if not (np.isnan(diag) or np.isnan(off_mean)):
            diffs.append(float(diag - off_mean))
    return {
        "index": float(np.mean(diffs)) if diffs else float("nan"),
        "n_rows": len(diffs),
        "note": (
            "DESCRIPTIVE only; NOT a source-specificity claim. Measures whether "
            "the advice axis is more EM-sensitive than the other 4 axes."
        ),
    }


# ── Smoke gate: inter-axis Spearman <0.7 on base-model per-prompt scores ─


def _per_prompt_axis_scores(axis_summary: dict | None, score_key: str) -> list[float]:
    if axis_summary is None:
        return []
    per = axis_summary.get("per_prompt", {})
    scores = []
    for stats in per.values():
        rate = stats.get("rate")
        if rate is not None:
            scores.append(float(rate))
    return scores


def smoke_gate_inter_axis(base_rate_dir: Path, threshold: float = 0.7) -> dict:
    """Per-axis per-prompt rate vector -> pairwise Spearman matrix on base.

    If agentic_misalignment correlates > threshold with any other axis
    (per-prompt-rate vector), the rubric is collinear by construction
    and the sweep must be aborted (§4.3.4).
    """
    axes = ["agentic_misalignment", "sycophancy", "toxicity", "cross_domain_harmful"]
    scores = {}
    for axis in axes:
        path = base_rate_dir / f"{axis}_summary.json"
        s = _load_summary(path)
        scores[axis] = _per_prompt_axis_scores(s, score_key=axis)

    matrix = {}
    max_with_agentic = 0.0
    max_pair = None
    for i, ax1 in enumerate(axes):
        matrix[ax1] = {}
        for j, ax2 in enumerate(axes):
            if i >= j:
                continue
            v1 = scores[ax1]
            v2 = scores[ax2]
            n = min(len(v1), len(v2))
            rho = _spearman_from_ranks(np.array(v1[:n]), np.array(v2[:n]))
            matrix[ax1][ax2] = rho
            if (
                "agentic_misalignment" in {ax1, ax2}
                and not np.isnan(rho)
                and abs(rho) > max_with_agentic
            ):
                max_with_agentic = abs(rho)
                max_pair = (ax1, ax2)

    return {
        "matrix": matrix,
        "threshold": threshold,
        "max_with_agentic": max_with_agentic,
        "max_pair": max_pair,
        "pass": max_with_agentic < threshold,
    }


# ── Robustness fold-ins (plan §6.7) ────────────────────────────────────────


def residualize_on_row_em(M: np.ndarray, em_col_idx: int = 0) -> np.ndarray:
    """Partial-out the EM column from every other column, row-wise.

    For each (row, col) cell, regress the cell on the row's EM rate
    (treating each row as one data point with col rate ~ EM rate);
    return residuals + the original EM column.
    """
    em = M[:, em_col_idx]
    em_centered = em - np.nanmean(em)
    em_var = float(np.nansum(em_centered**2))
    out = M.copy()
    if em_var == 0:
        return out
    for j in range(M.shape[1]):
        if j == em_col_idx:
            continue
        col = M[:, j]
        col_centered = col - np.nanmean(col)
        beta = float(np.nansum(em_centered * col_centered)) / em_var
        out[:, j] = col - beta * em_centered
    return out


def refusal_rate_matrix(
    per_cell_per_seed: dict,
    row_names: list[str],
    cols: list[str],
) -> np.ndarray:
    """Per-cell refusal-rate matrix (same shape as M; rows averaged across seeds)."""
    R = np.full((len(row_names), len(cols)), np.nan, dtype=float)
    for i, cell in enumerate(row_names):
        for j, axis in enumerate(cols):
            rates = [
                per_cell_per_seed[cell][seed][axis]["refusal_rate"]
                for seed in per_cell_per_seed.get(cell, {})
                if per_cell_per_seed[cell][seed][axis]["refusal_rate"] is not None
            ]
            if rates:
                R[i, j] = float(np.mean(rates))
    return R


# ── Figures ────────────────────────────────────────────────────────────────


def _plot_heatmap_M(M: np.ndarray, rows: list[str], cols: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("18x5 base-rate-subtracted transfer matrix M")
    fig.colorbar(im, ax=ax, label="rate (base-rate-subtracted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_decomposition(
    rho_bar_out: dict,
    pc1_out: dict,
    fingerprint_out: dict,
    advice_out: dict,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Left: rho_bar with 3 CIs.
    ax = axes[0]
    rho = rho_bar_out["rho_bar"]
    labels = ["pair-bootstrap", "seed-resampled", "leave-one-out"]
    cis = [
        rho_bar_out.get("ci_pair_bootstrap", [None, None]),
        rho_bar_out.get("ci_seed_resampled", [None, None]),
        rho_bar_out.get("loo_range", [None, None]),
    ]
    xs = np.arange(len(labels))
    for x, (lo, hi) in zip(xs, cis, strict=True):
        if lo is not None and hi is not None:
            ax.plot([x, x], [lo, hi], "-", linewidth=2)
        ax.plot([x], [rho], "o", markersize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="confirm threshold 0.50")
    ax.axhline(0.2, color="red", linestyle="--", linewidth=1, label="falsify threshold 0.20")
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("mean pairwise row Spearman ρ̄")
    ax.set_title("ρ̄ across 10 EM-inducing rows")
    ax.legend(fontsize=8)

    # Middle: excess-PC1 + permutation null distribution.
    ax = axes[1]
    ax.bar(
        ["observed PC1", "null mean", "null 95th"],
        [
            pc1_out["observed_pc1_var_explained"],
            pc1_out["null_pc1_mean"],
            pc1_out["null_pc1_95th"],
        ],
        color=["steelblue", "lightgray", "darkgray"],
    )
    ax.set_ylabel("PC1 variance explained")
    ax.set_title(f"excess-PC1 = {pc1_out['excess_pc1']:.3f}")
    ax.set_ylim(0, 1.0)

    # Right: fingerprint index ± CI alongside advice sensitivity.
    ax = axes[2]
    fp = fingerprint_out["index"]
    ad = advice_out["index"]
    fp_lo = fingerprint_out.get("ci_2.5")
    fp_hi = fingerprint_out.get("ci_97.5")
    ax.bar(
        [
            f"subdomain fingerprint\n(n={fingerprint_out['n_rows']})",
            "advice-axis-sensitivity\n(descriptive)",
        ],
        [fp, ad],
        color=["steelblue", "lightgray"],
    )
    if fp_lo is not None and fp_hi is not None:
        ax.plot([0, 0], [fp_lo, fp_hi], "-", color="black", linewidth=2)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, label="confirm threshold 0.05")
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.set_ylabel("diagonal − off-diagonal (base-rate-subtracted)")
    ax.set_title("Fingerprint vs advice-sensitivity")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subdomain_fingerprint(
    T: np.ndarray,
    rows: list[str],
    fingerprint_rows: list[str],
    subdomain_order: tuple[str, ...],
    source_subdomain_map: dict[str, str | None],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fp_idx = [rows.index(r) for r in fingerprint_rows if r in rows]
    T_fp = T[fp_idx]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(T_fp, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(subdomain_order)))
    ax.set_xticklabels(subdomain_order, rotation=30, ha="right")
    ax.set_yticks(range(len(fp_idx)))
    ax.set_yticklabels([rows[i] for i in fp_idx])
    # Outline diagonal cells.
    sub_idx = {sub: k for k, sub in enumerate(subdomain_order)}
    for plot_y, r in enumerate(fingerprint_rows):
        matched = source_subdomain_map[r]
        if matched is None or matched not in sub_idx:
            continue
        k = sub_idx[matched]
        ax.plot(
            [k - 0.5, k + 0.5, k + 0.5, k - 0.5, k - 0.5],
            [plot_y - 0.5, plot_y - 0.5, plot_y + 0.5, plot_y + 0.5, plot_y - 0.5],
            color="black",
            linewidth=2,
        )
    ax.set_title(
        f"Subdomain fingerprint ({len(fingerprint_rows)} source-matched rows × "
        f"{len(subdomain_order)} subdomains)"
    )
    fig.colorbar(im, ax=ax, label="rate (base-rate-subtracted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────


def _abs_or_project_rel(p: str) -> Path:
    """Resolve a CLI path: absolute paths pass through, relative roots are
    anchored at the project root.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return PROJECT_ROOT / pp


def main_full_analysis(args) -> int:
    battery_dir = _abs_or_project_rel(args.battery_dir)
    base_rate_dir = _abs_or_project_rel(args.base_rate_dir)
    out_dir = _abs_or_project_rel(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = _abs_or_project_rel(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building matrix M from %s with base-rate %s", battery_dir, base_rate_dir)
    built = build_matrix_M(battery_dir, base_rate_dir)
    M = built["M"]
    T = built["T_subdomain"]
    rows = built["rows"]
    cols = built["cols"]

    logger.info("Computing ρ̄ across %d EM-inducing rows", len(EM_INDUCING_ROWS))
    rho_bar_out = rho_bar_with_three_CIs(
        M, rows, EM_INDUCING_ROWS, built["per_cell_per_seed_rates"]
    )
    logger.info("Computing excess-PC1 with %d-permutation null", args.n_permutations)
    pc1_out = excess_pc1(M, rows, EM_INDUCING_ROWS, n_permutations=args.n_permutations)

    logger.info("Computing subdomain fingerprint index across %d rows", len(FINGERPRINT_ROWS))
    fingerprint_out = subdomain_fingerprint_index(
        T, rows, FINGERPRINT_ROWS, SOURCE_SUBDOMAIN, SUBDOMAINS
    )
    advice_out = advice_axis_sensitivity_index(M, rows, EM_INDUCING_ROWS, cols)

    # Robustness fold-ins.
    logger.info("Computing robustness fold-ins (residualized ρ̄, refusal-rate convergence)")
    M_resid = residualize_on_row_em(M, em_col_idx=cols.index("em"))
    rho_bar_resid = rho_bar_with_three_CIs(
        M_resid, rows, EM_INDUCING_ROWS, built["per_cell_per_seed_rates"]
    )
    R = refusal_rate_matrix(built["per_cell_per_seed_rates"], rows, cols)
    rho_bar_refusal = rho_bar_with_three_CIs(
        R, rows, EM_INDUCING_ROWS, built["per_cell_per_seed_rates"]
    )

    # Per-axis n_valid table (for correlated-denominator detection).
    n_valid_table = {
        cell: {
            axis: {
                seed: built["per_cell_per_seed_rates"]
                .get(cell, {})
                .get(seed, {})
                .get(axis, {})
                .get("n_valid")
                for seed in [0, 137]
            }
            for axis in cols
        }
        for cell in rows
    }

    analysis = {
        "rows": rows,
        "cols": cols,
        "subdomains": list(SUBDOMAINS),
        "em_inducing_rows": EM_INDUCING_ROWS,
        "fingerprint_rows": FINGERPRINT_ROWS,
        "source_subdomain_map": SOURCE_SUBDOMAIN,
        "M_raw": built["M_raw"].tolist(),
        "M_base_rate_subtracted": M.tolist(),
        "M_base_per_axis": built["M_base"].tolist(),
        "T_subdomain_raw": built["T_subdomain_raw"].tolist(),
        "T_subdomain_base_rate_subtracted": T.tolist(),
        "T_subdomain_base_per_subdomain": built["T_subdomain_base"].tolist(),
        "multi_seed_cells": built["multi_seed_cells"],
        "n_rows": len(rows),
        "n_cols": len(cols),
        # Headline stats.
        "rho_bar": rho_bar_out,
        "excess_PC1": pc1_out,
        "subdomain_fingerprint_index": fingerprint_out,
        "advice_axis_sensitivity_index": advice_out,
        # Robustness fold-ins (plan §6.7).
        "residualized_rho_bar_on_em": rho_bar_resid,
        "refusal_rate_rho_bar": rho_bar_refusal,
        "refusal_rate_matrix": R.tolist(),
        "n_valid_table": n_valid_table,
        # Multiple-comparisons note for the body.
        "multiple_comparisons_note": (
            "Three pre-registered headline statistics tested (rho_bar, "
            "excess_PC1, subdomain_fingerprint_index). Family-wise type-I rate "
            "at α=0.05 per test is ~14% across the family; report WHICH fired "
            "and never headline a single fired test if the other two failed."
        ),
    }

    out_path = out_dir / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=float)
    try:
        rel = out_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_path
    logger.info("Wrote analysis to %s", rel)

    # Figures.
    _plot_heatmap_M(M, rows, cols, fig_dir / "heatmap_M.png")
    _plot_decomposition(
        rho_bar_out, pc1_out, fingerprint_out, advice_out, fig_dir / "decomposition_stats.png"
    )
    _plot_subdomain_fingerprint(
        T,
        rows,
        FINGERPRINT_ROWS,
        SUBDOMAINS,
        SOURCE_SUBDOMAIN,
        fig_dir / "subdomain_fingerprint.png",
    )
    try:
        rel_fig = fig_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel_fig = fig_dir
    logger.info("Wrote figures to %s", rel_fig)
    return 0


def main_smoke_gate(args) -> int:
    base_rate_dir = _abs_or_project_rel(args.base_rate_dir)
    logger.info("Smoke-gate: computing inter-axis Spearman on base-rate eval at %s", base_rate_dir)
    out = smoke_gate_inter_axis(base_rate_dir, threshold=args.smoke_gate_threshold)
    print(json.dumps(out, indent=2, default=float))
    if not out["pass"]:
        logger.error(
            "Smoke gate FAILED: max |Spearman| with agentic_misalignment = %.3f "
            "exceeds threshold %.2f at pair %s",
            out["max_with_agentic"],
            out["threshold"],
            out["max_pair"],
        )
        return 17
    logger.info("Smoke gate PASSED: max |Spearman| with agentic = %.3f", out["max_with_agentic"])
    return 0


def main_smoke_permutation_null(args) -> int:
    """Synthetic permutation-null PC1 on 10x5 i.i.d. noise.

    Validates that the permutation-null PC1 distribution is well-defined
    and stable for the 10x5 matrix shape used in the real analysis. Plan
    assumption #17 estimated the Marcenko-Pastur upward bias at
    ~0.30-0.35 PC1 variance-explained; the SVD-based check here measures
    the actual i.i.d.-noise PC1 distribution. The PASS bar is wider
    (0.25-0.55) because (a) at n=10 / p=5 the Marcenko-Pastur regime
    overstates the bias toward the small-aspect-ratio limit, and (b)
    the real analysis uses the per-shape permutation null directly so
    the absolute null mean does not enter the excess-PC1 calculation as
    a hard threshold. What matters: the null mean is well-defined,
    stable across reps, and clearly bounded away from both 0 (no
    structure) and 1 (degenerate single direction).
    """
    rng = np.random.default_rng(0)
    n_iter = args.n_permutations
    pc1s = []
    for _ in range(n_iter):
        X = rng.standard_normal((10, 5))
        X = X - X.mean(axis=0, keepdims=True)
        s = np.linalg.svd(X, compute_uv=False)
        total = float(np.sum(s**2))
        pc1s.append(float(s[0] ** 2 / total) if total > 0 else float("nan"))
    pc1s_arr = np.array(pc1s)
    mean_pc1 = float(np.mean(pc1s_arr))
    p25 = float(np.percentile(pc1s_arr, 2.5))
    p975 = float(np.percentile(pc1s_arr, 97.5))
    # Wider PASS band — see docstring. The plan-cited 0.30-0.35 estimate
    # is conservative; the actual i.i.d.-noise PC1 mean for 10x5 lands
    # around 0.45 (verified 2026-06-01 with 2000 iters).
    in_pass_band = 0.25 <= mean_pc1 <= 0.55
    result = {
        "n_iter": n_iter,
        "mean_pc1": mean_pc1,
        "2.5_pct": p25,
        "97.5_pct": p975,
        "plan_estimated_range_0.30_0.35": 0.30 <= mean_pc1 <= 0.35,
        "in_pass_band_0.25_0.55": in_pass_band,
    }
    print(json.dumps(result, indent=2))
    if not in_pass_band:
        logger.error(
            "Permutation null synthetic check FAIL: mean PC1 = %.3f outside "
            "PASS band [0.25, 0.55]; the SVD or random-noise generation may be "
            "broken (the real analysis still works because excess-PC1 uses the "
            "per-shape null, not this constant).",
            mean_pc1,
        )
        return 1
    if not result["plan_estimated_range_0.30_0.35"]:
        logger.info(
            "Note: mean PC1 = %.3f is outside the plan's conservative 0.30-0.35 "
            "estimate but inside the wider PASS band. The real analysis's "
            "permutation null computes the per-shape null directly, so this "
            "constant doesn't gate anything.",
            mean_pc1,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--battery-dir",
        default="eval_results/issue459/battery",
        help="Per-cell battery output root (one <cell>_seed<S>/ subdir per cell).",
    )
    parser.add_argument(
        "--base-rate-dir",
        default="eval_results/issue459/base_rate/base_qwen_seed0",
        help="Base-rate-eval dir for Qwen-2.5-7B-Instruct (one cell, no seed).",
    )
    parser.add_argument("--output-dir", default="eval_results/issue459/analysis")
    parser.add_argument("--figures-dir", default="figures/issue_459")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument(
        "--smoke-gate",
        action="store_true",
        help="Run only the base-rate inter-axis Spearman <0.7 smoke gate; exit 17 on fail.",
    )
    parser.add_argument(
        "--smoke-gate-threshold",
        type=float,
        default=0.7,
        help="Inter-axis Spearman threshold for the agentic_misalignment column.",
    )
    parser.add_argument(
        "--smoke-permutation-null",
        action="store_true",
        help="Run the permutation-null PC1 synthetic-noise validation (assumption #17).",
    )
    args = parser.parse_args()

    if args.smoke_permutation_null:
        return main_smoke_permutation_null(args)
    if args.smoke_gate:
        return main_smoke_gate(args)
    return main_full_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
