"""Issue #537 registered estimators -- v6 question-level noise machinery.

Plan v6 §3 / §6: every noise correction is QUESTION-level (single seed -- the
seed-split estimators of the #524-v6 import are not computable at 1 seed and
are NOT implemented here). Registered forms:

- per-cell noise variance: question-level bootstrap of the cell mean
  (B=2000) -- :func:`question_bootstrap_var` / :func:`question_bootstrap_ci`;
- split-half cross-check: the same K=200 random half-splits applied to every
  cell, ``Var_noise(G_full) = E_splits[(G_A - G_B)²] / 4`` (exact for
  independent question draws) -- :func:`split_half_noise_var`;
- EM row exception (plan A37): 8-question pool makes split-half degenerate →
  the EM noise floor uses the question-cluster response-level bootstrap --
  :func:`cluster_bootstrap_var`;
- H-structure read: per-train-row between-eval-context variance of G[i→·]
  (off-diagonal, per row i, then averaged over i -- NOT the variance of the
  i-averaged mean profile) vs the mean per-cell question-noise floor --
  :func:`h_structure_read`;
- antisymmetric fraction, question-split corrected: A[i,j] computed on two
  disjoint eval-question halves, cross-half covariance over off-diagonal
  pairs -- :func:`antisym_fraction_split_half` (raw fraction from
  :func:`decompose_sym_anti`, provenance ``scripts/issue502_deltaG_symmetry.py``);
- question-split reliability, Spearman-Brown corrected, + disattenuated
  cross-row Spearman -- :func:`spearman_brown` / :func:`disattenuated_spearman`
  (bias direction: question-split OVER-estimates reliability → UNDER-corrects;
  callers must carry the single-seed caveat).

All functions are pure numpy and unit-testable on synthetic per-question data
with known variance (plan A37 harness unit test, exercised by the P3 smoke).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

__all__ = [
    "antisym_fraction_split_half",
    "cluster_bootstrap_var",
    "decompose_sym_anti",
    "disattenuated_spearman",
    "h_structure_read",
    "question_bootstrap_ci",
    "question_bootstrap_var",
    "question_split_reliability",
    "spearman_brown",
    "split_half_noise_var",
]


def question_bootstrap_var(per_q: np.ndarray, b: int = 2000, seed: int = 0) -> float:
    """Bootstrap variance of the cell mean under question resampling.

    Args:
        per_q: shape (n_q,) per-question G values for one cell.
        b: bootstrap replicates (plan: B=2000).
        seed: RNG seed (deterministic per cell; callers derive from cell id).

    Returns:
        Variance (not SD) of the bootstrap distribution of the mean.
    """
    per_q = np.asarray(per_q, dtype=float)
    assert per_q.ndim == 1 and per_q.size >= 2, per_q.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_q.size, size=(b, per_q.size))
    means = per_q[idx].mean(axis=1)
    return float(means.var(ddof=1))


def question_bootstrap_ci(
    per_q: np.ndarray, b: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI (lo, hi) of the cell mean under question resampling."""
    per_q = np.asarray(per_q, dtype=float)
    assert per_q.ndim == 1 and per_q.size >= 2, per_q.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_q.size, size=(b, per_q.size))
    means = per_q[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def split_half_noise_var(per_q: np.ndarray, k: int = 200, seed: int = 0) -> float:
    """Split-half noise-variance estimate of the FULL-pool cell mean.

    ``Var_noise(G_full) = E_splits[(G_A - G_B)²] / 4`` -- exact for independent
    question draws: the two half-means are independent with 2x the full-mean
    variance each, so E[(G_A - G_B)²] = 4 · Var(G_full).

    Args:
        per_q: shape (n_q,) per-question values; n_q must be ≥ 4 (the EM row's
            8-question pool is excepted upstream per plan A37 -- its floor uses
            :func:`cluster_bootstrap_var` instead).
        k: number of random half-splits (plan: K=200, same splits every cell --
            callers pass the same seed for the shared-split protocol).
        seed: RNG seed.
    """
    per_q = np.asarray(per_q, dtype=float)
    assert per_q.ndim == 1 and per_q.size >= 4, per_q.shape
    n = per_q.size
    half = n // 2
    rng = np.random.default_rng(seed)
    sq_diffs = np.empty(k, dtype=float)
    for s in range(k):
        perm = rng.permutation(n)
        g_a = per_q[perm[:half]].mean()
        g_b = per_q[perm[half : 2 * half]].mean()
        sq_diffs[s] = (g_a - g_b) ** 2
    return float(sq_diffs.mean() / 4.0)


def cluster_bootstrap_var(
    per_response: np.ndarray, question_ids: np.ndarray, b: int = 2000, seed: int = 0
) -> float:
    """Question-cluster response-level bootstrap variance of the cell mean.

    Resamples QUESTIONS (clusters) with replacement and keeps every response
    nested within each drawn question (the EM row's 8 Q x 5 samples shape;
    plan §6 per-cell metadata + A37 exception).

    Args:
        per_response: shape (n_responses,) per-response values.
        question_ids: shape (n_responses,) cluster id per response.
    """
    per_response = np.asarray(per_response, dtype=float)
    question_ids = np.asarray(question_ids)
    assert per_response.shape == question_ids.shape, (
        per_response.shape,
        question_ids.shape,
    )
    uniq = np.unique(question_ids)
    assert uniq.size >= 2, f"need >=2 question clusters, got {uniq.size}"
    groups = [per_response[question_ids == q] for q in uniq]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, uniq.size, size=(b, uniq.size))
    means = np.empty(b, dtype=float)
    for i in range(b):
        means[i] = np.concatenate([groups[j] for j in draws[i]]).mean()
    return float(means.var(ddof=1))


def h_structure_read(g_matrix: np.ndarray, per_cell_noise_var: np.ndarray) -> dict[str, float]:
    """H-structure (plan §3, the marker-row kill read).

    Between-eval-context variance of G[i→·] computed PER TRAIN ROW i over its
    off-diagonal eval contexts, then averaged over i (NOT the variance of the
    i-averaged profile, whose floor would scale as per-cell/16), compared to
    the mean per-cell question-noise floor.

    Args:
        g_matrix: shape (n_train, n_eval) cell means; NaN cells are excluded.
            Diagonal cells (same cid trained→evaled) must already be NaN'd by
            the caller when n_train ≠ n_eval indexing applies.
        per_cell_noise_var: same shape; per-cell question-level noise variance.

    Returns:
        dict with ``between_context_var`` (mean over rows of within-row
        variance), ``noise_floor`` (mean per-cell noise var over the same
        cells), ``corrected_var`` (between - floor), ``ratio``
        (corrected / floor) and ``pass_2x`` (ratio ≥ 2 → structure clears the
        kill threshold; single-seed caveat carried by the caller).
    """
    g = np.asarray(g_matrix, dtype=float)
    nv = np.asarray(per_cell_noise_var, dtype=float)
    assert g.shape == nv.shape and g.ndim == 2, (g.shape, nv.shape)
    row_vars = []
    cell_noises = []
    for i in range(g.shape[0]):
        row = g[i]
        mask = ~np.isnan(row)
        if mask.sum() < 3:
            continue
        row_vars.append(row[mask].var(ddof=1))
        cell_noises.append(nv[i][mask])
    assert row_vars, "no usable rows in g_matrix"
    between = float(np.mean(row_vars))
    floor = float(np.nanmean(np.concatenate(cell_noises)))
    corrected = between - floor
    ratio = corrected / floor if floor > 0 else float("inf")
    return {
        "between_context_var": between,
        "noise_floor": floor,
        "corrected_var": corrected,
        "ratio": float(ratio),
        "pass_2x": bool(ratio >= 2.0),
    }


def decompose_sym_anti(m: np.ndarray) -> dict[str, float]:
    """Raw symmetric/antisymmetric variance decomposition (off-diag, mean-removed).

    Provenance: ``scripts/issue502_deltaG_symmetry.py::decompose`` (#502/#524).
    Var(G) = Var(S) + Var(A) with S = (M + Mᵀ)/2, A = (M - Mᵀ)/2.
    The RAW fraction is a diagnostic only at 1 seed (plan §3 H-asymmetry).
    """
    m = np.asarray(m, dtype=float)
    assert m.ndim == 2 and m.shape[0] == m.shape[1], m.shape
    n = m.shape[0]
    s = 0.5 * (m + m.T)
    a = 0.5 * (m - m.T)
    mask = ~np.eye(n, dtype=bool)
    mu = m[mask].mean()
    total = float(((m[mask] - mu) ** 2).mean())
    sym = float(((s - mu)[mask] ** 2).mean())
    anti = float((a[mask] ** 2).mean())
    return {"anti_frac": anti / total, "sym_var": sym, "anti_var": anti, "total_var": total}


def antisym_fraction_split_half(g_half_a: np.ndarray, g_half_b: np.ndarray) -> dict[str, float]:
    """Question-split corrected antisymmetric fraction (plan §3 H-asymmetry, v6).

    Compute A[i,j] = ½(G[i→j] - G[j→i]) separately on two disjoint
    eval-question halves; the cross-half covariance over off-diagonal pairs
    kills question-level measurement noise (the same cross-correlation algebra
    the seed-split form used, applied to question halves). The corrected
    fraction is Cov(A_A, A_B) / Cov(G_A, G_B) over off-diagonal entries.

    Caveat (carried by every caller, verbatim from the plan): A[i,j] mixes
    training noise from BOTH adapters of the pair, which question splits
    cannot remove -- single-seed caveat on every read.

    Args:
        g_half_a / g_half_b: shape (n, n) G matrices on the 16x16
            shared-instance block, each computed from one question half.
    """
    a_mat = np.asarray(g_half_a, dtype=float)
    b_mat = np.asarray(g_half_b, dtype=float)
    assert a_mat.shape == b_mat.shape and a_mat.ndim == 2, (a_mat.shape, b_mat.shape)
    assert a_mat.shape[0] == a_mat.shape[1], a_mat.shape
    n = a_mat.shape[0]
    mask = ~np.eye(n, dtype=bool)

    def _anti(m: np.ndarray) -> np.ndarray:
        return (0.5 * (m - m.T))[mask]

    def _centered(m: np.ndarray) -> np.ndarray:
        v = m[mask]
        return v - v.mean()

    anti_a, anti_b = _anti(a_mat), _anti(b_mat)
    g_a, g_b = _centered(a_mat), _centered(b_mat)
    cov_anti = float(np.mean(anti_a * anti_b))  # anti part is mean-free already
    cov_total = float(np.mean(g_a * g_b))
    assert cov_total != 0.0, "degenerate G halves (zero cross-covariance)"
    return {
        "anti_frac_corrected": cov_anti / cov_total,
        "cov_anti": cov_anti,
        "cov_total": cov_total,
    }


def spearman_brown(r_half: float) -> float:
    """Spearman-Brown correction of a half-pool reliability to full pool length."""
    assert -1.0 <= r_half <= 1.0, r_half
    denom = 1.0 + r_half
    if denom == 0.0:
        return -1.0
    return 2.0 * r_half / denom


def question_split_reliability(g_half_a: np.ndarray, g_half_b: np.ndarray) -> float:
    """Per-row question-split reliability of the off-diagonal G matrix.

    Spearman between the off-diagonal entries of the two half-pool G matrices,
    Spearman-Brown corrected to full pool length. OVER-estimates reliability
    (omits training noise) → downstream disattenuation UNDER-corrects; the
    cross-row read is registered DESCRIPTIVE only (plan §3 H-behavior-dependence).
    """
    a_mat = np.asarray(g_half_a, dtype=float)
    b_mat = np.asarray(g_half_b, dtype=float)
    assert a_mat.shape == b_mat.shape and a_mat.shape[0] == a_mat.shape[1], (
        a_mat.shape,
        b_mat.shape,
    )
    mask = ~np.eye(a_mat.shape[0], dtype=bool)
    r_half = float(spearmanr(a_mat[mask], b_mat[mask]).statistic)
    return spearman_brown(r_half)


def disattenuated_spearman(rho_xy: float, rel_x: float, rel_y: float) -> float:
    """Disattenuate a cross-row Spearman by the two rows' reliabilities.

    rho_corrected = rho_xy / sqrt(rel_x · rel_y). With question-split reliabilities
    this is a LOWER bound on the noise-free cross-row similarity
    (anti-conservative for the < 0.7 read -- plan §3 names the bias direction).
    """
    assert rel_x > 0 and rel_y > 0, (rel_x, rel_y)
    return float(rho_xy / np.sqrt(rel_x * rel_y))
