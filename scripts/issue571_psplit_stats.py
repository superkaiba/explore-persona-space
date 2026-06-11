# ruff: noqa: RUF002
# Intentional Unicode (Δ) in scientific docstrings.
"""Task #571 persona-split-composition — shared registered-inference helpers.

Used by both ``issue571_psplit_stage1_analysis.py`` (Stage 1, context-typed
geometry join) and ``issue571_psplit_analysis.py`` (Stage 2, persona-typed
arms). One implementation so the two stages' registered statistics are
computed identically (plan v2 §4.1 / §6 Inference):

- ``spearman``                plain Spearman rho (scipy, average ranks).
- ``partial_spearman``        rank-based partial correlation x ⊥ y | z.
- ``residualized_spearman``   the §4.1 primary statistic under collinearity:
                              Spearman(y, resid(x)), resid(x) = residuals of
                              x on a linear+quadratic fit of z (RAW values).
- ``perm_p``                  two-sided permutation p, permuting the LEAKAGE
                              vector only (persona as the exchangeable unit),
                              vectorized; 10,000 draws default.
- ``bootstrap_ci``            percentile bootstrap CI on any statistic,
                              resampling personas with replacement (full
                              statistic recompute per resample).
- ``holm``                    Holm step-down correction over a p-value dict.

Degenerate inputs (zero-variance vectors — e.g. a hijack channel pinned at
ceiling) return ``nan`` statistics with ``degenerate=True`` rather than
crashing: a pinned channel is a reportable outcome, not an error.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata, spearmanr

N_PERM_DEFAULT = 10_000
N_BOOT_DEFAULT = 10_000
SEED_DEFAULT = 42


def _is_degenerate(*vecs: np.ndarray) -> bool:
    """True when any input vector has (near-)zero variance (ties everywhere)."""
    return any(float(np.std(v)) < 1e-12 for v in vecs)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Plain Spearman rho (nan on degenerate input)."""
    if _is_degenerate(x, y):
        return float("nan")
    return float(spearmanr(x, y)[0])


def _rank_center(v: np.ndarray) -> np.ndarray:
    r = rankdata(v)
    return r - r.mean()


def _proj_residual_matrix(z: np.ndarray) -> np.ndarray:
    """M = I − Z(Z'Z)⁻¹Z' for Z = [1, rank(z)] — rank-residualizer for partials."""
    n = len(z)
    Z = np.column_stack([np.ones(n), rankdata(z)])
    return np.eye(n) - Z @ np.linalg.pinv(Z)


def partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Rank-based partial correlation of y and x controlling z.

    Pearson correlation of the residuals of rank(y) and rank(x) after
    regressing each on [1, rank(z)] (the standard partial-Spearman
    construction). nan on degenerate input.
    """
    if _is_degenerate(x, y, z):
        return float("nan")
    M = _proj_residual_matrix(z)
    ry, rx = M @ rankdata(y), M @ rankdata(x)
    denom = float(np.linalg.norm(ry) * np.linalg.norm(rx))
    if denom < 1e-12:
        return float("nan")
    return float(ry @ rx / denom)


def quad_residuals(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residuals of x on a linear+quadratic fit of z (RAW values, §4.1)."""
    Z = np.column_stack([np.ones(len(z)), z, z**2])
    beta, *_ = np.linalg.lstsq(Z, x, rcond=None)
    return x - Z @ beta


def residualized_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """The §4.1 registered primary: Spearman(y, resid(x on quad(z)))."""
    if _is_degenerate(x, y, z):
        return float("nan")
    return spearman(y, quad_residuals(x, z))


def _perm_matrix(n: int, n_perm: int, seed: int) -> np.ndarray:
    """(n_perm, n) row-permutation index matrix."""
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((n_perm, n)), axis=1)


def perm_p(
    stat_fn: Callable[[np.ndarray], float],
    leakage: np.ndarray,
    observed: float,
    *,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> float:
    """Two-sided permutation p permuting the leakage vector (generic, loop)."""
    if not np.isfinite(observed):
        return float("nan")
    idx = _perm_matrix(len(leakage), n_perm, seed)
    hits = 1
    for row in idx:
        s = stat_fn(leakage[row])
        if np.isfinite(s) and abs(s) >= abs(observed) - 1e-12:
            hits += 1
    return hits / (n_perm + 1)


def perm_p_corr_vs_fixed(
    leakage: np.ndarray,
    fixed_rank_side: np.ndarray,
    observed: float,
    *,
    proj: np.ndarray | None = None,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> float:
    """Vectorized two-sided permutation p for rank-correlation statistics.

    Covers every registered Stage-1/2 read: each is a Pearson correlation
    between (optionally projected) rank(leakage) and a FIXED vector
    ``fixed_rank_side`` (rank of resid(d_nn), projected rank(d_nn), rank of
    d_src, ...). Permuting leakage permutes its ranks, so the null
    distribution is computable as one matrix product over ``n_perm``
    row-permutations. ``proj`` (the partial-correlation residualizer M) is
    applied to the permuted rank rows when given.
    """
    if not np.isfinite(observed):
        return float("nan")
    rl = rankdata(leakage)
    perms = rl[_perm_matrix(len(leakage), n_perm, seed)]  # (n_perm, n)
    if proj is not None:
        perms = perms @ proj.T
    a = fixed_rank_side - (0.0 if proj is not None else fixed_rank_side.mean())
    perms = perms - (0.0 if proj is not None else perms.mean(axis=1, keepdims=True))
    num = perms @ a
    denom = np.linalg.norm(perms, axis=1) * float(np.linalg.norm(a))
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = num / denom
    finite = np.isfinite(stats)
    hits = 1 + int(np.sum(np.abs(stats[finite]) >= abs(observed) - 1e-12))
    return hits / (n_perm + 1)


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    n: int,
    *,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
    ci: float = 95.0,
) -> tuple[float, float, int]:
    """Percentile bootstrap CI resampling persona indices with replacement.

    ``stat_fn`` receives an index array (with repeats) and recomputes the
    FULL statistic on the resample (including any residualization fit).
    Returns (lo, hi, n_dropped_nan); degenerate resamples are dropped and
    counted.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([stat_fn(row) for row in idx], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    n_dropped = int(len(vals) - len(finite))
    if len(finite) < max(100, n_boot // 10):
        return float("nan"), float("nan"), n_dropped
    lo, hi = np.percentile(finite, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi), n_dropped


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down correction; nan p-values pass through as nan."""
    items = [(k, v) for k, v in p_values.items() if np.isfinite(v)]
    out: dict[str, float] = {k: float("nan") for k in p_values}
    m = len(items)
    prev = 0.0
    for rank, (k, p) in enumerate(sorted(items, key=lambda kv: kv[1])):
        adj = min(1.0, max(prev, (m - rank) * p))
        out[k] = adj
        prev = adj
    return out


def ci_excludes_zero(lo: float, hi: float) -> bool:
    """True when a finite CI excludes 0."""
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
