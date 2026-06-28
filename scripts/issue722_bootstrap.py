#!/usr/bin/env python3
"""Issue #722 — family-clustered bootstrap for the headline scalar-Δ DV.

The headline statistic for #722 is a per-cell SCALAR location statistic (the
median over the c-grid of ``|Δ(c)·r̂_B|``), NOT a Spearman correlation between
two paired arrays. #667's ``clustered_bootstrap_spearman`` computes a CI on
``Spearman(x, y)`` and would error on the shape assert
(``x.shape == y.shape == fams.shape``) if fed a single value array — see plan
§4 MF#1. This module provides the matching family-resampling CI helper for a
single value array, plus ``make_refit_pair``, the shared harness that builds the
three refit/shift floors through IDENTICAL bootstrap+random-init refit logic so
refit noise cancels in the floor and the Δ stays interpretable even when the
fitted map M is individually weak (plan §3 / §4.5.1 / §12).

This file is intentionally fit-machinery-agnostic: ``make_refit_pair`` takes a
``fit_fn`` callback so the same harness drives both the ridge (closed-form) and
MLP (gradient-descent) refits from ``issue722_fit_M.py`` without importing it
(no circular import).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

_AGG = {"median": np.median, "mean": np.mean}


def clustered_bootstrap_scalar(
    values: Sequence[float],
    families: Sequence[str],
    *,
    statistic: str = "median",
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Family-clustered percentile CI on a SCALAR location statistic of ``values``.

    Mirrors ``gate_chain.clustered_bootstrap_spearman``'s family-resampling loop
    (resample whole ``target_cid`` families with replacement, so the CI respects
    the ~7-family cluster structure) but aggregates a single value array with
    ``statistic`` (``median`` default; ``mean`` exposed for the robustness dump)
    instead of correlating two arrays.

    ``values`` and ``families`` are parallel arrays (one entry per cell). Returns
    ``{"point", "ci_lo", "ci_hi", "n_families"}`` as a percentile CI. A
    degenerate input (<2 distinct families, or empty) returns a point-only CI.
    """
    vals = np.asarray(values, dtype=float)
    fams = np.asarray(list(families), dtype=object)
    assert vals.shape == fams.shape, (vals.shape, fams.shape)
    if statistic not in _AGG:
        raise ValueError(f"unknown statistic {statistic!r} (want one of {sorted(_AGG)})")
    agg = _AGG[statistic]
    if vals.size == 0:
        return {
            "point": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_families": 0,
        }
    point = float(agg(vals))
    uniq = sorted({str(f) for f in fams})
    if len(uniq) < 2:
        return {"point": point, "ci_lo": point, "ci_hi": point, "n_families": len(uniq)}
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    n_fam = len(uniq)
    for r in range(n_resamples):
        chosen = rng.choice(uniq, size=n_fam, replace=True)
        idx = np.concatenate([fam_to_idx[f] for f in chosen])
        boot[r] = agg(vals[idx])
    return {
        "point": point,
        "ci_lo": float(np.percentile(boot, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "n_families": n_fam,
    }


def floor_sd(values: Sequence[float]) -> float:
    """SD of a floor distribution (used to express Δ_med in floor-SD units)."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1))


def _resample_family_idx(
    fam_to_idx: dict[str, np.ndarray], uniq: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    """Resample whole FAMILY labels with replacement → concatenated row indices.

    Mirrors ``clustered_bootstrap_scalar`` / ``gate_chain.clustered_bootstrap_spearman``:
    draw ``len(uniq)`` family labels with replacement, then concatenate the rows
    belonging to each sampled label. The returned index array can be longer or
    shorter than ``n`` (families have uneven sizes) — that is correct
    family-clustered resampling, and ``fit_fn`` handles an arbitrary row count.
    """
    chosen = rng.choice(list(uniq), size=len(uniq), replace=True)
    return np.concatenate([fam_to_idx[str(f)] for f in chosen])


def make_refit_pair(
    X: np.ndarray,
    Y: np.ndarray,
    fit_fn: Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray],
    eval_grid: np.ndarray,
    r_hat: np.ndarray,
    families: Sequence[str],
    *,
    n_pairs: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Build a refit-floor distribution of per-pair median projected distances.

    The IDENTICAL bootstrap+random-init refit harness behind all three floors
    (``floor_M0_refit``, ``floor_Mplus_refit``, ``floor_shifted``; plan §4.5.1).
    For each of ``n_pairs`` pairs, draw TWO independent **family-clustered**
    resamples of ``(X, Y)`` (resample whole ``target_cid`` families with
    replacement — MF#6, NOT a row-i.i.d. bootstrap) and fit TWO maps with
    INDEPENDENT random inits (the per-call ``np.random.Generator`` reseeds the
    fit). The pair's statistic is
    ``median_c |(fit_a(eval_grid) - fit_b(eval_grid))·r̂_B|`` — two equally-weak
    refits of the SAME underlying map, so refit noise (NOT a true function
    change) drives it. Returns the (n_pairs,) array; the caller takes its 95th
    percentile as the floor and its SD for floor-SD units.

    ``families`` is a parallel array (one label per row of ``X`` / ``Y``); the
    SAME family-resampling unit the headline Δ CI (``clustered_bootstrap_scalar``)
    uses, so the H_function gate compares a family-clustered numerator against a
    family-clustered floor (mismatched sampling units biased the gate — MF#6).

    ``fit_fn(X_boot, Y_boot, rng)`` must fit a map on the bootstrap sample and
    return predictions on ``eval_grid`` of shape ``(n_grid, P)`` (P == Y.shape[1]).
    The two pair members differ ONLY by their independent ``rng`` AND their
    independent family-clustered resample, exactly the refit/bootstrap noise
    lever (the store is seed42-only, so there is no cross-seed lever — plan §5).

    Degenerate fallback: with <2 distinct families the family resample cannot
    vary the cluster mix, so it falls back to an i.i.d. ROW bootstrap (a single
    family means clustering is a no-op) — keeps tiny smokes runnable while the
    production grid (~7 families) always takes the clustered path.
    """
    n = X.shape[0]
    r_hat = np.asarray(r_hat, dtype=float)
    fams = np.asarray(list(families), dtype=object)
    assert fams.shape == (n,), (fams.shape, n)
    uniq = sorted({str(f) for f in fams})
    clustered = len(uniq) >= 2
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n_pairs, dtype=float)
    for p in range(n_pairs):
        if clustered:
            idx_a = _resample_family_idx(fam_to_idx, uniq, rng)
            idx_b = _resample_family_idx(fam_to_idx, uniq, rng)
        else:
            idx_a = rng.integers(0, n, size=n)
            idx_b = rng.integers(0, n, size=n)
        rng_a = np.random.default_rng(rng.integers(0, 2**31 - 1))
        rng_b = np.random.default_rng(rng.integers(0, 2**31 - 1))
        pred_a = fit_fn(X[idx_a], Y[idx_a], rng_a)  # (n_grid, P)
        pred_b = fit_fn(X[idx_b], Y[idx_b], rng_b)
        delta = pred_a - pred_b  # (n_grid, P)
        proj = np.abs(delta @ r_hat)  # (n_grid,)
        out[p] = float(np.median(proj))
    return out
