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

import logging
from collections.abc import Callable, Sequence

import numpy as np

logger = logging.getLogger("issue722.bootstrap")

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
    fit_fn: Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray] | None,
    eval_grid: np.ndarray,
    r_hat: np.ndarray,
    families: Sequence[str],
    *,
    n_pairs: int = 100,
    seed: int = 0,
    skip_counter: dict | None = None,
    batched_chain_fn: Callable[
        [list[np.ndarray], list[np.random.Generator]], list[np.ndarray | None]
    ]
    | None = None,
    per_draw_out: dict | None = None,
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

    SVD-non-convergence guard (issue #722 round 3): a heavily-duplicated
    family-clustered resample is mean-centered to a rank-deficient ``Vc`` inside
    ``fit_fn`` (the ridge refit's ``_pca_basis_v0``). ``_pca_basis_v0`` already
    falls back from LAPACK ``gesdd`` to the robust ``gesvd`` driver, but on the
    rare resample where EVEN ``gesvd`` cannot converge (or any other degenerate
    linear-algebra failure surfaces in the refit) a single bad pair would
    otherwise crash the whole production fit. So each pair is wrapped: a
    ``LinAlgError`` from either fit SKIPS that pair (logged) rather than aborting
    the run — losing 1-2 of ``n_pairs`` pairs to non-convergence is acceptable
    bootstrap noise. The returned array is the SURVIVING pairs (length
    ``n_pairs - n_skipped``); the caller's 95th-percentile floor is unbiased over
    the survivors. ``skip_counter`` (if passed, a mutable dict) records
    ``{"n_attempted", "n_skipped"}`` so the caller can surface the skip RATE — a
    skip rate above ~5% means the resample geometry is pathological and is raised
    as a CONCERN, not silently absorbed. (The crash class: round-2's unguarded
    ``np.linalg.svd`` crashed the GCP run at sycophancy L7 on exactly such a
    resample; the 3 em cells had fit cleanly.)

    #811 batched path (plan §4.3 item 10): when ``batched_chain_fn`` is passed,
    the per-pair fits are delegated to it IN BATCH — it receives every resample
    index array (+ a per-fit ``Generator``, unused by deterministic fits) and
    returns per-resample chain projections ``fit(X[idx], Y[idx]).predict(grid)
    @ r_hat`` (or ``None`` = skip, the LinAlgError semantics). The resample/seed
    STREAM is identical to the serial path (pre-drawn in the same rng order), so
    a seeded serial-oracle equivalence check is a pure numerics comparison.
    ``fit_fn`` may then be ``None``. ``per_draw_out`` (a mutable dict) receives
    ``{"stats": (n_pairs,) float}`` with NaN at skipped pairs — DRAW-ALIGNED
    across summaries at the same seed (the #811 ``bootstrap_draws_*.npz`` dump /
    selection-null escape input). The serial ``fit_fn`` path is retained
    verbatim as the equivalence ORACLE; this module stays fit-machinery-agnostic
    either way.
    """
    n = X.shape[0]
    r_hat = np.asarray(r_hat, dtype=float)
    fams = np.asarray(list(families), dtype=object)
    assert fams.shape == (n,), (fams.shape, n)
    assert fit_fn is not None or batched_chain_fn is not None, (
        "make_refit_pair needs fit_fn (serial oracle) or batched_chain_fn (batched path)"
    )
    uniq = sorted({str(f) for f in fams})
    clustered = len(uniq) >= 2
    fam_to_idx = {f: np.where(fams.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    # Pre-draw ALL resample index pairs + per-fit rng seeds in the EXACT stream
    # order the historical serial loop used (idx_a, idx_b, seed_a, seed_b per
    # pair) — so the serial oracle and the batched path see BIT-IDENTICAL
    # resamples/seeds and the #811 seeded serial-oracle equivalence gate is a
    # pure numerics comparison (plan §4.3 item 10 / §13 smoke 7).
    idx_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    seed_pairs: list[tuple[int, int]] = []
    for _p in range(n_pairs):
        if clustered:
            idx_a = _resample_family_idx(fam_to_idx, uniq, rng)
            idx_b = _resample_family_idx(fam_to_idx, uniq, rng)
        else:
            idx_a = rng.integers(0, n, size=n)
            idx_b = rng.integers(0, n, size=n)
        seed_a = int(rng.integers(0, 2**31 - 1))
        seed_b = int(rng.integers(0, 2**31 - 1))
        idx_pairs.append((idx_a, idx_b))
        seed_pairs.append((seed_a, seed_b))
    # per_draw keeps DRAW-ALIGNED stats (NaN at skipped pairs) — the same draw
    # index maps to the same family resample across every summary fit at the same
    # seed, which is what makes the #811 per-draw dump's max-over-summaries
    # selection-null escape a pure read (plan §6 / §6.5).
    per_draw = np.full(n_pairs, np.nan, dtype=float)
    n_skipped = 0
    if batched_chain_fn is not None:
        # Batched Gram/dual-space path: the callee receives ALL 2·n_pairs resample
        # index arrays at once (the batch axis) and returns per-resample CHAIN
        # projections fit(X[idx], Y[idx]).predict(eval_grid) @ r_hat — (n_grid,)
        # each, or None for a LinAlgError-equivalent skip. The pair statistic
        # median|chain_a - chain_b| equals the serial median|(pred_a - pred_b)·r̂|
        # exactly (the dot product distributes over the subtraction). This module
        # stays fit-machinery-agnostic — the Gram/dual-space knowledge lives in
        # the callee (issue722_fit_M.make_batched_refit_chain_fn).
        flat_idx = [i for pair in idx_pairs for i in pair]
        flat_rngs = [np.random.default_rng(s) for pair in seed_pairs for s in pair]
        chains = batched_chain_fn(flat_idx, flat_rngs)
        assert len(chains) == 2 * n_pairs, (len(chains), n_pairs)
        for p in range(n_pairs):
            ca, cb = chains[2 * p], chains[2 * p + 1]
            if ca is None or cb is None:
                n_skipped += 1
                logger.warning(
                    "[phase=fit_M] make_refit_pair(batched): skipping bootstrap "
                    "pair %d/%d (degenerate refit); %d skipped so far",
                    p + 1,
                    n_pairs,
                    n_skipped,
                )
                continue
            per_draw[p] = float(np.median(np.abs(np.asarray(ca) - np.asarray(cb))))
    else:
        for p, ((idx_a, idx_b), (seed_a, seed_b)) in enumerate(
            zip(idx_pairs, seed_pairs, strict=True)
        ):
            rng_a = np.random.default_rng(seed_a)
            rng_b = np.random.default_rng(seed_b)
            try:
                pred_a = fit_fn(X[idx_a], Y[idx_a], rng_a)  # (n_grid, P)
                pred_b = fit_fn(X[idx_b], Y[idx_b], rng_b)
            except np.linalg.LinAlgError as e:
                # Defensive: _pca_basis_v0 already retries gesdd->gesvd, so this
                # fires only on the rare resample where even gesvd cannot converge
                # (or another degenerate refit). Skip the pair; never crash the fit.
                n_skipped += 1
                logger.warning(
                    "[phase=fit_M] make_refit_pair: skipping bootstrap pair %d/%d "
                    "after LinAlgError in the refit (%s); %d skipped so far",
                    p + 1,
                    n_pairs,
                    e,
                    n_skipped,
                )
                continue
            delta = pred_a - pred_b  # (n_grid, P)
            proj = np.abs(delta @ r_hat)  # (n_grid,)
            per_draw[p] = float(np.median(proj))
    survivors = per_draw[~np.isnan(per_draw)]
    if skip_counter is not None:
        skip_counter["n_attempted"] = n_pairs
        skip_counter["n_skipped"] = n_skipped
    if per_draw_out is not None:
        per_draw_out["stats"] = per_draw
    if survivors.size == 0:
        # Every pair failed — a genuinely degenerate fit; fail loud rather than
        # return an empty floor that the caller's np.percentile would crash on.
        raise np.linalg.LinAlgError(
            f"make_refit_pair: all {n_pairs} refit pairs failed with LinAlgError "
            "(the resample geometry is fully degenerate — cannot build a floor)"
        )
    return survivors.astype(float)
