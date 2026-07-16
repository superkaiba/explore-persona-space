"""Held-out cross-fitted per-dim affine recalibration — math cores (#1336, plan v9).

The E1-validated recalibration adopted as the PRIMARY within-stage read on the
Llama family after the `resume_on_recalibrated_dv` route (plan v9 §4 route 1):
for each fold k, per-dim (a_j, b_j) are fit by least squares on the OTHER
folds' (out-of-fold prediction, truth) pairs and applied to fold k held-out —
no row's recalibration is ever fit on itself. Raw pooled R^2 (fold-local test
mean, the committed pooled convention) is always reported as companion; the
two scales are separate reads, never blended.

These functions were built + code-reviewed in the E1 round inside
``scripts/issue1336_recal_verdict.py`` and are EXTRACTED here (verbatim
bodies) so the production ladder drivers (``issue1336_fit_cells.py``,
``issue1336_ladder_alignment.py``) can import them without pulling the whole
E1 verdict driver (whose module chain imports ``issue1336_diagnose_g1`` ->
``issue1336_fit_cells`` — a circular import from the fit driver).
``issue1336_recal_verdict.py`` re-aliases them under the historical
underscore names — single source of truth, no copy-paste.
"""

from __future__ import annotations

import numpy as np

VAR_EPS = 1e-12  # per-dim variance guard (matches _perdim_from_preds)

__all__ = [
    "VAR_EPS",
    "crossfit_offset_only_ss",
    "crossfit_recal_direct",
    "crossfit_scalar_recal_r2",
    "fold_rows",
    "insample_recal_r2",
    "raw_pooled_r2",
    "recal_r2_from_stats",
    "suff_stats_observed",
]


def fold_rows(folds: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    """Sorted fold ids + per-fold row-index arrays for a fold-assignment vector."""
    ids = sorted(set(int(v) for v in folds))
    return ids, [np.flatnonzero(folds == k) for k in ids]


def suff_stats_observed(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> dict:
    """Per-fold sufficient statistics (K, d) + counts (K,) for the recal math."""
    ids, rows = fold_rows(folds)
    K, d = len(ids), P.shape[1]
    out = {k: np.empty((K, d)) for k in ("s_p", "s_y", "s_pp", "s_yy", "s_py")}
    n = np.empty(K)
    for ki, r in enumerate(rows):
        Pk = P[r].astype(np.float64)
        Yk = Y[r].astype(np.float64)
        out["s_p"][ki] = Pk.sum(0)
        out["s_y"][ki] = Yk.sum(0)
        out["s_pp"][ki] = (Pk * Pk).sum(0)
        out["s_yy"][ki] = (Yk * Yk).sum(0)
        out["s_py"][ki] = (Pk * Yk).sum(0)
        n[ki] = len(r)
    return {**out, "n": n}


def recal_r2_from_stats(s_p, s_y, s_pp, s_yy, s_py, n) -> np.ndarray:
    """Pooled held-out cross-fitted per-dim affine-recal R^2 from per-fold stats.

    Broadcast-batched: stats are (..., K, d), counts (..., K); returns (...,).
    Train moments for fold k = totals - fold-k (leave-fold-out); eval ss_res
    expands in the fold's own sums, ss_tot uses the fold-local test mean (the
    committed pooled convention). Empty folds contribute zero.
    """
    s_p, s_y, s_pp, s_yy, s_py = (
        np.asarray(a, dtype=np.float64) for a in (s_p, s_y, s_pp, s_yy, s_py)
    )
    n = np.asarray(n, dtype=np.float64)
    t_p, t_y, t_pp, _t_yy, t_py = (
        a.sum(axis=-2, keepdims=True) for a in (s_p, s_y, s_pp, s_yy, s_py)
    )
    t_n = n.sum(axis=-1, keepdims=True)
    tr_n = (t_n - n)[..., None]  # (..., K, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mp = (t_p - s_p) / tr_n
        my = (t_y - s_y) / tr_n
        var_p = (t_pp - s_pp) / tr_n - mp * mp
        cov = (t_py - s_py) / tr_n - mp * my
        a = np.where(var_p > VAR_EPS, cov / np.maximum(var_p, VAR_EPS), 0.0)
        b = my - a * mp
        ss_res = (s_yy - 2.0 * a * s_py + a * a * s_pp - 2.0 * b * s_y + 2.0 * a * b * s_p).sum(
            axis=-1
        ) + n * (b * b).sum(axis=-1)
        ss_tot = (s_yy - (s_y * s_y) / np.maximum(n[..., None], 1.0)).sum(axis=-1)
    ok = (n > 0) & (tr_n[..., 0] >= 2)
    ss_res = np.where(ok, ss_res, 0.0)
    ss_tot = np.where(ok, ss_tot, 0.0)
    tot = ss_tot.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(1.0 - ss_res.sum(axis=-1) / np.where(tot > 0, tot, np.nan))


def crossfit_recal_direct(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> dict:
    """Reference cross-fitted per-dim affine recal (fold loop, vectorized dims).

    No row's recalibration is fit on itself: fold k's (a_j, b_j) come from the
    OTHER folds' (pred, truth) pairs only. Returns r2 (pooled, fold-local test
    mean), per-fold (a, b) (K, d), per-fold ss, and the recalibrated preds.
    """
    ids, rows = fold_rows(folds)
    K, d = len(ids), P.shape[1]
    a_all = np.zeros((K, d))
    b_all = np.zeros((K, d))
    ss_res = np.zeros(K)
    ss_tot = np.zeros(K)
    pred_recal = np.zeros_like(P, dtype=np.float64)
    for ki, r in enumerate(rows):
        tr = np.setdiff1d(np.arange(len(folds)), r, assume_unique=True)
        Ptr = P[tr].astype(np.float64)
        Ytr = Y[tr].astype(np.float64)
        mp, my = Ptr.mean(0), Ytr.mean(0)
        var_p = ((Ptr - mp) ** 2).mean(0)
        cov = ((Ptr - mp) * (Ytr - my)).mean(0)
        a = np.where(var_p > VAR_EPS, cov / np.maximum(var_p, VAR_EPS), 0.0)
        b = my - a * mp
        a_all[ki], b_all[ki] = a, b
        pr = a * P[r].astype(np.float64) + b
        true = Y[r].astype(np.float64)
        mu = true.mean(0)
        ss_res[ki] = float(((true - pr) ** 2).sum())
        ss_tot[ki] = float(((true - mu) ** 2).sum())
        pred_recal[r] = pr
    r2 = float(1.0 - ss_res.sum() / ss_tot.sum()) if ss_tot.sum() > 0 else float("nan")
    return {
        "r2": r2,
        "a": a_all,
        "b": b_all,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
        "pred_recal": pred_recal,
        "fold_ids": ids,
    }


def crossfit_offset_only_ss(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> float:
    """SS_res of the cross-fitted OFFSET-ONLY correction (a=1, b free per dim)."""
    _, rows = fold_rows(folds)
    ss = 0.0
    for r in rows:
        tr = np.setdiff1d(np.arange(len(folds)), r, assume_unique=True)
        b = Y[tr].astype(np.float64).mean(0) - P[tr].astype(np.float64).mean(0)
        pr = P[r].astype(np.float64) + b
        ss += float(((Y[r].astype(np.float64) - pr) ** 2).sum())
    return ss


def crossfit_scalar_recal_r2(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> float:
    """Cross-fitted GLOBAL-SCALAR affine recal (one a, b across all dims)."""
    _, rows = fold_rows(folds)
    ss_res = ss_tot = 0.0
    for r in rows:
        tr = np.setdiff1d(np.arange(len(folds)), r, assume_unique=True)
        Ptr = P[tr].astype(np.float64)
        Ytr = Y[tr].astype(np.float64)
        mp, my = float(Ptr.mean()), float(Ytr.mean())
        var_p = float(((Ptr - mp) ** 2).mean())
        cov = float(((Ptr - mp) * (Ytr - my)).mean())
        a = cov / var_p if var_p > VAR_EPS else 0.0
        b = my - a * mp
        pr = a * P[r].astype(np.float64) + b
        true = Y[r].astype(np.float64)
        mu = true.mean(0)
        ss_res += float(((true - pr) ** 2).sum())
        ss_tot += float(((true - mu) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def insample_recal_r2(P: np.ndarray, Y: np.ndarray) -> float:
    """The committed in-sample recal (pooled-GLOBAL convention — exact replica
    of _perdim_from_preds's affine_recalibrated_r2_pooled_global)."""
    Pm = P.astype(np.float64)
    T = Y.astype(np.float64)
    pm, tm = Pm.mean(0), T.mean(0)
    var_p = ((Pm - pm) ** 2).mean(0)
    cov = ((Pm - pm) * (T - tm)).mean(0)
    a = np.where(var_p > VAR_EPS, cov / np.maximum(var_p, VAR_EPS), 0.0)
    resid = T - (a * Pm + (tm - a * pm))
    return float(1.0 - (resid**2).sum() / ((T - tm) ** 2).sum())


def raw_pooled_r2(P: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> float:
    """Raw pooled R^2, fold-local test mean (the committed pooled convention —
    the DG-E0 consumer-side recompute of the stored predictions)."""
    _, rows = fold_rows(folds)
    ss_res = ss_tot = 0.0
    for r in rows:
        true = Y[r].astype(np.float64)
        pred = P[r].astype(np.float64)
        mu = true.mean(0)
        ss_res += float(((true - pred) ** 2).sum())
        ss_tot += float(((true - mu) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
