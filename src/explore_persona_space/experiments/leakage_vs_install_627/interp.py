"""Task #627 — #606 interpolation + bootstrap-guard mechanics, ported VERBATIM.

Plan §13 item 1 (BINDING): ``_interp_at`` (nearest-anchor extrapolation +
zero-denominator fallback) and the >=50%-finite-replicates ``RuntimeError``
guard are ported VERBATIM from ``origin/issue-606:scripts/issue_606/
i606_analyze.py`` — NOT re-derived. The #606 module is branch-only (never
merged to main), so the port lives here.

``out_of_bracket_rate`` is the §13 item 1 diagnostic: the share of bootstrap
replicates whose resampled install anchors no longer straddle the target (so
``_interp_at`` extrapolated from the two nearest anchors instead of
interpolating). Reported per cell alongside the headline CI.
"""

from __future__ import annotations

import numpy as np

# --- BEGIN verbatim port: origin/issue-606:scripts/issue_606/i606_analyze.py ---


def _interp_at(xs: np.ndarray, ys: np.ndarray, target: float) -> np.ndarray | float:
    """Piecewise-linear interpolation across (xs, ys) at ``target``, with
    extrapolation from the two nearest anchors outside the range (the #508
    ``_linear_interp`` convention). Vectorized over leading axes.

    xs shape (..., A), ys shape (..., A[, P]) sorted handled internally.
    """
    order = np.argsort(xs, axis=-1)
    xs_s = np.take_along_axis(xs, order, axis=-1)
    if ys.ndim == xs.ndim:
        ys_s = np.take_along_axis(ys, order, axis=-1)
    else:  # ys has trailing persona axis
        ys_s = np.take_along_axis(ys, order[..., None], axis=-2)
    n_anchor = xs_s.shape[-1]
    if n_anchor < 2:
        return np.full(ys_s.shape[:-1] if ys.ndim == xs.ndim else ys_s.shape[:-2], np.nan)
    pos = (xs_s < target).sum(axis=-1)
    hi = np.clip(pos, 1, n_anchor - 1)
    lo = hi - 1
    x_lo = np.take_along_axis(xs_s, lo[..., None], axis=-1)[..., 0]
    x_hi = np.take_along_axis(xs_s, hi[..., None], axis=-1)[..., 0]
    denom = x_hi - x_lo
    frac = np.where(denom == 0, 0.0, (target - x_lo) / np.where(denom == 0, 1.0, denom))
    if ys.ndim == xs.ndim:
        y_lo = np.take_along_axis(ys_s, lo[..., None], axis=-1)[..., 0]
        y_hi = np.take_along_axis(ys_s, hi[..., None], axis=-1)[..., 0]
        return y_lo + frac * (y_hi - y_lo)
    y_lo = np.take_along_axis(ys_s, lo[..., None, None], axis=-2)[..., 0, :]
    y_hi = np.take_along_axis(ys_s, hi[..., None, None], axis=-2)[..., 0, :]
    return y_lo + frac[..., None] * (y_hi - y_lo)


# --- END verbatim port ---


def assert_finite_replicates(rep: np.ndarray, *, b: int, label: str) -> np.ndarray:
    """The #606 >=50%-finite-replicates guard (i606_analyze.py, ported logic):
    fewer than ``b/2`` finite bootstrap replicates means degenerate cells /
    empty denominators — fail LOUD, never quantile over a husk."""
    valid = rep[np.isfinite(rep)]
    if len(valid) < 0.5 * b:
        raise RuntimeError(
            f"[{label}] >{b // 2} bootstrap replicates non-finite — degenerate cells "
            f"or empty clean denominators; inspect per-cell tables"
        )
    return valid


def out_of_bracket_rate(xs_rep: np.ndarray, target: float) -> float:
    """Share of replicates whose resampled anchors do NOT straddle ``target``
    (=> ``_interp_at`` extrapolated). ``xs_rep`` shape (B, A)."""
    lo = xs_rep.min(axis=-1)
    hi = xs_rep.max(axis=-1)
    out = (target < lo) | (target > hi)
    return float(out.mean())
