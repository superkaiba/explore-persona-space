"""SVD-based direction-constancy analyses for #519.

Headline metrics on `M_seed = [Delta_v_b(c_1), ..., Delta_v_b(c_N)] in R^{H x N}`:

- ``s_1 / sum(s)``  — fraction of variance captured by the top singular value
- mean per-context ``cos(Delta_v_b(c_i), U_1)``  — per-context alignment to U_1
- ``cos(U_1, v_steer)``  — geometric identity to an independently extracted CAA steering vector
- per-arm Spearman ``rho(magnitude_DV, base_cosine)`` and ``rho(||Delta_v_b||, base_cosine)``

Two null distributions for ``s_1 / sum(s)``:

1. **Row-shuffle null** (primary, replaces the v1 column-shuffle that was
   degenerate per the Statistics Claude lens-1 critique). For each row of
   ``M`` (``H`` rows of length ``N``) INDEPENDENTLY permute its ``N``
   entries. This breaks per-context structure while preserving per-feature
   variance — the natural null for "is there a coherent across-context
   direction?".
2. **Sign-flip null** (secondary). Multiply each column by an independent
   uniform-random plus/minus 1. Preserves the column norm but breaks any
   coherent across-column direction.

Both nulls run 1000 reps; the 95th percentile of ``s_1 / sum(s)`` under
each is the calibrated significance threshold. The two should agree
within ~10% on the 95-percentile; if they disagree by more we flag in
the analyzer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SVDSummary(TypedDict):
    """Per-seed SVD summary."""

    s: np.ndarray  # (min(H, N),) all singular values
    s_top1_frac: float  # s[0] / sum(s)
    U1: np.ndarray  # (H,) top left-singular vector
    cos_to_U1: np.ndarray  # (N,) per-context cos(M[:, i], U1)
    M_shape: tuple[int, int]
    N: int  # number of columns / panel contexts


class NullSummary(TypedDict):
    """One-null-name -> distribution + critical thresholds."""

    name: str
    n_reps: int
    null_top1_frac: np.ndarray  # (n_reps,)
    p95: float
    p99: float


def assemble_M(
    shifts: dict[str, dict[str, torch.Tensor]],
    *,
    persona_order: Sequence[str] | None = None,
    use_mean_resp: bool = False,
    tensor_key: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assemble M (H x N) from the per-persona shift dict.

    Parameters
    ----------
    shifts
        Output of `activation_shift.extract_per_context_shifts` —
        dict {persona_name: {"delta_v": (H,) tensor, ...}}.
    persona_order
        Optional fixed column order. If None, uses sorted persona names
        for deterministic output.
    use_mean_resp
        If True, use ``shifts[p]["delta_v_mean_resp"]`` instead of
        ``shifts[p]["delta_v"]``. EM-arm secondary read only.
    tensor_key
        Explicit per-persona tensor key (e.g. ``"delta_v_l7"`` for the
        layer-7 read). Mutually exclusive with ``use_mean_resp``; when
        None (default) behavior is unchanged.

    Returns
    -------
    (M, persona_names_in_order)
        M: (H, N) float32 numpy array.
        persona_names_in_order: list of persona names in column order.
    """
    if tensor_key is not None and use_mean_resp:
        raise ValueError("pass either tensor_key or use_mean_resp, not both")
    if persona_order is None:
        persona_order = sorted(shifts.keys())
    else:
        for p in persona_order:
            if p not in shifts:
                raise KeyError(f"persona {p!r} not present in shifts dict")

    if tensor_key is not None:
        key = tensor_key
    else:
        key = "delta_v_mean_resp" if use_mean_resp else "delta_v"
    cols = []
    for p in persona_order:
        entry = shifts[p]
        if key not in entry:
            raise KeyError(f"persona {p!r}: missing key {key!r} in shifts entry")
        v = entry[key]
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"persona {p!r}: expected torch.Tensor, got {type(v)}")
        cols.append(v.detach().float().cpu().numpy())

    M = np.stack(cols, axis=1)  # (H, N)
    assert M.ndim == 2, M.shape
    return M.astype(np.float32), list(persona_order)


def svd_summary(M: np.ndarray) -> SVDSummary:
    """Compute the SVD-based direction-constancy summary on M (H x N)."""
    assert M.ndim == 2, f"expected 2-D M, got {M.shape}"
    H, N = M.shape
    if N < 2:
        raise ValueError(f"need N >= 2 columns for SVD, got {N}")

    U, s, _ = np.linalg.svd(M, full_matrices=False)
    s_top1_frac = float(s[0] / s.sum())
    U1 = U[:, 0].astype(np.float32)
    # Sign convention: orient so the mean column has nonnegative projection.
    mean_col = M.mean(axis=1)
    if float(np.dot(mean_col, U1)) < 0:
        U1 = -U1

    # Per-column cosine to U1.
    col_norms = np.linalg.norm(M, axis=0)
    col_norms_safe = np.where(col_norms > 0, col_norms, 1.0)
    cos_to_U1 = (M.T @ U1) / col_norms_safe  # (N,)
    # NaN-safe — if a column was exactly zero, we report 0 cosine for it.
    cos_to_U1 = np.where(col_norms > 0, cos_to_U1, 0.0).astype(np.float32)

    return SVDSummary(
        s=s.astype(np.float32),
        s_top1_frac=s_top1_frac,
        U1=U1,
        cos_to_U1=cos_to_U1,
        M_shape=(H, N),
        N=N,
    )


def row_shuffle_null(
    M: np.ndarray,
    n_reps: int = 1000,
    seed: int = 0,
) -> NullSummary:
    """Row-shuffle null: independently permute each row's N entries.

    Recompute SVD on the shuffled matrix, take ``s_1 / sum(s)``, repeat
    n_reps times. Returns the null distribution + 95/99 percentiles.

    Plan §6.4 / §11 row 24. Replaces the v1 column-shuffle null which
    was degenerate (column permutation preserves the singular spectrum).
    """
    rng = np.random.default_rng(seed)
    _H, _N = M.shape  # shape asserted by caller; bound for documentation only
    null_vals = np.empty(n_reps, dtype=np.float32)
    M_shuffled = np.empty_like(M)
    for r in range(n_reps):
        # Per-row independent permutation of the N entries.
        # numpy.random.Generator.permuted with axis=1 does exactly this.
        M_shuffled[:] = rng.permuted(M, axis=1)
        s = np.linalg.svd(M_shuffled, compute_uv=False)
        null_vals[r] = float(s[0] / s.sum())
    return NullSummary(
        name="row_shuffle",
        n_reps=n_reps,
        null_top1_frac=null_vals,
        p95=float(np.percentile(null_vals, 95)),
        p99=float(np.percentile(null_vals, 99)),
    )


def sign_flip_null(
    M: np.ndarray,
    n_reps: int = 1000,
    seed: int = 0,
) -> NullSummary:
    """Sign-flip null: multiply each ENTRY of M by an independent +/-1.

    Plan §6.4 secondary null. Per-entry sign-flip breaks both per-row
    and per-column coherent direction.

    NOTE on the degenerate per-column variant: applying a single +/-1
    sign per column transforms M -> M @ D where D = diag(s_1..s_N) is
    orthogonal; M^T M -> D M^T M D has identical eigenvalues, so the
    singular spectrum is invariant. The entrywise variant is the
    correct sanity-check.
    """
    rng = np.random.default_rng(seed)
    null_vals = np.empty(n_reps, dtype=np.float32)
    for r in range(n_reps):
        signs = rng.choice([-1.0, 1.0], size=M.shape).astype(np.float32)
        M_signed = M * signs
        s = np.linalg.svd(M_signed, compute_uv=False)
        null_vals[r] = float(s[0] / s.sum())
    return NullSummary(
        name="sign_flip_entrywise",
        n_reps=n_reps,
        null_top1_frac=null_vals,
        p95=float(np.percentile(null_vals, 95)),
        p99=float(np.percentile(null_vals, 99)),
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Plain cosine similarity between two 1-D vectors, NaN-safe."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; in-house to avoid the scipy import overhead."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: x={x.shape}, y={y.shape}")
    if x.size < 2:
        return 0.0

    def _rankdata(v: np.ndarray) -> np.ndarray:
        # Average-rank ties (matches scipy.stats.rankdata 'average').
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
        # Tie-handle: collapse to averages.
        sorted_v = v[order]
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
                j += 1
            if j > i:
                avg = (i + j + 2) / 2.0  # mean of ranks (i+1..j+1)
                for k in range(i, j + 1):
                    ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _rankdata(x)
    ry = _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def shift_norm_vs_cosine_regression(
    M: np.ndarray,
    base_cosines: Sequence[float],
) -> dict[str, float]:
    """Mechanism A-vs-B test: regress ``||Delta_v_b(c)||`` on ``cos_base(source, c)``.

    Returns the Spearman rho + a basic OLS slope/intercept (in case the
    analyzer wants a linear fit alongside the rank correlation).
    """
    norms = np.linalg.norm(M, axis=0)
    cos = np.asarray(base_cosines, dtype=np.float64)
    if norms.size != cos.size:
        raise ValueError(
            f"shape mismatch: ||Delta_v_b|| has {norms.size} entries, base_cosines has {cos.size}"
        )
    rho = spearman_rho(norms, cos)
    # OLS slope on raw (not rank) data.
    if cos.std() > 0:
        slope = float(np.cov(norms, cos)[0, 1] / cos.var())
        intercept = float(norms.mean() - slope * cos.mean())
    else:
        slope = 0.0
        intercept = float(norms.mean())
    return {
        "spearman_rho": rho,
        "ols_slope": slope,
        "ols_intercept": intercept,
        "n_points": int(norms.size),
    }


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (median, lo, hi) for the (1 - alpha) bootstrap CI on a 1-D list."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        raise ValueError("empty bootstrap input")
    if n == 1:
        v = float(arr[0])
        return v, v, v
    medians = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        medians[r] = float(np.median(arr[idx]))
    lo = float(np.percentile(medians, 100 * alpha / 2))
    hi = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    return float(np.median(arr)), lo, hi
