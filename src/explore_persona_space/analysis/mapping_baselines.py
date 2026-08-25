# ruff: noqa: RUF002
"""Standing baselines + retrieval metric for representation-mapping experiments.

Two reads every fitted map ``v_X -> v_Y`` reports alongside held-out R²
(standing rule, 2026-07-22; first applied to the #779 context→answer and the
#658-battery prefix-level context→answer maps):

- :func:`identity_bias_predict` — the **W = identity, learned-bias** baseline
  ``v̂ = x + b`` with ``b = train-mean(Y − X)``. Isolates how much of a map's
  R² a context-independent constant shift already explains (a shared
  position/formatting offset). Requires ``d_in == d_out`` (same-space maps);
  callers whose input and output spaces differ state that inapplicability
  instead.
- :func:`knn_retrieval` — the **retrieval metric**: P(true target within the
  ``k`` nearest neighbors of the prediction) among a candidate pool (default:
  the held-out true targets), with chance = ``k / n_pool``. A scale-invariant
  recall@k companion to R² (R² can look mediocre while predictions still
  single out the right target among hundreds, and vice versa).
"""

from __future__ import annotations

import numpy as np

__all__ = ["identity_bias_predict", "identity_bias_predict_blocked", "knn_retrieval"]


def identity_bias_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray
) -> np.ndarray:
    """W=identity, learned-bias baseline: ``pred = x_eval + mean(y_train − x_train)``.

    The bias is the train-set mean residual — the least-squares solution for
    ``b`` under a frozen identity ``W``. Requires matching input/output dims.
    """
    xtr = np.asarray(x_train, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    xev = np.asarray(x_eval, dtype=np.float64)
    if xtr.shape != ytr.shape:
        raise ValueError(
            f"identity+bias baseline needs matching train shapes, got {xtr.shape} vs {ytr.shape}"
        )
    if xev.shape[1:] != xtr.shape[1:]:
        raise ValueError(f"x_eval dim {xev.shape[1:]} != train dim {xtr.shape[1:]}")
    return xev + (ytr - xtr).mean(axis=0)


def identity_bias_predict_blocked(
    x_train: np.ndarray,
    y_train: np.ndarray,
    tr_idx: np.ndarray,
    x_eval: np.ndarray,
    block: int = 65_536,
    *,
    return_bias: bool = False,
):
    """Blocked identity+bias baseline over ``tr_idx`` rows of the full stores.

    Computes ``b = fp64 mean over x_train[tr_idx]/y_train[tr_idx] of (y − x)``
    by accumulating ``(y_blk − x_blk).sum(0)`` over index blocks, then returns
    ``x_eval + b`` — numerically the exact-helper result up to fp64
    summation-order roundoff (≤1e-6 asserted in tests). Unlike
    ``identity_bias_predict(x_train[tr_idx], y_train[tr_idx], x_eval)``, it
    never materializes fp64 train copies or full fp32 advanced-index copies:
    at n_train=963k × d=3584 the exact route peaks ≈138 GB where this route's
    per-block temporaries peak ≈5.6 GB at the default block (65,536 rows ×
    3,584 dims × 8 B ≈ 1.9 GB each for the xb/yb fp64 copies + their
    difference) (#1901 mlp-scaling-densify, plan §9 LARGEST-CELL keying).

    ``x_train`` / ``y_train`` are the FULL (N, d) stores (any indexable
    array-like of matching row dim); ``tr_idx`` selects the train rows.
    ``return_bias=True`` additionally returns the fp64 bias vector ``b``
    (so callers can persist it without a second train pass).
    """
    tr_idx = np.asarray(tr_idx, dtype=np.int64)
    if tr_idx.ndim != 1 or tr_idx.size == 0:
        raise ValueError(f"tr_idx must be a non-empty 1-D index array, got shape {tr_idx.shape}")
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")
    n_rows = int(x_train.shape[0])
    if int(y_train.shape[0]) != n_rows:
        raise ValueError(
            f"identity+bias baseline needs matching train row counts, got "
            f"{n_rows} vs {y_train.shape[0]}"
        )
    lo, hi = int(tr_idx.min()), int(tr_idx.max())
    if lo < 0:
        raise ValueError(f"tr_idx must be non-negative (no negative indexing), got min {lo}")
    if hi >= n_rows:
        raise ValueError(f"tr_idx out of bounds: max {hi} >= n_rows {n_rows}")
    d_tr = tuple(x_train.shape[1:])
    if tuple(y_train.shape[1:]) != d_tr:
        raise ValueError(
            f"identity+bias baseline needs matching train dims, got {x_train.shape[1:]} "
            f"vs {y_train.shape[1:]}"
        )
    xev = np.asarray(x_eval, dtype=np.float64)
    if tuple(xev.shape[1:]) != d_tr:
        raise ValueError(f"x_eval dim {xev.shape[1:]} != train dim {d_tr}")
    acc = np.zeros(d_tr, dtype=np.float64)
    for s in range(0, tr_idx.size, block):
        idx = tr_idx[s : s + block]
        xb = np.asarray(x_train[idx], dtype=np.float64)
        yb = np.asarray(y_train[idx], dtype=np.float64)
        acc += (yb - xb).sum(axis=0)
    b = acc / float(tr_idx.size)
    pred = xev + b
    return (pred, b) if return_bias else pred


def _pairwise_dist(pred: np.ndarray, pool: np.ndarray, metric: str) -> np.ndarray:
    """(n_pred, n_pool) distance matrix; ``euclidean`` (squared, rank-equivalent)
    or ``cosine`` (1 − cosine similarity)."""
    if metric == "euclidean":
        # squared euclidean via GEMM — monotone in euclidean, so rank-identical.
        p2 = (pred**2).sum(1)[:, None]
        q2 = (pool**2).sum(1)[None, :]
        return p2 + q2 - 2.0 * (pred @ pool.T)
    if metric == "cosine":
        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
        qn = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
        return 1.0 - pn @ qn.T
    raise ValueError(f"unknown metric {metric!r}")


def knn_retrieval(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    metric: str = "euclidean",
    pool: np.ndarray | None = None,
    true_pool_idx: np.ndarray | None = None,
) -> dict:
    """P(true target within the k nearest pool neighbors of the prediction).

    ``pool`` defaults to ``true`` (the held-out targets are their own candidate
    set); ``true_pool_idx[i]`` is the pool row holding row ``i``'s true target
    (defaults to ``arange(n)``, the pool==true case). Ties get MID-RANKS
    (tolerance-based). A degenerate constant predictor (predict-the-mean)
    scores EXACTLY chance = k / n_pool when pool == true — every pool row gets
    a unique rank in its fixed ordering — so the ``chance_at_k`` field is both
    the floor and the constant-predictor read. Returns acc@k per k, median
    rank, MRR, n_pool.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pool_arr = true if pool is None else np.asarray(pool, dtype=np.float64)
    n, n_pool = pred.shape[0], pool_arr.shape[0]
    idx = np.arange(n) if true_pool_idx is None else np.asarray(true_pool_idx)
    if idx.shape[0] != n or idx.max() >= n_pool:
        raise ValueError(f"true_pool_idx invalid: n={n}, n_pool={n_pool}")
    d = _pairwise_dist(pred, pool_arr, metric)
    d_true = d[np.arange(n), idx]
    # mid-rank: 1 + #closer + (#tied-others)/2. Ties are tolerance-based (the GEMM
    # distance path leaves ~1e-13 relative float noise on genuinely-identical rows,
    # which would otherwise rank a degenerate constant predictor arbitrarily).
    tol = 1e-9 * np.maximum(np.abs(d_true)[:, None], 1e-12)
    closer = (d < d_true[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - d_true[:, None]) <= tol).sum(axis=1) - 1  # excl. the true target
    ranks = 1.0 + closer + 0.5 * tied
    return {
        "metric": metric,
        "n": int(n),
        "n_pool": int(n_pool),
        "acc_at_k": {int(k): float((ranks <= k).mean()) for k in ks},
        "chance_at_k": {int(k): float(k / n_pool) for k in ks},
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
    }
