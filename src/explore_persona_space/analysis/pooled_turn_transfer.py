"""Primal replay of the legacy #825 GCV ridge recipe for pooled source turns."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from explore_persona_space.analysis.turn_transfer_calibration import LAMBDAS, BatchedRidge


def fit_primal_gcv(
    x: np.ndarray, y: np.ndarray, train_indices: Sequence[np.ndarray]
) -> BatchedRidge:
    """Fit folds together in feature space, preserving the parent GCV objective.

    GCV is the legacy rowwise training selector, not grouped inner validation.
    The caller holds out whole conversations for the final performance estimate.
    Feature-space eigendecomposition bounds the pooled solve at 3,584 dimensions.
    Lambda updates and target dimensions are vectorized over one factorization.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not x.shape[1] or not y.shape[1]:
        raise ValueError("ridge inputs must be aligned nonempty feature matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite")
    indices = [np.asarray(i) for i in train_indices]
    if not indices:
        raise ValueError("at least one fold is required")
    for i in indices:
        if (
            i.ndim != 1
            or i.dtype.kind not in "iu"
            or len(i) < 3
            or i.min() < 0
            or i.max() >= len(x)
            or len(np.unique(i)) != len(i)
        ):
            raise ValueError("invalid training indices")
    sizes = np.array([len(i) for i in indices], dtype=np.int64)
    b, d, p = len(indices), x.shape[1], y.shape[1]
    means = torch.empty((b, d), dtype=torch.float64)
    stds = torch.empty_like(means)
    targets = torch.empty((b, p), dtype=torch.float64)
    gram = torch.empty((b, d, d), dtype=torch.float64)
    cross = torch.empty((b, d, p), dtype=torch.float64)
    total = torch.empty(b, dtype=torch.float64)
    # Chunk-local centering avoids padded pooled banks; the expensive solves and
    # the lambda/target axes are batched. The reductions are one GEMM per fold.
    for f, i in enumerate(indices):
        xx = torch.as_tensor(x[i], dtype=torch.float64)
        yy = torch.as_tensor(y[i], dtype=torch.float64)
        means[f], stds[f], targets[f] = xx.mean(0), xx.std(0) + 1e-9, yy.mean(0)
        xx = (xx - means[f]) / stds[f]
        yy = yy - targets[f]
        gram[f] = xx.T @ xx
        cross[f] = xx.T @ yy
        total[f] = yy.square().sum()
        del xx, yy
    eigen, vectors = torch.linalg.eigh(gram)
    del gram
    eigen.clamp_(min=0)
    projected = vectors.transpose(1, 2) @ cross
    del cross
    projected_norm = projected.square().sum(2)
    lam = torch.as_tensor(LAMBDAS, dtype=torch.float64)
    inverse = 1 / (eigen[:, None, :] + lam[None, :, None])
    explained = 2 * inverse - eigen[:, None, :] * inverse.square()
    rss = total[:, None] - (explained * projected_norm[:, None, :]).sum(2)
    df = (eigen[:, None, :] * inverse).sum(2)
    denominator = (torch.as_tensor(sizes)[:, None] - df).square()
    scores = torch.where(denominator > 1e-12, rss / denominator, torch.inf)
    if torch.isnan(scores).any() or not torch.isfinite(scores).any(1).all():
        raise ValueError("GCV has no finite regularization choice")
    chosen = lam[scores.argmin(1)]
    beta = vectors @ (projected / (eigen + chosen[:, None])[:, :, None])
    return BatchedRidge(
        beta=beta.numpy(),
        xmu=means.numpy(),
        xsd=stds.numpy(),
        ymu=targets.numpy(),
        lambdas=chosen.numpy(),
        n_train=sizes,
        gcv_scores=scores.numpy(),
    )


def row_metrics(predictions: np.ndarray, truth: np.ndarray) -> dict:
    """Return per-conversation errors and exact fold-local retrieval hit indicators."""
    if predictions.ndim != 3 or predictions.shape[1:] != truth.shape or truth.ndim != 2:
        raise ValueError("prediction/target shapes disagree")
    if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError("nonfinite predictions/targets")
    norms = np.linalg.norm(predictions, axis=-1)
    ynorm = np.linalg.norm(truth, axis=-1)
    if np.any(norms == 0) or np.any(ynorm == 0):
        raise ValueError("zero norm makes cosine retrieval undefined")
    dot = np.einsum("mnd,vd->mnv", predictions, truth, optimize=True)
    cosine = dot / (norms[:, :, None] * ynorm[None, None, :])
    distance = norms[:, :, None] ** 2 + ynorm[None, None, :] ** 2 - 2 * dot
    row = np.arange(len(truth))
    return {
        "sse": np.square(predictions - truth).sum(-1),
        "sst": np.square(truth - truth.mean(0)).sum(-1),
        "cosine_hit": (cosine.argmax(-1) == row).astype(np.uint8),
        "euclidean_hit": (distance.argmin(-1) == row).astype(np.uint8),
    }
