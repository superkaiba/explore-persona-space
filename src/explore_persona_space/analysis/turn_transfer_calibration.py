"""Replay #825 source-turn ridge fits and fit target-turn affine calibration.

The ridge recipe is the legacy unguarded GCV estimator in
``issue825_crossmodel_map_transfer`` at 9ed29304644087843f640df7afff7a433e0f8204.
It is deliberately pinned for comparison with existing transfer results, rather
than adopting the newer estimator defaults. All operations use CPU float64.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

# Source: #825 turn-dynamics-allturns-5000, archived production estimator.
LAMBDAS = np.logspace(-2, 4, 13)


@dataclass(frozen=True)
class BatchedRidge:
    """Frozen standardized-coordinate coefficients and their training statistics."""

    beta: np.ndarray
    xmu: np.ndarray
    xsd: np.ndarray
    ymu: np.ndarray
    lambdas: np.ndarray
    n_train: np.ndarray
    gcv_scores: np.ndarray

    def predict(self, fold: int, x: np.ndarray) -> np.ndarray:
        """Apply one frozen source fit without estimating any target statistics."""
        xx = np.asarray(x, dtype=np.float64)
        if not 0 <= fold < len(self.beta):
            raise ValueError(f"invalid fold index: {fold}")
        if xx.ndim != 2 or xx.shape[1] != self.beta.shape[1]:
            raise ValueError("prediction inputs have the wrong feature dimension")
        if not np.isfinite(xx).all():
            raise ValueError("prediction inputs must be finite")
        return ((xx - self.xmu[fold]) / self.xsd[fold]) @ self.beta[fold] + self.ymu[
            fold
        ]


def fit_batched_gcv(
    x: np.ndarray,
    y: np.ndarray,
    train_indices: Sequence[np.ndarray],
    *,
    pad_to: int | None = None,
) -> BatchedRidge:
    """Fit independent source folds in one padded, dual-eigendecomposition batch.

    ``x`` and ``y`` contain one source turn. Indices select each fold's actual
    training rows. Only centered zero rows are padded. Real row counts determine
    means, unbiased standard deviations and GCV denominators. Padding therefore
    changes neither the ridge objective nor its selected regularization.

    The returned beta acts on standardized inputs. Input banks need not be copied
    to float64 by callers. At six folds of 4,167 rows and 3,584 features the main
    centered batches occupy about 1.43 GB, each batched Gram/eigenvector array
    about 0.83 GB, and the returned coefficients about 0.62 GB (decimal units).
    LAPACK workspace and caller-owned banks are additional.
    """
    xx, yy = np.asarray(x), np.asarray(y)
    if xx.ndim != 2 or yy.ndim != 2 or len(xx) != len(yy):
        raise ValueError("ridge inputs must be aligned matrices")
    if not xx.shape[1] or not yy.shape[1]:
        raise ValueError("ridge matrices must have nonempty feature dimensions")
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        raise ValueError("ridge inputs must be finite")
    indices = [np.asarray(idx) for idx in train_indices]
    if not indices:
        raise ValueError("at least one training fold is required")
    for idx in indices:
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or len(idx) < 3:
            raise ValueError("each fold needs at least three integer row indices")
        if idx.min() < 0 or idx.max() >= len(xx) or len(np.unique(idx)) != len(idx):
            raise ValueError("training indices must be unique and in bounds")
    sizes = np.asarray([len(idx) for idx in indices], dtype=np.int64)
    padded_n = int(sizes.max()) if pad_to is None else pad_to
    if not isinstance(padded_n, (int, np.integer)) or padded_n < sizes.max():
        raise ValueError("pad_to must be an integer covering every training fold")
    batch, d_in, d_out = len(indices), xx.shape[1], yy.shape[1]
    xn = torch.zeros((batch, padded_n, d_in), dtype=torch.float64, device="cpu")
    yc = torch.zeros((batch, padded_n, d_out), dtype=torch.float64, device="cpu")
    xmu = torch.empty((batch, d_in), dtype=torch.float64, device="cpu")
    xsd = torch.empty_like(xmu)
    ymu = torch.empty((batch, d_out), dtype=torch.float64, device="cpu")
    for fold, idx in enumerate(indices):
        xtr = torch.as_tensor(xx[idx], dtype=torch.float64, device="cpu")
        ytr = torch.as_tensor(yy[idx], dtype=torch.float64, device="cpu")
        xmu[fold], xsd[fold], ymu[fold] = xtr.mean(0), xtr.std(0) + 1e-9, ytr.mean(0)
        xn[fold, : len(idx)] = (xtr - xmu[fold]) / xsd[fold]
        yc[fold, : len(idx)] = ytr - ymu[fold]
        del xtr, ytr
    gram = xn @ xn.transpose(1, 2)
    w, vectors = torch.linalg.eigh(gram)
    del gram
    w.clamp_(min=0.0)
    projected_y = vectors.transpose(1, 2) @ yc
    total = yc.square().sum((1, 2))
    del yc
    projected_y2 = projected_y.square().sum(2)
    lambdas = torch.as_tensor(LAMBDAS, dtype=torch.float64, device="cpu")
    filt = w[:, None, :] / (w[:, None, :] + lambdas[None, :, None])
    rss = total[:, None] - ((2 * filt - filt.square()) * projected_y2[:, None, :]).sum(
        2
    )
    n_actual = torch.as_tensor(sizes, dtype=torch.float64, device="cpu")
    denominator = (n_actual[:, None] - filt.sum(2)).square()
    gcv = torch.where(denominator > 1e-12, rss / denominator, torch.inf)
    if torch.isnan(gcv).any() or not torch.isfinite(gcv).any(1).all():
        raise ValueError("GCV has no finite regularization choice")
    selected = lambdas[gcv.argmin(1)]
    del filt, rss, denominator, total, projected_y2
    projected_y /= (w + selected[:, None])[:, :, None]
    dual = vectors @ projected_y
    del vectors, projected_y, w
    beta = xn.transpose(1, 2) @ dual
    del xn, dual
    return BatchedRidge(
        beta=beta.numpy(),
        xmu=xmu.numpy(),
        xsd=xsd.numpy(),
        ymu=ymu.numpy(),
        lambdas=selected.numpy(),
        n_train=sizes,
        gcv_scores=gcv.numpy(),
    )


def calibrate(train_prediction: np.ndarray, train_target: np.ndarray) -> dict:
    """Fit a vector bias and a global scalar plus vector intercept by least squares.

    This is the #2054 K5 transfer-calibration helper's mathematical recipe. The
    caller must supply training rows only, disjoint from every evaluated fold.
    A constant prediction cannot identify the scalar and raises explicitly.
    """
    p, y = (
        np.asarray(train_prediction, dtype=np.float64),
        np.asarray(train_target, np.float64),
    )
    if p.shape != y.shape or p.ndim != 2 or len(p) < 2 or not p.shape[1]:
        raise ValueError("invalid calibration arrays")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite calibration values")
    pm, ym = p.mean(0), y.mean(0)
    pc, yc = p - pm, y - ym
    denominator = float(np.einsum("ij,ij->", pc, pc))
    if denominator <= 0:
        raise ValueError("constant predictions cannot identify scalar scaling")
    gain = float(np.einsum("ij,ij->", pc, yc) / denominator)
    return {"bias": ym - pm, "gain": gain, "prediction_mean": pm, "target_mean": ym}


def adapted_predictions(
    prediction: np.ndarray, coefficients: dict
) -> dict[str, np.ndarray]:
    """Apply fixed calibration coefficients to prediction rows without target labels."""
    p = np.asarray(prediction, dtype=np.float64)
    c = coefficients
    if p.ndim != 2 or p.shape[1:] != np.asarray(c["bias"]).shape:
        raise ValueError("prediction and calibration dimensions differ")
    if not np.isfinite(p).all():
        raise ValueError("predictions must be finite")
    return {
        "bias": p + c["bias"],
        "bias_scale": c["gain"] * (p - c["prediction_mean"]) + c["target_mean"],
    }
