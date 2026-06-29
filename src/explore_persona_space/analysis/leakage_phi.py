# ruff: noqa: RUF002, RUF003
# Intentional Unicode (φ, σ, Δ, →, ≤) in scientific docstrings.
"""Per-behavior affine→[0,1] link φ for the issue-666 leakage predictor (plan §4g).

φ maps the unbounded latent leakage score (L̂ / Δs scale) to the bounded behavior
``[0,1]`` rate scale. It is a MONOTONE link (paper "Bounding expression to [0,1]"):
a logistic σ fit on the TRAIN partition ONLY, clamped to ``[0, 1]``. The latent Δs
scale is unbounded and needs NO φ (only the behavior-rate MAE secondary uses it).

``fit_phi(x_train, y_train)`` is a PURE function of the train data (no test-row
leakage, plan §4g). It returns a frozen ``PhiLink`` whose ``params`` are
reproducible; ``apply_phi(phi, x)`` maps new scores into ``[0, 1]`` without
refitting. CPU-only; numpy + scipy.optimize. No store, no network, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhiLink:
    """A fitted monotone affine→logistic link. ``params`` = (slope a, intercept b).

    The map is ``φ(x) = clip(σ(a·x + b), 0, 1)`` with ``σ`` the logistic sigmoid.
    Monotone non-decreasing iff ``a ≥ 0`` (the fit constrains the slope to be
    non-negative so φ is a valid monotone calibration link).
    """

    params: tuple[float, float]

    @property
    def slope(self) -> float:
        return self.params[0]

    @property
    def intercept(self) -> float:
        return self.params[1]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def fit_phi(x_train, y_train) -> PhiLink:
    """Fit the monotone logistic link φ on the TRAIN partition only (plan §4g).

    Least-squares fit of ``σ(a·x + b)`` to the train targets via a small bounded
    optimization (slope ``a`` constrained ``≥ 0`` so φ is monotone non-decreasing).
    A PURE function of ``(x_train, y_train)`` — no test data ever enters the fit,
    so the same train rows always yield the same ``params`` regardless of any
    held-out test set (the train-only-φ discipline the LOBO/LOCO drivers rely on).

    Returns a frozen ``PhiLink``.
    """
    from scipy.optimize import least_squares

    x = np.asarray(x_train, dtype=np.float64).reshape(-1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x_train {x.shape} and y_train {y.shape} must match")
    if x.size == 0:
        raise ValueError("fit_phi needs at least one train row")

    # Init: scale a from the data spread, b to center the logistic on the mean.
    sx = float(x.std()) or 1.0
    a0 = 1.0 / sx
    b0 = -a0 * float(x.mean())

    def resid(theta: np.ndarray) -> np.ndarray:
        a, b = theta
        return _sigmoid(a * x + b) - y

    # slope a >= 0 (monotone non-decreasing); intercept b free.
    res = least_squares(
        resid, x0=np.array([a0, b0]), bounds=([0.0, -np.inf], [np.inf, np.inf]), max_nfev=2000
    )
    a, b = float(res.x[0]), float(res.x[1])
    return PhiLink(params=(a, b))


def apply_phi(phi: PhiLink, x) -> np.ndarray:
    """Apply a fitted φ link to new scores, clamped to ``[0, 1]`` (no refit).

    ``φ(x) = clip(σ(a·x + b), 0, 1)``. Returns an array the same shape as ``x``.
    Idempotent — applying φ to any test set never re-fits the link (plan §4g).
    """
    a, b = phi.params
    arr = np.asarray(x, dtype=np.float64)
    out = _sigmoid(a * arr + b)
    return np.clip(out, 0.0, 1.0)
