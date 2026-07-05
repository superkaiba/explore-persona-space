# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ≈, ×, ≫) in scientific docstrings.
"""Shared train-fold PCA reduction for the issue #763 LOCO predictors.

The GLM (PRIMARY) and ridge (COMPARATOR) arms of #763 BOTH read the registered
``ρ_ridge − ρ_GLM`` optimism delta (plan §1/§5/§11: "ridge was optimistic by
0.05–0.11 at m=8"). For that delta to be apples-to-apples the two arms MUST have
MATCHED model capacity — i.e. fit on the SAME train-fold PCA-reduced design at
the SAME nested-CV-selected dimensionality. The GLM PCA-reduced its train fold
to a nested-CV ``d`` (the d≫n probe-memorization guard at H=3584, n=50); pre-fix
the ridge fit on the raw 3584-d ``x``, so it got the FULL capacity the GLM did
not — corrupting exactly the headline statistic the task exists to compute
(BLOCKER ridge-pca-comparator).

This module factors the GLM's train-fold PCA selection into one helper both arms
call, so both consume the identically-reduced features per LOCO fold:

- ``_pca_fit`` / ``_pca_transform`` are the same centered-SVD basis the GLM used
  (kept here verbatim so the GLM's selection is unchanged when it switches to
  this shared path).
- ``nested_cv_pca_reduce(x_train, x_test, *, y_train, w_train, d_grid)`` selects
  ``d`` by nested-CV inner-LOO squared error on the TRAIN fold ONLY (no held-out
  leakage), fits the basis on the train fold, and projects BOTH the train fold
  and the held-out test row(s) onto it → ``(x_train_red, x_test_red, d_chosen)``.

The inner-CV criterion is a precision-weighted binomial-GLM inner-LOO (the GLM's
own criterion) so the chosen ``d`` is selected for the SAME objective both arms
fit; the ridge arm then fits its own closed-form LOCO on those reduced features
with its own nested-CV λ. (Selecting ``d`` by the GLM criterion and λ by the
ridge criterion keeps each arm's regularization on its own terms while matching
the feature space — the capacity match that makes the optimism delta honest.)
"""

from __future__ import annotations

import warnings

import numpy as np

# The plan §11 d_eff grid, shared by both arms (kept identical to the GLM's grid
# so the capacity match is exact).
PCA_DIM_GRID: tuple[int, ...] = (2, 4, 6, 8, 10, 15, 20)


def _pca_fit(x_train: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a PCA basis on the training rows; return (mean, components (d, H)).

    Centered SVD. ``d`` is clamped to the rank the train fold supports so a tiny
    smoke slice (n=3) cannot request more components than the rank.
    """
    mu = x_train.mean(axis=0)
    xc = x_train - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    d_eff = min(d, vt.shape[0])
    return mu, vt[:d_eff]


def _pca_transform(x: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """Project rows of x onto the fitted PCA basis -> (n, d)."""
    return (x - mu) @ comps.T


def _fit_binomial_glm(z: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Fit a precision-weighted binomial GLM; return the fitted results or None.

    Used ONLY as the nested-CV inner criterion for dim selection (the outer fit
    lives in the GLM arm). None on a singular / non-converging fit (the caller
    falls back to the train-mean prediction).
    """
    import statsmodels.api as sm

    zc = sm.add_constant(z, has_constant="add")
    yc = np.clip(y, 1e-6, 1.0 - 1e-6)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(yc, zc, family=sm.families.Binomial(), var_weights=w)
            res = model.fit(maxiter=100)
        if not np.all(np.isfinite(res.params)):
            return None
        return res
    except Exception:
        return None


def _inner_loo_mse(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Mean inner-LOO squared error for one candidate PCA dim on the train set."""
    n = z.shape[0]
    errs: list[float] = []
    for k in range(n):
        tr = [j for j in range(n) if j != k]
        res = _fit_binomial_glm(z[tr], y[tr], w[tr])
        if res is None:
            pred = float(np.mean(y[tr]))
        else:
            import statsmodels.api as sm

            zk = sm.add_constant(z[k : k + 1], has_constant="add")
            pred = float(res.predict(zk)[0])
        errs.append((pred - y[k]) ** 2)
    return float(np.mean(errs)) if errs else np.inf


def select_pca_dim(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    *,
    d_grid: tuple[int, ...] = PCA_DIM_GRID,
) -> int:
    """Nested-CV PCA dim selection on the TRAIN fold only (no held-out leakage).

    Caps candidate dims at ~1/5 the train size (the p≪n overfit guard the GLM
    used): a fit with d≫n/5 over-fits even under LOCO, and the nested-CV inner
    criterion — itself estimated on n-1 rows — can pick a too-high dim that fits
    training noise and leaks a spurious held-out rank order. Returns the chosen
    integer dim.
    """
    y = np.asarray(y_train, dtype=np.float64)
    w = np.asarray(w_train, dtype=np.float64)
    w = np.where(w < 1, 1.0, w)
    n = x_train.shape[0]
    d_max = max(2, n // 5)
    best_d, best_mse = d_grid[0], np.inf
    for d in d_grid:
        if d > d_max:  # p ≪ n guard (overfit + speed)
            continue
        mu_d, comps_d = _pca_fit(x_train, d)
        z_tr = _pca_transform(x_train, mu_d, comps_d)
        mse = _inner_loo_mse(z_tr, y, w)
        if mse < best_mse:
            best_mse, best_d = mse, d
    return best_d


def nested_cv_pca_reduce(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    y_train: np.ndarray,
    w_train: np.ndarray,
    d_grid: tuple[int, ...] = PCA_DIM_GRID,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select d by nested-CV on the train fold, fit the basis, project BOTH sides.

    The single shared reduction both the GLM and ridge LOCO loops call so the
    ``ρ_ridge − ρ_GLM`` optimism delta is apples-to-apples (matched capacity).

    Args:
        x_train: (n_tr, H) the LOCO-fold training activations at one layer.
        x_test:  (n_te, H) the held-out row(s) at the same layer (n_te ≥ 1).
        y_train: (n_tr,) the train-fold rate (the nested-CV inner objective).
        w_train: (n_tr,) the train-fold precision weight (judged counts).
        d_grid:  candidate PCA dims for the nested-CV selection (plan §11).

    Returns:
        ``(x_train_red (n_tr, d), x_test_red (n_te, d), d_chosen)`` — the PCA
        basis is fit on ``x_train`` ONLY (no held-out leakage) and applied to
        both sides at the selected dim.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    if x_test.ndim == 1:
        x_test = x_test.reshape(1, -1)
    d_chosen = select_pca_dim(x_train, y_train, w_train, d_grid=d_grid)
    mu, comps = _pca_fit(x_train, d_chosen)
    x_train_red = _pca_transform(x_train, mu, comps)
    x_test_red = _pca_transform(x_test, mu, comps)
    return x_train_red, x_test_red, d_chosen
