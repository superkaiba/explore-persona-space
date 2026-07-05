# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ≈, ×) in scientific docstrings.
"""Precision-weighted binomial GLM LOCO predictor (issue #763, PRIMARY read).

The in-session 2026-06-30 GLM-vs-ridge finding (plan §2 / §11): at m=8 the
closed-form ridge ``v0(C,B)→E0(C,B)`` read was OPTIMISTIC by 0.05–0.11 vs a
correctly-specified binomial GLM, with the gap growing under floor/ceiling
skew. The GLM is the correctly-specified estimator for a RATE response with
per-cell binomial variance, so #763 registers it as PRIMARY (ridge stays the
comparator, computed by ``issue658_fit_predictors._ridge_predict_loco``).

This module provides the GLM LOCO predictor only — the per-behavior fit
driver (``scripts/issue763_fit_predictors.py``) calls ``glm_predict_loco`` per
(behavior × layer), with the ridge / PV / bootstrap / reliability / nulls
wired around it from the reused #658 functions + ``issue_763_reliability``.

Design:

- **Response:** the per-context observed rate ``E0(C,B) ∈ [0, 1]`` modelled
  with ``family=Binomial`` (logit link). statsmodels' Binomial GLM accepts a
  fractional endog when paired with ``var_weights`` = the per-context judged
  count ``n_judged`` (the precision weight) — high-m contexts get more weight.
- **Overdispersion:** after the full-data fit, if ``deviance / df_resid >
  ~1.5`` the per-context binomial variance is too tight for the spread, so the
  CI is anti-conservative; we record the dispersion + flag ``quasibinomial``
  (statsmodels reports the scale via ``scale="X2"`` Pearson dispersion — we
  recompute predictions identically, the flag is for the CI interpretation
  the analysis layer reads).
- **PCA-reduce:** ``v0`` (n_ctx × H) is PCA-reduced to ``d`` components BEFORE
  the GLM (d≫n probe-memorization guard at H=3584, n=50). ``d`` is selected by
  NESTED CV per held-out context (inner-LOO deviance over the grid), NOT fixed
  at k=10 (the plan §11 in-session k-sweep: k=10 conservative, k≥~40 degenerate
  at n=50). The PCA basis is fit on the TRAINING contexts only per LOCO fold
  (no held-out leakage).
- **LOCO:** leave-one-context-out; the prediction for held-out context ``i`` is
  the GLM fit on the other n-1 contexts evaluated at ``v0[i]``. Held-out
  Spearman ρ of the n predictions vs the n observed rates is the headline.
"""

from __future__ import annotations

import warnings

import numpy as np

from explore_persona_space.analysis.issue_763_pca import (
    PCA_DIM_GRID,
    _pca_fit,
    _pca_transform,
    nested_cv_pca_reduce,
)

OVERDISPERSION_THRESHOLD = 1.5


def _fit_binomial_glm(z: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Fit a precision-weighted binomial GLM; return the fitted results or None.

    ``z`` (n, d) design (PCA scores), ``y`` (n,) fractional rate in [0,1],
    ``w`` (n,) var_weights (judged counts). Adds an intercept column. Returns
    None on a singular / non-converging fit (the caller falls back to the
    train-mean prediction — never a crash).
    """
    import statsmodels.api as sm

    zc = sm.add_constant(z, has_constant="add")
    yc = np.clip(y, 1e-6, 1.0 - 1e-6)  # logit link needs strictly-interior endog
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


def glm_predict_loco(
    x: np.ndarray,
    y: np.ndarray,
    n_judged: np.ndarray,
    *,
    pca_dim_grid: tuple[int, ...] = PCA_DIM_GRID,
) -> dict:
    """LOCO precision-weighted binomial GLM predictions of E0(C,B) from v0(C,B).

    Args:
        x: (n_ctx, H) the v0(C,B) per-context activation summary at one layer.
        y: (n_ctx,) observed E0(C,B) rate in [0, 1].
        n_judged: (n_ctx,) per-context judged count (the precision weight).
        pca_dim_grid: candidate PCA dims for the nested-CV selection.

    Returns:
        ``{"pred": (n_ctx,) held-out predictions, "chosen_dims": [d per fold],
           "overdispersion": float, "quasibinomial": bool}``. The held-out
        Spearman ρ + cluster-bootstrap CI are computed by the caller from
        ``pred`` (reusing ``issue658_fit_predictors._rho`` / ``_cluster_bootstrap_rho``).
    """
    n = x.shape[0]
    w = np.asarray(n_judged, dtype=np.float64)
    w = np.where(w < 1, 1.0, w)
    y = np.asarray(y, dtype=np.float64)
    preds = np.zeros(n, dtype=np.float64)
    chosen_dims: list[int] = []

    for i in range(n):
        tr = [j for j in range(n) if j != i]
        x_tr, y_tr, w_tr = x[tr], y[tr], w[tr]
        # SHARED nested-CV PCA reduction (#763 BLOCKER ridge-pca-comparator):
        # the SAME helper the ridge arm calls, so both arms fit on identically-
        # reduced features and the ρ_ridge − ρ_GLM optimism delta is apples-to-
        # apples. The helper selects d by nested-CV inner-LOO on the train fold
        # only (no held-out leakage), caps candidate dims at ~1/5 the train size
        # (the p≪n overfit guard — d=20 on n=49 read |ρ|≈0.64 on PURE NOISE),
        # fits the basis on the train fold, and projects BOTH sides.
        z_tr, z_held, best_d = nested_cv_pca_reduce(
            x_tr, x[i : i + 1], y_train=y_tr, w_train=w_tr, d_grid=pca_dim_grid
        )
        chosen_dims.append(best_d)
        # outer fit at the selected dim, predict the held-out context
        res = _fit_binomial_glm(z_tr, y_tr, w_tr)
        if res is None:
            preds[i] = float(np.mean(y_tr))
        else:
            import statsmodels.api as sm

            zc = sm.add_constant(z_held, has_constant="add")
            preds[i] = float(res.predict(zc)[0])

    # overdispersion check from a full-data fit at the modal chosen dim
    modal_d = max(set(chosen_dims), key=chosen_dims.count) if chosen_dims else pca_dim_grid[0]
    overdisp = 1.0
    quasibinomial = False
    mu_f, comps_f = _pca_fit(x, min(modal_d, n - 1))
    z_f = _pca_transform(x, mu_f, comps_f)
    res_f = _fit_binomial_glm(z_f, y, w)
    if res_f is not None and getattr(res_f, "df_resid", 0) > 0:
        overdisp = float(res_f.deviance / res_f.df_resid)
        quasibinomial = overdisp > OVERDISPERSION_THRESHOLD

    return {
        "pred": preds,
        "chosen_dims": chosen_dims,
        "overdispersion": overdisp,
        "quasibinomial": quasibinomial,
    }


def glm_predict_loco_fixed_dim(
    x: np.ndarray,
    y: np.ndarray,
    n_judged: np.ndarray,
    dim: int,
) -> np.ndarray:
    """LOCO binomial-GLM predictions at a FIXED PCA dim (the null fast path).

    Identical to :func:`glm_predict_loco` EXCEPT the per-fold PCA dim is the
    passed ``dim`` (fit on the train fold via ``_pca_fit`` / ``_pca_transform``)
    instead of being re-selected by ``nested_cv_pca_reduce``'s inner-LOO per
    fold. The basis is STILL fit on the train fold only (no held-out leakage);
    only the dim NUMBER is fixed. Used inside the shuffle / control nulls of the
    BINARY-companion read (issue763 BLOCKER analysis-null-infeasible-at-scale):
    the PCA dim is a regularization-capacity hyperparameter, NOT the permuted
    label, so it is chosen ONCE on the observed data per layer and held fixed
    across permutations — removing the ~245x inner-LOO-nested-CV-per-fold cost.
    ``nested_cv_pca_reduce`` is untouched. Returns (n_ctx,) held-out predictions.
    """
    n = x.shape[0]
    w = np.asarray(n_judged, dtype=np.float64)
    w = np.where(w < 1, 1.0, w)
    y = np.asarray(y, dtype=np.float64)
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu, comps = _pca_fit(x[tr], min(dim, len(tr) - 1))
        z_tr = _pca_transform(x[tr], mu, comps)
        z_held = _pca_transform(x[i : i + 1], mu, comps)
        res = _fit_binomial_glm(z_tr, y[tr], w[tr])
        if res is None:
            preds[i] = float(np.mean(y[tr]))
        else:
            import statsmodels.api as sm

            zc = sm.add_constant(z_held, has_constant="add")
            preds[i] = float(res.predict(zc)[0])
    return preds
