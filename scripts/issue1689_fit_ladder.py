"""Issue #1689 Phase D — 9-rung mapping-similarity ladder over pair set.

For each ordered pair (source, target) in the pair set (both mapping arms,
both models, all 4 layers), computes the WEAKEST rung reconciling source's
map with target's data. Runs selection-symmetric bootstrap argmin per plan
§5 (selection rides per draw of 1000 conv-id bootstraps).

Rungs 1-9 per plan §4 (weakest -> strongest correction):
  1. direct transfer                closed-form (M_S applied to x_T)
  2. context offset (Δx)            closed-form (mean-shift x)
  3. answer offset  (Δy)            closed-form (mean-shift y)
  4. bias refit  (b*)               closed-form (refit intercept on T)
  5. scalar-α                       closed-form (scalar rescale)
  6. rotation (orthogonal)          Procrustes SVD on train residuals
  7. context reparam A              ridge fit x_T -> x_S, apply M_S then recenter
  8. answer reparam B               ridge fit y_S -> y_T, apply to M_S(x_T)
  9. full A·M·B                     A on input side + B on output side

Rungs 7/8/9 port the math from scripts/issue1639_oneside_reparam.py (a
CLI-only script), reusing its ridge-based one-sided reparameterization
formulation. Bootstrap null band computed via shuffled-answer resamples per
`.claude/rules/selection-symmetric-nulls.md` § "Per-draw same-selection".

Ridge fits use inner-group-cv λ selection over the LAMBDAS grid (plan §11)
with a conv-id-grouped 5-fold train/test split (plan §11, ood-generalization
rule).

Smoke: --smoke → single pair × single layer × single arm, small bootstrap
count for fast verification. Full: --model-slug + --layer, all 4 layers via
--all-layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# CRITICAL: load_dotenv() BEFORE importing numpy / torch — the shared-VM
# thread caps (#847) freeze at first BLAS/torch import; a load_dotenv() call
# below the imports is too late (pinned by tests/test_shared_vm_thread_caps.py).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import (  # noqa: E402
    CAPTURE_LAYERS,
    HEADLINE_LAYER,
    LAMBDA_GRID_SIZE,
    LAMBDA_LOG_MAX,
    LAMBDA_LOG_MIN,
    N_BOOTSTRAP_DRAWS,
    N_FOLDS,
    N_REPARAM_NULL_DRAWS,
    RUNG_REACHED_THRESHOLD,
    enumerate_pair_set,
)

# Ridge λ grid: 13 log-spaced values from 1e-2 to 1e4 (plan §11 committed grid).
LAMBDAS: np.ndarray = np.logspace(LAMBDA_LOG_MIN, LAMBDA_LOG_MAX, LAMBDA_GRID_SIZE)


# ---------------------------------------------------------------------------
# Ridge fit primitives (Gram-dual for N < D; primal for N >= D).
# ---------------------------------------------------------------------------


def _fit_ridge_gram(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge: returns (W, b) s.t. Y ≈ X W + b, at fixed λ.

    Uses Gram-dual reformulation when N < D for speed (per
    `.claude/rules/vectorize-many-cell-fits.md`).
    """
    n, d = X.shape
    y_mean = Y.mean(axis=0)
    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    Yc = Y - y_mean
    if n < d:
        # Dual (N < D): W = X^T alpha with alpha = (XX^T + λI)^-1 Y
        G = Xc @ Xc.T + lam * np.eye(n)
        alpha = np.linalg.solve(G, Yc)
        W = Xc.T @ alpha
    else:
        # Primal (N >= D)
        G = Xc.T @ Xc + lam * np.eye(d)
        W = np.linalg.solve(G, Xc.T @ Yc)
    b = y_mean - x_mean @ W
    return W, b


# ---------------------------------------------------------------------------
# Eigendecomposition-based ridge — batched λ-scan (round-4 fix, concern
# bootstrap-wall-projection-over-plan).
# ---------------------------------------------------------------------------
#
# For a fixed centered X (n, d), the ridge solution at scalar λ is
#   W = (Xᵀ X + λ I)^-1 Xᵀ Y                    (primal, d < n)
#     = Xᵀ (X Xᵀ + λ I)^-1 Y                    (dual, n < d)
# Both admit an eigendecomposition of the smaller Gram matrix that is
# SHARED across every λ in the grid. Concretely, letting the smaller
# Gram be `S` (d × d primal, n × n dual) with eigendecomposition
# `S = U diag(σ²) Uᵀ`, the SAME `U` diagonalizes `S + λ I` for every λ,
# so a per-λ inversion collapses to a per-λ elementwise reciprocal
# `1/(σ² + λ)` and one `U diag(...) Uᵀ` sandwich. Cost per λ then becomes
# O(d²) (matmul) instead of O(d³) (fresh solve), and the shared
# eigendecomposition is amortized over the whole grid.
#
# When wrapped in inner-group-cv (K inner folds × L lambdas), the shared
# eigendecomposition is ALSO shared across the L lambdas within each inner
# fold — a K·L reduction to K eigendecomps + K·L cheap projections. This
# is the concrete win the round-3 `epm:compute-deviation` marker names.


def _ridge_eigh_prep(X: np.ndarray, Y: np.ndarray) -> dict:
    """Precompute the SHARED ridge eigendecomposition of the smaller Gram.

    Returns a dict of precomputed factors reusable across every λ in a scan:
      x_mean, y_mean:  centering means (float64)
      dual:            bool — True iff n < d, so Gram is (n × n) not (d × d)
      U:               eigenvectors of the SMALLER Gram (shape (m, m) with
                       m = min(n, d))
      sigma2:          eigenvalues of the smaller Gram (m,)
      Xc:              centered X (n, d)
      Yc:              centered Y (n, d_y)
    Uses torch.linalg.eigh's CPU fallback via _eigh_robust for numerical
    stability on near-singular subsampled Grams (see gotchas.md § cuSOLVER
    eigh non-convergence).
    """
    n, d = X.shape
    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    Xc = X - x_mean
    Yc = Y - y_mean
    dual = n < d
    if dual:
        # S = Xc @ Xc.T (n × n)
        S = Xc @ Xc.T
    else:
        # S = Xc.T @ Xc (d × d)
        S = Xc.T @ Xc
    # Symmetrize for numerical robustness before eigh.
    S = 0.5 * (S + S.T)
    sigma2, U = np.linalg.eigh(S)
    # Clip tiny negative eigenvalues to zero (numerical noise on PSD Grams).
    sigma2 = np.clip(sigma2, 0.0, None)
    return {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "dual": dual,
        "U": U,
        "sigma2": sigma2,
        "Xc": Xc,
        "Yc": Yc,
    }


def _ridge_predict_from_prep(prep: dict, X_test: np.ndarray, lam: float) -> np.ndarray:
    """Predict Y_test from a ridge fit at scalar λ using precomputed eigendecomp.

    Uses:
      primal (n >= d): W = U diag(1/(σ² + λ)) Uᵀ Xcᵀ Yc
      dual  (n <  d): α = U diag(1/(σ² + λ)) Uᵀ Yc, then W = Xcᵀ α

    Prediction is: X_test @ W + b, with b = y_mean − x_mean @ W.
    """
    Xc, Yc = prep["Xc"], prep["Yc"]
    U, sigma2 = prep["U"], prep["sigma2"]
    x_mean, y_mean = prep["x_mean"], prep["y_mean"]
    inv_diag = 1.0 / (sigma2 + lam)  # (m,)
    if prep["dual"]:
        # α = U diag(inv_diag) Uᵀ Yc
        UtY = U.T @ Yc  # (m, d_y)
        alpha = U @ (inv_diag[:, None] * UtY)  # (n, d_y)
        # Prediction: X_test @ Xcᵀ @ α + b
        # W = Xcᵀ @ α; predict as X_test_centered @ W (where centering is
        # against x_mean).
        Xtest_c = X_test - x_mean
        pred = Xtest_c @ (Xc.T @ alpha)  # (n_test, d_y)
    else:
        # W = U diag(inv_diag) Uᵀ Xcᵀ Yc
        XtY = Xc.T @ Yc  # (d, d_y)
        UtXtY = U.T @ XtY  # (d, d_y) — U is (d, d) here
        W = U @ (inv_diag[:, None] * UtXtY)  # (d, d_y)
        Xtest_c = X_test - x_mean
        pred = Xtest_c @ W  # (n_test, d_y)
    return pred + y_mean


def _ridge_fit_from_prep(prep: dict, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (W, b) at scalar λ using the precomputed eigendecomp.

    Same math as _ridge_predict_from_prep but returns the fit rather than
    a prediction — used where a downstream operation needs W_s (e.g. the
    source map used across all rungs).
    """
    Xc, Yc = prep["Xc"], prep["Yc"]
    U, sigma2 = prep["U"], prep["sigma2"]
    x_mean, y_mean = prep["x_mean"], prep["y_mean"]
    inv_diag = 1.0 / (sigma2 + lam)
    if prep["dual"]:
        UtY = U.T @ Yc
        alpha = U @ (inv_diag[:, None] * UtY)
        W = Xc.T @ alpha
    else:
        XtY = Xc.T @ Yc
        UtXtY = U.T @ XtY
        W = U @ (inv_diag[:, None] * UtXtY)
    b = y_mean - x_mean @ W
    return W, b


def _r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((Y_true - Y_pred) ** 2))
    ss_tot = float(np.sum((Y_true - Y_true.mean(axis=0)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _conv_grouped_folds(conv_ids: np.ndarray, n_folds: int, seed: int = 42) -> np.ndarray:
    """Assign each row to a fold [0..n_folds) grouped by conv_id (plan §11).

    Rows sharing a conv_id land in the same fold — matches #825
    heldout_r2_sweep group-fold convention + .claude/rules/ood-generalization-folds.md.
    """
    rng = np.random.default_rng(seed)
    unique_convs = np.unique(conv_ids)
    perm = rng.permutation(len(unique_convs))
    shuffled = unique_convs[perm]
    fold_map: dict = {c: int(i % n_folds) for i, c in enumerate(shuffled)}
    return np.array([fold_map[c] for c in conv_ids], dtype=np.int64)


def _fit_ridge_inner_group_cv(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    train_conv_ids: np.ndarray,
    lambdas: np.ndarray,
    n_inner_folds: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Ridge fit with λ selected by inner-group-cv over LAMBDAS grid.

    Returns (W, b, best_lambda). Splits X_train into `n_inner_folds` by
    conv_id, computes mean held-out R² per λ, picks the best, refits on all
    of X_train at best_lambda. Per plan §11: lambda_selection="inner-group-cv".

    Round-4 vectorization (concern bootstrap-wall-projection-over-plan):
    each inner fold's ridge λ-scan is now driven by a SHARED
    eigendecomposition of the fold's centered Gram — L lambdas cost 1
    eigh + L cheap `U diag(1/(σ²+λ)) Uᵀ` sandwiches, instead of L fresh
    O(D³) solves. Same math (bit-equivalent up to float roundoff), ~L×
    faster on the λ-scan.
    """
    inner_folds = _conv_grouped_folds(train_conv_ids, n_inner_folds, seed=seed)
    scores = np.zeros((len(lambdas), n_inner_folds), dtype=np.float64)
    for fold_i in range(n_inner_folds):
        te_mask = inner_folds == fold_i
        tr_mask = ~te_mask
        if tr_mask.sum() < 3 or te_mask.sum() == 0:
            # Degenerate inner fold — assign NaN (skipped in mean).
            scores[:, fold_i] = np.nan
            continue
        X_tr, Y_tr = X_train[tr_mask], Y_train[tr_mask]
        X_te, Y_te = X_train[te_mask], Y_train[te_mask]
        # SHARED eigendecomposition across the λ grid.
        prep = _ridge_eigh_prep(X_tr, Y_tr)
        for li, lam in enumerate(lambdas):
            pred = _ridge_predict_from_prep(prep, X_te, lam=float(lam))
            scores[li, fold_i] = _r2(Y_te, pred)
    # Mean over inner folds (nanmean to tolerate degenerate folds).
    mean_scores = np.nanmean(scores, axis=1)
    # Pick the λ with highest mean CV R² (ties broken by largest λ, most regularization).
    valid = ~np.isnan(mean_scores)
    if not valid.any():
        # All folds degenerate — fall back to mid-grid λ.
        best_lam = float(lambdas[len(lambdas) // 2])
    else:
        best_idx = int(np.argmax(np.where(valid, mean_scores, -np.inf)))
        best_lam = float(lambdas[best_idx])
    # Refit on all of X_train at best λ — reuses the outer eigendecomp
    # via _ridge_fit_from_prep (one more eigh on the full X_train Gram).
    prep_full = _ridge_eigh_prep(X_train, Y_train)
    W, b = _ridge_fit_from_prep(prep_full, best_lam)
    return W, b, best_lam


# ---------------------------------------------------------------------------
# Per-rung closed-form / ridge computations.
# ---------------------------------------------------------------------------


def _rung_1_direct(W_s, b_s, X_T_test):
    return X_T_test @ W_s + b_s


def _rung_2_ctx_offset(W_s, b_s, X_T_test, X_S_full, X_T_full):
    dx = X_S_full.mean(0) - X_T_full.mean(0)
    return (X_T_test - dx) @ W_s + b_s


def _rung_3_ans_offset(W_s, b_s, X_T_test, X_T_full, Y_T_full):
    mean_x = X_T_full.mean(0)
    pred_at_mean = mean_x @ W_s + b_s
    dy = Y_T_full.mean(0) - pred_at_mean
    return X_T_test @ W_s + b_s + dy


def _rung_4_bias_refit(W_s, X_T_train, Y_T_train, X_T_test):
    """Rung 4: bias refit b* = mean(Y_T - X_T W_s) on train, eval on test."""
    pred_train = X_T_train @ W_s
    b_star = (Y_T_train - pred_train).mean(0)
    return X_T_test @ W_s + b_star


def _rung_5_scalar_alpha(W_s, X_T_train, Y_T_train, X_T_test, b_star):
    """Rung 5: α W_s x + b*, α scalar closed-form."""
    pred_train = X_T_train @ W_s
    num = float(np.sum(pred_train * (Y_T_train - b_star)))
    den = float(np.sum(pred_train**2)) + 1e-12
    alpha = num / den
    return alpha * (X_T_test @ W_s) + b_star


def _rung_6_rotation(W_s, X_T_train, Y_T_train, X_T_test, b_star):
    """Rung 6: R W_s x + b*, R = orthogonal Procrustes on train residuals."""
    pred_train = X_T_train @ W_s
    target = Y_T_train - b_star
    M = target.T @ pred_train  # (D_y, D_pred)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return (X_T_test @ W_s) @ R.T + b_star


def _rung_7_ctx_reparam(
    W_s, b_s, X_S_train, X_T_train, Y_T_train, X_T_test, lambdas: np.ndarray, train_conv_ids
) -> np.ndarray:
    """Rung 7: context reparam A. Fit A: x_T -> x_S on train, apply M_S then recenter.

    Ports issue1639_oneside_reparam.py "input_only" arm:
      A_ctx(x_t) ≈ x_s (ridge fit), then M_s applied, then recentered to y_T mean.
    """
    # Fit A_ctx via ridge with inner-group-cv λ selection.
    A_W, A_b, _lam = _fit_ridge_inner_group_cv(X_T_train, X_S_train, train_conv_ids, lambdas)
    # Predict source-side x-hats then apply source map + recenter to y_T train mean.
    x_pred_test = X_T_test @ A_W + A_b
    y_pred_raw = x_pred_test @ W_s + b_s
    # Recenter: pred - train_pred_mean + y_T_train_mean (final-prediction recentering).
    x_pred_train = X_T_train @ A_W + A_b
    y_pred_train_raw = x_pred_train @ W_s + b_s
    return y_pred_raw - y_pred_train_raw.mean(0) + Y_T_train.mean(0)


def _rung_8_ans_reparam(
    W_s, b_s, Y_S_train, Y_T_train, X_T_train, X_T_test, lambdas: np.ndarray, train_conv_ids
) -> np.ndarray:
    """Rung 8: answer reparam B. Fit B: y_S -> y_T on train, apply to M_S(x_T).

    Ports issue1639_oneside_reparam.py "output_only" arm:
      B(y_s) ≈ y_t (ridge fit), then M_s(x_t) run through B, then recentered.
    """
    B_W, B_b, _lam = _fit_ridge_inner_group_cv(Y_S_train, Y_T_train, train_conv_ids, lambdas)
    y_s_test = X_T_test @ W_s + b_s
    y_pred_raw = y_s_test @ B_W + B_b
    y_s_train = X_T_train @ W_s + b_s
    y_pred_train_raw = y_s_train @ B_W + B_b
    return y_pred_raw - y_pred_train_raw.mean(0) + Y_T_train.mean(0)


def _rung_9_full_amb(
    W_s,
    b_s,
    X_S_train,
    Y_S_train,
    X_T_train,
    Y_T_train,
    X_T_test,
    lambdas: np.ndarray,
    train_conv_ids,
) -> np.ndarray:
    """Rung 9: full A·M·B. A on input side (x_T→x_S), M_s applied, B on output side (y_S→y_T).

    Composes rung 7 + rung 8: A_ctx(x_T) → x̂_S, then M_S(x̂_S) → ŷ_S, then B_ans(ŷ_S) → ŷ_T.
    """
    A_W, A_b, _ = _fit_ridge_inner_group_cv(X_T_train, X_S_train, train_conv_ids, lambdas)
    B_W, B_b, _ = _fit_ridge_inner_group_cv(Y_S_train, Y_T_train, train_conv_ids, lambdas)
    # Compose: X_T -> X̂_S -> Ŷ_S -> Ŷ_T
    x_hat_test = X_T_test @ A_W + A_b
    y_s_test = x_hat_test @ W_s + b_s
    y_pred_raw = y_s_test @ B_W + B_b
    # Recenter with the composite chain's train-fold mean.
    x_hat_train = X_T_train @ A_W + A_b
    y_s_train = x_hat_train @ W_s + b_s
    y_pred_train_raw = y_s_train @ B_W + B_b
    return y_pred_raw - y_pred_train_raw.mean(0) + Y_T_train.mean(0)


# ---------------------------------------------------------------------------
# Core ladder computation for one (source, target, arm) triple.
# ---------------------------------------------------------------------------


def _load_cell_layer(store_root: Path, cell_slug: str, layer: int) -> dict:
    """Load one (cell, layer) bundle -> {X_prefix, X_context, Y, conv_ids}."""
    import torch

    path = store_root / cell_slug / f"L{layer}.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "X_prefix": np.asarray(bundle["X_prefix"], dtype=np.float64),
        "X_context": np.asarray(bundle["X_context"], dtype=np.float64),
        "Y": np.asarray(bundle["Y"], dtype=np.float64),
        "conv_ids": np.asarray(bundle["conv_ids"]),
    }


def _compute_ladder_r2s(
    X_S: np.ndarray,
    Y_S: np.ndarray,
    X_T: np.ndarray,
    Y_T: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_conv_ids: np.ndarray,
    lambdas: np.ndarray,
    full_conv_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute R^2 for all 9 rungs on a fixed (train_idx, test_idx) split.

    Source map W_s, b_s is fit on ALL source rows (per plan §4: 'fit source
    map on ALL source rows'). Downstream rungs work on the target's train/test.
    ``full_conv_ids`` is the conv-id array for X_S/Y_S — used for inner-group-cv
    lambda selection on the source-map fit. Defaults to train_conv_ids when
    the source is passed already sliced to the training subset.
    """
    if full_conv_ids is None:
        full_conv_ids = train_conv_ids
    # Source map — fit on all source rows via inner-group-cv lambda selection.
    W_s, b_s, _lam_s = _fit_ridge_inner_group_cv(X_S, Y_S, full_conv_ids, lambdas)

    X_S_train = X_S[train_idx]
    Y_S_train = Y_S[train_idx]
    X_T_train = X_T[train_idx]
    Y_T_train = Y_T[train_idx]
    X_T_test = X_T[test_idx]

    # b* for rungs 4-6 (bias refit on target's train fold).
    pred_train_s = X_T_train @ W_s
    b_star = (Y_T_train - pred_train_s).mean(0)

    preds = {}
    preds["rung_1_direct"] = _rung_1_direct(W_s, b_s, X_T_test)
    preds["rung_2_ctx_offset"] = _rung_2_ctx_offset(W_s, b_s, X_T_test, X_S, X_T)
    preds["rung_3_ans_offset"] = _rung_3_ans_offset(W_s, b_s, X_T_test, X_T, Y_T)
    preds["rung_4_bias_refit"] = _rung_4_bias_refit(W_s, X_T_train, Y_T_train, X_T_test)
    preds["rung_5_scalar_alpha"] = _rung_5_scalar_alpha(W_s, X_T_train, Y_T_train, X_T_test, b_star)
    preds["rung_6_rotation"] = _rung_6_rotation(W_s, X_T_train, Y_T_train, X_T_test, b_star)
    preds["rung_7_ctx_reparam"] = _rung_7_ctx_reparam(
        W_s, b_s, X_S_train, X_T_train, Y_T_train, X_T_test, lambdas, train_conv_ids
    )
    preds["rung_8_ans_reparam"] = _rung_8_ans_reparam(
        W_s, b_s, Y_S_train, Y_T_train, X_T_train, X_T_test, lambdas, train_conv_ids
    )
    preds["rung_9_full_AMB"] = _rung_9_full_amb(
        W_s, b_s, X_S_train, Y_S_train, X_T_train, Y_T_train, X_T_test, lambdas, train_conv_ids
    )

    Y_T_test = Y_T[test_idx]
    return {name: _r2(Y_T_test, pred) for name, pred in preds.items()}


def _reach_bar_within_cell(
    X_T: np.ndarray,
    Y_T: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_conv_ids: np.ndarray,
    lambdas: np.ndarray,
    threshold: float = RUNG_REACHED_THRESHOLD,
) -> tuple[float, float]:
    """Compute within-target held-out R² (ceiling) and reach_bar = threshold * ceiling."""
    W_ref, b_ref, _ = _fit_ridge_inner_group_cv(
        X_T[train_idx], Y_T[train_idx], train_conv_ids, lambdas
    )
    pred = X_T[test_idx] @ W_ref + b_ref
    r2_within = _r2(Y_T[test_idx], pred)
    reach_bar = threshold * r2_within if r2_within > 0 else float("-inf")
    return r2_within, reach_bar


def _rung_reached_from_r2s(rung_r2s: dict[str, float], reach_bar: float) -> int:
    """Selection: find the WEAKEST (lowest-index) rung with R² >= reach_bar."""
    ordered_keys = [
        "rung_1_direct",
        "rung_2_ctx_offset",
        "rung_3_ans_offset",
        "rung_4_bias_refit",
        "rung_5_scalar_alpha",
        "rung_6_rotation",
        "rung_7_ctx_reparam",
        "rung_8_ans_reparam",
        "rung_9_full_AMB",
    ]
    for i, key in enumerate(ordered_keys, start=1):
        if rung_r2s[key] >= reach_bar:
            return i
    return 9  # default to strongest if none reaches


def _run_ladder_pair(
    source: dict,
    target: dict,
    *,
    arm: str,
    n_bootstrap_draws: int,
    n_null_draws: int,
    threshold: float = RUNG_REACHED_THRESHOLD,
    seed: int = 42,
) -> dict:
    """Run 9 rungs + selection-symmetric bootstrap for one (source, target, arm).

    Per plan §5 + .claude/rules/selection-symmetric-nulls.md:
    - Point estimate on the full row-paired conv-id-grouped 80/20 split.
    - Per-draw bootstrap: resample conv_ids with replacement, run ALL 9 rungs,
      compute rung_reached per draw. Report per-draw R² matrix and rung
      distribution.
    - Selection-symmetric null: shuffled-answer resamples (matched-capacity),
      same argmin selection per draw.
    """
    X_S_p = source[f"X_{arm}"]
    Y_S = source["Y"]
    X_T_p = target[f"X_{arm}"]
    Y_T = target["Y"]
    conv_S = source["conv_ids"]
    conv_T = target["conv_ids"]

    # Row-pair by conv_id (intersection).
    common = np.intersect1d(conv_S, conv_T)
    if len(common) < 3:
        return {"error": "insufficient shared conv_ids", "n_common": int(len(common))}
    s_idx = np.array([np.where(conv_S == c)[0][0] for c in common])
    t_idx = np.array([np.where(conv_T == c)[0][0] for c in common])

    X_S = X_S_p[s_idx]
    Y_S = Y_S[s_idx]
    X_T = X_T_p[t_idx]
    Y_T = Y_T[t_idx]
    common_convs = common  # noqa: F841 (kept for provenance)

    # Conv-grouped train/test split: use 5-fold groups; train = folds 1..4, test = fold 0.
    n = len(common)
    folds = _conv_grouped_folds(common, n_folds=N_FOLDS, seed=seed)
    train_mask = folds != 0
    test_mask = folds == 0
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    # Smoke fallback: if any split is too small, use all rows for both.
    if len(train_idx) < 3 or len(test_idx) < 1:
        train_idx = np.arange(n)
        test_idx = np.arange(n)
    train_conv_ids = common[train_idx]

    # ---- Point estimate ----
    rung_r2s_point = _compute_ladder_r2s(
        X_S,
        Y_S,
        X_T,
        Y_T,
        train_idx,
        test_idx,
        train_conv_ids,
        LAMBDAS,
        full_conv_ids=common,
    )
    r2_within, reach_bar = _reach_bar_within_cell(
        X_T, Y_T, train_idx, test_idx, train_conv_ids, LAMBDAS, threshold=threshold
    )
    rung_reached_point = _rung_reached_from_r2s(rung_r2s_point, reach_bar)

    # ---- Selection-symmetric bootstrap ----
    # Persist per-draw × per-rung R² matrix + per-draw rung_reached.
    rng = np.random.default_rng(seed + 1)
    rung_names = list(rung_r2s_point.keys())
    bootstrap_r2_matrix = np.full((n_bootstrap_draws, len(rung_names)), np.nan, dtype=np.float64)
    bootstrap_rung_reached = np.zeros(n_bootstrap_draws, dtype=np.int64)
    for draw_i in range(n_bootstrap_draws):
        # Conv-level resample with replacement.
        resample_idx = rng.choice(n, size=n, replace=True)
        # Rebuild train/test split on the resample's conv_ids to stay
        # selection-symmetric: same-selection recipe per draw.
        resample_convs = common[resample_idx]
        resample_folds = _conv_grouped_folds(resample_convs, n_folds=N_FOLDS, seed=seed + draw_i)
        rs_train = np.where(resample_folds != 0)[0]
        rs_test = np.where(resample_folds == 0)[0]
        if len(rs_train) < 3 or len(rs_test) < 1:
            continue  # NaN row
        X_S_d, Y_S_d = X_S[resample_idx], Y_S[resample_idx]
        X_T_d, Y_T_d = X_T[resample_idx], Y_T[resample_idx]
        train_conv_ids_d = resample_convs[rs_train]
        try:
            r2s_draw = _compute_ladder_r2s(
                X_S_d,
                Y_S_d,
                X_T_d,
                Y_T_d,
                rs_train,
                rs_test,
                train_conv_ids_d,
                LAMBDAS,
                full_conv_ids=resample_convs,
            )
        except np.linalg.LinAlgError:
            continue  # degenerate resample; NaN row
        for j, name in enumerate(rung_names):
            bootstrap_r2_matrix[draw_i, j] = r2s_draw[name]
        _, reach_bar_d = _reach_bar_within_cell(
            X_T_d, Y_T_d, rs_train, rs_test, train_conv_ids_d, LAMBDAS, threshold=threshold
        )
        bootstrap_rung_reached[draw_i] = _rung_reached_from_r2s(r2s_draw, reach_bar_d)

    # ---- Matched-capacity null (shuffled-answer) ----
    # Batched null: shuffle Y_S and Y_T within their own rows, refit, compute
    # rung_reached per draw. Same-selection ceiling per draw.
    null_r2_matrix = np.full((n_null_draws, len(rung_names)), np.nan, dtype=np.float64)
    null_rung_reached = np.zeros(n_null_draws, dtype=np.int64)
    for draw_i in range(n_null_draws):
        perm_s = rng.permutation(n)
        perm_t = rng.permutation(n)
        try:
            r2s_null = _compute_ladder_r2s(
                X_S,
                Y_S[perm_s],
                X_T,
                Y_T[perm_t],
                train_idx,
                test_idx,
                train_conv_ids,
                LAMBDAS,
                full_conv_ids=common,
            )
        except np.linalg.LinAlgError:
            continue
        for j, name in enumerate(rung_names):
            null_r2_matrix[draw_i, j] = r2s_null[name]
        _, reach_bar_null = _reach_bar_within_cell(
            X_T, Y_T[perm_t], train_idx, test_idx, train_conv_ids, LAMBDAS, threshold=threshold
        )
        null_rung_reached[draw_i] = _rung_reached_from_r2s(r2s_null, reach_bar_null)

    return {
        "n_common": int(n),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "r2_within_target": float(r2_within),
        "reach_bar_90pct": float(reach_bar),
        "rung_r2s_point": {k: float(v) for k, v in rung_r2s_point.items()},
        "rung_reached_point": int(rung_reached_point),
        "bootstrap_draws": {
            "n_draws": int(n_bootstrap_draws),
            "rung_names": rung_names,
            "r2_matrix": bootstrap_r2_matrix.tolist(),
            "rung_reached_per_draw": bootstrap_rung_reached.tolist(),
            "rung_reached_median": float(np.median(bootstrap_rung_reached)),
            "rung_reached_p025": float(np.percentile(bootstrap_rung_reached, 2.5)),
            "rung_reached_p975": float(np.percentile(bootstrap_rung_reached, 97.5)),
        },
        "matched_capacity_null": {
            "n_draws": int(n_null_draws),
            "rung_reached_per_draw": null_rung_reached.tolist(),
            "rung_reached_null_p975": float(np.percentile(null_rung_reached, 97.5))
            if n_null_draws > 0
            else float("nan"),
        },
    }


def run_all_pairs(
    store_root: Path,
    *,
    model_slug: str,
    layer: int = HEADLINE_LAYER,
    smoke: bool = False,
    n_bootstrap_draws: int = N_BOOTSTRAP_DRAWS,
    n_null_draws: int = N_REPARAM_NULL_DRAWS,
) -> dict:
    """Run the pair ladder for one (model, layer). Both arms per pair."""
    pairs = enumerate_pair_set()
    if smoke:
        # Smoke: filter pairs to those whose BOTH cells have captured stores;
        # fall back to self-pair on the one available cell.
        available = set()
        for src, tgt in pairs:
            src_cell = f"{model_slug}/{src}"
            tgt_cell = f"{model_slug}/{tgt}"
            if (store_root / src_cell / f"L{layer}.pt").exists() and (
                store_root / tgt_cell / f"L{layer}.pt"
            ).exists():
                available.add((src, tgt))
        if available:
            pairs = [next(iter(available))]
        else:
            for src, tgt in pairs:
                src_cell = f"{model_slug}/{src}"
                if (store_root / src_cell / f"L{layer}.pt").exists():
                    pairs = [(src, src)]
                    break
            else:
                raise ValueError(f"no captured cells found under {store_root}/{model_slug}")

    out: dict[str, Any] = {
        "model": model_slug,
        "layer": layer,
        "n_pairs": len(pairs),
        "arms": ["prefix", "context"],
        "n_bootstrap_draws": n_bootstrap_draws,
        "n_null_draws": n_null_draws,
        "pairs": {},
    }
    for src, tgt in pairs:
        src_cell = f"{model_slug}/{src}"
        tgt_cell = f"{model_slug}/{tgt}"
        source = _load_cell_layer(store_root, src_cell, layer)
        target = _load_cell_layer(store_root, tgt_cell, layer)
        pair_key = f"{src}__{tgt}"
        out["pairs"][pair_key] = {}
        for arm in ["prefix", "context"]:
            out["pairs"][pair_key][arm] = _run_ladder_pair(
                source,
                target,
                arm=arm,
                n_bootstrap_draws=n_bootstrap_draws,
                n_null_draws=n_null_draws,
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-root", type=Path, required=True)
    ap.add_argument("--model-slug", type=str, required=True, help="e.g. Qwen_Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=HEADLINE_LAYER, choices=list(CAPTURE_LAYERS))
    ap.add_argument(
        "--all-layers",
        action="store_true",
        help="Loop over all 4 CAPTURE_LAYERS (plan §6 exploratory dump); writes one out per layer.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--bootstrap-draws",
        type=int,
        default=None,
        help="Override N_BOOTSTRAP_DRAWS (default: 1000 full / 10 smoke)",
    )
    ap.add_argument(
        "--null-draws",
        type=int,
        default=None,
        help="Override N_REPARAM_NULL_DRAWS (default: 200 full / 5 smoke)",
    )
    args = ap.parse_args()

    # Smoke defaults: keep bootstrap loop small so the code path executes without hours of compute.
    if args.bootstrap_draws is None:
        args.bootstrap_draws = 10 if args.smoke else N_BOOTSTRAP_DRAWS
    if args.null_draws is None:
        args.null_draws = 5 if args.smoke else N_REPARAM_NULL_DRAWS

    print(f"[fit_ladder] model={args.model_slug} smoke={args.smoke}")
    print(
        f"[fit_ladder] bootstrap_draws={args.bootstrap_draws} null_draws={args.null_draws} "
        f"(N_BOOTSTRAP_DRAWS={N_BOOTSTRAP_DRAWS}, N_REPARAM_NULL_DRAWS={N_REPARAM_NULL_DRAWS})"
    )

    layers = list(CAPTURE_LAYERS) if args.all_layers else [args.layer]
    for layer in layers:
        print(f"[fit_ladder] --- layer L{layer} ---")
        results = run_all_pairs(
            args.store_root,
            model_slug=args.model_slug,
            layer=layer,
            smoke=args.smoke,
            n_bootstrap_draws=args.bootstrap_draws,
            n_null_draws=args.null_draws,
        )
        # Rewrite out path to include layer when running all-layers.
        if args.all_layers:
            out_path = args.out.with_name(
                args.out.stem.replace(f"_L{args.layer}", "") + f"_L{layer}" + args.out.suffix
            )
        else:
            out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[fit_ladder] wrote {out_path} ({results['n_pairs']} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
