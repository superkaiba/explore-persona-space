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
import time
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
    LAMBDA_GRIDS,
    LAMBDA_LOG_MAX,
    LAMBDA_LOG_MIN,
    N_BOOTSTRAP_DRAWS,
    N_FOLDS,
    N_REPARAM_NULL_DRAWS,
    RUNG_REACHED_THRESHOLD,
    enumerate_pair_set,
    resolve_lambda_grid,
)

# Ridge λ grid: 13 log-spaced values from 1e-2 to 1e4 (plan §11 committed grid).
# The wider-lambda-ceilings follow-up threads a NAMED alternative grid
# ("wide19", issue1689_common.LAMBDA_GRIDS) through run_all_pairs; the module
# default stays byte-identical.
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


# ---------------------------------------------------------------------------
# Torch engine (R16) — numerics-preserving port of the numpy ladder.
#
# The numpy path above runs the whole ladder serially on CPU BLAS; the dense
# kernels (Gram matmuls, n x n eigh, the rung-6 d x d SVD) dominate wall-time
# and are far faster on one H100 in float64. ALL rng / fold-assignment /
# lambda-selection logic STAYS in numpy with an IDENTICAL call order, so fold
# splits, resample indices and lambda picks are bit-identical across engines;
# only the dense linear algebra moves to torch (fp64). Equivalence is gated
# by --verify-equivalence (runs BOTH engines on real pairs, compares every
# rung R^2 + rung_reached). The numpy path is retained verbatim above as the
# equivalence reference (vectorize-many-cell-fits.md Supersede contract:
# contained reference, not a silently-importable twin — same entrypoint).
# ---------------------------------------------------------------------------


def _eigh_robust_t(S):
    """torch.linalg.eigh with CPU fallback on cuSOLVER non-convergence.

    The #1335 `_eigh_robust` pattern (gotchas.md): cuda syevd can fail on
    near-singular resampled Grams that CPU LAPACK decomposes fine; the
    fallback is an exact numerical-backend swap, not a semantic change.
    """
    import torch

    try:
        sigma2, U = torch.linalg.eigh(S)
    except torch.linalg.LinAlgError:
        print("[fit_ladder] eigh: cuSOLVER non-convergence -> CPU fallback", flush=True)
        sigma2, U = torch.linalg.eigh(S.cpu())
        sigma2 = sigma2.to(S.device)
        U = U.to(S.device)
    return sigma2, U


def _svd_robust_t(M):
    """torch.linalg.svd (full_matrices=False) with CPU fallback (same rationale)."""
    import torch

    try:
        U, s, Vt = torch.linalg.svd(M, full_matrices=False)
    except torch.linalg.LinAlgError:
        print("[fit_ladder] svd: cuSOLVER non-convergence -> CPU fallback", flush=True)
        U, s, Vt = torch.linalg.svd(M.cpu(), full_matrices=False)
        U = U.to(M.device)
        s = s.to(M.device)
        Vt = Vt.to(M.device)
    return U, s, Vt


def _ridge_eigh_prep_t(X, Y) -> dict:
    """Torch mirror of _ridge_eigh_prep (same keys, torch tensors)."""
    n, d = X.shape
    x_mean = X.mean(dim=0)
    y_mean = Y.mean(dim=0)
    Xc = X - x_mean
    Yc = Y - y_mean
    dual = n < d
    S = Xc @ Xc.T if dual else Xc.T @ Xc
    S = 0.5 * (S + S.T)
    sigma2, U = _eigh_robust_t(S)
    sigma2 = sigma2.clamp_min(0.0)
    return {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "dual": dual,
        "U": U,
        "sigma2": sigma2,
        "Xc": Xc,
        "Yc": Yc,
    }


def _ridge_predict_from_prep_t(prep: dict, X_test, lam: float):
    """Torch mirror of _ridge_predict_from_prep."""
    Xc, Yc = prep["Xc"], prep["Yc"]
    U, sigma2 = prep["U"], prep["sigma2"]
    x_mean, y_mean = prep["x_mean"], prep["y_mean"]
    inv_diag = 1.0 / (sigma2 + lam)
    Xtest_c = X_test - x_mean
    if prep["dual"]:
        UtY = U.T @ Yc
        alpha = U @ (inv_diag[:, None] * UtY)
        pred = Xtest_c @ (Xc.T @ alpha)
    else:
        XtY = Xc.T @ Yc
        UtXtY = U.T @ XtY
        W = U @ (inv_diag[:, None] * UtXtY)
        pred = Xtest_c @ W
    return pred + y_mean


def _ridge_fit_from_prep_t(prep: dict, lam: float):
    """Torch mirror of _ridge_fit_from_prep — returns (W, b) tensors."""
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


def _r2_t(Y_true, Y_pred) -> float:
    """Torch mirror of _r2 (returns python float)."""
    ss_res = float(((Y_true - Y_pred) ** 2).sum().item())
    ss_tot = float(((Y_true - Y_true.mean(dim=0)) ** 2).sum().item())
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _fit_ridge_inner_group_cv_t(
    X_train,
    Y_train,
    train_conv_ids: np.ndarray,
    lambdas: np.ndarray,
    n_inner_folds: int = 3,
    seed: int = 42,
):
    """Torch mirror of _fit_ridge_inner_group_cv.

    Fold assignment + lambda selection stay in numpy (identical rng + argmax
    tie-breaking); only the per-fold prep/predict math runs on the device.
    """
    import torch

    dev = X_train.device
    inner_folds = _conv_grouped_folds(train_conv_ids, n_inner_folds, seed=seed)
    scores = np.zeros((len(lambdas), n_inner_folds), dtype=np.float64)
    for fold_i in range(n_inner_folds):
        te_mask = inner_folds == fold_i
        tr_mask = ~te_mask
        if tr_mask.sum() < 3 or te_mask.sum() == 0:
            scores[:, fold_i] = np.nan
            continue
        tr_idx = torch.from_numpy(np.where(tr_mask)[0]).to(dev)
        te_idx = torch.from_numpy(np.where(te_mask)[0]).to(dev)
        X_tr, Y_tr = X_train[tr_idx], Y_train[tr_idx]
        X_te, Y_te = X_train[te_idx], Y_train[te_idx]
        prep = _ridge_eigh_prep_t(X_tr, Y_tr)
        for li, lam in enumerate(lambdas):
            pred = _ridge_predict_from_prep_t(prep, X_te, lam=float(lam))
            scores[li, fold_i] = _r2_t(Y_te, pred)
    mean_scores = np.nanmean(scores, axis=1)
    valid = ~np.isnan(mean_scores)
    if not valid.any():
        best_lam = float(lambdas[len(lambdas) // 2])
    else:
        best_idx = int(np.argmax(np.where(valid, mean_scores, -np.inf)))
        best_lam = float(lambdas[best_idx])
    prep_full = _ridge_eigh_prep_t(X_train, Y_train)
    W, b = _ridge_fit_from_prep_t(prep_full, best_lam)
    return W, b, best_lam


def _compute_ladder_r2s_t(
    X_S,
    Y_S,
    X_T,
    Y_T,
    train_idx,
    test_idx,
    train_conv_ids: np.ndarray,
    lambdas: np.ndarray,
    full_conv_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Torch mirror of _compute_ladder_r2s (tensors on device; idx torch long)."""
    if full_conv_ids is None:
        full_conv_ids = train_conv_ids
    W_s, b_s, _lam_s = _fit_ridge_inner_group_cv_t(X_S, Y_S, full_conv_ids, lambdas)

    X_S_train = X_S[train_idx]
    Y_S_train = Y_S[train_idx]
    X_T_train = X_T[train_idx]
    Y_T_train = Y_T[train_idx]
    X_T_test = X_T[test_idx]

    pred_train_s = X_T_train @ W_s
    b_star = (Y_T_train - pred_train_s).mean(dim=0)

    preds = {}
    # rung 1: direct
    preds["rung_1_direct"] = X_T_test @ W_s + b_s
    # rung 2: ctx offset
    dx = X_S.mean(dim=0) - X_T.mean(dim=0)
    preds["rung_2_ctx_offset"] = (X_T_test - dx) @ W_s + b_s
    # rung 3: ans offset
    mean_x = X_T.mean(dim=0)
    pred_at_mean = mean_x @ W_s + b_s
    dy = Y_T.mean(dim=0) - pred_at_mean
    preds["rung_3_ans_offset"] = X_T_test @ W_s + b_s + dy
    # rung 4: bias refit (recomputes b_star internally in the numpy path too)
    b_star4 = (Y_T_train - X_T_train @ W_s).mean(dim=0)
    preds["rung_4_bias_refit"] = X_T_test @ W_s + b_star4
    # rung 5: scalar alpha
    pred_train5 = X_T_train @ W_s
    num = float((pred_train5 * (Y_T_train - b_star)).sum().item())
    den = float((pred_train5**2).sum().item()) + 1e-12
    alpha = num / den
    preds["rung_5_scalar_alpha"] = alpha * (X_T_test @ W_s) + b_star
    # rung 6: rotation (Procrustes on train residuals)
    target6 = Y_T_train - b_star
    M6 = target6.T @ pred_train5
    U6, _s6, Vt6 = _svd_robust_t(M6)
    R6 = U6 @ Vt6
    preds["rung_6_rotation"] = (X_T_test @ W_s) @ R6.T + b_star
    # rung 7: ctx reparam A
    A_W, A_b, _ = _fit_ridge_inner_group_cv_t(X_T_train, X_S_train, train_conv_ids, lambdas)
    x_pred_test = X_T_test @ A_W + A_b
    y_pred_raw7 = x_pred_test @ W_s + b_s
    x_pred_train = X_T_train @ A_W + A_b
    y_pred_train_raw7 = x_pred_train @ W_s + b_s
    preds["rung_7_ctx_reparam"] = (
        y_pred_raw7 - y_pred_train_raw7.mean(dim=0) + Y_T_train.mean(dim=0)
    )
    # rung 8: ans reparam B
    B_W, B_b, _ = _fit_ridge_inner_group_cv_t(Y_S_train, Y_T_train, train_conv_ids, lambdas)
    y_s_test = X_T_test @ W_s + b_s
    y_pred_raw8 = y_s_test @ B_W + B_b
    y_s_train = X_T_train @ W_s + b_s
    y_pred_train_raw8 = y_s_train @ B_W + B_b
    preds["rung_8_ans_reparam"] = (
        y_pred_raw8 - y_pred_train_raw8.mean(dim=0) + Y_T_train.mean(dim=0)
    )
    # rung 9: full A.M.B. The numpy path REFITS A and B with byte-identical
    # inputs to the rung-7/8 fits (deterministic — no rng in the fit), so
    # reusing (A_W, A_b) + (B_W, B_b) is mathematically identical and saves
    # ~25% of the per-eval eigh work.
    x_hat_test = X_T_test @ A_W + A_b
    y_s_test9 = x_hat_test @ W_s + b_s
    y_pred_raw9 = y_s_test9 @ B_W + B_b
    x_hat_train = X_T_train @ A_W + A_b
    y_s_train9 = x_hat_train @ W_s + b_s
    y_pred_train_raw9 = y_s_train9 @ B_W + B_b
    preds["rung_9_full_AMB"] = y_pred_raw9 - y_pred_train_raw9.mean(dim=0) + Y_T_train.mean(dim=0)

    Y_T_test = Y_T[test_idx]
    return {name: _r2_t(Y_T_test, pred) for name, pred in preds.items()}


def _reach_bar_within_cell_t(
    X_T,
    Y_T,
    train_idx,
    test_idx,
    train_conv_ids: np.ndarray,
    lambdas: np.ndarray,
    threshold: float = RUNG_REACHED_THRESHOLD,
) -> tuple[float, float]:
    """Torch mirror of _reach_bar_within_cell."""
    W_ref, b_ref, _ = _fit_ridge_inner_group_cv_t(
        X_T[train_idx], Y_T[train_idx], train_conv_ids, lambdas
    )
    pred = X_T[test_idx] @ W_ref + b_ref
    r2_within = _r2_t(Y_T[test_idx], pred)
    reach_bar = threshold * r2_within if r2_within > 0 else float("-inf")
    return r2_within, reach_bar


def _ladder_pair_core_torch(
    X_S: np.ndarray,
    Y_S: np.ndarray,
    X_T: np.ndarray,
    Y_T: np.ndarray,
    common: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_conv_ids: np.ndarray,
    *,
    n_bootstrap_draws: int,
    n_null_draws: int,
    threshold: float,
    seed: int,
    device: str,
    lambdas: np.ndarray | None = None,
) -> dict:
    """Torch-engine core of _run_ladder_pair (point + bootstrap + null).

    rng call ORDER is kept identical to the numpy core so resample indices,
    fold splits and permutations are bit-identical across engines.
    ``lambdas=None`` (default) uses the module ``LAMBDAS`` grid byte-for-byte;
    the wider-lambda-ceilings follow-up threads a custom grid through every
    ridge fit (point + bootstrap + null — selection stays symmetric).
    """
    import torch

    lams = LAMBDAS if lambdas is None else np.asarray(lambdas, dtype=np.float64)
    dev = torch.device(device)
    tX_S = torch.from_numpy(np.ascontiguousarray(X_S)).to(dev)
    tY_S = torch.from_numpy(np.ascontiguousarray(Y_S)).to(dev)
    tX_T = torch.from_numpy(np.ascontiguousarray(X_T)).to(dev)
    tY_T = torch.from_numpy(np.ascontiguousarray(Y_T)).to(dev)
    t_train = torch.from_numpy(train_idx).to(dev)
    t_test = torch.from_numpy(test_idx).to(dev)
    n = len(common)

    # ---- Point estimate ----
    rung_r2s_point = _compute_ladder_r2s_t(
        tX_S, tY_S, tX_T, tY_T, t_train, t_test, train_conv_ids, lams, full_conv_ids=common
    )
    r2_within, reach_bar = _reach_bar_within_cell_t(
        tX_T, tY_T, t_train, t_test, train_conv_ids, lams, threshold=threshold
    )
    rung_reached_point = _rung_reached_from_r2s(rung_r2s_point, reach_bar)

    # ---- Selection-symmetric bootstrap ----
    rng = np.random.default_rng(seed + 1)
    rung_names = list(rung_r2s_point.keys())
    bootstrap_r2_matrix = np.full((n_bootstrap_draws, len(rung_names)), np.nan, dtype=np.float64)
    bootstrap_rung_reached = np.zeros(n_bootstrap_draws, dtype=np.int64)
    for draw_i in range(n_bootstrap_draws):
        resample_idx = rng.choice(n, size=n, replace=True)
        resample_convs = common[resample_idx]
        resample_folds = _conv_grouped_folds(resample_convs, n_folds=N_FOLDS, seed=seed + draw_i)
        rs_train = np.where(resample_folds != 0)[0]
        rs_test = np.where(resample_folds == 0)[0]
        if len(rs_train) < 3 or len(rs_test) < 1:
            continue  # NaN row
        t_resample = torch.from_numpy(resample_idx).to(dev)
        X_S_d, Y_S_d = tX_S[t_resample], tY_S[t_resample]
        X_T_d, Y_T_d = tX_T[t_resample], tY_T[t_resample]
        train_conv_ids_d = resample_convs[rs_train]
        try:
            r2s_draw = _compute_ladder_r2s_t(
                X_S_d,
                Y_S_d,
                X_T_d,
                Y_T_d,
                torch.from_numpy(rs_train).to(dev),
                torch.from_numpy(rs_test).to(dev),
                train_conv_ids_d,
                lams,
                full_conv_ids=resample_convs,
            )
        except torch.linalg.LinAlgError:
            continue  # degenerate resample; NaN row
        for j, name in enumerate(rung_names):
            bootstrap_r2_matrix[draw_i, j] = r2s_draw[name]
        _, reach_bar_d = _reach_bar_within_cell_t(
            X_T_d,
            Y_T_d,
            torch.from_numpy(rs_train).to(dev),
            torch.from_numpy(rs_test).to(dev),
            train_conv_ids_d,
            lams,
            threshold=threshold,
        )
        bootstrap_rung_reached[draw_i] = _rung_reached_from_r2s(r2s_draw, reach_bar_d)

    # ---- Matched-capacity null (shuffled-answer) ----
    null_r2_matrix = np.full((n_null_draws, len(rung_names)), np.nan, dtype=np.float64)
    null_rung_reached = np.zeros(n_null_draws, dtype=np.int64)
    for draw_i in range(n_null_draws):
        perm_s = rng.permutation(n)
        perm_t = rng.permutation(n)
        t_perm_s = torch.from_numpy(perm_s).to(dev)
        t_perm_t = torch.from_numpy(perm_t).to(dev)
        try:
            r2s_null = _compute_ladder_r2s_t(
                tX_S,
                tY_S[t_perm_s],
                tX_T,
                tY_T[t_perm_t],
                t_train,
                t_test,
                train_conv_ids,
                lams,
                full_conv_ids=common,
            )
        except torch.linalg.LinAlgError:
            continue
        for j, name in enumerate(rung_names):
            null_r2_matrix[draw_i, j] = r2s_null[name]
        _, reach_bar_null = _reach_bar_within_cell_t(
            tX_T, tY_T[t_perm_t], t_train, t_test, train_conv_ids, lams, threshold=threshold
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
            "rung_reached_median": float(np.median(bootstrap_rung_reached))
            if n_bootstrap_draws > 0
            else float("nan"),
            "rung_reached_p025": float(np.percentile(bootstrap_rung_reached, 2.5))
            if n_bootstrap_draws > 0
            else float("nan"),
            "rung_reached_p975": float(np.percentile(bootstrap_rung_reached, 97.5))
            if n_bootstrap_draws > 0
            else float("nan"),
        },
        "matched_capacity_null": {
            "n_draws": int(n_null_draws),
            "rung_reached_per_draw": null_rung_reached.tolist(),
            "rung_reached_null_p975": float(np.percentile(null_rung_reached, 97.5))
            if n_null_draws > 0
            else float("nan"),
        },
    }


def _run_ladder_pair(
    source: dict,
    target: dict,
    *,
    arm: str,
    n_bootstrap_draws: int,
    n_null_draws: int,
    threshold: float = RUNG_REACHED_THRESHOLD,
    seed: int = 42,
    engine: str = "numpy",
    device: str = "cpu",
    lambdas: np.ndarray | None = None,
) -> dict:
    """Run 9 rungs + selection-symmetric bootstrap for one (source, target, arm).

    ``lambdas=None`` (default) keeps the module ``LAMBDAS`` grid byte-for-byte;
    a custom grid (wider-lambda-ceilings follow-up) threads through BOTH
    engines' point/bootstrap/null fits identically.

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
    lams = LAMBDAS if lambdas is None else np.asarray(lambdas, dtype=np.float64)

    if engine == "torch":
        return _ladder_pair_core_torch(
            X_S,
            Y_S,
            X_T,
            Y_T,
            common,
            train_idx,
            test_idx,
            train_conv_ids,
            n_bootstrap_draws=n_bootstrap_draws,
            n_null_draws=n_null_draws,
            threshold=threshold,
            seed=seed,
            device=device,
            lambdas=lams,
        )

    # ---- Point estimate (numpy reference engine) ----
    rung_r2s_point = _compute_ladder_r2s(
        X_S,
        Y_S,
        X_T,
        Y_T,
        train_idx,
        test_idx,
        train_conv_ids,
        lams,
        full_conv_ids=common,
    )
    r2_within, reach_bar = _reach_bar_within_cell(
        X_T, Y_T, train_idx, test_idx, train_conv_ids, lams, threshold=threshold
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
                lams,
                full_conv_ids=resample_convs,
            )
        except np.linalg.LinAlgError:
            continue  # degenerate resample; NaN row
        for j, name in enumerate(rung_names):
            bootstrap_r2_matrix[draw_i, j] = r2s_draw[name]
        _, reach_bar_d = _reach_bar_within_cell(
            X_T_d, Y_T_d, rs_train, rs_test, train_conv_ids_d, lams, threshold=threshold
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
                lams,
                full_conv_ids=common,
            )
        except np.linalg.LinAlgError:
            continue
        for j, name in enumerate(rung_names):
            null_r2_matrix[draw_i, j] = r2s_null[name]
        _, reach_bar_null = _reach_bar_within_cell(
            X_T, Y_T[perm_t], train_idx, test_idx, train_conv_ids, lams, threshold=threshold
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
            "rung_reached_median": float(np.median(bootstrap_rung_reached))
            if n_bootstrap_draws > 0
            else float("nan"),
            "rung_reached_p025": float(np.percentile(bootstrap_rung_reached, 2.5))
            if n_bootstrap_draws > 0
            else float("nan"),
            "rung_reached_p975": float(np.percentile(bootstrap_rung_reached, 97.5))
            if n_bootstrap_draws > 0
            else float("nan"),
        },
        "matched_capacity_null": {
            "n_draws": int(n_null_draws),
            "rung_reached_per_draw": null_rung_reached.tolist(),
            "rung_reached_null_p975": float(np.percentile(null_rung_reached, 97.5))
            if n_null_draws > 0
            else float("nan"),
        },
    }


def _pair_ckpt_meta(
    model_slug: str,
    layer: int,
    n_bootstrap_draws: int,
    n_null_draws: int,
    lambda_grid: str = "ladder13",
) -> dict:
    """Regime key for per-pair checkpoints — every output-affecting knob.

    ``lambda_grid`` joined the key with the wider-lambda-ceilings follow-up:
    a wide19 rerun must never silently resume a ladder13 checkpoint (the
    resume-metadata rule — every output-affecting regime key is pinned).
    """
    return {
        "model": model_slug,
        "layer": int(layer),
        "n_bootstrap_draws": int(n_bootstrap_draws),
        "n_null_draws": int(n_null_draws),
        "threshold": float(RUNG_REACHED_THRESHOLD),
        "seed": 42,
        "lambda_grid": str(lambda_grid),
    }


def _ckpt_meta_satisfies(prior: dict | None, want: dict) -> bool:
    """A checkpoint satisfies the requested regime when every key matches
    EXCEPT the draw counts, where a SUPERSET (prior >= want) is acceptable:
    a 200/40-draw checkpoint carries strictly more information than a 0/40
    request, so recomputing it would only discard CIs (2026-07-28 user-chat
    descope: BOOT 200 -> 0 mid-run; the 51 completed full-CI pairs must
    RESUME, not recompute). A 10/2 smoke checkpoint still never satisfies a
    production request (10 < 200, and 10 < 40-null production floors).

    Back-compat: parent-round checkpoints predate the ``lambda_grid`` key and
    were all produced under the ladder13 module default, so a missing key
    reads as "ladder13" — a ladder13 request still resumes them; a wide19
    request never does."""
    if not isinstance(prior, dict):
        return False
    for k, v in want.items():
        if k in ("n_bootstrap_draws", "n_null_draws"):
            try:
                if int(prior.get(k, -1)) < int(v):
                    return False
            except (TypeError, ValueError):
                return False
        elif k == "lambda_grid":
            if prior.get(k, "ladder13") != v:
                return False
        elif prior.get(k) != v:
            return False
    return True


def run_all_pairs(
    store_root: Path,
    *,
    model_slug: str,
    layer: int = HEADLINE_LAYER,
    smoke: bool = False,
    n_bootstrap_draws: int = N_BOOTSTRAP_DRAWS,
    n_null_draws: int = N_REPARAM_NULL_DRAWS,
    engine: str = "numpy",
    device: str = "cpu",
    num_shards: int = 1,
    shard_index: int = 0,
    checkpoint_dir: Path | None = None,
    lambda_grid: str = "ladder13",
    pairs_subset: list[tuple[str, str]] | None = None,
) -> dict:
    """Run the pair ladder for one (model, layer). Both arms per pair.

    R16: pairs are independent, so the loop shards (`pairs[shard_index::
    num_shards]`) for multi-worker parallelism, and every completed pair is
    persisted to `checkpoint_dir/<src>__<tgt>.json` the moment it finishes
    (code-style.md checkpoint-per-unit: 126 pairs x 2 arms >> the ~50-unit
    trigger; the pre-R16 loop accumulated everything in memory with one
    terminal write). A resume run skips pairs whose checkpoint matches the
    regime meta (draws/layer/model/threshold/seed/lambda_grid).

    wider-lambda-ceilings knobs: ``lambda_grid`` names the ridge grid
    (issue1689_common.LAMBDA_GRIDS; "ladder13" = the module default,
    byte-identical); ``pairs_subset`` restricts to a validated subset of
    ``enumerate_pair_set()`` (the Stage-2 affected-pairs re-read).
    """
    pairs = enumerate_pair_set()
    if pairs_subset is not None:
        known = set(pairs)
        bad = [p for p in pairs_subset if tuple(p) not in known]
        if bad:
            raise ValueError(f"pairs_subset contains {len(bad)} unknown pairs (first: {bad[:3]})")
        pairs = sorted({tuple(p) for p in pairs_subset})
    lams = LAMBDAS if lambda_grid == "ladder13" else resolve_lambda_grid(lambda_grid)
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

    shard_pairs = pairs[shard_index::num_shards] if num_shards > 1 else pairs

    out: dict[str, Any] = {
        "model": model_slug,
        "layer": layer,
        "n_pairs": len(pairs),
        "arms": ["prefix", "context"],
        "n_bootstrap_draws": n_bootstrap_draws,
        "n_null_draws": n_null_draws,
        "lambda_grid": lambda_grid,
        "pairs": {},
    }
    meta = _pair_ckpt_meta(model_slug, layer, n_bootstrap_draws, n_null_draws, lambda_grid)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    n_shard = len(shard_pairs)
    for i, (src, tgt) in enumerate(shard_pairs):
        pair_key = f"{src}__{tgt}"
        ckpt_path = checkpoint_dir / f"{pair_key}.json" if checkpoint_dir is not None else None

        # Resume predicate — keyed on the regime meta with a draw-count
        # SUPERSET accepted (a 10/2-draw smoke checkpoint never satisfies a
        # 200/40 production run, but a 200/40 checkpoint satisfies a 0/40
        # descope re-run — _ckpt_meta_satisfies).
        if ckpt_path is not None and ckpt_path.exists():
            try:
                prior = json.loads(ckpt_path.read_text())
            except (json.JSONDecodeError, OSError):
                prior = None
            if prior is not None and _ckpt_meta_satisfies(prior.get("meta"), meta):
                out["pairs"][pair_key] = prior["arms"]
                print(
                    f"[fit_ladder]   pair {i + 1}/{n_shard}: {src} -> {tgt} RESUME (checkpoint)",
                    flush=True,
                )
                continue
            if prior is not None:
                print(
                    f"[fit_ladder]   pair {i + 1}/{n_shard}: {src} -> {tgt} "
                    f"regime-mismatch checkpoint -> recompute",
                    flush=True,
                )

        # Flushed per-pair progress print — bypasses libc's 4KB block buffer
        # so a long serial ladder loop is observable instead of appearing hung
        # (R14 fix, issue #1689 crash-fix round: pre-fix stdout was buffered,
        # so per-layer prints only surfaced at process exit).
        print(
            f"[fit_ladder]   pair {i + 1}/{n_shard}: {src} -> {tgt}",
            flush=True,
        )
        pair_t0 = time.perf_counter()
        src_cell = f"{model_slug}/{src}"
        tgt_cell = f"{model_slug}/{tgt}"
        source = _load_cell_layer(store_root, src_cell, layer)
        target = _load_cell_layer(store_root, tgt_cell, layer)
        out["pairs"][pair_key] = {}
        for arm in ["prefix", "context"]:
            out["pairs"][pair_key][arm] = _run_ladder_pair(
                source,
                target,
                arm=arm,
                n_bootstrap_draws=n_bootstrap_draws,
                n_null_draws=n_null_draws,
                engine=engine,
                device=device,
                lambdas=lams,
            )
        pair_dt = time.perf_counter() - pair_t0
        print(
            f"[fit_ladder]   pair {i + 1}/{n_shard}: {src} -> {tgt} done elapsed={pair_dt:.1f}s",
            flush=True,
        )
        if ckpt_path is not None:
            # Atomic same-dir tmp + replace (EXDEV rule: tmp INSIDE dest dir).
            tmp_path = ckpt_path.with_name(f".{ckpt_path.name}.tmp")
            with tmp_path.open("w") as fh:
                json.dump({"meta": meta, "arms": out["pairs"][pair_key]}, fh)
            tmp_path.replace(ckpt_path)
    return out


def merge_pair_checkpoints(
    checkpoint_dir: Path,
    *,
    model_slug: str,
    layer: int,
    n_bootstrap_draws: int,
    n_null_draws: int,
    lambda_grid: str = "ladder13",
    pairs_subset: list[tuple[str, str]] | None = None,
) -> dict:
    """Assemble the final ladder JSON (same schema as run_all_pairs) from
    per-pair checkpoints. Fails loud on missing pairs or regime mismatches.
    ``lambda_grid`` / ``pairs_subset`` mirror run_all_pairs (a wide19 merge
    only accepts wide19-regime checkpoints; a subset merge only requires the
    subset's pairs)."""
    pairs = enumerate_pair_set()
    if pairs_subset is not None:
        known = set(pairs)
        bad = [p for p in pairs_subset if tuple(p) not in known]
        if bad:
            raise ValueError(f"pairs_subset contains {len(bad)} unknown pairs (first: {bad[:3]})")
        pairs = sorted({tuple(p) for p in pairs_subset})
    meta = _pair_ckpt_meta(model_slug, layer, n_bootstrap_draws, n_null_draws, lambda_grid)
    out: dict[str, Any] = {
        "model": model_slug,
        "layer": layer,
        "n_pairs": len(pairs),
        "arms": ["prefix", "context"],
        "n_bootstrap_draws": n_bootstrap_draws,
        "n_null_draws": n_null_draws,
        "lambda_grid": lambda_grid,
        "pairs": {},
    }
    missing: list[str] = []
    mismatched: list[str] = []
    for src, tgt in pairs:
        pair_key = f"{src}__{tgt}"
        p = checkpoint_dir / f"{pair_key}.json"
        if not p.exists():
            missing.append(pair_key)
            continue
        d = json.loads(p.read_text())
        if not _ckpt_meta_satisfies(d.get("meta"), meta):
            mismatched.append(pair_key)
            continue
        out["pairs"][pair_key] = d["arms"]
    if missing or mismatched:
        raise RuntimeError(
            f"merge incomplete: {len(missing)} missing, {len(mismatched)} regime-mismatched "
            f"of {len(pairs)} pairs under {checkpoint_dir} "
            f"(first missing: {missing[:3]}; first mismatched: {mismatched[:3]})"
        )
    return out


# ---------------------------------------------------------------------------
# Cross-model generalized pair addressing (derived-vs-free-answer-map round).
#
# EXTENSION CONTRACT (consistency-checker WARN discharge): everything below is
# ADDITIVE — new functions + a new main() branch gated on the pairs-file
# schema. The within-model path (enumerate_pair_set -> run_all_pairs ->
# _run_ladder_pair) is byte-untouched; the generalized runner routes every
# pair through the SAME _run_ladder_pair, so the fit math is shared, not
# forked (pinned by tests/test_issue1689_derived_vs_free.py
# ::test_generalized_runner_matches_run_all_pairs_within_model).
# ---------------------------------------------------------------------------

PairSpec = tuple[tuple[str, str], tuple[str, str]]  # ((src_model, src_cond), (tgt_model, tgt_cond))


def pair_rows_by_conv(
    conv_S: np.ndarray, conv_T: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standalone copy of _run_ladder_pair's inline conv-id row pairing.

    Returns (common, s_idx, t_idx): sorted shared conv ids + the FIRST-occurrence
    row index per conv on each side (identical semantics to the inline code —
    _run_ladder_pair itself is deliberately left untouched for byte-identity).
    """
    common = np.intersect1d(conv_S, conv_T)
    s_idx = np.array([np.where(conv_S == c)[0][0] for c in common])
    t_idx = np.array([np.where(conv_T == c)[0][0] for c in common])
    return common, s_idx, t_idx


def parse_pair_specs(entries: list, default_model: str | None = None) -> list[PairSpec]:
    """Parse pairs-file entries into ((src_model, src_cond), (tgt_model, tgt_cond)) specs.

    Two accepted element shapes (mixable in one file):
      ["src_cond", "tgt_cond"]              -> within-model (requires default_model)
      [["m1", "cond_a"], ["m2", "cond_b"]]  -> cross-model (explicit model per side)
    """
    specs: list[PairSpec] = []
    for e in entries:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            raise ValueError(f"pair entry must have exactly 2 elements: {e!r}")
        a, b = e
        if isinstance(a, str) and isinstance(b, str):
            if default_model is None:
                raise ValueError(f"string pair {e!r} needs a default model (--model-slug)")
            specs.append(((default_model, a), (default_model, b)))
        elif (
            isinstance(a, (list, tuple))
            and len(a) == 2
            and isinstance(b, (list, tuple))
            and len(b) == 2
        ):
            specs.append(((str(a[0]), str(a[1])), (str(b[0]), str(b[1]))))
        else:
            raise ValueError(f"unrecognized pair entry shape: {e!r}")
    return specs


def pairs_file_is_generalized(loaded) -> bool:
    """True when a flat pairs-file list carries >=1 nested [[m,c],[m,c]] entry."""
    return isinstance(loaded, list) and any(
        isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], (list, tuple))
        for e in loaded
    )


def pair_spec_key(spec: PairSpec) -> str:
    """Checkpoint key: parent-style 'src__tgt' within-model; 'm@c__m@c' cross-model."""
    (sm, sc), (tm, tc) = spec
    if sm == tm:
        return f"{sc}__{tc}"
    return f"{sm}@{sc}__{tm}@{tc}"


def crossmodel_pair_specs(model_a: str, model_b: str) -> list[PairSpec]:
    """The 21 same-condition cross-model ordered pairs x 2 directions (scope item 9)."""
    from scripts.issue1689_common import CONDITION_TABLE

    conds = sorted({c.slug for c in CONDITION_TABLE})
    specs: list[PairSpec] = []
    for cond in conds:
        specs.append(((model_a, cond), (model_b, cond)))
        specs.append(((model_b, cond), (model_a, cond)))
    return specs


def _pair_ckpt_meta_generalized(
    spec: PairSpec,
    layer: int,
    n_bootstrap_draws: int,
    n_null_draws: int,
    lambda_grid: str,
) -> dict:
    """Generalized regime key: the within-model meta + per-side model slugs."""
    (sm, _sc), (tm, _tc) = spec
    return {
        "src_model": sm,
        "tgt_model": tm,
        "layer": int(layer),
        "n_bootstrap_draws": int(n_bootstrap_draws),
        "n_null_draws": int(n_null_draws),
        "threshold": float(RUNG_REACHED_THRESHOLD),
        "seed": 42,
        "lambda_grid": str(lambda_grid),
    }


def run_pairs_generalized(
    store_root: Path,
    pair_specs: list[PairSpec],
    *,
    layer: int = HEADLINE_LAYER,
    n_bootstrap_draws: int,
    n_null_draws: int,
    engine: str = "numpy",
    device: str = "cpu",
    num_shards: int = 1,
    shard_index: int = 0,
    checkpoint_dir: Path | None = None,
    lambda_grid: str = "ladder13",
) -> dict:
    """Cross-model-capable sibling of run_all_pairs (same per-pair math).

    Cells address as (model_slug, condition) per SIDE; every pair runs through
    the SAME _run_ladder_pair as the parent path (fit math shared). Per-pair
    checkpoints + regime-keyed resume mirror run_all_pairs.
    """
    lams = LAMBDAS if lambda_grid == "ladder13" else resolve_lambda_grid(lambda_grid)
    shard_specs = pair_specs[shard_index::num_shards] if num_shards > 1 else list(pair_specs)
    out: dict[str, Any] = {
        "layer": layer,
        "n_pairs": len(pair_specs),
        "arms": ["prefix", "context"],
        "n_bootstrap_draws": n_bootstrap_draws,
        "n_null_draws": n_null_draws,
        "lambda_grid": lambda_grid,
        "generalized": True,
        "pairs": {},
    }
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    n_shard = len(shard_specs)
    for i, spec in enumerate(shard_specs):
        (sm, sc), (tm, tc) = spec
        pair_key = pair_spec_key(spec)
        meta = _pair_ckpt_meta_generalized(
            spec, layer, n_bootstrap_draws, n_null_draws, lambda_grid
        )
        ckpt_path = checkpoint_dir / f"{pair_key}.json" if checkpoint_dir is not None else None
        if ckpt_path is not None and ckpt_path.exists():
            try:
                prior = json.loads(ckpt_path.read_text())
            except (json.JSONDecodeError, OSError):
                prior = None
            if prior is not None and _ckpt_meta_satisfies(prior.get("meta"), meta):
                out["pairs"][pair_key] = prior["arms"]
                print(
                    f"[fit_ladder]   xpair {i + 1}/{n_shard}: {pair_key} RESUME (checkpoint)",
                    flush=True,
                )
                continue
            if prior is not None:
                print(
                    f"[fit_ladder]   xpair {i + 1}/{n_shard}: {pair_key} "
                    f"regime-mismatch checkpoint -> recompute",
                    flush=True,
                )
        print(f"[fit_ladder]   xpair {i + 1}/{n_shard}: {pair_key}", flush=True)
        pair_t0 = time.perf_counter()
        source = _load_cell_layer(store_root, f"{sm}/{sc}", layer)
        target = _load_cell_layer(store_root, f"{tm}/{tc}", layer)
        arms_out: dict[str, Any] = {}
        for arm in ["prefix", "context"]:
            arms_out[arm] = _run_ladder_pair(
                source,
                target,
                arm=arm,
                n_bootstrap_draws=n_bootstrap_draws,
                n_null_draws=n_null_draws,
                engine=engine,
                device=device,
                lambdas=lams,
            )
        out["pairs"][pair_key] = arms_out
        pair_dt = time.perf_counter() - pair_t0
        print(
            f"[fit_ladder]   xpair {i + 1}/{n_shard}: {pair_key} done elapsed={pair_dt:.1f}s",
            flush=True,
        )
        if ckpt_path is not None:
            tmp_path = ckpt_path.with_name(f".{ckpt_path.name}.tmp")
            with tmp_path.open("w") as fh:
                json.dump({"meta": meta, "arms": arms_out}, fh)
            tmp_path.replace(ckpt_path)
    return out


def merge_pairs_generalized(
    checkpoint_dir: Path,
    pair_specs: list[PairSpec],
    *,
    layer: int,
    n_bootstrap_draws: int,
    n_null_draws: int,
    lambda_grid: str = "ladder13",
) -> dict:
    """Fail-loud merge of generalized per-pair checkpoints (run_pairs_generalized)."""
    out: dict[str, Any] = {
        "layer": layer,
        "n_pairs": len(pair_specs),
        "arms": ["prefix", "context"],
        "n_bootstrap_draws": n_bootstrap_draws,
        "n_null_draws": n_null_draws,
        "lambda_grid": lambda_grid,
        "generalized": True,
        "pairs": {},
    }
    missing: list[str] = []
    mismatched: list[str] = []
    for spec in pair_specs:
        pair_key = pair_spec_key(spec)
        meta = _pair_ckpt_meta_generalized(
            spec, layer, n_bootstrap_draws, n_null_draws, lambda_grid
        )
        p = checkpoint_dir / f"{pair_key}.json"
        if not p.exists():
            missing.append(pair_key)
            continue
        d = json.loads(p.read_text())
        if not _ckpt_meta_satisfies(d.get("meta"), meta):
            mismatched.append(pair_key)
            continue
        out["pairs"][pair_key] = d["arms"]
    if missing or mismatched:
        raise RuntimeError(
            f"generalized merge incomplete: {len(missing)} missing, "
            f"{len(mismatched)} regime-mismatched of {len(pair_specs)} pairs under "
            f"{checkpoint_dir} (first missing: {missing[:3]}; first mismatched: {mismatched[:3]})"
        )
    return out


def fit_rung78_corrections_t(
    source: dict,
    target: dict,
    *,
    arm: str,
    seed: int = 42,
    device: str = "cpu",
    lambdas: np.ndarray | None = None,
    row_limit: int | None = None,
    dim_limit: int | None = None,
) -> dict:
    """Refit + EXPOSE the parent ladder's rung-7/8 correction operators (item 8).

    Parent conventions preserved EXACTLY (this is the ladder's own math, exposed
    for in-process consumption instead of being discarded inside
    _compute_ladder_r2s_t): row-pairing by conv-id intersection, conv-grouped
    5-fold split (seed 42; train = folds != 0, test = fold 0 — the ladder's
    point split), W_s fit on ALL common rows (the ladder's registered all-rows
    convention), A (rung 7): x_T -> x_S on train rows, B (rung 8): y_S -> y_T
    on train rows; every fit inner-group-cv on the given lambda grid.

    ``row_limit`` / ``dim_limit`` are smoke-scale knobs (slice common convs /
    leading dims); production leaves both None. Returns numpy operators + the
    paired arrays + split indices so the rank-k sweep evaluates on the same
    held-out rows.
    """
    import torch

    lams = LAMBDAS if lambdas is None else np.asarray(lambdas, dtype=np.float64)
    conv_S = source["conv_ids"]
    conv_T = target["conv_ids"]
    common, s_idx, t_idx = pair_rows_by_conv(conv_S, conv_T)
    if row_limit is not None:
        common = common[:row_limit]
        s_idx = s_idx[:row_limit]
        t_idx = t_idx[:row_limit]
    if len(common) < 3:
        return {"error": "insufficient shared conv_ids", "n_common": int(len(common))}
    dsl = slice(None) if dim_limit is None else slice(0, dim_limit)
    X_S = source[f"X_{arm}"][s_idx][:, dsl]
    Y_S = source["Y"][s_idx][:, dsl]
    X_T = target[f"X_{arm}"][t_idx][:, dsl]
    Y_T = target["Y"][t_idx][:, dsl]

    n = len(common)
    folds = _conv_grouped_folds(common, n_folds=N_FOLDS, seed=seed)
    train_idx = np.where(folds != 0)[0]
    test_idx = np.where(folds == 0)[0]
    if len(train_idx) < 3 or len(test_idx) < 1:
        # train==test is a leakage-by-construction fallback tolerable ONLY in
        # the explicit smoke regime (row_limit/dim_limit are the documented
        # smoke-scale knobs; production passes both as None). In production
        # shape a degenerate conv-grouped split means the pair's data is
        # broken — fail loud instead of silently scoring on the train fold.
        if row_limit is None and dim_limit is None:
            raise RuntimeError(
                f"degenerate conv-grouped split in production shape (arm={arm}: "
                f"train={len(train_idx)}, test={len(test_idx)}, n_common={n}) — "
                "refusing the train==test smoke fallback"
            )
        train_idx = np.arange(n)
        test_idx = np.arange(n)
    train_conv_ids = common[train_idx]

    dev = torch.device(device)
    tX_S = torch.from_numpy(np.ascontiguousarray(X_S)).to(dev)
    tY_S = torch.from_numpy(np.ascontiguousarray(Y_S)).to(dev)
    tX_T = torch.from_numpy(np.ascontiguousarray(X_T)).to(dev)
    tY_T = torch.from_numpy(np.ascontiguousarray(Y_T)).to(dev)
    t_train = torch.from_numpy(train_idx).to(dev)

    # W_s: ALL common rows (parent ladder convention — Gate-1 parity anchors it).
    W_s, b_s, lam_ws = _fit_ridge_inner_group_cv_t(tX_S, tY_S, common, lams)
    # Rung-7 correction A: x_T -> x_S on the train fold.
    A_W, A_b, lam_a = _fit_ridge_inner_group_cv_t(
        tX_T[t_train], tX_S[t_train], train_conv_ids, lams
    )
    # Rung-8 correction B: y_S -> y_T on the train fold.
    B_W, B_b, lam_b = _fit_ridge_inner_group_cv_t(
        tY_S[t_train], tY_T[t_train], train_conv_ids, lams
    )
    return {
        "common": common,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_S": X_S,
        "Y_S": Y_S,
        "X_T": X_T,
        "Y_T": Y_T,
        "W_s": W_s.cpu().numpy(),
        "b_s": b_s.cpu().numpy(),
        "A_W": A_W.cpu().numpy(),
        "A_b": A_b.cpu().numpy(),
        "B_W": B_W.cpu().numpy(),
        "B_b": B_b.cpu().numpy(),
        "lambdas_chosen": {"W_s": lam_ws, "A": lam_a, "B": lam_b},
        "n_common": int(n),
    }


# ---------------------------------------------------------------------------
# Target-varying inner-group-cv fits with a CACHED X-side eigendecomposition.
#
# Matched-capacity null draws permute only the TARGET rows, so the Gram
# eigendecomposition (a function of X alone) is identical across draws —
# recomputing it per draw is the #823 "redundant recompute of shareable
# work" class. These helpers cache the X-side preps ONCE and rerun the
# EXACT _fit_ridge_inner_group_cv_t math per draw (same eigh, same fold
# assignment, same lambda argmax tie-breaking; predictions are linear in
# the centered target). Bit-equivalence is pinned by
# tests/test_issue1689_derived_vs_free.py::test_prepped_inner_cv_matches_plain.
# ---------------------------------------------------------------------------


def _prep_x_t(X) -> dict:
    """X-side half of _ridge_eigh_prep_t (no target): mean, centered X, eigh."""
    n, d = X.shape
    x_mean = X.mean(dim=0)
    Xc = X - x_mean
    dual = n < d
    S = Xc @ Xc.T if dual else Xc.T @ Xc
    S = 0.5 * (S + S.T)
    sigma2, U = _eigh_robust_t(S)
    sigma2 = sigma2.clamp_min(0.0)
    return {"x_mean": x_mean, "Xc": Xc, "dual": dual, "U": U, "sigma2": sigma2}


def _predict_prepped_t(prepx: dict, Y_tr, X_test, lam: float):
    """_ridge_predict_from_prep_t with the target supplied per call."""
    y_mean = Y_tr.mean(dim=0)
    Yc = Y_tr - y_mean
    inv_diag = 1.0 / (prepx["sigma2"] + lam)
    Xtest_c = X_test - prepx["x_mean"]
    U, Xc = prepx["U"], prepx["Xc"]
    if prepx["dual"]:
        UtY = U.T @ Yc
        alpha = U @ (inv_diag[:, None] * UtY)
        pred = Xtest_c @ (Xc.T @ alpha)
    else:
        XtY = Xc.T @ Yc
        UtXtY = U.T @ XtY
        W = U @ (inv_diag[:, None] * UtXtY)
        pred = Xtest_c @ W
    return pred + y_mean


def _fit_prepped_t(prepx: dict, Y_tr, lam: float):
    """_ridge_fit_from_prep_t with the target supplied per call -> (W, b)."""
    y_mean = Y_tr.mean(dim=0)
    Yc = Y_tr - y_mean
    inv_diag = 1.0 / (prepx["sigma2"] + lam)
    U, Xc = prepx["U"], prepx["Xc"]
    if prepx["dual"]:
        UtY = U.T @ Yc
        alpha = U @ (inv_diag[:, None] * UtY)
        W = Xc.T @ alpha
    else:
        XtY = Xc.T @ Yc
        UtXtY = U.T @ XtY
        W = U @ (inv_diag[:, None] * UtXtY)
    b = y_mean - prepx["x_mean"] @ W
    return W, b


def build_inner_cv_cache_t(
    X_train,
    train_conv_ids: np.ndarray,
    n_inner_folds: int = 3,
    seed: int = 42,
) -> dict:
    """Precompute the fold masks + X-side eigh preps _fit_ridge_inner_group_cv_t
    would build, for reuse across target-varying (null-draw) fits."""
    import torch

    dev = X_train.device
    inner_folds = _conv_grouped_folds(train_conv_ids, n_inner_folds, seed=seed)
    folds = []
    for fold_i in range(n_inner_folds):
        te_mask = inner_folds == fold_i
        tr_mask = ~te_mask
        if tr_mask.sum() < 3 or te_mask.sum() == 0:
            folds.append(None)
            continue
        tr_idx = torch.from_numpy(np.where(tr_mask)[0]).to(dev)
        te_idx = torch.from_numpy(np.where(te_mask)[0]).to(dev)
        folds.append({"tr": tr_idx, "te": te_idx, "prep": _prep_x_t(X_train[tr_idx])})
    return {
        "folds": folds,
        "full": _prep_x_t(X_train),
        "X": X_train,  # kept by reference for bit-exact X_te slicing
        "n_inner_folds": n_inner_folds,
    }


def fit_inner_group_cv_cached_t(cache: dict, Y_train, lambdas: np.ndarray):
    """_fit_ridge_inner_group_cv_t math on a cached X-side prep (target varies).

    Same scores matrix, same nanmean, same argmax tie-breaking, same refit —
    only the (X-only) eigendecompositions are reused across calls.
    """
    n_inner = cache["n_inner_folds"]
    scores = np.zeros((len(lambdas), n_inner), dtype=np.float64)
    for fold_i, fold in enumerate(cache["folds"]):
        if fold is None:
            scores[:, fold_i] = np.nan
            continue
        Y_tr = Y_train[fold["tr"]]
        Y_te = Y_train[fold["te"]]
        X_te = cache["X"][fold["te"]]
        for li, lam in enumerate(lambdas):
            pred = _predict_prepped_t(fold["prep"], Y_tr, X_te, lam=float(lam))
            scores[li, fold_i] = _r2_t(Y_te, pred)
    mean_scores = np.nanmean(scores, axis=1)
    valid = ~np.isnan(mean_scores)
    if not valid.any():
        best_lam = float(lambdas[len(lambdas) // 2])
    else:
        best_idx = int(np.argmax(np.where(valid, mean_scores, -np.inf)))
        best_lam = float(lambdas[best_idx])
    W, b = _fit_prepped_t(cache["full"], Y_train, best_lam)
    return W, b, best_lam


def _compare_pair_results(ref: dict, new: dict, atol: float = 1e-4) -> tuple[bool, list[str]]:
    """Compare numpy-engine vs torch-engine results for one (pair, arm).

    Returns (ok, messages). Scalars/matrices compare with atol (nan-equal);
    the point rung_reached must match exactly; per-draw rung_reached flips
    are tolerated up to max(2%, 2 draws) (draws sitting float-epsilon from
    the reach bar can legitimately flip across fp64 backends).

    atol CALIBRATION (measured, 2026-07-28 gate on pod-1689, real pair
    assistant_chat->assistant_naturalistic L19): worst cross-backend
    (numpy LAPACK vs cuSOLVER fp64) R^2 deviation was 4.476e-6 on the
    bootstrap matrix and 1.5e-6 on the point estimate, concentrated in the
    SVD-bearing rung 6 (Procrustes rotation — singular vectors of
    near-tied singular values rotate freely across backends). Bar = 1e-4:
    ~22x the measured worst (gotchas.md gate-calibration rule: >=4x
    headroom on the noisiest quantity class), still 3+ orders of magnitude
    below the real-bug regime (R^2 differences of ~0.1-1).
    """
    msgs: list[str] = []
    ok = True

    def _close(a, b) -> bool:
        return bool(np.all(np.isclose(np.asarray(a), np.asarray(b), atol=atol, equal_nan=True)))

    for k in ("r2_within_target", "reach_bar_90pct"):
        if not _close(ref[k], new[k]):
            ok = False
            msgs.append(f"{k}: {ref[k]} vs {new[k]}")
    for k, v in ref["rung_r2s_point"].items():
        if not _close(v, new["rung_r2s_point"][k]):
            ok = False
            msgs.append(f"point {k}: {v} vs {new['rung_r2s_point'][k]}")
    if int(ref["rung_reached_point"]) != int(new["rung_reached_point"]):
        ok = False
        msgs.append(
            f"rung_reached_point: {ref['rung_reached_point']} vs {new['rung_reached_point']}"
        )
    ref_m = np.asarray(ref["bootstrap_draws"]["r2_matrix"], dtype=np.float64)
    new_m = np.asarray(new["bootstrap_draws"]["r2_matrix"], dtype=np.float64)
    if not _close(ref_m, new_m):
        diff = np.nanmax(np.abs(ref_m - new_m))
        ok = False
        msgs.append(f"bootstrap r2_matrix max |diff| = {diff:.3e} > atol {atol}")
    for section in ("bootstrap_draws", "matched_capacity_null"):
        rr_ref = np.asarray(ref[section]["rung_reached_per_draw"], dtype=np.int64)
        rr_new = np.asarray(new[section]["rung_reached_per_draw"], dtype=np.int64)
        n_flip = int((rr_ref != rr_new).sum())
        if n_flip:
            frac = n_flip / max(1, len(rr_ref))
            msg = f"{section} rung_reached flips: {n_flip}/{len(rr_ref)} ({frac:.1%})"
            # max(2%, 2 draws): at smoke draw counts a single epsilon-adjacent
            # flip would otherwise dominate the fraction (1/10 = 10%).
            if frac > 0.02 and n_flip > 2:
                ok = False
                msgs.append(msg)
            else:
                msgs.append(f"WARN (tolerated) {msg}")
    return ok, msgs


def _verify_equivalence(args) -> int:
    """Run BOTH engines on real pairs and compare — the vectorize-rule
    equivalence gate, dispatched through the same _run_ladder_pair the
    production path calls (no hollow-gate sibling helper)."""
    pairs = enumerate_pair_set()
    avail: list[tuple[str, str]] = []
    for src, tgt in pairs:
        if (args.store_root / f"{args.model_slug}/{src}" / f"L{args.layer}.pt").exists() and (
            args.store_root / f"{args.model_slug}/{tgt}" / f"L{args.layer}.pt"
        ).exists():
            avail.append((src, tgt))
        if len(avail) >= args.verify_equivalence:
            break
    if not avail:
        print("[fit_ladder] verify: no pairs with captured stores found", flush=True)
        return 1
    all_ok = True
    for src, tgt in avail:
        source = _load_cell_layer(args.store_root, f"{args.model_slug}/{src}", args.layer)
        target = _load_cell_layer(args.store_root, f"{args.model_slug}/{tgt}", args.layer)
        for arm in ["prefix", "context"]:
            t0 = time.perf_counter()
            ref = _run_ladder_pair(
                source,
                target,
                arm=arm,
                n_bootstrap_draws=args.bootstrap_draws,
                n_null_draws=args.null_draws,
                engine="numpy",
            )
            t_np = time.perf_counter() - t0
            t0 = time.perf_counter()
            new = _run_ladder_pair(
                source,
                target,
                arm=arm,
                n_bootstrap_draws=args.bootstrap_draws,
                n_null_draws=args.null_draws,
                engine="torch",
                device=args.device,
            )
            t_torch = time.perf_counter() - t0
            if "error" in ref or "error" in new:
                match = ref == new
                all_ok &= match
                print(
                    f"[fit_ladder] verify {src}->{tgt} [{arm}]: error-result match={match}",
                    flush=True,
                )
                continue
            ok, msgs = _compare_pair_results(ref, new)
            all_ok &= ok
            n_evals = 1 + args.bootstrap_draws + args.null_draws
            print(
                f"[fit_ladder] verify {src}->{tgt} [{arm}]: {'PASS' if ok else 'FAIL'} | "
                f"numpy={t_np:.1f}s torch({args.device})={t_torch:.1f}s "
                f"speedup={t_np / max(t_torch, 1e-9):.1f}x | "
                f"per-eval torch={t_torch / n_evals:.2f}s ({n_evals} evals)",
                flush=True,
            )
            for m in msgs:
                print(f"[fit_ladder]   verify-detail: {m}", flush=True)
    print(f"[fit_ladder] verify-equivalence VERDICT: {'PASS' if all_ok else 'FAIL'}", flush=True)
    return 0 if all_ok else 1


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
    ap.add_argument(
        "--engine",
        choices=["numpy", "torch"],
        default="numpy",
        help="Dense-math engine. numpy = pre-R16 reference path (byte-identical); "
        "torch = fp64 port (GPU-capable), equivalence-gated via --verify-equivalence.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="torch device for --engine torch (e.g. cuda; pin the GPU via CUDA_VISIBLE_DEVICES).",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Shard the 126 pairs across N workers (pairs[shard::N]); workers write per-pair "
        "checkpoints only — assemble with --merge.",
    )
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Per-pair checkpoint dir (default: <out_parent>/pairs_<model>_L<layer>).",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="Assemble the final ladder JSON from per-pair checkpoints (run after all shards).",
    )
    ap.add_argument(
        "--verify-equivalence",
        type=int,
        default=0,
        metavar="N_PAIRS",
        help="Run N pairs through BOTH engines at the given draw counts, compare, exit 0/1.",
    )
    ap.add_argument(
        "--lambda-grid",
        choices=sorted(LAMBDA_GRIDS),
        default="ladder13",
        help="ridge lambda grid (issue1689_common.LAMBDA_GRIDS); ladder13 = the committed "
        "module default (byte-identical), wide19 = logspace(-2,7,19) superset "
        "(wider-lambda-ceilings Stage 2).",
    )
    ap.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help="JSON restricting the pair set: either a flat [[src,tgt],...] list or the "
        "wider-lambda-ceilings affected_pairs.json shape "
        "({model_slug: {arm: [[src,tgt],...]}} — union over arms for --model-slug).",
    )
    args = ap.parse_args()

    # Smoke defaults: keep bootstrap loop small so the code path executes without hours of compute.
    if args.bootstrap_draws is None:
        args.bootstrap_draws = 10 if args.smoke else N_BOOTSTRAP_DRAWS
    if args.null_draws is None:
        args.null_draws = 5 if args.smoke else N_REPARAM_NULL_DRAWS

    if args.engine == "torch" and args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--engine torch --device cuda but torch.cuda.is_available() False")

    # flush=True on every progress print: pod stdout is block-buffered by
    # default (no PYTHONUNBUFFERED=1 in the R11-R13 launcher), so unflushed
    # per-layer prints stayed in the 4KB libc buffer for 8+ hours (R14 fix).
    print(
        f"[fit_ladder] model={args.model_slug} smoke={args.smoke} engine={args.engine} "
        f"device={args.device} shard={args.shard_index}/{args.num_shards}",
        flush=True,
    )
    print(
        f"[fit_ladder] bootstrap_draws={args.bootstrap_draws} null_draws={args.null_draws} "
        f"(N_BOOTSTRAP_DRAWS={N_BOOTSTRAP_DRAWS}, N_REPARAM_NULL_DRAWS={N_REPARAM_NULL_DRAWS})",
        flush=True,
    )

    if args.verify_equivalence:
        return _verify_equivalence(args)

    pairs_subset = None
    generalized_specs = None
    if args.pairs_file is not None:
        loaded = json.loads(args.pairs_file.read_text())
        if pairs_file_is_generalized(loaded):
            # Cross-model pair specs ([[model, cond], [model, cond]] entries):
            # route to the ADDITIVE generalized runner; the within-model path
            # below stays byte-untouched.
            generalized_specs = parse_pair_specs(loaded, default_model=args.model_slug)
            if not generalized_specs:
                raise ValueError(f"--pairs-file {args.pairs_file} resolves to an EMPTY pair set")
            print(
                f"[fit_ladder] GENERALIZED pair specs: {len(generalized_specs)} via --pairs-file",
                flush=True,
            )
        elif isinstance(loaded, dict):
            # affected_pairs.json shape: {model_slug: {arm: [[src, tgt], ...]}}
            if args.model_slug not in loaded:
                raise ValueError(
                    f"--pairs-file {args.pairs_file} has no entry for model "
                    f"{args.model_slug!r} (keys: {sorted(loaded)})"
                )
            pairs_subset = sorted(
                {tuple(p) for arm_pairs in loaded[args.model_slug].values() for p in arm_pairs}
            )
        else:
            pairs_subset = [tuple(p) for p in loaded]
        if generalized_specs is None:
            if not pairs_subset:
                raise ValueError(f"--pairs-file {args.pairs_file} resolves to an EMPTY pair set")
            print(
                f"[fit_ladder] pairs restricted to {len(pairs_subset)} via --pairs-file", flush=True
            )

    if generalized_specs is not None:
        if args.all_layers:
            raise ValueError("--all-layers is not supported with generalized (cross-model) pairs")
        grid_suffix = "" if args.lambda_grid == "ladder13" else f"_{args.lambda_grid}"
        ckpt_dir = args.checkpoint_dir or args.out.parent / f"xpairs_L{args.layer}{grid_suffix}"
        if args.merge:
            results = merge_pairs_generalized(
                ckpt_dir,
                generalized_specs,
                layer=args.layer,
                n_bootstrap_draws=args.bootstrap_draws,
                n_null_draws=args.null_draws,
                lambda_grid=args.lambda_grid,
            )
        else:
            results = run_pairs_generalized(
                args.store_root,
                generalized_specs,
                layer=args.layer,
                n_bootstrap_draws=args.bootstrap_draws,
                n_null_draws=args.null_draws,
                engine=args.engine,
                device=args.device,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
                checkpoint_dir=ckpt_dir,
                lambda_grid=args.lambda_grid,
            )
            if args.num_shards > 1:
                print("[fit_ladder] shard done (generalized); assemble with --merge", flush=True)
                return 0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[fit_ladder] wrote {args.out} ({results['n_pairs']} generalized pairs)", flush=True)
        return 0

    layers = list(CAPTURE_LAYERS) if args.all_layers else [args.layer]
    for layer in layers:
        # Compute the per-layer output path FIRST so we can skip a
        # completed layer without paying the run_all_pairs cost. This is
        # the core resume affordance: once R14 completes a layer, that
        # layer's JSON is durable, and future crashes/restarts skip it.
        if args.all_layers:
            out_path = args.out.with_name(
                args.out.stem.replace(f"_L{args.layer}", "") + f"_L{layer}" + args.out.suffix
            )
        else:
            out_path = args.out

        # Non-default grids get a suffixed default checkpoint dir so a wide19
        # run can never land its (regime-mismatching) checkpoints inside the
        # parent's ladder13 pairs_* dir (parent artifacts never overwritten).
        grid_suffix = "" if args.lambda_grid == "ladder13" else f"_{args.lambda_grid}"
        ckpt_dir = (
            args.checkpoint_dir
            or out_path.parent / f"pairs_{args.model_slug}_L{layer}{grid_suffix}"
        )

        if args.merge:
            results = merge_pair_checkpoints(
                ckpt_dir,
                model_slug=args.model_slug,
                layer=layer,
                n_bootstrap_draws=args.bootstrap_draws,
                n_null_draws=args.null_draws,
                lambda_grid=args.lambda_grid,
                pairs_subset=pairs_subset,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as fh:
                json.dump(results, fh, indent=2)
            print(f"[fit_ladder] MERGE wrote {out_path} ({results['n_pairs']} pairs)", flush=True)
            continue

        if out_path.exists():
            print(
                f"[fit_ladder] SKIP layer L{layer} — output {out_path} already exists",
                flush=True,
            )
            continue

        print(f"[fit_ladder] --- layer L{layer} ---", flush=True)
        layer_t0 = time.perf_counter()
        results = run_all_pairs(
            args.store_root,
            model_slug=args.model_slug,
            layer=layer,
            smoke=args.smoke,
            n_bootstrap_draws=args.bootstrap_draws,
            n_null_draws=args.null_draws,
            engine=args.engine,
            device=args.device,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            checkpoint_dir=ckpt_dir,
            lambda_grid=args.lambda_grid,
            pairs_subset=pairs_subset,
        )
        layer_elapsed = time.perf_counter() - layer_t0
        print(
            f"[fit_ladder] layer L{layer} shard {args.shard_index}/{args.num_shards} done in "
            f"{layer_elapsed:.1f}s ({len(results['pairs'])} pairs)",
            flush=True,
        )
        if args.num_shards > 1:
            # Shard workers persist per-pair checkpoints only; the final JSON
            # is assembled by the --merge invocation after all shards join.
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(results, fh, indent=2)
        print(f"[fit_ladder] wrote {out_path} ({results['n_pairs']} pairs)", flush=True)
    return 0


if __name__ == "__main__":
    import os

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGBART pointer. main()'s writes are
    # already flushed via explicit fh.close(); atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
