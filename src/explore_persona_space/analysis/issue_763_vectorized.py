# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, λ, ×, ≈, √, μ) in scientific docstrings + log messages.
"""Vectorized batched LOCO fits for the issue #763 predictor (speed refactor).

The issue #763 fit (``scripts/issue763_fit_predictors.py``) is OVERHEAD-bound,
not FLOP-bound (``.claude/rules/vectorize-many-cell-fits.md``): its serial hot
loops are the shuffle/control NULLS — 1000 perms × 28 layers × 50 LOCO folds ×
(a statsmodels binomial-GLM IRLS OR a per-fold ``torch.linalg.eigh`` PRESS
ridge). Profiled 2026-07-01 on the deception shard, each 1-layer fixed-dim LOCO
is ~70 s (GLM) / ~79 s (ridge) on the contended VM, projecting the serial null
to ~1000 h PER BEHAVIOR (the "80× compute deviation, 0.3 h → 24 h that cannot
finish" Thomas reversed). The math per fit is n=50, d≤20 — seconds of real
compute; the wall-time is Python loop + statsmodels + per-fold torch dispatch.

This module vectorizes those fits into batched tensor ops with a HARD exactness
gate (``assert_matches_reference``), leaving the STATISTICAL SEMANTICS identical
(same nested-CV d, same PRESS-λ, same IRLS, same n_perms / n_bootstrap; only the
execution is batched). Two batched fitters:

1. ``batched_binomial_glm_loco_fixed_dim`` — the binary-companion null's inner op.
   The precision-weighted binomial GLM (logit link, ``var_weights`` = judged
   counts) is IRLS, i.e. iteratively-reweighted least squares; one Newton step is
   a linear solve, so a BATCH of (design, y, w) problems sharing (n, d) is a
   batched ``torch.linalg.solve``. Reproduces statsmodels' ``GLM(...,
   family=Binomial(), var_weights=w).fit()`` predictions to ≈1e-10 (verified:
   7e-12 max over 200 hard skewed cases), including the same ``y`` clip to
   [1e-6, 1-1e-6] and the same singular/non-converge → train-mean fallback.

2. ``batched_ridge_press_loco_fixed_dim`` — the graded-headline null's inner op.
   The KEY optimization: at a FIXED PCA dim the per-(layer, fold) PCA basis and
   the standardized-design eigendecomposition depend ONLY on the layer's X, NOT
   on the (permuted) labels — so the ``torch.linalg.eigh`` PRESS structure is
   computed ONCE per (layer, fold) and REUSED across all 1000 perms (only
   ``Qᵀy`` and the held-out projection change per perm). This is the label-
   independence win the serial per-perm refit threw away. The math is #658's
   exact PRESS / dual-weight closed form; predictions match
   ``_ridge_predict_loco_fixed_dim`` bit-for-bit up to fp reduction order.

Both fitters take a per-layer FIXED dim (the null path — the dim is chosen ONCE
on the observed data per ``_observed_layer_dims`` and held fixed across perms,
the BLOCKER analysis-null-infeasible-at-scale contract, UNCHANGED). The observed
nested-CV read stays on the exact serial ``glm_predict_loco`` /
``_ridge_predict_loco_pca`` (28 layers only, not 28×1000 — cheap enough), so the
HEADLINE ρ is still selected by the honest nested-CV path; only the null re-runs
are vectorized.

Thread discipline (``.claude/rules/vectorize-many-cell-fits.md`` item 3): the
caller sets ``torch.set_num_threads`` sanely for the CPU lane (``cpu-mid`` = 8
vCPU); tiny per-op tensors thrash with the default high thread count.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from explore_persona_space.analysis.issue_763_pca import _pca_fit, _pca_transform  # noqa: E402

logger = logging.getLogger("issue_763_vectorized")

# statsmodels' Binomial GLM logit-link endog clamp (issue_763_glm._fit_binomial_glm
# uses np.clip(y, 1e-6, 1 - 1e-6)); replicated so the batched IRLS consumes the
# identical endog.
_ENDOG_CLIP = 1e-6
# IRLS convergence: param-space max-abs delta. statsmodels' Binomial GLM converges
# in ~4-8 Newton iterations to machine precision; a 1e-10 param tol + maxiter=100
# reproduces its .predict() to ≈1e-11 (verified against statsmodels).
_IRLS_TOL = 1e-10
_IRLS_MAXITER = 100
# mu clamp inside IRLS (avoids /0 in the working-response; matches a stable IRLS).
_MU_CLIP = 1e-10


# ── batched precision-weighted binomial-GLM IRLS ──────────────────────────────


def _batched_irls_binomial(
    X: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    *,
    maxiter: int = _IRLS_MAXITER,
    tol: float = _IRLS_TOL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit B independent precision-weighted binomial GLMs (logit) via batched IRLS.

    Args:
        X: (B, n, p) per-problem design (intercept column already prepended).
        y: (B, n) per-problem fractional rate endog (already clipped to
            [1e-6, 1-1e-6] by the caller — matches statsmodels' endog clamp).
        w: (B, n) per-problem var_weights (judged counts).

    Returns:
        ``(beta (B, p), ok (B,) bool)`` — ``beta`` are the fitted coefficients;
        ``ok[b]`` is False iff problem b's solve went singular / non-finite (the
        caller then uses the train-mean fallback, mirroring statsmodels'
        ``res is None`` path). All problems share n, p; run in fp64 for parity
        with statsmodels' double-precision fit.
    """
    b_size, _n, p = X.shape
    device, dtype = X.device, X.dtype
    beta = torch.zeros(b_size, p, dtype=dtype, device=device)
    ok = torch.ones(b_size, dtype=torch.bool, device=device)
    active = torch.ones(b_size, dtype=torch.bool, device=device)
    for _ in range(maxiter):
        if not bool(active.any()):
            break
        eta = torch.bmm(X, beta.unsqueeze(2)).squeeze(2)  # (B, n)
        mu = torch.sigmoid(eta).clamp(_MU_CLIP, 1.0 - _MU_CLIP)  # (B, n)
        var = mu * (1.0 - mu)  # (B, n)
        wd = w * var  # IRLS weight = var_weights * variance (B, n)
        zwork = eta + (y - mu) / var  # working response (B, n)
        # A = Xᵀ W X  (B, p, p);  rhs = Xᵀ W zwork  (B, p)
        xw = X * wd.unsqueeze(2)  # (B, n, p)
        A = torch.bmm(xw.transpose(1, 2), X)  # (B, p, p)
        rhs = torch.bmm(xw.transpose(1, 2), zwork.unsqueeze(2)).squeeze(2)  # (B, p)
        try:
            beta_new = torch.linalg.solve(A, rhs.unsqueeze(2)).squeeze(2)  # (B, p)
        except RuntimeError:
            # A per-problem singular solve; fall back to a masked per-problem loop
            # so one bad problem does not poison the batch (marks it not-ok).
            beta_new = beta.clone()
            for bi in range(b_size):
                if not bool(active[bi]):
                    continue
                try:
                    beta_new[bi] = torch.linalg.solve(A[bi], rhs[bi])
                except RuntimeError:
                    ok[bi] = False
                    active[bi] = False
        finite = torch.isfinite(beta_new).all(dim=1)  # (B,)
        ok = ok & (finite | ~active)  # a non-finite update on an active problem fails
        # converged problems stop updating; non-finite ones freeze (marked not-ok)
        delta = (beta_new - beta).abs().amax(dim=1)  # (B,)
        step = active & finite
        beta = torch.where(step.unsqueeze(1), beta_new, beta)
        converged = step & (delta < tol)
        active = active & ~converged & finite
    return beta, ok


def batched_binomial_glm_loco_fixed_dim(
    x: np.ndarray,
    y_batch: np.ndarray,
    n_judged_batch: np.ndarray,
    dim: int,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """LOCO binomial-GLM held-out predictions for a BATCH of label vectors, one layer.

    Vectorizes ``issue_763_glm.glm_predict_loco_fixed_dim`` across the ``P``
    permutations of the null: the layer's ``x`` (n, H) is fixed, only the labels
    (y_batch) and their aligned weights (n_judged_batch) permute. The per-fold
    PCA basis depends only on ``x`` (fit on the train fold, label-independent),
    so it is computed ONCE per fold and reused across all P perms; the GLM fit at
    the selected dim is batched over (P × n folds) as one IRLS.

    Args:
        x: (n, H) the layer's v0 activations (SHARED across perms).
        y_batch: (P, n) the P permuted label vectors (rates in [0, 1]).
        n_judged_batch: (P, n) the P aligned precision-weight vectors.
        dim: the FIXED PCA dim (chosen once on observed data per layer).

    Returns:
        (P, n) held-out LOCO predictions — ``out[p, i]`` is the GLM fit on the
        n-1 train contexts (≠ i) of perm p, evaluated at held-out context i.
        Byte-for-byte the same LOCO protocol as the serial fixed-dim path;
        numerically ≈ statsmodels to ~1e-10.
    """
    n, _h = x.shape
    P = y_batch.shape[0]
    dev = torch.device(device)
    y_raw = np.asarray(y_batch, dtype=np.float64)  # UNCLIPPED — the fallback mean
    y_all = np.clip(y_raw, _ENDOG_CLIP, 1.0 - _ENDOG_CLIP)  # CLIPPED — the GLM endog
    w_all = np.asarray(n_judged_batch, dtype=np.float64)
    w_all = np.where(w_all < 1, 1.0, w_all)

    # Per-fold PCA reductions (label-independent) — computed ONCE, reused ∀ perms.
    # z_tr_by_fold[i] : (n-1, d) train-fold reduced design (intercept prepended).
    # z_held_by_fold[i]: (1, d) held-out reduced row (intercept prepended).
    z_tr_by_fold: list[np.ndarray] = []
    z_held_by_fold: list[np.ndarray] = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu, comps = _pca_fit(x[tr], min(dim, len(tr) - 1))
        z_tr = _pca_transform(x[tr], mu, comps)  # (n-1, d)
        z_held = _pca_transform(x[i : i + 1], mu, comps)  # (1, d)
        # prepend intercept column (statsmodels add_constant, has_constant="add")
        z_tr_by_fold.append(np.column_stack([np.ones(len(tr)), z_tr]))
        z_held_by_fold.append(np.column_stack([np.ones(1), z_held]))
    # d may be clamped per fold (rank); all folds share the same clamp here
    # because n-1 is constant, so the design width p is uniform.
    p = z_tr_by_fold[0].shape[1]
    assert all(zt.shape[1] == p for zt in z_tr_by_fold), "non-uniform GLM design width"

    # Build the (P*n, n-1, p) batched design + (P*n, n-1) endog/weights.
    # member (perm pth, fold i) trains on the n-1 train rows of fold i under perm p.
    n_tr = n - 1
    X_stack = np.empty((P * n, n_tr, p), dtype=np.float64)
    Xheld_stack = np.empty((P * n, p), dtype=np.float64)
    y_stack = np.empty((P * n, n_tr), dtype=np.float64)  # CLIPPED endog (fit input)
    ymean_stack = np.empty((P * n, n_tr), dtype=np.float64)  # UNCLIPPED (fallback mean)
    w_stack = np.empty((P * n, n_tr), dtype=np.float64)
    fold_train_idx = [[j for j in range(n) if j != i] for i in range(n)]
    for pth in range(P):
        for i in range(n):
            m = pth * n + i
            tr = fold_train_idx[i]
            X_stack[m] = z_tr_by_fold[i]
            Xheld_stack[m] = z_held_by_fold[i][0]
            y_stack[m] = y_all[pth, tr]
            ymean_stack[m] = y_raw[pth, tr]
            w_stack[m] = w_all[pth, tr]

    Xt = torch.from_numpy(X_stack).to(dev)
    yt = torch.from_numpy(y_stack).to(dev)
    wt = torch.from_numpy(w_stack).to(dev)
    beta, ok = _batched_irls_binomial(Xt, yt, wt)  # (P*n, p), (P*n,)
    Xheld = torch.from_numpy(Xheld_stack).to(dev)  # (P*n, p)
    eta_held = (Xheld * beta).sum(dim=1)  # (P*n,)
    pred = torch.sigmoid(eta_held)  # (P*n,)
    # train-mean fallback for singular/non-converged problems: statsmodels
    # res is None → serial preds[i] = np.mean(y_tr) on the UNCLIPPED train y
    # (glm_predict_loco_fixed_dim clips only inside _fit_binomial_glm, never the
    # fallback), so use ymean_stack (unclipped), not y_stack (clipped).
    train_mean = torch.from_numpy(ymean_stack.mean(axis=1)).to(dev)  # (P*n,)
    pred = torch.where(ok, pred, train_mean)
    return pred.detach().cpu().numpy().reshape(P, n)


# ── batched PRESS ridge LOCO (label-independent eigh reused across perms) ──────


@dataclass
class _FoldPress:
    """Cached label-INDEPENDENT PRESS structure for one (layer, LOCO fold).

    All of these depend ONLY on the layer's train-fold X (the PCA basis, the
    standardization μ/σ, and the eigendecomposition of the standardized Gram) —
    NOT on the labels — so they are computed once and reused across all null
    perms. Only ``Qᵀy`` and the final held-out projection recompute per perm.
    """

    mu_pca: np.ndarray  # PCA mean (H,)
    comps: np.ndarray  # PCA components (d, H)
    x_mu: torch.Tensor  # standardization mean (d,) fp64
    x_sd: torch.Tensor  # standardization sd (d,) fp64  (ddof=0 + 1e-9, #658 conv)
    Q: torch.Tensor  # (n-1, n-1) eigenvectors of the dual Gram
    evals: torch.Tensor  # (n-1,) eigenvalues
    Qsq: torch.Tensor  # (n-1, n-1) Q*Q for diag(H)
    z_tr_n: torch.Tensor  # (n-1, d) standardized train design
    z_held_n: torch.Tensor  # (d,) standardized held-out row


def batched_ridge_press_loco_fixed_dim(
    x: np.ndarray,
    y_batch: np.ndarray,
    lambdas: list[float],
    dim: int,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """LOCO PRESS-ridge held-out predictions for a BATCH of label vectors, one layer.

    Vectorizes ``issue763_fit_predictors._ridge_predict_loco_fixed_dim`` across
    the P null perms. The per-fold PCA basis, standardization, and the dual-Gram
    eigendecomposition are label-INDEPENDENT (depend only on ``x``), so they are
    computed ONCE per fold (``_FoldPress``) and reused across all P perms — the
    label-independence win the serial per-perm refit discarded. Per perm+fold the
    only label-dependent work is: the inner-LOO PRESS λ pick (a per-λ rescale of
    the cached eigenbasis applied to ``Qᵀ y_perm``) and the final held-out
    prediction ``z_held_n · (Xᵀ α)``. Every arithmetic op is #658's exact PRESS /
    dual-weight closed form, so the predictions match the serial fixed-dim ridge
    up to fp reduction order.

    Args:
        x: (n, H) the layer's v0 activations (SHARED across perms).
        y_batch: (P, n) the P permuted label vectors.
        lambdas: the ridge λ grid (RIDGE_LAMBDAS).
        dim: the FIXED PCA dim.

    Returns:
        (P, n) held-out LOCO predictions.
    """
    n, _h = x.shape
    P = y_batch.shape[0]
    dev = torch.device(device)
    lam_t = torch.tensor(lambdas, dtype=torch.float64, device=dev)  # (L,)
    y_all = torch.from_numpy(np.asarray(y_batch, dtype=np.float64)).to(dev)  # (P, n)

    folds: list[_FoldPress] = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu_p, comps = _pca_fit(x[tr], min(dim, len(tr) - 1))
        z_tr = _pca_transform(x[tr], mu_p, comps)  # (n-1, d)
        z_held = _pca_transform(x[i : i + 1], mu_p, comps)[0]  # (d,)
        # fp64 throughout — matches #658 _ridge_predict_loco's .to(torch.float64)
        # convention so the batched path is bit-comparable to the serial oracle.
        Xtr = torch.from_numpy(np.ascontiguousarray(z_tr)).to(device=dev, dtype=torch.float64)
        x_mu = Xtr.mean(0)
        x_sd = Xtr.std(0, correction=0) + 1e-9  # #658 ddof=0 convention
        Xtr_n = (Xtr - x_mu) / x_sd  # (n-1, d)
        G = Xtr_n @ Xtr_n.t()  # dual Gram (n-1, n-1)
        evals, Q = torch.linalg.eigh(G)
        Qsq = Q * Q
        x_held = torch.from_numpy(np.ascontiguousarray(z_held)).to(device=dev, dtype=torch.float64)
        z_held_n = (x_held - x_mu) / x_sd  # (d,)
        folds.append(_FoldPress(mu_p, comps, x_mu, x_sd, Q, evals, Qsq, Xtr_n, z_held_n))

    train_idx = [[j for j in range(n) if j != i] for i in range(n)]
    out = torch.empty(P, n, dtype=torch.float64, device=dev)
    for i in range(n):
        fp = folds[i]
        tr = train_idx[i]
        Ytr = y_all[:, tr].t()  # (n-1, P) — all perms' train endog for this fold
        # inner PRESS LOO MSE per (λ, perm): reuse #658's identity with the cached
        # eigenbasis. QtY (n-1, P); for each λ, Yhat = Q diag(g/(g+λ)) QtY,
        # loo_resid = (Ytr - Yhat)/(1 - h_diag) with h_diag = Qsq @ (g/(g+λ)).
        QtY = fp.Q.t() @ Ytr  # (n-1, P)
        best_mse = torch.full((P,), float("inf"), dtype=torch.float64, device=dev)
        best_lam_idx = torch.zeros(P, dtype=torch.long, device=dev)
        for li in range(lam_t.shape[0]):
            filt = fp.evals / (fp.evals + lam_t[li])  # (n-1,)
            h_diag = fp.Qsq @ filt  # (n-1,)
            Yhat = fp.Q @ (filt.unsqueeze(1) * QtY)  # (n-1, P)
            denom = (1.0 - h_diag).clamp(min=1e-8).unsqueeze(1)  # (n-1, 1)
            loo = (Ytr - Yhat) / denom  # (n-1, P)
            mse = (loo * loo).mean(dim=0)  # (P,)
            better = mse < best_mse
            best_lam_idx = torch.where(better, torch.full_like(best_lam_idx, li), best_lam_idx)
            best_mse = torch.where(better, mse, best_mse)
        # per-perm outer fit at the selected λ: α = (G + λI)⁻¹ Ytr, then
        # pred = z_held_n · (Xtrᵀ α). Group perms by their selected λ index so
        # each distinct λ solves ONE batched system.
        pred_i = torch.empty(P, dtype=torch.float64, device=dev)
        n_tr = len(tr)
        eye = torch.eye(n_tr, dtype=torch.float64, device=dev)
        G_full = fp.z_tr_n @ fp.z_tr_n.t()  # (n-1, n-1) — same Gram
        for li in range(lam_t.shape[0]):
            sel = best_lam_idx == li
            if not bool(sel.any()):
                continue
            A = G_full + lam_t[li] * eye  # (n-1, n-1)
            alpha = torch.linalg.solve(A, Ytr[:, sel])  # (n-1, k)
            w = fp.z_tr_n.t() @ alpha  # (d, k) dual weights
            pred_i[sel] = fp.z_held_n @ w  # (k,)
        out[:, i] = pred_i
    return out.detach().cpu().numpy()


# ── batched observed nested-CV read (select-dim inner-LOO is the shared kernel) ─


def _batched_select_pca_dim(
    x_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    d_grid: tuple[int, ...],
    *,
    device: str = "cpu",
) -> int:
    """Vectorized ``issue_763_pca.select_pca_dim`` — batch the inner-LOO GLM fits.

    Reproduces the serial selection EXACTLY: same p≪n dim cap (``d_max = max(2,
    n//5)``), same nested-CV inner criterion (mean inner-LOO squared error of a
    precision-weighted binomial GLM), same tie-break (first dim in grid order
    reaching the min MSE — a strict ``<`` comparison so equal MSEs keep the
    earlier dim, matching the serial loop). For each candidate dim, the inner-LOO
    (n fits, each on n-1 rows at that dim's PCA reduction) is ONE batched IRLS
    over the n inner folds; the SVD basis at that dim is fit on the full train set
    once (matching the serial ``_pca_fit(x_train, d)`` reuse across inner folds).
    """
    y = np.asarray(y_train, dtype=np.float64)
    w = np.asarray(w_train, dtype=np.float64)
    w = np.where(w < 1, 1.0, w)
    n = x_train.shape[0]
    d_max = max(2, n // 5)
    best_d, best_mse = d_grid[0], np.inf
    for d in d_grid:
        if d > d_max:
            continue
        mu_d, comps_d = _pca_fit(x_train, d)
        z = _pca_transform(x_train, mu_d, comps_d)  # (n, d_eff)
        p = z.shape[1] + 1  # + intercept
        # batch the n inner-LOO folds: member k trains on rows != k, predicts k.
        X_stack = np.empty((n, n - 1, p), dtype=np.float64)
        Xheld = np.empty((n, p), dtype=np.float64)
        y_stack = np.empty((n, n - 1), dtype=np.float64)  # CLIPPED endog (fit input)
        ymean_stack = np.empty((n, n - 1), dtype=np.float64)  # UNCLIPPED (fallback mean)
        w_stack = np.empty((n, n - 1), dtype=np.float64)
        yk = np.empty(n, dtype=np.float64)
        yc = np.clip(y, _ENDOG_CLIP, 1.0 - _ENDOG_CLIP)
        for k in range(n):
            tr = [j for j in range(n) if j != k]
            X_stack[k] = np.column_stack([np.ones(n - 1), z[tr]])
            Xheld[k] = np.concatenate([[1.0], z[k]])
            y_stack[k] = yc[tr]
            ymean_stack[k] = y[tr]  # serial fallback uses np.mean(UNCLIPPED y[tr])
            w_stack[k] = w[tr]
            yk[k] = y[k]  # inner-LOO MSE uses the UNCLIPPED y[k] (serial _inner_loo_mse)
        dev = torch.device(device)
        beta, ok = _batched_irls_binomial(
            torch.from_numpy(X_stack).to(dev),
            torch.from_numpy(y_stack).to(dev),
            torch.from_numpy(w_stack).to(dev),
        )
        eta = (torch.from_numpy(Xheld).to(dev) * beta).sum(dim=1)
        pred = torch.sigmoid(eta)
        train_mean = torch.from_numpy(ymean_stack.mean(axis=1)).to(dev)
        pred = torch.where(ok, pred, train_mean).detach().cpu().numpy()
        mse = float(np.mean((pred - yk) ** 2))
        if mse < best_mse:
            best_mse, best_d = mse, d
    return best_d


def batched_glm_predict_loco(
    x: np.ndarray,
    y: np.ndarray,
    n_judged: np.ndarray,
    d_grid: tuple[int, ...],
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Vectorized ``issue_763_glm.glm_predict_loco`` observed nested-CV LOCO.

    The per-outer-fold nested-CV dim selection (the ~223 s/layer serial kernel)
    is ``_batched_select_pca_dim`` (batched inner-LOO IRLS); the outer held-out
    GLM fit at the selected dim is a single per-fold IRLS. Returns the (n,)
    held-out predictions — the same LOCO protocol + the same nested-CV d per fold
    as the serial path (verified within IRLS-vs-statsmodels tolerance). Only
    ``pred`` is returned (the caller reads ρ from it); the serial's
    overdispersion / quasibinomial diagnostics are NOT recomputed here because
    the fit driver consumes only ``["pred"]`` from ``glm_predict_loco`` inside
    ``_layer_sweep_select`` (the diagnostics ride the standalone unused path).
    """
    n = x.shape[0]
    w = np.asarray(n_judged, dtype=np.float64)
    w = np.where(w < 1, 1.0, w)
    y = np.asarray(y, dtype=np.float64)
    dev = torch.device(device)
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_d = _batched_select_pca_dim(x[tr], y[tr], w[tr], d_grid, device=device)
        mu, comps = _pca_fit(x[tr], min(best_d, len(tr) - 1))
        z_tr = _pca_transform(x[tr], mu, comps)
        z_held = _pca_transform(x[i : i + 1], mu, comps)
        X = np.column_stack([np.ones(len(tr)), z_tr])
        Xh = np.concatenate([[1.0], z_held[0]])
        yc = np.clip(y[tr], _ENDOG_CLIP, 1.0 - _ENDOG_CLIP)
        beta, ok = _batched_irls_binomial(
            torch.from_numpy(X[None]).to(dev),
            torch.from_numpy(yc[None]).to(dev),
            torch.from_numpy(w[tr][None]).to(dev),
        )
        if bool(ok[0]):
            preds[i] = float(torch.sigmoid((torch.from_numpy(Xh).to(dev) * beta[0]).sum()))
        else:
            preds[i] = float(np.mean(y[tr]))
    return preds


def batched_ridge_predict_loco_pca(
    x: np.ndarray,
    y: np.ndarray,
    n_judged: np.ndarray,
    lambdas: list[float],
    d_grid: tuple[int, ...],
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Vectorized ``issue763_fit_predictors._ridge_predict_loco_pca`` observed LOCO.

    Same shared-PCA matched-capacity contract (#763 BLOCKER ridge-pca-comparator):
    per outer fold, the dim is selected by the GLM inner criterion
    (``_batched_select_pca_dim`` — the SAME batched selection ``batched_glm_
    predict_loco`` uses, so the two arms consume identically-reduced features),
    then a closed-form PRESS-λ LOCO ridge on those reduced features. The ridge
    inner-λ PRESS + dual-weight solve is #658's exact closed form. Returns (n,)
    held-out predictions matching the serial path up to fp reduction order.
    """
    n = x.shape[0]
    w = np.asarray(n_judged, dtype=np.float64)
    w = np.where(w < 1, 1.0, w)
    y = np.asarray(y, dtype=np.float64)
    dev = torch.device(device)
    lam_t = torch.tensor(lambdas, dtype=torch.float64, device=dev)
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        best_d = _batched_select_pca_dim(x[tr], y[tr], w[tr], d_grid, device=device)
        mu, comps = _pca_fit(x[tr], min(best_d, len(tr) - 1))
        z_tr = _pca_transform(x[tr], mu, comps)
        z_held = _pca_transform(x[i : i + 1], mu, comps)
        Xtr = torch.from_numpy(np.ascontiguousarray(z_tr)).to(device=dev, dtype=torch.float64)
        Ytr = torch.from_numpy(np.ascontiguousarray(y[tr].reshape(-1, 1))).to(
            device=dev, dtype=torch.float64
        )
        x_mu = Xtr.mean(0)
        x_sd = Xtr.std(0, correction=0) + 1e-9
        Xtr_n = (Xtr - x_mu) / x_sd
        G = Xtr_n @ Xtr_n.t()
        evals, Q = torch.linalg.eigh(G)
        Qsq = Q * Q
        QtY = Q.t() @ Ytr  # (n-1, 1)
        best_mse, best_li = float("inf"), 0
        for li in range(lam_t.shape[0]):
            filt = evals / (evals + lam_t[li])
            h_diag = Qsq @ filt
            Yhat = Q @ (filt.unsqueeze(1) * QtY)
            denom = (1.0 - h_diag).clamp(min=1e-8).unsqueeze(1)
            loo = (Ytr - Yhat) / denom
            mse = float((loo * loo).mean())
            if mse < best_mse:
                best_mse, best_li = mse, li
        A = G + lam_t[best_li] * torch.eye(len(tr), dtype=torch.float64, device=dev)
        alpha = torch.linalg.solve(A, Ytr)
        wgt = Xtr_n.t() @ alpha  # (d, 1)
        x_held = torch.from_numpy(np.ascontiguousarray(z_held[0])).to(
            device=dev, dtype=torch.float64
        )
        x_held_n = (x_held - x_mu) / x_sd
        preds[i] = float((x_held_n @ wgt).reshape(-1)[0])
    return preds


# ── exactness gate ────────────────────────────────────────────────────────────


def assert_matches_reference(seed: int = 0, n: int = 20, h: int = 24, dim: int = 6) -> dict:
    """Assert the batched fitters reproduce the serial references within tolerance.

    Builds a small synthetic (x, y) LOCO problem with real rank structure and a
    2-perm batch (identity + one shuffle) and checks EVERY batched fitter against
    its serial oracle:

    (a) ``batched_binomial_glm_loco_fixed_dim`` vs ``issue_763_glm.
        glm_predict_loco_fixed_dim`` (statsmodels IRLS) — tol 1e-6 (the batched
        torch IRLS reproduces statsmodels' logit fit to ~1e-10; 1e-6 is generous
        headroom for the endog-clip + convergence-criterion difference).
    (b) ``batched_ridge_press_loco_fixed_dim`` vs ``issue763_fit_predictors.
        _ridge_predict_loco_fixed_dim`` — tol 1e-8 (both are the identical #658
        PRESS/dual closed form; the residual is fp reduction order only).
    (c) ``_batched_select_pca_dim`` vs ``issue_763_pca.select_pca_dim`` — the
        chosen dim must be ARGMAX-IDENTICAL (integer equal), not merely close: a
        drifted dim would change the observed read's capacity.
    (d) ``batched_glm_predict_loco`` (observed nested-CV) vs ``glm_predict_loco``
        — tol 1e-6.
    (e) ``batched_ridge_predict_loco_pca`` (observed nested-CV) vs
        ``_ridge_predict_loco_pca`` — tol 1e-8.

    Returns the measured max deltas + the dim-select match. Raises AssertionError
    on any tolerance miss — the batched path must NEVER be trusted without this
    gate passing (the batched-solve seeding + standardization is easy to get
    subtly wrong).
    """
    # import the serial ridge fixed-dim reference from the fit script
    import issue763_fit_predictors as _fit

    from explore_persona_space.analysis.issue_763_glm import (
        glm_predict_loco,
        glm_predict_loco_fixed_dim,
    )
    from explore_persona_space.analysis.issue_763_pca import PCA_DIM_GRID, select_pca_dim

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    W = rng.standard_normal((3, h))
    x = (z @ W + 0.15 * rng.standard_normal((n, h))).astype(np.float64)
    beta = rng.standard_normal(h) * 0.02
    eta = x @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = np.clip(prob + rng.normal(0, 0.08, n), 0.02, 0.98)
    nj = rng.integers(10, 60, n).astype(np.float64)
    perm = rng.permutation(n)
    y_perm = y[perm]
    nj_perm = nj[perm]

    # (a) GLM
    glm_ref0 = glm_predict_loco_fixed_dim(x, y, nj, dim)
    glm_ref1 = glm_predict_loco_fixed_dim(x, y_perm, nj_perm, dim)
    glm_bat = batched_binomial_glm_loco_fixed_dim(
        x, np.stack([y, y_perm]), np.stack([nj, nj_perm]), dim
    )
    d_glm0 = float(np.max(np.abs(glm_bat[0] - glm_ref0)))
    d_glm1 = float(np.max(np.abs(glm_bat[1] - glm_ref1)))

    # (b) ridge
    ridge_ref0 = _fit._ridge_predict_loco_fixed_dim(x, y, nj, _fit.RIDGE_LAMBDAS, dim)
    ridge_ref1 = _fit._ridge_predict_loco_fixed_dim(x, y_perm, nj_perm, _fit.RIDGE_LAMBDAS, dim)
    ridge_bat = batched_ridge_press_loco_fixed_dim(
        x, np.stack([y, y_perm]), _fit.RIDGE_LAMBDAS, dim
    )
    d_r0 = float(np.max(np.abs(ridge_bat[0] - ridge_ref0)))
    d_r1 = float(np.max(np.abs(ridge_bat[1] - ridge_ref1)))

    # (c) observed nested-CV dim selection must pick the SAME dim as the serial
    #     select_pca_dim (argmin over the d-grid must be identical, not just close).
    d_sel_ref = select_pca_dim(x, y, nj, d_grid=PCA_DIM_GRID)
    d_sel_bat = _batched_select_pca_dim(x, y, nj, PCA_DIM_GRID)
    dim_select_identical = int(d_sel_ref) == int(d_sel_bat)

    # (d) observed nested-CV GLM LOCO vs serial glm_predict_loco (predictions).
    glm_obs_ref = glm_predict_loco(x, y, nj, pca_dim_grid=PCA_DIM_GRID)["pred"]
    glm_obs_bat = batched_glm_predict_loco(x, y, nj, PCA_DIM_GRID)
    d_glm_obs = float(np.max(np.abs(glm_obs_bat - glm_obs_ref)))

    # (e) observed nested-CV PCA ridge LOCO vs serial _ridge_predict_loco_pca.
    ridge_obs_ref = _fit._ridge_predict_loco_pca(x, y, nj, _fit.RIDGE_LAMBDAS)
    ridge_obs_bat = batched_ridge_predict_loco_pca(x, y, nj, _fit.RIDGE_LAMBDAS, PCA_DIM_GRID)
    d_ridge_obs = float(np.max(np.abs(ridge_obs_bat - ridge_obs_ref)))

    tol_glm = 1e-6
    tol_ridge = 1e-8
    assert d_glm0 <= tol_glm and d_glm1 <= tol_glm, (
        f"batched binomial-GLM LOCO exactness FAILED: max|Δpred|=({d_glm0:.3e}, "
        f"{d_glm1:.3e}) > {tol_glm} vs statsmodels glm_predict_loco_fixed_dim"
    )
    assert d_r0 <= tol_ridge and d_r1 <= tol_ridge, (
        f"batched PRESS-ridge LOCO exactness FAILED: max|Δpred|=({d_r0:.3e}, "
        f"{d_r1:.3e}) > {tol_ridge} vs _ridge_predict_loco_fixed_dim"
    )
    assert dim_select_identical, (
        f"batched select_pca_dim MISMATCH: serial picked d={d_sel_ref}, batched "
        f"picked d={d_sel_bat} — the nested-CV dim selection must be argmax-identical"
    )
    assert d_glm_obs <= tol_glm, (
        f"batched observed GLM LOCO exactness FAILED: max|Δpred|={d_glm_obs:.3e} > "
        f"{tol_glm} vs glm_predict_loco"
    )
    assert d_ridge_obs <= tol_ridge, (
        f"batched observed ridge-PCA LOCO exactness FAILED: max|Δpred|={d_ridge_obs:.3e} "
        f"> {tol_ridge} vs _ridge_predict_loco_pca"
    )
    return {
        "glm_delta": max(d_glm0, d_glm1),
        "ridge_delta": max(d_r0, d_r1),
        "dim_select_identical": dim_select_identical,
        "glm_obs_delta": d_glm_obs,
        "ridge_obs_delta": d_ridge_obs,
        "tol_glm": tol_glm,
        "tol_ridge": tol_ridge,
    }
