# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ², λ, ȳ) in scientific docstrings.
"""Issue #810 — batched shuffle-null re-fits (vectorize-many-cell-fits mandate).

The r1 code re-invoked the SERIAL closed-form LOCO ridge (a 50-fold Python loop
per cell) ONCE PER PERMUTATION PER CELL for the 1000-perm selection-symmetric
null — the exact #722 overhead-bound anti-pattern (`.claude/rules/
vectorize-many-cell-fits.md`), projecting 231 wall-h for Phase D (308× the
plan's 0.75h). This module BATCHES the null: the design X is FIXED across
permutations (only the target Y is row-permuted), so every X-only quantity —
the per-fold train-row standardization, the dual Gram eigendecomposition (PRESS
λ-select), and the per-λ dual-solve factor `(G + λI)⁻¹` — is computed ONCE per
cell, and all 1000 permutations' Y-dependent parts run as batched matmuls over
a stacked `(n_perms, n_train, P)` tensor. NO Python-level per-perm loop, NO
per-perm re-fit.

The batched closed form IS the serial refit (same PRESS / dual identities from
``issue658_fit_predictors``), so the per-draw skill / ρ are NUMERICALLY
IDENTICAL to the r1 serial null (a throughput win, not a numerical change) — the
smoke asserts this byte-for-byte-close against the serial path.

Three null shapes:

- ``batched_ridge_loco_null_skill`` — RECON null (DV a): fixed c_C design ``Xc``,
  row-permute the PCA target ``Y_pca``, re-fit the LOCO ridge, per-draw
  skill-over-mean R². Matches ``vectorized_mlp_skill.ridge_predict_loco_centered``
  + ``skill_over_mean_r2`` on the permuted target.
- ``batched_ridge_loco_null_rho`` — READOUT trained-ridge null (DV b): fixed
  PCA-reduced summary design ``Xp``, row-permute the scalar E0 ``y``, re-fit the
  LOCO ridge, per-draw Spearman ρ. Matches ``ridge_predict_loco_centered``
  (1-column target) + Spearman.
- ``batched_projection_null_rho`` — READOUT fixed-r_B null (DV b): ZERO-parameter
  projection (no re-fit); permute the (E0, prediction) pairing and re-Spearman.
  Trivially batched (a rank-correlation over a stacked permuted-y matrix).

All match the SAME centered-ridge convention as
``vectorized_mlp_skill.ridge_predict_loco_centered``: per-fold train-only X
standardization (torch ``.std(0, correction=0) + 1e-9``, numpy ddof=0), per-fold
train-only target centering (``ymu = Ytr.mean(0)``, prediction ``ymu + x_held @
w``), inner PRESS nested-CV λ over ``RIDGE_LAMBDAS`` (no λ leakage), fp64.
"""

from __future__ import annotations

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue658_fit_predictors import RIDGE_LAMBDAS  # noqa: E402

# ── permutation matrix ────────────────────────────────────────────────────────


def make_perm_matrix(n: int, n_perms: int, rng: np.random.Generator) -> np.ndarray:
    """(n_perms, n) row-permutation index matrix drawn from ``rng``.

    Draw order matches the serial null's ``rng.permutation(n)`` per draw so the
    batched path consumes the SAME permutations as the serial reference for a
    like-seeded ``rng`` — the exactness contract in the smoke.
    """
    return np.stack([rng.permutation(n) for _ in range(n_perms)]).astype(np.int64)


# ── shared X-only LOCO ridge precompute ───────────────────────────────────────


class _LocoRidgeXCache:
    """Per-fold X-only LOCO ridge factors, computed ONCE per cell (fixed X).

    For each held-out fold i (train rows tr = all but i), caches the train-row
    standardization (mu_x, sd_x), the standardized train design ``Xtr_n``, the
    dual Gram eigendecomposition (``evals``, ``Q`` for the PRESS λ-select), the
    per-λ dual-solve factor ``(G + λ I)⁻¹`` (so the null's dual weights are a
    single batched matmul against permuted Y), and the standardized held-out row
    ``x_held``. Everything here is INDEPENDENT of Y → shared across all
    permutations. Matches ``ridge_predict_loco_centered`` exactly (ddof=0 std,
    1e-9 floor, dual/PRESS identities).
    """

    def __init__(self, X: np.ndarray, lambdas, device: str = "cpu") -> None:
        self.n = int(X.shape[0])
        self.d = int(X.shape[1])
        self.lambdas = list(lambdas)
        dev = torch.device(device)
        self.device = dev
        Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device=dev, dtype=torch.float64)
        self.tr_idx: list[torch.Tensor] = []
        self.evals: list[torch.Tensor] = []  # (m,) per fold
        self.Q: list[torch.Tensor] = []  # (m, m) per fold
        self.Qsq: list[torch.Tensor] = []  # (m, m) per fold (diag(H))
        self.Ainv: list[torch.Tensor] = []  # (n_lambda, m, m) per fold
        # x_held_dot = x_held @ Xtr_nᵀ (m,) per fold: the DUAL prediction contracts
        # the d-dim design out ONCE (X-only), so the held-out read is
        # ``ymu + x_held_dot @ alpha`` — never materializing the (B, d, P) primal
        # weight ``w`` (d = c_C H = 3584 → a (1000, 3584, 48) blowup avoided; the
        # 143s→sub-second RECON speedup). x_held @ (Xtr_nᵀ @ alpha) ==
        # (x_held @ Xtr_nᵀ) @ alpha by associativity, so numerically identical.
        self.x_held_dot: list[torch.Tensor] = []  # (m,) per fold
        m = self.n - 1
        eye_m = torch.eye(m, dtype=torch.float64, device=dev)
        for i in range(self.n):
            tr = [j for j in range(self.n) if j != i]
            tr_t = torch.tensor(tr, device=dev)
            Xtr = Xt[tr_t]
            mu = Xtr.mean(0)
            sd = Xtr.std(0, correction=0) + 1e-9  # numpy ddof=0 (#658 convention)
            Xtr_n = (Xtr - mu) / sd
            G = Xtr_n @ Xtr_n.t()  # (m, m) dual Gram
            evals, Q = torch.linalg.eigh(G)
            self.tr_idx.append(tr_t)
            self.evals.append(evals)
            self.Q.append(Q)
            self.Qsq.append(Q * Q)
            # per-λ dual-solve inverse (G + λI)⁻¹, reused across all permutations.
            Ainv_lam = torch.stack(
                [torch.linalg.inv(G + lam * eye_m) for lam in self.lambdas]
            )  # (n_lambda, m, m)
            self.Ainv.append(Ainv_lam)
            x_held = (Xt[i] - mu) / sd  # (d,)
            self.x_held_dot.append(x_held @ Xtr_n.t())  # (m,) dual read vector


def _loco_ridge_pred_batched(cache: _LocoRidgeXCache, Yperm: torch.Tensor) -> torch.Tensor:
    """Batched LOCO-ridge held-out predictions for a stack of permuted targets.

    ``Yperm`` (B, n, P) — B permuted target matrices sharing ``cache``'s fixed X.
    Returns (B, n, P) held-out predictions, one per permutation, matching
    ``ridge_predict_loco_centered`` per (permutation, fold) EXACTLY (train-only
    centering, PRESS λ-select, dual weights, add train mean back). Every
    Y-dependent step is a batched matmul; the per-λ argmin is a batched reduce.
    """
    B, n, P = Yperm.shape
    assert n == cache.n, (n, cache.n)
    dev = cache.device
    lambdas = cache.lambdas
    nlam = len(lambdas)
    preds = torch.zeros((B, n, P), dtype=torch.float64, device=dev)
    for i in range(n):
        tr_t = cache.tr_idx[i]
        Ytr = Yperm[:, tr_t, :]  # (B, m, P)
        ymu = Ytr.mean(dim=1, keepdim=True)  # (B, 1, P) train predict-the-mean
        Ytr_c = Ytr - ymu  # (B, m, P) train-centered target
        # PRESS LOO MSE per λ over the SHARED eigenbasis (X-only), batched over B.
        Q = cache.Q[i]  # (m, m)
        evals = cache.evals[i]  # (m,)
        Qsq = cache.Qsq[i]  # (m, m)
        QtY = torch.einsum("ij,bjp->bip", Q.t(), Ytr_c)  # (B, m, P)
        # filt (n_lambda, m); h_diag (n_lambda, m); Yhat (n_lambda, B, m, P).
        filt = evals.unsqueeze(0) / (
            evals.unsqueeze(0) + torch.tensor(lambdas, dtype=torch.float64, device=dev).unsqueeze(1)
        )  # (n_lambda, m)
        h_diag = filt @ Qsq.t()  # (n_lambda, m)  == sum_j Qsq[k,j] filt[l,j]
        # Yhat[l,b] = Q diag(filt[l]) QtY[b]  → (n_lambda, B, m, P). The filtered
        # dual coeffs are (n_lambda, B, m, P) = filt[l] ⊙ QtY[b] broadcast, then
        # Q applied over the m (row) dim.
        filt_QtY = filt.view(nlam, 1, -1, 1) * QtY.unsqueeze(0)  # (n_lambda, B, m, P)
        Yhat = torch.einsum("ij,lbjp->lbip", Q, filt_QtY)  # (n_lambda, B, m, P)
        resid = Ytr_c.unsqueeze(0) - Yhat  # (n_lambda, B, m, P)
        denom = (1.0 - h_diag).clamp(min=1e-8).view(nlam, 1, -1, 1)  # (n_lambda,1,m,1)
        loo = resid / denom
        # mean over the m LOO folds AND P outputs (matches _press_loo_mse_per_lambda).
        mse = (loo * loo).mean(dim=(2, 3))  # (n_lambda, B)
        best = torch.argmin(mse, dim=0)  # (B,) best λ index per permutation
        # dual coeffs at each permutation's selected λ: alpha = (G+λI)⁻¹ Ytr_c.
        Ainv_sel = cache.Ainv[i][best]  # (B, m, m)
        alpha = torch.bmm(Ainv_sel, Ytr_c)  # (B, m, P)
        # DUAL held-out read: ymu + (x_held @ Xtr_nᵀ) @ alpha — contracts the
        # d-dim design out via the precomputed (m,) vector, avoiding the (B, d, P)
        # primal weight. Identical to ymu + x_held @ (Xtr_nᵀ @ alpha).
        xhd = cache.x_held_dot[i]  # (m,)
        pred_i = ymu.squeeze(1) + torch.einsum("m,bmp->bp", xhd, alpha)  # (B, P)
        preds[:, i, :] = pred_i
    return preds


# ── RECON null: skill-over-mean R² per permutation ────────────────────────────


def batched_ridge_loco_null_skill(
    Xc: np.ndarray,
    Y_pca: np.ndarray,
    perm: np.ndarray,
    device: str = "cpu",
) -> list[float]:
    """Per-draw ridge skill-over-mean R² for the RECON label-shuffle null (batched).

    ``Xc`` (n, d) fixed c_C design; ``Y_pca`` (n, P) PCA target; ``perm``
    (n_perms, n) row permutations (rows of BOTH the prediction target and the
    train-mean baseline are permuted — the serial ``skill_over_mean_r2(pred,
    Y_pca[perm])`` scores the ridge fit against the SAME permuted target). Returns
    a list of n_perms skill values, numerically identical to the r1 serial null.
    """
    n = Y_pca.shape[0]
    dev = torch.device(device)
    cache = _LocoRidgeXCache(Xc, RIDGE_LAMBDAS, device=device)
    Yt = torch.from_numpy(np.ascontiguousarray(Y_pca)).to(device=dev, dtype=torch.float64)
    perm_t = torch.from_numpy(np.ascontiguousarray(perm)).to(device=dev, dtype=torch.long)
    Yperm = Yt[perm_t]  # (B, n, P)
    preds = _loco_ridge_pred_batched(cache, Yperm)  # (B, n, P)
    # skill_over_mean_r2 on the SAME permuted target, per draw.
    # LOO train mean per fold: (total - row_i) / (n-1), batched over B.
    total = Yperm.sum(dim=1, keepdim=True)  # (B, 1, P)
    tmean = (total - Yperm) / (n - 1)  # (B, n, P)
    ss_res = ((Yperm - preds) ** 2).sum(dim=(1, 2))  # (B,)
    ss_tot = ((Yperm - tmean) ** 2).sum(dim=(1, 2))  # (B,)
    skill = torch.where(
        ss_tot < 1e-12, torch.full_like(ss_tot, float("nan")), 1.0 - ss_res / ss_tot
    )
    return [float(s) for s in skill.detach().cpu().numpy()]


# ── READOUT trained-ridge null: Spearman ρ per permutation ────────────────────


def _spearman_batched(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Batched Spearman ρ between (B, n) prediction rows and (B, n) target rows.

    Matches ``scipy.stats.spearmanr`` for tie-free ranks (the shuffle-null targets
    are distinct floats → no ties in practice); a degenerate (constant) row
    yields NaN (guarded by the caller's 0.0-fallback, matching the serial
    ``_rho`` None→0.0 convention).
    """
    B, n = pred.shape

    def _rank(a: torch.Tensor) -> torch.Tensor:
        order = a.argsort(dim=1)
        ranks = torch.zeros_like(a, dtype=torch.float64)
        arange = torch.arange(n, dtype=torch.float64, device=a.device).expand(B, n)
        ranks.scatter_(1, order, arange)
        return ranks

    rp = _rank(pred)
    ry = _rank(y)
    rp = rp - rp.mean(dim=1, keepdim=True)
    ry = ry - ry.mean(dim=1, keepdim=True)
    num = (rp * ry).sum(dim=1)
    den = torch.sqrt((rp * rp).sum(dim=1) * (ry * ry).sum(dim=1))
    return num / den


def batched_ridge_loco_null_rho(
    Xp: np.ndarray,
    y: np.ndarray,
    perm: np.ndarray,
    device: str = "cpu",
) -> list[float]:
    """Per-draw Spearman ρ for the READOUT trained-ridge label-shuffle null (batched).

    ``Xp`` (n, k) fixed PCA-reduced summary design; ``y`` (n,) scalar E0; ``perm``
    (n_perms, n). Per draw: permute E0 rows, re-fit the LOCO ridge, Spearman(pred,
    permuted-y) — matching the serial ``_trained_ridge_pred`` + ``_rho`` on the
    permuted target. A degenerate (constant-pred / constant-y) draw → 0.0 (the
    serial ``dr if dr is not None else 0.0`` convention). Returns n_perms ρ.
    """
    dev = torch.device(device)
    cache = _LocoRidgeXCache(Xp, RIDGE_LAMBDAS, device=device)
    yt = torch.from_numpy(np.ascontiguousarray(y.reshape(-1, 1))).to(
        device=dev, dtype=torch.float64
    )
    perm_t = torch.from_numpy(np.ascontiguousarray(perm)).to(device=dev, dtype=torch.long)
    Yperm = yt[perm_t]  # (B, n, 1)
    preds = _loco_ridge_pred_batched(cache, Yperm)[:, :, 0]  # (B, n)
    yperm2d = Yperm[:, :, 0]  # (B, n)
    # Guard degenerate draws (serial _rho returns None → caller stores 0.0).
    pred_std = preds.std(dim=1)
    y_std = yperm2d.std(dim=1)
    rho = _spearman_batched(preds, yperm2d)
    bad = (pred_std < 1e-9) | (y_std < 1e-9) | torch.isnan(rho)
    rho = torch.where(bad, torch.zeros_like(rho), rho)
    return [float(r) for r in rho.detach().cpu().numpy()]


# ── READOUT fixed-r_B null: zero-parameter projection, re-Spearman ────────────


def batched_projection_null_rho(
    pred: np.ndarray,
    y: np.ndarray,
    perm: np.ndarray,
    device: str = "cpu",
) -> list[float]:
    """Per-draw Spearman ρ for the fixed-r_B projection null (batched, no re-fit).

    The fixed-r_B read-out is a ZERO-parameter projection ``pred = X @ r`` — the
    correct null permutes the (E0, prediction) PAIRING and re-Spearmans the SAME
    ``pred`` against the permuted E0 (breaks alignment, preserves marginals). No
    re-fit → this is a batched rank correlation over the stacked permuted-y.
    Matches the serial ``_rho(pred, y[perm])`` (None→0.0). Returns n_perms ρ.
    """
    dev = torch.device(device)
    predt = torch.from_numpy(np.ascontiguousarray(pred)).to(device=dev, dtype=torch.float64)
    yt = torch.from_numpy(np.ascontiguousarray(y)).to(device=dev, dtype=torch.float64)
    perm_t = torch.from_numpy(np.ascontiguousarray(perm)).to(device=dev, dtype=torch.long)
    B, n = perm.shape
    yperm = yt[perm_t]  # (B, n)
    predB = predt.unsqueeze(0).expand(B, n)  # (B, n) same pred every draw
    pred_std = predt.std()
    y_std = yperm.std(dim=1)
    rho = _spearman_batched(predB, yperm)
    bad = (pred_std < 1e-9) | (y_std < 1e-9) | torch.isnan(rho)
    rho = torch.where(bad, torch.zeros_like(rho), rho)
    return [float(r) for r in rho.detach().cpu().numpy()]
