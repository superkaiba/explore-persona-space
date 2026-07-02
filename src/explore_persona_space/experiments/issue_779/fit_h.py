"""Issue #779 R1 fitters: the context->profile map ``h`` + the direct predictor ``g``.

``h: c_x -> v(x)`` reconstructs the mean-response profile from the pre-generation
context vector (behavior-agnostic). ``g: c_x -> trait_score`` is the
matched-capacity direct predictor (behavior-specific, the pivotal control).

Both are trained on the LMSYS train contexts (pass B) and applied to the PV
eval contexts (pass A), held out. Uses the #722 vectorized helpers
(``ridge_predict_loco_raw`` closed-form ridge, ``fit_batched_loco_mlp_multihead``
batched MLP with a PCA-k target head) — NEVER a serial per-cell fit
(vectorize-many-cell-fits.md).

Fit-then-apply shape (NOT the helper's LOCO shape): here ``h``/``g`` train on the
train corpus and predict on a DISJOINT eval set, so we fit ONE model on the full
train set and apply it to eval. The vectorized MLP helper is LOCO-shaped
(leave-one-context-out on a single set); for the train->eval application we use a
plain batched fit (train on all train rows, predict eval rows) built on the same
primitives, plus a small LOCO reference on 2-3 train cells for the
vectorized-reproduces-serial parity check (rule 5).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from explore_persona_space.analysis.vectorized_mlp_skill import (
    MLP_HIDDEN,
    MLP_LR,
    MLP_MAX_EPOCHS,
    MLP_WD,
    robust_pca_basis,
)

logger = logging.getLogger("issue779.fit_h")


# ── ridge (closed-form, GCV lambda) ───────────────────────────────────────────


class RidgeFitCore:
    """Eigh-based closed-form GCV ridge fit (standardize-X / center-Y), reusable.

    Ports the equivalence-verified torch-eigh recipe of
    ``scripts/issue779_percontext_recon.py::_ridge_fit_predict_fast`` (v79 fix 4:
    identical predictions + identical selected lambda vs the numpy-SVD path) and
    extends it with:

      (a) a PRIMAL (H x H Gram) form for N > H, so the eigh is never larger than
          min(N, H) — the numpy ``gesdd`` SVD ran at ~11 GFLOPS effective on this
          VM vs ~165 GFLOPS for the Gram GEMMs (16:12Z audit);
      (b) low-rank W factors ``P (H, r)`` / ``Q (r, D)`` with ``W = P @ Q`` for
          the pv_pinv compact SVD (``scaling_grid.pv_pinv_svd``);
      (c) ``predict_scalar`` so a cell's direct predictor g shares the SAME
          decomposition as its h fit (v79 fix 3 — no second SVD on the same X).

    GCV RSS uses the exact identity (v79 fix 2 — no per-lambda (N, D) Yhat
    materialization, which was ~78% of the old fit FLOPs):

        rss(lam) = ||Y_c||_F^2 - sum_k (2 f_k - f_k^2) * e_k,
        f_k = w_k / (w_k + lam),  e_k = per-eigendirection target energy
        (= ||U^T Y_c||_k^2 in SVD terms; w_k = s_k^2).

    Standardization matches the numpy path exactly (POPULATION std,
    ``correction=0`` — torch's default sample std would rescale lambda by
    n/(n-1) at small n). Deterministic; float64 throughout. Numerical agreement
    with the SVD reference is gated by ``scaling_grid.verify_live_ridge`` (v79
    fix 6) and pinned in ``tests/test_issue779_scaling_grid.py``.

    Attributes: ``n, h, form ('dual'|'primal'), gram_n, lam, xmu, xsd, ymu
    (numpy), P, Q (numpy)``.
    """

    def __init__(
        self, X_train: np.ndarray, Y_train: np.ndarray, *, lambdas: np.ndarray | None = None
    ):
        if lambdas is None:
            lambdas = np.logspace(-2, 4, 13)
        Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float64)
        Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64)
        if Ytr.ndim == 1:
            Ytr = Ytr[:, None]
        n, h = Xtr.shape
        self.n, self.h = int(n), int(h)
        xmu = Xtr.mean(0)
        xsd = Xtr.std(0, correction=0) + 1e-9  # POPULATION std == numpy .std (exactness)
        Xn = (Xtr - xmu) / xsd
        ymu = Ytr.mean(0)
        Yc = Ytr - ymu
        self._Xn = Xn
        self._lambdas = np.asarray(lambdas, dtype=np.float64)

        if n <= h:
            # DUAL form: eigh of the (N, N) Gram X X^T; eigenvalues w == s^2.
            self.form = "dual"
            G = Xn @ Xn.T
            w, V = torch.linalg.eigh(G)
            w = torch.clamp(w, min=0.0)
            self.gram_n = int(n)
            VtY = V.T @ Yc  # (n, D) == U^T Y_c in SVD terms
            e = (VtY**2).sum(1)
            lam = self._gcv_select(w, e, float((Yc**2).sum()))
            inv = 1.0 / (w + lam)
            # W = Xn^T (G + lam I)^{-1} Yc = P @ Q, rank <= n.
            P_t = Xn.T @ V  # (h, n)
            Q_t = inv[:, None] * VtY  # (n, D)
        else:
            # PRIMAL form: eigh of the (H, H) Gram X^T X; eigenvalues w == s^2.
            self.form = "primal"
            G = Xn.T @ Xn
            w, V = torch.linalg.eigh(G)
            w = torch.clamp(w, min=0.0)
            self.gram_n = int(h)
            A = Xn.T @ Yc  # (h, D)
            B = V.T @ A  # (h, D); B_k = s_k * (U^T Y_c)_k
            e = self._primal_energies(w, B)
            lam = self._gcv_select(w, e, float((Yc**2).sum()))
            inv = 1.0 / (w + lam)
            # W = V diag(1/(w+lam)) V^T A = P @ Q.
            P_t = V  # (h, h)
            Q_t = inv[:, None] * B  # (h, D)
        self.lam = float(lam)
        self._w, self._V = w, V
        self._P_t, self._Q_t = P_t, Q_t
        self.xmu = xmu.numpy()
        self.xsd = xsd.numpy()
        self.ymu = ymu.numpy().squeeze() if ymu.numel() > 1 else float(ymu)
        self.P = P_t.numpy()
        self.Q = Q_t.numpy()
        self._W_cache: np.ndarray | None = None

    @staticmethod
    def _primal_energies(w: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """e_k = ||B_k||^2 / w_k with a zero-eigenvalue guard (B_k -> 0 as w_k -> 0,
        matching the SVD path where a zero singular value contributes nothing)."""
        sq = (B**2).sum(1)
        w_max = float(w.max()) if w.numel() else 0.0
        if w_max <= 0.0:
            return torch.zeros_like(sq)
        safe = torch.where(w > w_max * 1e-12, w, torch.inf)
        return sq / safe

    def _gcv_select(self, w: torch.Tensor, e: torch.Tensor, tot: float) -> float:
        """First-strictly-smaller GCV lambda over the grid (matches the SVD loop)."""
        n = self.n
        best_lam = float(self._lambdas[0])
        best_gcv = float("inf")
        for lam in self._lambdas:
            filt = w / (w + float(lam))
            rss = tot - float(((2.0 * filt - filt**2) * e).sum())
            dof = float(filt.sum())
            denom = (n - dof) ** 2
            gcv = rss / denom if denom > 1e-12 else float("inf")
            if gcv < best_gcv:
                best_gcv = gcv
                best_lam = float(lam)
        return best_lam

    def W(self) -> np.ndarray:
        """The (H, D) standardized-input weight matrix, formed lazily (P @ Q).

        Grid cells never call this (v79 fix — they read predictions via the
        factors); only ``run_arm_comparison`` needs W for the pv_pinv reads."""
        if self._W_cache is None:
            self._W_cache = (self._P_t @ self._Q_t).numpy()
        return self._W_cache

    def predict(self, X_eval: np.ndarray) -> np.ndarray:
        """Ridge predictions on X_eval via the low-rank factors (never forms W)."""
        Xev = torch.as_tensor(np.asarray(X_eval), dtype=torch.float64)
        Xev_n = (Xev - torch.as_tensor(self.xmu)) / torch.as_tensor(self.xsd)
        pred = (Xev_n @ self._P_t) @ self._Q_t + torch.as_tensor(np.atleast_1d(self.ymu))
        return pred.numpy()

    def predict_scalar(self, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
        """Scalar-target ridge (the direct predictor g) SHARING this fit's
        decomposition — mathematically identical to ``ridge_fit_predict`` on the
        SAME rows (same standardization, same GCV grid, own selected lambda).
        Requires ``len(y_train) == self.n`` (the h fit's rows)."""
        y = torch.as_tensor(np.asarray(y_train), dtype=torch.float64)
        assert y.shape == (self.n,), (y.shape, self.n)
        ymu_s = y.mean()
        yc = y - ymu_s
        Xev = torch.as_tensor(np.asarray(X_eval), dtype=torch.float64)
        Xev_n = (Xev - torch.as_tensor(self.xmu)) / torch.as_tensor(self.xsd)
        if self.form == "dual":
            Vty = self._V.T @ yc  # (n,)
            e = Vty**2
            lam = self._gcv_select(self._w, e, float((yc**2).sum()))
            w_vec = self._P_t @ (Vty / (self._w + lam))  # (h,)
        else:
            a = self._Xn.T @ yc  # (h,)
            b = self._V.T @ a
            e = self._primal_energies(self._w, b[:, None])
            lam = self._gcv_select(self._w, e, float((yc**2).sum()))
            w_vec = self._V @ (b / (self._w + lam))
        return (Xev_n @ w_vec + ymu_s).numpy()


def ridge_fit_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
) -> np.ndarray:
    """Ridge fit on (X_train, Y_train), predict X_eval. GCV lambda selection.

    Standardizes X on train stats, centers Y on train mean, picks lambda by
    Generalized Cross-Validation (GCV) over ``lambdas``, returns un-centered
    predictions (N_eval, D_out). Deterministic closed form. Handles multi-output
    Y (D_out >= 1) — the same ridge weights predict all output dims.
    """
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 13)
    Xtr = np.asarray(X_train, dtype=np.float64)
    Ytr = np.asarray(Y_train, dtype=np.float64)
    Xev = np.asarray(X_eval, dtype=np.float64)
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
        squeeze = True
    else:
        squeeze = False
    n = Xtr.shape[0]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu

    # GCV: pick lambda minimizing mean GCV over output dims via the hat-matrix
    # trace on the train Gram. Use the SVD of Xtr_n for efficiency. RSS per
    # lambda uses the exact identity
    #   rss(lam) = ||Y_c||_F^2 - sum_j (2 f_j - f_j^2) ||UtY_j||^2,
    # f_j = s_j^2/(s_j^2+lam) — algebraically equal to the materialized
    # ||Y_c - U diag(f) UtY||_F^2 (residual = U (I-f) UtY + Y_perp, Y_perp
    # orthogonal to col(U)), WITHOUT the per-lambda (N, D) Yhat GEMM that was
    # ~78% of the fit FLOPs (v79 fix 2; pinned by
    # test_ridge_fit_predict_gcv_identity_matches_materialized_reference).
    U, s, _Vt = np.linalg.svd(Xtr_n, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c  # (r, D_out)
    e = np.sum(UtY**2, axis=1)  # (r,) per-direction target energy
    tot = float(np.sum(Ytr_c**2))
    best_lam = lambdas[0]
    best_gcv = np.inf
    for lam in lambdas:
        filt = s2 / (s2 + lam)  # (r,)
        rss = tot - float(np.sum((2.0 * filt - filt**2) * e))
        dof = float(np.sum(filt))
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else np.inf
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam
    # dual ridge weights at best_lam: w = Vt.T diag(s/(s2+lam)) UtY  (d, D_out)
    # predict eval: Xev_n @ w + ymu
    filt = s / (s2 + best_lam)
    W = (_Vt.T * filt) @ UtY  # (d, D_out)
    preds = Xev_n @ W + ymu
    return preds[:, 0] if squeeze else preds


# ── MLP (batched multi-head, train->eval application) ─────────────────────────


def mlp_fit_predict(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    hidden: int = MLP_HIDDEN,
    lr: float = MLP_LR,
    wd: float = MLP_WD,
    max_epochs: int = MLP_MAX_EPOCHS,
    pca_k: int = 64,
    seed: int = 42,
    device: str = "cpu",
    val_frac: float = 0.1,
    num_threads: int | None = 8,
) -> np.ndarray:
    """1-hidden-512 GELU MLP fit on train, predict eval; PCA-k multi-output head.

    Matches the #722/#658 MLP recipe (AdamW lr 1e-3, wd, <=300 epochs early-stop
    on a val split, PCA-k target head). Multi-output (D_out >= 1) — the target is
    reduced to its top-``pca_k`` PCA components (basis fit on TRAIN), predicted
    jointly, then un-PCA'd. For a scalar target (D_out=1, the direct predictor g)
    pca_k is clamped to 1 and no PCA is applied. Vectorized: ONE batched training
    loop over all output dims (never a per-dim serial net).
    """
    if num_threads is not None and device == "cpu":
        torch.set_num_threads(int(num_threads))
    Xtr = np.asarray(X_train, dtype=np.float32)
    Ytr = np.asarray(Y_train, dtype=np.float32)
    Xev = np.asarray(X_eval, dtype=np.float32)
    scalar = Ytr.ndim == 1
    if scalar:
        Ytr = Ytr[:, None]

    n, d_in = Xtr.shape
    dev = torch.device(device)

    # Standardize X on train stats.
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-6
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd

    # PCA-reduce the target on TRAIN (skip for scalar).
    if scalar or Ytr.shape[1] <= pca_k:
        y_mu = Ytr.mean(0)
        Y_target = Ytr - y_mu
        comps = None
        p = Y_target.shape[1]
    else:
        y_mu, comps, _fb = robust_pca_basis(Ytr, pca_k)  # comps (k', H)
        Y_target = (Ytr - y_mu) @ comps.T  # (n, k')
        p = Y_target.shape[1]

    # Train/val split for early stopping.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, round(val_frac * n))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    Xt = torch.from_numpy(Xtr_n).to(dev)
    Yt = torch.from_numpy(Y_target.astype(np.float32)).to(dev)
    Xe = torch.from_numpy(Xev_n).to(dev)

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, p)
    ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()

    tr_t = torch.from_numpy(tr_idx).to(dev)
    val_t = torch.from_numpy(val_idx).to(dev)
    best_val = float("inf")
    best_state = None
    patience, bad = 20, 0
    for _ep in range(max_epochs):
        net.train()
        opt.zero_grad(set_to_none=True)
        pred = net(Xt[tr_t])
        loss = loss_fn(pred, Yt[tr_t])
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vloss = float(loss_fn(net(Xt[val_t]), Yt[val_t]).item())
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        pred_ev = net(Xe).cpu().numpy()  # (N_eval, p)

    # Un-PCA + un-center.
    pred_full = (pred_ev @ comps + y_mu) if comps is not None else (pred_ev + y_mu)
    return pred_full[:, 0] if scalar else pred_full


# ── readouts ──────────────────────────────────────────────────────────────────


def dot_readout(pred_profile: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """<h(c_x), r_B> — the dot readout (N,). pred_profile (N, H), r_b (H,)."""
    return np.asarray(pred_profile, dtype=np.float64) @ np.asarray(r_b, dtype=np.float64)


def cosine_readout(pred_profile: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """cos(h(c_x), r_B) — the cosine readout (N,) (#493: cosine was the winner)."""
    P = np.asarray(pred_profile, dtype=np.float64)
    rb = np.asarray(r_b, dtype=np.float64)
    num = P @ rb
    den = (np.linalg.norm(P, axis=1) + 1e-12) * (np.linalg.norm(rb) + 1e-12)
    return num / den


# ── reconstruction quality (R3(a) / R1 diagnostic) ────────────────────────────


def reconstruction_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """R2 + mean cosine of predicted vs true profile (N, H). The A4-map test."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    num = np.sum(pred * true, axis=1)
    den = (np.linalg.norm(pred, axis=1) + 1e-12) * (np.linalg.norm(true, axis=1) + 1e-12)
    cos = float(np.mean(num / den))
    return {"r2": r2, "mean_cosine": cos}
