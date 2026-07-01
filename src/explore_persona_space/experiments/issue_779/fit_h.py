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
    # trace on the train Gram. Use the SVD of Xtr_n for efficiency.
    U, s, _Vt = np.linalg.svd(Xtr_n, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c  # (r, D_out)
    best_lam = lambdas[0]
    best_gcv = np.inf
    for lam in lambdas:
        filt = s2 / (s2 + lam)  # (r,)
        # fitted train values: U @ diag(filt) @ UtY
        Yhat_tr = U @ (filt[:, None] * UtY)
        rss = float(np.sum((Ytr_c - Yhat_tr) ** 2))
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
