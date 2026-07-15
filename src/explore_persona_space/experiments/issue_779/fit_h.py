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


def ridge_fit_predict_fast(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Torch-eigh Gram-space ridge — fast APPROXIMATE twin of
    :func:`ridge_fit_predict` (same standardize-X / center-Y / GCV-lambda-select /
    un-center recipe). PARITY IS SIZE-DEPENDENT: the #779 Read-1 gate measured
    ~8e-13 agreement at n_train=500, but a live #823 full-size slice
    (2026-07-02, n_train~3998, H=3584) measured max rel diff ~1.7e-5 vs the SVD
    path — the Gram squares the condition number, so precision degrades with n.
    NOT a default-on substitute: callers MUST run a full-size slow-vs-fast
    parity gate on their own inputs (e.g. run_823._ridge_equivalence_gate) and
    ship it only when the gate passes their tolerance.

    Ported VERBATIM from ``scripts/issue779_percontext_recon.py::_ridge_fit_predict_fast``
    (the #779 fast path that cut 140 CV fits from ~5 h to ~28 min) into the shared
    fitter module for the #823 phase-4 perf patch, with one addition: an optional
    torch ``device`` for the eigh/matmuls (float64 throughout; predictions return
    as CPU numpy). ``ridge_fit_predict`` runs a full ``numpy.linalg.svd`` of the
    (N_tr, H) train matrix plus 13 full-size GCV matmuls per call; this computes
    the DUAL ridge via ``torch.linalg.eigh`` of the (N_tr, N_tr) Gram and
    evaluates GCV RSS in eigen-coefficient space (no full train-fit
    reconstruction). Callers MUST engage a slow-vs-fast parity gate on first use
    per input family (vectorize-many-cell-fits.md rule 5).
    """
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 13)
    dev = torch.device(device)
    Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float64).to(dev)
    Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64).to(dev)
    Xev = torch.as_tensor(np.asarray(X_eval), dtype=torch.float64).to(dev)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9  # matches ridge_fit_predict's 1e-9 (numpy .std is population)
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    ntr = Xtr.shape[0]

    # Dual ridge: (G + lam I) alpha = Ytr_c, G = Xtr_n Xtr_n^T = V diag(w) V^T.
    G = Xtr_n @ Xtr_n.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    VtY = V.T @ Ytr_c  # (ntr, H)
    Kev = Xev_n @ Xtr_n.T  # (n_ev, ntr) cross-kernel
    KevV = Kev @ V
    sqVtY = (VtY**2).sum(1)  # per-eigencomponent target energy
    tot = float((Ytr_c**2).sum())

    # GCV: RSS(lam) = ||Y||^2 - sum_k (2 f_k - f_k^2) sqVtY_k with f = w/(w+lam),
    # dof = sum_k f_k (hat-matrix trace); GCV = RSS / (ntr - dof)^2.
    best_lam = float(lambdas[0])
    best_gcv = float("inf")
    for lam in lambdas:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = float(lam)
    filt = 1.0 / (w + best_lam)
    pred = (KevV * filt) @ VtY + ymu
    return pred.cpu().numpy()


def ridge_fit_predict_fast_layer_batched(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    *,
    lambdas: np.ndarray | None = None,
    device: str = "cpu",
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """LAYER-BATCHED Gram-eigh ridge — one batched eigh over a leading axis.

    Vectorizes :func:`ridge_fit_predict_fast` over a leading layer/cell axis
    (the #1332 source-module fix per artifact-reuse check (i): the parent
    scripts loop the per-(family, layer, fold) fits serially; here the layer
    axis is ONE batched ``torch.linalg.eigh`` + batched matmuls). Numerically
    the SAME recipe per slice: standardize-X on train stats, center-Y, GCV
    lambda selected PER SLICE over ``lambdas`` (slices may select different
    lambdas, matching per-call GCV), dual Gram-space solve, un-centered
    predictions. float64 throughout.

    PARITY IS SIZE-DEPENDENT (same caveat as the fast twin): callers MUST run
    a slow-vs-fast parity gate vs :func:`ridge_fit_predict` on >=3 slices at
    their production shape (tolerance per the fast twin's docstring; #1332
    uses max rel diff <= 1e-4 at n_train~320, d=3584) and fall back to the
    canonical solver when the gate fails.

    Args:
        X_train: (L, n_tr, d) inputs per slice.
        Y_train: (L, n_tr, d_out) targets per slice.
        X_eval: (L, n_ev, d) eval inputs per slice.
        lambdas: GCV grid (default logspace(-2, 4, 13) — the #823 grid).
        device: torch device for the eigh/matmuls.
        return_weights: also return standardized-input-space primal weights
            ``W`` (L, d, d_out) reconstructed from the dual coefficients
            (descriptive weight-space reads ONLY — the #1332 settled decision).

    Returns:
        preds (L, n_ev, d_out) as CPU numpy; optionally (preds, W).
    """
    if lambdas is None:
        lambdas = np.logspace(-2, 4, 13)
    dev = torch.device(device)
    Xtr = torch.as_tensor(np.asarray(X_train), dtype=torch.float64).to(dev)
    Ytr = torch.as_tensor(np.asarray(Y_train), dtype=torch.float64).to(dev)
    Xev = torch.as_tensor(np.asarray(X_eval), dtype=torch.float64).to(dev)
    assert Xtr.ndim == 3 and Ytr.ndim == 3 and Xev.ndim == 3, (Xtr.shape, Ytr.shape, Xev.shape)
    n_slices, ntr, _d = Xtr.shape

    xmu = Xtr.mean(dim=1, keepdim=True)  # (L, 1, d)
    xsd = Xtr.std(dim=1, keepdim=True, unbiased=False) + 1e-9  # population std (twin parity)
    Xtr_n = (Xtr - xmu) / xsd
    Xev_n = (Xev - xmu) / xsd
    ymu = Ytr.mean(dim=1, keepdim=True)  # (L, 1, d_out)
    Ytr_c = Ytr - ymu

    G = Xtr_n @ Xtr_n.transpose(1, 2)  # (L, n_tr, n_tr)
    w, V = torch.linalg.eigh(G)  # (L, n_tr), (L, n_tr, n_tr)
    w = torch.clamp(w, min=0.0)
    VtY = V.transpose(1, 2) @ Ytr_c  # (L, n_tr, d_out)
    Kev = Xev_n @ Xtr_n.transpose(1, 2)  # (L, n_ev, n_tr)
    sqVtY = (VtY**2).sum(dim=2)  # (L, n_tr)
    tot = (Ytr_c**2).sum(dim=(1, 2))  # (L,)

    # GCV per slice: rss(lam) = tot - sum_k (2 f_k - f_k^2) sqVtY_k, f = w/(w+lam).
    gcv_all = torch.empty((n_slices, len(lambdas)), dtype=torch.float64, device=dev)
    for li, lam in enumerate(lambdas):
        filt = w / (w + float(lam))  # (L, n_tr)
        rss = tot - ((2 * filt - filt**2) * sqVtY).sum(dim=1)  # (L,)
        dof = filt.sum(dim=1)  # (L,)
        denom = (ntr - dof) ** 2
        gcv_all[:, li] = torch.where(denom > 1e-12, rss / denom, torch.full_like(rss, float("inf")))
    best_idx = gcv_all.argmin(dim=1)  # (L,)
    best_lam = torch.as_tensor(np.asarray(lambdas), dtype=torch.float64, device=dev)[best_idx]

    filt = 1.0 / (w + best_lam[:, None])  # (L, n_tr)
    alpha = V @ (filt[:, :, None] * VtY)  # (L, n_tr, d_out) dual coefficients
    preds = Kev @ alpha + ymu  # (L, n_ev, d_out)
    if not return_weights:
        return preds.cpu().numpy()
    W = Xtr_n.transpose(1, 2) @ alpha  # (L, d, d_out) standardized-input-space weights
    return preds.cpu().numpy(), W.cpu().numpy()


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
