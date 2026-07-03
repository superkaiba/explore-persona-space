# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, λ, γ, σ, ×, ≈) in scientific docstrings.
"""Matched-protocol co-fit machinery for issue #763 `neutral-contrast-and-cofit`.

The plan §4.2 UNIFORM protocol, vectorized (the many-cell/many-draw batching
rules — a serial per-cell/per-draw loop is a review blocker):

- **Target:** graded E0 rank-transformed WITHIN each training fold (average
  ranks scaled to [0, 1] — the rank-ridge move for the Spearman objective),
  UNWEIGHTED (uniform per-context; matches the parent's graded ridge, §4.2).
- **Folds:** LOCO over the n contexts, one shared fold list, seed 763.
- **Subspace methods** (ridge / kernel ridge / prompt-side ridge): per-fold PCA
  at FIXED d=10 (`issue_763_pca._pca_fit`, train-fold basis — no held-out
  leakage), standardized (ddof=0 + 1e-9, the #658 convention), λ selected per
  fold by training-fold PRESS (exact LOO identity on the dual Gram); kernel
  ridge = RBF at bandwidth σ = c × median heuristic (training-fold pairwise
  distances), c ∈ {0.5, 1, 2}, (c, λ) selected jointly by the same PRESS.
- **Direction methods** (pv_rA / pv_rC / pv_neutral / crude diff-means / random
  floor): per-layer scalar projection + per-fold 1-D linear fit (sign + scale
  on the training contexts) on the rank target.
- **Selection-symmetric nulls** (#778 rule): EVERY permutation draw re-runs the
  full per-fold refit across ALL layers; the per-draw × per-layer ρ matrix is
  returned for persistence so the layer-max band is spot-checkable.

VECTORIZATION SHAPE: everything label-INDEPENDENT — the per-(layer, fold) PCA
basis, standardization, the linear dual Gram + RBF kernel eigendecompositions,
and the diff-means cross-Gram — is computed ONCE per (layer, fold)
(``LayerCache``) and REUSED across all P permutation draws; per draw only the
rank-transformed train targets (one shared ``fold_rank_targets`` tensor) thread
through cached-eigenbasis PRESS scans, batched solves, and GEMMs. The
``assert_cofit_matches_reference`` exactness gate verifies the batched paths
against naive serial oracles before any behavior is fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import rankdata

from explore_persona_space.analysis.issue_763_pca import _pca_fit, _pca_transform

# The uniform-protocol constants (plan §4.2; λ grid = #763 plans/v3 §6 as-run).
COFIT_PCA_DIM = 10
COFIT_LAMBDAS: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)
COFIT_RBF_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 2.0)


def rank01(y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average-rank transform scaled to [0, 1] along ``axis`` (Conover-Iman)."""
    y = np.asarray(y, dtype=np.float64)
    r = rankdata(y, method="average", axis=axis)
    n = y.shape[axis]
    return (r - 1.0) / max(n - 1, 1)


def batched_spearman(preds: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Row-wise Spearman ρ for ``(P, n)`` prediction/target batches (NaN-guarded).

    Ranks both sides (average ties) then computes the row-wise Pearson of the
    ranks. Rows where either side has no variance return NaN (the caller drops
    them, matching the serial ``_rho`` None-guard).
    """
    P, _n = preds.shape
    rp = rankdata(preds, method="average", axis=1)
    ry = rankdata(ys, method="average", axis=1)
    rp = rp - rp.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    num = (rp * ry).sum(axis=1)
    den = np.sqrt((rp**2).sum(axis=1) * (ry**2).sum(axis=1))
    out = np.full(P, np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def loco_train_indices(n: int) -> list[list[int]]:
    """The shared LOCO fold list: fold i trains on every context except i."""
    return [[j for j in range(n) if j != i] for i in range(n)]


def fold_rank_targets(y_batch: np.ndarray) -> np.ndarray:
    """Per-fold rank-transformed train targets, shared by every method.

    ``y_batch``: (P, n). Returns ``R`` (P, n, n-1) with ``R[p, i] =
    rank01(y_batch[p, tr_i])`` — the WITHIN-training-fold rank transform the
    uniform protocol mandates, computed once and reused across all layers and
    methods (the rank transform is layer-independent).
    """
    P, n = y_batch.shape
    tr_idx = loco_train_indices(n)
    R = np.empty((P, n, n - 1), dtype=np.float64)
    for i in range(n):
        R[:, i, :] = rank01(y_batch[:, tr_idx[i]], axis=1)
    return R


# ── per-(layer, fold) label-independent caches ────────────────────────────────


@dataclass
class _FoldKernelCache:
    """Label-independent PRESS structure for one (layer, fold), all kernels.

    ``kernels[k] = (Q, evals, Qsq, K, k_held, label)`` — the eigendecomposed
    train-fold kernel matrix + the held-out kernel row, for the linear dual
    Gram (ridge) and each RBF candidate (kernel ridge). All depend only on the
    layer's X (PCA basis, standardization, pairwise distances), never labels.
    """

    kernels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]]
    median_dist: float


@dataclass
class LayerCache:
    """All label-independent per-fold structure for one (behavior, layer).

    Built once per layer and shared across every method, the observed read, and
    every null battery (the label-independence win — #778 / the vectorize
    rule). ``G_folds[i]`` is the diff-means cross-Gram ``x @ x[tr_i].T``.
    """

    folds: list[_FoldKernelCache]
    G_folds: list[np.ndarray]  # per fold: (n, n-1) cross-Gram for diff-means
    n: int

    @property
    def device(self) -> torch.device:
        """The device every cached kernel tensor lives on — THE battery fit device.

        Every label/target tensor entering the battery math derives its device
        from here (crash-fix r4, pod-763: labels built on a default-"cpu"
        ``device`` kwarg crashed ``Q.t() @ Ytr`` against cuda:0 kernels).
        """
        assert self.folds and self.folds[0].kernels, "empty LayerCache — no device to derive"
        return self.folds[0].kernels[0][0].device

    @staticmethod
    def build(
        x_layer: np.ndarray,
        *,
        dim: int = COFIT_PCA_DIM,
        rbf_multipliers: tuple[float, ...] = COFIT_RBF_MULTIPLIERS,
        device: str = "cpu",
    ) -> LayerCache:
        """Fit per-fold PCA + standardization + kernel eigendecompositions.

        ``x_layer``: (n, H) fp64. PCA at min(dim, rank) per train fold (the
        FIXED d=10 protocol; clamped only by tiny smoke slices), standardize
        (ddof=0 + 1e-9), then eigendecompose the linear dual Gram and each RBF
        kernel at σ = c × median(train pairwise distances).
        """
        dev = torch.device(device)
        x = np.asarray(x_layer, dtype=np.float64)
        n = x.shape[0]
        tr_idx = loco_train_indices(n)
        folds: list[_FoldKernelCache] = []
        G_folds: list[np.ndarray] = []
        for i in range(n):
            tr = tr_idx[i]
            mu_p, comps = _pca_fit(x[tr], min(dim, len(tr) - 1))
            z_tr = _pca_transform(x[tr], mu_p, comps)  # (t, d)
            z_held = _pca_transform(x[i : i + 1], mu_p, comps)[0]  # (d,)
            Xtr = torch.from_numpy(np.ascontiguousarray(z_tr)).to(dev, torch.float64)
            x_mu = Xtr.mean(0)
            x_sd = Xtr.std(0, correction=0) + 1e-9  # #658 ddof=0 convention
            Xtr_n = (Xtr - x_mu) / x_sd  # (t, d)
            zh = torch.from_numpy(np.ascontiguousarray(z_held)).to(dev, torch.float64)
            zh_n = (zh - x_mu) / x_sd  # (d,)

            kernels = []
            # linear dual Gram (the ridge method's kernel)
            G = Xtr_n @ Xtr_n.t()
            evals, Q = torch.linalg.eigh(G)
            kernels.append((Q, evals, Q * Q, G, Xtr_n @ zh_n, "linear"))
            # RBF kernels at c × median heuristic (train pairwise distances)
            d2 = torch.cdist(Xtr_n, Xtr_n).pow(2)  # (t, t)
            dists = d2.sqrt()
            triu = torch.triu_indices(d2.shape[0], d2.shape[1], offset=1, device=dev)
            off = dists[triu[0], triu[1]]
            pos = off[off > 0]
            med = float(pos.median()) if pos.numel() else 1.0
            d2_held = (Xtr_n - zh_n.unsqueeze(0)).pow(2).sum(dim=1)  # (t,)
            for c in rbf_multipliers:
                sigma = max(c * med, 1e-9)
                K = torch.exp(-d2 / (2.0 * sigma**2))
                ev_k, Q_k = torch.linalg.eigh(K)
                k_held = torch.exp(-d2_held / (2.0 * sigma**2))
                kernels.append((Q_k, ev_k, Q_k * Q_k, K, k_held, f"rbf_c{c}"))
            folds.append(_FoldKernelCache(kernels=kernels, median_dist=med))
            G_folds.append(x @ x[tr].T)  # (n, t) diff-means cross-Gram
        return LayerCache(folds=folds, G_folds=G_folds, n=n)


def kernel_loco_preds(
    cache: LayerCache,
    R: np.ndarray,
    *,
    kernel_labels: tuple[str, ...],
    lambdas: tuple[float, ...] = COFIT_LAMBDAS,
) -> np.ndarray:
    """Batched PRESS-selected (kernel, λ) LOCO predictions for all P draws.

    ``R``: (P, n, n-1) fold rank targets. For each fold and each draw, the
    (kernel, λ) pair is selected by the exact LOO PRESS identity on the cached
    eigenbasis (``Ŷ = Q diag(g/(g+λ)) Qᵀ Y``, ``h = Q² (g/(g+λ))``, ``loo =
    (Y − Ŷ)/(1 − h)``), then the held-out prediction is ``k_heldᵀ α`` with
    ``α = (K + λI)⁻¹ Y`` — draws grouped by their selected pair so each distinct
    pair solves ONE batched system. ``kernel_labels=("linear",)`` is the ridge
    method; ``("rbf_c0.5", "rbf_c1.0", "rbf_c2.0")`` is the kernel-ridge method.

    The compute device is DERIVED from the cache (``cache.device``) — there is
    deliberately NO ``device`` kwarg a caller could desync: the r4 production
    crash (pod-763) came from a default-"cpu" kwarg leaving ``Ytr`` on cpu
    against cuda:0 cached kernels under ``EPM_FIT_DEVICE=cuda``.
    """
    dev = cache.device
    P, n, _t = R.shape
    lam_t = torch.tensor(lambdas, dtype=torch.float64, device=dev)
    out = torch.empty(P, n, dtype=torch.float64, device=dev)
    for i in range(n):
        fold = cache.folds[i]
        kernels = [k for k in fold.kernels if k[5] in kernel_labels]
        assert kernels, f"no kernels matching {kernel_labels}"
        Ytr = torch.from_numpy(np.ascontiguousarray(R[:, i, :].T)).to(dev)  # (t, P)
        t = Ytr.shape[0]
        best_mse = torch.full((P,), float("inf"), dtype=torch.float64, device=dev)
        best_kern = torch.zeros(P, dtype=torch.long, device=dev)
        best_lam = torch.zeros(P, dtype=torch.long, device=dev)
        for ki, (Q, evals, Qsq, _K, _kh, _label) in enumerate(kernels):
            QtY = Q.t() @ Ytr  # (t, P)
            for li in range(lam_t.shape[0]):
                filt = evals / (evals + lam_t[li])
                h_diag = Qsq @ filt  # (t,)
                Yhat = Q @ (filt.unsqueeze(1) * QtY)  # (t, P)
                denom = (1.0 - h_diag).clamp(min=1e-8).unsqueeze(1)
                loo = (Ytr - Yhat) / denom
                mse = (loo * loo).mean(dim=0)  # (P,)
                better = mse < best_mse
                best_kern = torch.where(better, torch.full_like(best_kern, ki), best_kern)
                best_lam = torch.where(better, torch.full_like(best_lam, li), best_lam)
                best_mse = torch.where(better, mse, best_mse)
        pred_i = torch.empty(P, dtype=torch.float64, device=dev)
        eye = torch.eye(t, dtype=torch.float64, device=dev)
        for ki, (_Q, _ev, _Qsq, K, k_held, _label) in enumerate(kernels):
            for li in range(lam_t.shape[0]):
                sel = (best_kern == ki) & (best_lam == li)
                if not bool(sel.any()):
                    continue
                A = K + lam_t[li] * eye
                alpha = torch.linalg.solve(A, Ytr[:, sel])  # (t, k)
                pred_i[sel] = k_held @ alpha  # (k,)
        out[:, i] = pred_i
    return out.detach().cpu().numpy()


def direction_loco_preds(s: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Per-fold 1-D linear fit (sign + scale) of the rank target on a projection.

    ``s``: (n,) the fixed per-layer scalar projection ``r[ℓ] · v0(C)[ℓ]``;
    ``R``: (P, n, n-1) fold rank targets. Closed-form simple regression per
    (draw, fold), fully broadcast: ``a = cov(s_tr, y_r)/var(s_tr)``,
    ``b = ȳ_r − a·s̄_tr``, prediction ``a·s_i + b``. A zero-variance train
    projection degrades to the train-mean prediction (a = 0).
    """
    _p, n, _t = R.shape
    tr_idx = loco_train_indices(n)
    S_tr = np.stack([s[tr] for tr in tr_idx])  # (n, t)
    s_mean = S_tr.mean(axis=1)  # (n,)
    s_var = (S_tr**2).mean(axis=1) - s_mean**2  # (n,)
    y_mean = R.mean(axis=2)  # (P, n)
    cov = (R * S_tr[None]).mean(axis=2) - y_mean * s_mean[None]  # (P, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(s_var[None] > 1e-18, cov / s_var[None], 0.0)
    b = y_mean - a * s_mean[None]
    return a * s[None, :] + b  # (P, n)


def diffmeans_loco_preds(cache: LayerCache, R: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
    """Crude supervised diff-means (mass-mean, Marks & Tegmark 2310.06824), fold-refit.

    Within each training fold the contexts are median-split on the (permuted)
    graded target: ``direction = mean(v0 | top half) − mean(v0 | bottom half)``
    — expressed as a weight vector over train rows, so the projection of EVERY
    context is one GEMM against the cached cross-Gram ``x @ x[tr]ᵀ``. Split
    rule (deterministic): sort the t=n−1 train contexts by target; bottom =
    first ⌊t/2⌋, top = last ⌊t/2⌋ (the middle context of an odd fold is in
    neither half). The projection then gets the same per-fold 1-D fit as every
    other direction method.
    """
    P, n, t = R.shape
    tr_idx = loco_train_indices(n)
    k = t // 2
    preds = np.empty((P, n), dtype=np.float64)
    w = np.zeros((P, t), dtype=np.float64)
    for i in range(n):
        y_tr = y_batch[:, tr_idx[i]]  # (P, t)
        order = np.argsort(y_tr, axis=1, kind="stable")  # (P, t)
        w[:] = 0.0
        np.put_along_axis(w, order[:, :k], -1.0 / k, axis=1)
        np.put_along_axis(w, order[:, t - k :], 1.0 / k, axis=1)
        proj = w @ cache.G_folds[i].T  # (P, n) — every context's projection
        s_tr = proj[:, tr_idx[i]]  # (P, t)
        y_r = R[:, i, :]  # (P, t)
        s_mean = s_tr.mean(axis=1)
        s_var = (s_tr**2).mean(axis=1) - s_mean**2
        y_mean = y_r.mean(axis=1)
        cov = (y_r * s_tr).mean(axis=1) - y_mean * s_mean
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(s_var > 1e-18, cov / s_var, 0.0)
        b = y_mean - a * s_mean
        preds[:, i] = a * proj[:, i] + b
    return preds


def random_unit_directions(k: int, hidden: int, seed: int) -> np.ndarray:
    """K seeded Gaussian unit directions in the hidden space -> (K, H)."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((k, hidden))
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


# ── exactness gate (batched vs naive serial oracles) ─────────────────────────


def _serial_rank_ridge_reference(
    x: np.ndarray, y: np.ndarray, lambdas: tuple[float, ...], dim: int
) -> np.ndarray:
    """Naive serial oracle for the rank-target PRESS ridge (exactness gate only)."""
    n = x.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        mu_p, comps = _pca_fit(x[tr], min(dim, len(tr) - 1))
        z_tr = _pca_transform(x[tr], mu_p, comps)
        z_held = _pca_transform(x[i : i + 1], mu_p, comps)[0]
        Xtr = torch.from_numpy(z_tr).to(torch.float64)
        x_mu = Xtr.mean(0)
        x_sd = Xtr.std(0, correction=0) + 1e-9
        Xn = (Xtr - x_mu) / x_sd
        y_r = torch.from_numpy(rank01(y[tr])).to(torch.float64).unsqueeze(1)
        G = Xn @ Xn.t()
        evals, Q = torch.linalg.eigh(G)
        best_mse, best_lam = float("inf"), lambdas[0]
        for lam in lambdas:
            filt = evals / (evals + lam)
            h = (Q * Q) @ filt
            Yhat = Q @ (filt.unsqueeze(1) * (Q.t() @ y_r))
            loo = (y_r - Yhat) / (1.0 - h).clamp(min=1e-8).unsqueeze(1)
            mse = float((loo * loo).mean())
            if mse < best_mse:
                best_mse, best_lam = mse, lam
        alpha = torch.linalg.solve(G + best_lam * torch.eye(len(tr), dtype=torch.float64), y_r)
        zh = torch.from_numpy(z_held).to(torch.float64)
        zh_n = (zh - x_mu) / x_sd
        preds[i] = float((Xn @ zh_n) @ alpha)
    return preds


def _serial_direction_reference(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Naive serial oracle for the per-fold 1-D direction fit (exactness gate)."""
    n = s.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        y_r = rank01(y[tr])
        s_tr = s[tr]
        var = s_tr.var()
        a = ((s_tr * y_r).mean() - s_tr.mean() * y_r.mean()) / var if var > 1e-18 else 0.0
        b = y_r.mean() - a * s_tr.mean()
        preds[i] = a * s[i] + b
    return preds


def assert_cofit_matches_reference(
    seed: int = 0, n: int = 18, h: int = 30, *, device: str = "cpu"
) -> dict:
    """HARD exactness gate: batched co-fit paths vs naive serial oracles.

    Small synthetic LOCO problem, 2-draw batch (identity + one shuffle):
    (a) batched rank-target PRESS ridge (linear kernel) vs the serial oracle —
        tol 1e-8 (identical closed form, fp reduction order only);
    (b) batched per-fold 1-D direction fit vs the serial oracle — tol 1e-10.
    Raises AssertionError on any miss; the driver runs this at start-up before
    any behavior is fit (the batched-path-never-trusted-ungated discipline).

    ``device`` is the ON-LANE leg (crash-fix r4): the driver passes
    ``FIT_DEVICE`` so the cache is built — and ``kernel_loco_preds`` therefore
    runs — on the SAME device the production battery uses, while the serial
    oracles stay cpu. A future device-threading miss on the battery path fails
    HERE (seconds, at start-up), not at battery time.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    W = rng.standard_normal((3, h))
    x = (z @ W + 0.2 * rng.standard_normal((n, h))).astype(np.float64)
    y = (x @ rng.standard_normal(h) * 0.1 + rng.standard_normal(n) * 0.5).astype(np.float64)
    perm = rng.permutation(n)
    y_batch = np.stack([y, y[perm]])
    R = fold_rank_targets(y_batch)
    cache = LayerCache.build(x, dim=5, device=device)

    ridge_bat = kernel_loco_preds(cache, R, kernel_labels=("linear",))
    ridge_ref0 = _serial_rank_ridge_reference(x, y, COFIT_LAMBDAS, dim=5)
    ridge_ref1 = _serial_rank_ridge_reference(x, y[perm], COFIT_LAMBDAS, dim=5)
    d_r = max(
        float(np.max(np.abs(ridge_bat[0] - ridge_ref0))),
        float(np.max(np.abs(ridge_bat[1] - ridge_ref1))),
    )
    assert d_r <= 1e-8, f"batched rank-ridge exactness FAILED: max|Δpred|={d_r:.3e} > 1e-8"

    direction = rng.standard_normal(h)
    s = x @ direction
    dir_bat = direction_loco_preds(s, R)
    dir_ref0 = _serial_direction_reference(s, y)
    dir_ref1 = _serial_direction_reference(s, y[perm])
    d_d = max(
        float(np.max(np.abs(dir_bat[0] - dir_ref0))),
        float(np.max(np.abs(dir_bat[1] - dir_ref1))),
    )
    assert d_d <= 1e-10, f"batched direction-fit exactness FAILED: max|Δpred|={d_d:.3e} > 1e-10"
    return {"ridge_delta": d_r, "direction_delta": d_d, "device": str(cache.device)}
