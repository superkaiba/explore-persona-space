# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ℓ, σ, →, ‖·‖, ĥ) in scientific docstrings.
"""Next-activation maps + transport for issue #841.

The four function classes that predict the residual-stream update
``Δ_ℓ = h_{ℓ+1} − h_ℓ`` from the current state, plus the Stage-1 transport
(iterated one-step composition + the GRU roll), plus the atlas metrics.

Classes (all expose a torch ``apply(H) -> raw Δ̂`` for additive transport
composition ``ĥ_{ℓ+1} = ĥ_ℓ + apply(ĥ_ℓ)``):

- ``IdentityMap`` — Δ̂ = 0 (predict-zero null); its composition to ℓ* is
  ``ĥ_{ℓ*} = h_ℓ`` (the Stage-1 row-1b identity-transport input).
- ``RidgeMap`` — affine ``A h + c`` (closed-form ridge, bias + GCV λ over
  ``RIDGE_LAMBDAS``, reusing #658's PRESS / dual solvers).
- ``MLPMap`` — the ``d→hidden→d`` multi-output MLP fit by
  ``vectorized_mlp_skill.fit_batched_split_mlp``.
- ``DepthGRU`` — a 1-layer GRU over the depth axis (EXPLORATORY, prefix-informed;
  excluded from the H2 matched-information criterion).

Target spaces: raw ``Δ_ℓ`` and per-block-RMS-normalized ``Δ_ℓ / σ_m`` (σ_m the
ReSAE block-RMS from the norm curve). Per-transition identity-relative R² is
scale-invariant, so the raw/norm distinction is meaningful only through the
regularizer (ridge λ / MLP wd) and the shared-scale GRU fit. Stage-1 TRANSPORT
uses the RAW-target maps (additive composition needs raw Δ̂).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# parents: [0]=issue_841 [1]=experiments [2]=explore_persona_space [3]=src [4]=repo root.
# The package helper imports issue658_fit_predictors from scripts/, so the repo root
# (parents[4]) is the anchor — NOT parents[3] (=src/), which would yield the nonexistent
# src/scripts. Sentinel-assert so a wrong depth fails loud instead of silently no-op'ing
# (the stage scripts prepend scripts/ themselves, which would mask a broken path here).
_REPO_ROOT = Path(__file__).resolve().parents[4]
assert (_REPO_ROOT / "scripts" / "issue658_fit_predictors.py").is_file(), (
    f"maps.py repo-root anchor wrong: {_REPO_ROOT} has no scripts/issue658_fit_predictors.py"
)
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue658_fit_predictors as _i658  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    RIDGE_LAMBDAS,
    _press_loo_mse_per_lambda,
    _ridge_dual_weights,
)

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    SplitMLPGroup,
    fit_batched_split_mlp,
)

__all__ = [
    "RIDGE_LAMBDAS",
    "DepthGRU",
    "GruSourceOnlyMap",
    "IdentityMap",
    "MLPMap",
    "RidgeMap",
    "delta_error_percentiles",
    "deltas_at",
    "fit_depth_gru",
    "fit_depth_gru_source_only",
    "fit_direct_hop_ridge",
    "fit_ridge_primal",
    "fit_ridge_split",
    "fit_split_mlps",
    "gru_roll",
    "identity_relative_r2",
    "mean_centered_r2",
    "norm_curve",
    "transport_iterated",
]


# ── Δ + norm curve ─────────────────────────────────────────────────────────────


def deltas_at(cx: np.ndarray, transition: int) -> tuple[np.ndarray, np.ndarray]:
    """(h_ℓ, Δ_ℓ) for one transition ℓ=``transition`` from cx (N,28,H).

    Returns ``(h_source (N,H), delta (N,H))`` with ``delta = h_{ℓ+1} − h_ℓ``.
    """
    h = cx[:, transition, :]
    delta = cx[:, transition + 1, :] - cx[:, transition, :]
    return h, delta


def norm_curve(cx: np.ndarray) -> dict:
    """Per-layer/transition norm curve + per-block RMS σ_m (plan §4.1).

    From cx (N,28,H): per-layer ‖h_ℓ‖ (mean over contexts), per-transition
    ‖Δ_ℓ‖, the ratio ‖Δ_ℓ‖/‖h_ℓ‖, adjacent-layer cosine cos(h_ℓ, h_{ℓ+1}), and
    the ReSAE block-RMS σ_m = sqrt(mean Δ_ℓ²) (over contexts AND dims) — the
    normalizer for the RMS-normalized target space. All measured ON Qwen (not
    imported), per 2502.02732.
    """
    n_layers = cx.shape[1]
    h_norm = [float(np.mean(np.linalg.norm(cx[:, li, :], axis=1))) for li in range(n_layers)]
    d_norm, ratio, adj_cos, sigma = [], [], [], []
    for t in range(n_layers - 1):
        h, delta = deltas_at(cx, t)
        dn = np.linalg.norm(delta, axis=1)
        d_norm.append(float(np.mean(dn)))
        ratio.append(float(np.mean(dn / (np.linalg.norm(h, axis=1) + 1e-8))))
        h2 = cx[:, t + 1, :]
        cos = np.sum(h * h2, axis=1) / (
            np.linalg.norm(h, axis=1) * np.linalg.norm(h2, axis=1) + 1e-8
        )
        adj_cos.append(float(np.mean(cos)))
        sigma.append(float(np.sqrt(np.mean(delta**2))))
    return {
        "layers": list(range(n_layers)),
        "h_norm": h_norm,
        "delta_norm": d_norm,
        "delta_over_h_ratio": ratio,
        "adjacent_cosine": adj_cos,
        "sigma_block_rms": sigma,  # σ_m per transition ℓ=0..26
    }


# ── atlas metrics ──────────────────────────────────────────────────────────────


def identity_relative_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """1 − Σ‖pred−target‖² / Σ‖target‖² (uncentered ⇒ predict-zero scores 0).

    Sums over BOTH test contexts and all D dims (the total-Δ-energy denominator).
    """
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum(target**2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def mean_centered_r2(pred: np.ndarray, target: np.ndarray, target_mean: np.ndarray) -> float:
    """1 − Σ‖pred−target‖² / Σ‖target−μ_train‖² (JTC/ReSAE convention).

    ``target_mean`` (D,) is the per-dim TRAIN mean of Δ (no test leakage).
    """
    ss_res = float(np.sum((pred - target) ** 2))
    ss_tot = float(np.sum((target - target_mean) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def delta_error_percentiles(pred_raw: np.ndarray, target_raw: np.ndarray) -> dict:
    """Median / p90 / p99 of the per-context RAW Δ-error ‖pred−target‖ (2405.12250 tails)."""
    err = np.linalg.norm(pred_raw - target_raw, axis=1)
    return {
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p99": float(np.percentile(err, 99)),
    }


# ── map classes (torch apply for transport) ───────────────────────────────────


class IdentityMap:
    """Predict-zero null: Δ̂ = 0 (composition to ℓ* gives ĥ_{ℓ*} = h_ℓ)."""

    def apply(self, H: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(H)


@dataclass
class RidgeMap:
    """Affine one-step map Δ̂ = bias + ((H−μ)/σ_std) @ w, scaled to raw by ``sigma``.

    ``bias`` (the fit-space Δ mean ``ymu``) is the intercept c_ℓ of the plan's
    ``Δ̂ = A_ℓ h_ℓ + c_ℓ`` — Δ has a nonzero mean, so a no-intercept ridge
    (bias-free, forced through the origin) systematically underfits it. The
    weights are solved on the TRAIN-mean-centered target ``Y − ymu`` and the
    intercept is added back at prediction time (standard center-then-ridge with
    an unpenalized intercept; because ``(H−μ)/σ_std`` is zero-mean on train the
    intercept is exactly ``ymu``).
    """

    mu: torch.Tensor  # (d,)
    sd: torch.Tensor  # (d,) input standardization std
    w: torch.Tensor  # (d, p) dual weights (fit-space, on the CENTERED target)
    bias: torch.Tensor  # (p,) fit-space Δ mean ymu (the affine intercept c_ℓ)
    best_lam: float
    sigma: float  # target-space scale (σ_m if norm-space, else 1.0) → raw Δ̂

    def apply(self, H: torch.Tensor) -> torch.Tensor:
        hn = (H - self.mu) / self.sd
        return (self.bias + hn @ self.w) * self.sigma

    def to(self, device: str) -> RidgeMap:
        return RidgeMap(
            self.mu.to(device),
            self.sd.to(device),
            self.w.to(device),
            self.bias.to(device),
            self.best_lam,
            self.sigma,
        )


@dataclass
class MLPMap:
    """d→hidden→d multi-output MLP one-step map, scaled to raw by ``sigma``."""

    W1: torch.Tensor  # (hid, d)
    b1: torch.Tensor  # (hid,)
    W2: torch.Tensor  # (p, hid)
    b2: torch.Tensor  # (p,)
    mu: torch.Tensor  # (d,)
    sd: torch.Tensor  # (d,)
    sigma: float

    def apply(self, H: torch.Tensor) -> torch.Tensor:
        hn = (H - self.mu) / self.sd
        h = torch.nn.functional.gelu(hn @ self.W1.t() + self.b1)
        return (h @ self.W2.t() + self.b2) * self.sigma

    @classmethod
    def from_params(cls, params: dict, sigma: float, device: str) -> MLPMap:
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
        return cls(
            t(params["W1"]),
            t(params["b1"]),
            t(params["W2"]),
            t(params["b2"]),
            t(params["mu"]),
            t(params["sd"]),
            sigma,
        )


class DepthGRU(torch.nn.Module):
    """1-layer GRU over the depth axis predicting Δ_ℓ at each step (EXPLORATORY).

    Input at step ℓ is ``[h_ℓ, emb(ℓ)]`` (residual state + learned layer-index
    embedding); the output head predicts Δ_ℓ in the fit target space. One model
    for all transitions. Teacher-forced during fit; rolled autoregressively for
    transport (``gru_roll``). Prefix-informed (its state at ℓ has consumed
    h_0..h_ℓ) ⇒ excluded from the H2 matched-information criterion.
    """

    def __init__(
        self,
        d_state: int = 3584,
        gru_hidden: int = 1024,
        emb_dim: int = 32,
        n_transitions: int = 27,
    ) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(n_transitions, emb_dim)
        self.gru = torch.nn.GRU(d_state + emb_dim, gru_hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Linear(gru_hidden, d_state)
        self.n_transitions = n_transitions

    def forward(self, traj_inputs: torch.Tensor) -> torch.Tensor:
        """traj_inputs (B, T, d) = h_0..h_{T-1} → (B, T, d) predicted Δ_0..Δ_{T-1}."""
        b, t, _ = traj_inputs.shape
        idx = torch.arange(t, device=traj_inputs.device).unsqueeze(0).expand(b, t)
        x = torch.cat([traj_inputs, self.emb(idx)], dim=-1)
        out, _ = self.gru(x)
        return self.head(out)

    def forward_single(self, h: torch.Tensor, transition_idx: torch.Tensor | int) -> torch.Tensor:
        """Single-state (length-1 unroll) forward: predict Δ from ONE state h_ℓ.

        ``h`` (B, d) is the residual state at one layer; ``transition_idx`` is that
        transition's index — an int (all rows same transition) or a (B,) long tensor
        (a batch of mixed transitions). Runs the SAME ``self.gru`` on a length-1
        sequence ``[h, emb(transition_idx)]`` with a ZERO initial hidden state —
        numerically a single ``GRUCell`` step — and returns ``self.head(out)`` (B, d),
        the predicted Δ in the fit target space. This is the information-MATCHED
        analogue of ``forward``: the recurrence sees ONLY h_ℓ (+ the layer-index
        embedding), not the h_0..h_ℓ prefix (that matched-information property is why
        the source-only variant is comparable to the affine/MLP per-transition maps).
        """
        b = h.shape[0]
        if not torch.is_tensor(transition_idx):
            transition_idx = torch.full(
                (b,), int(transition_idx), device=h.device, dtype=torch.long
            )
        x = torch.cat([h, self.emb(transition_idx)], dim=-1)  # (B, d + emb_dim)
        out, _ = self.gru(x.unsqueeze(1))  # (B, 1, gru_hidden); zero init hidden ⇒ GRUCell step
        return self.head(out.squeeze(1))  # (B, d)


# ── ridge fit (fixed train/eval split; reuse #658 PRESS/dual) ─────────────────


def fit_ridge_split(
    x_train: np.ndarray,
    y_train_fit: np.ndarray,
    x_eval: np.ndarray,
    *,
    sigma: float,
    device: str,
    lambdas: list[float] = RIDGE_LAMBDAS,
) -> tuple[np.ndarray, RidgeMap]:
    """Closed-form ridge Δ-map on a FIXED train/eval split (ReSAE recipe).

    AFFINE with bias (plan §4.3 ``Δ̂ = A_ℓ h_ℓ + c_ℓ``; §11 rejects the no-bias
    JTC "mat" variant): Δ has a nonzero mean, so the target is TRAIN-mean-centered
    (``ymu``) before the ridge solve and the intercept ``ymu`` is added back at
    prediction time. Standardize X on train (ddof=0, matching #658), select λ by
    PRESS/LOO over the CENTERED train target (train-internal GCV — consistent with
    the centered solve), dual-solve the weights on ``Y − ymu``, predict
    ``ymu + Xn @ w`` on the eval slice. ``y_train_fit`` is the Δ target in the
    chosen fit space (raw or RMS-normalized). Returns ``(eval_pred_fitspace
    (n_eval,p), RidgeMap)`` — the eval preds feed the atlas R²; the RidgeMap is
    applied to eval-context activations in Stage-1 transport (its ``apply`` adds
    the bias then un-scales by ``sigma`` to raw).
    """
    _i658.DEVICE = device
    dev = torch.device(device)
    Xt = torch.from_numpy(np.ascontiguousarray(x_train)).to(device=dev, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(y_train_fit)).to(device=dev, dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9  # #658 numpy ddof=0 convention
    Xtr_n = (Xt - mu) / sd
    ymu = Yt.mean(0)  # (p,) train-mean of Δ = the affine intercept c_ℓ
    Ytr_c = Yt - ymu  # center the target so the ridge fits the residual, not the mean
    mse = _press_loo_mse_per_lambda(Xtr_n, Ytr_c, lambdas)
    best_lam = lambdas[int(torch.argmin(mse).item())]
    w = _ridge_dual_weights(Xtr_n, Ytr_c, best_lam)  # (d, p) fp64, on the centered target
    Xev = torch.from_numpy(np.ascontiguousarray(x_eval)).to(device=dev, dtype=torch.float64)
    Xev_n = (Xev - mu) / sd
    eval_pred = (ymu + Xev_n @ w).detach().cpu().numpy()  # fit-space eval prediction (WITH bias)
    rmap = RidgeMap(
        mu=mu.to(torch.float32),
        sd=sd.to(torch.float32),
        w=w.to(torch.float32),
        bias=ymu.to(torch.float32),
        best_lam=float(best_lam),
        sigma=float(sigma),
    )
    return eval_pred, rmap


def fit_ridge_primal(
    x_train: np.ndarray,
    y_train_fit: np.ndarray,
    x_eval: np.ndarray,
    *,
    sigma: float,
    device: str,
    lambdas: list[float] = RIDGE_LAMBDAS,
    gram_chunk: int = 20000,
) -> tuple[np.ndarray, RidgeMap]:
    """Primal closed-form ridge for n ≫ d — a numeric drop-in for ``fit_ridge_split``.

    ``fit_ridge_split`` builds the dual m×m Gram (m = n_fit); at n=100000 that is
    100000² fp64 ≈ 80 GB, infeasible. This primal path forms the d×d
    ``XₙᵀXₙ`` (d=3584, fp64 ≈ 103 MB) via a chunked GEMM and does the EXACT
    leave-one-out (PRESS) λ-selection in the PRIMAL eigenbasis — mathematically
    identical to the dual (both evaluate the SAME n×n hat matrix
    ``H(λ) = Xₙ(XₙᵀXₙ+λI)⁻¹Xₙᵀ`` via its closed form), so at a shared n the two
    solvers agree to fp64 precision (the KILL-B parity/cross-check gates that).

    Contract identical to ``fit_ridge_split``: AFFINE with bias — X standardized
    on train (ddof=0), the target train-mean-centered (``ymu``), the intercept
    ``ymu`` added back at prediction (unpenalized), λ by exact PRESS-LOO over the
    centered target, RETURNS ``(eval_pred_fitspace (n_eval,p), RidgeMap)`` whose
    ``apply`` un-scales by ``sigma`` to raw. ``y_train_fit`` is Δ in the chosen
    fit space (raw or RMS-normalized). fp64 throughout for parity with the dual.

    PRESS-LOO identity (primal eigenbasis). With ``XₙᵀXₙ = V diag(s) Vᵀ``
    (``eigh``, computed ONCE and reused across λ) and ``Z = Xₙ V`` (n×d):

        diag(H(λ))_k = Σ_j Z[k,j]² / (s_j+λ)              # = (Z² @ 1/(s+λ))_k
        Ŷ(λ)         = Z diag(1/(s+λ)) (Vᵀ XₙᵀYc)          # (n,P)
        LOO_resid_k  = (Yc_k − Ŷ_k) / (1 − diag(H)_k)

    The selected-λ weights reuse the SAME eigenbasis:
    ``w = V diag(1/(s+λ)) Vᵀ (XₙᵀYc)`` (d,P).
    """
    dev = torch.device(device)
    Xt = torch.from_numpy(np.ascontiguousarray(x_train)).to(device=dev, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(y_train_fit)).to(device=dev, dtype=torch.float64)
    n, d = Xt.shape
    assert n > 0 and d > 0, (n, d)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9  # #658 numpy ddof=0 convention (matches the dual path)
    ymu = Yt.mean(0)  # (P,) train-mean of Δ = the affine intercept c_ℓ
    p = Yt.shape[1]

    # Chunked accumulation of XₙᵀXₙ (d,d) and XₙᵀYc (d,P) — never materialize a
    # (n,d) standardized design when n is large; the reduction is streamed in
    # row-chunks so peak extra memory is O(chunk·d) beyond the d×d / d×P Grams.
    XtX = torch.zeros(d, d, dtype=torch.float64, device=dev)
    XtY = torch.zeros(d, p, dtype=torch.float64, device=dev)
    for lo in range(0, n, gram_chunk):
        xc = (Xt[lo : lo + gram_chunk] - mu) / sd  # (c,d) standardized
        yc = Yt[lo : lo + gram_chunk] - ymu  # (c,P) centered
        XtX += xc.t() @ xc
        XtY += xc.t() @ yc
    evals, V = torch.linalg.eigh(XtX)  # XₙᵀXₙ = V diag(evals) Vᵀ ; O(d³) ONCE
    evals = evals.clamp(min=0.0)  # PSD; guard tiny negative eigenvalues from round-off
    VtXtY = V.t() @ XtY  # (d,P), reused across λ and for the final weights

    # Exact PRESS-LOO, streamed in the SAME row-chunks (holds only a (c,d) Zc per
    # chunk, never the full (n,d) Z). Zc = xc @ V is computed ONCE per chunk and
    # reused across all λ (the λ-loop is the CHEAP inner loop), so the O(n·d²)
    # projection is paid once, not once-per-λ — at n=100k that is a 6× GEMM save.
    filts = [1.0 / (evals + lam) for lam in lambdas]  # each (d,)
    yhat_coefs = [f.unsqueeze(1) * VtXtY for f in filts]  # each (d,P)
    sse = [0.0] * len(lambdas)
    cnt = 0
    for lo in range(0, n, gram_chunk):
        xc = (Xt[lo : lo + gram_chunk] - mu) / sd  # (c,d)
        yc = Yt[lo : lo + gram_chunk] - ymu  # (c,P)
        zc = xc @ V  # (c,d) — ONCE per chunk
        z2 = zc * zc  # (c,d) reused across λ for diag(H)
        for li in range(len(lambdas)):
            h_diag = z2 @ filts[li]  # (c,)
            yhat = zc @ yhat_coefs[li]  # (c,P)
            denom = (1.0 - h_diag).clamp(min=1e-8).unsqueeze(1)  # (c,1)
            loo = (yc - yhat) / denom
            sse[li] += float((loo * loo).sum().item())
        cnt += yc.numel()
    best_li = int(np.argmin([s / cnt for s in sse]))
    best_lam = lambdas[best_li]

    filt = 1.0 / (evals + best_lam)
    w = V @ (filt.unsqueeze(1) * VtXtY)  # (d,P) selected-λ weights on the centered target
    Xev = torch.from_numpy(np.ascontiguousarray(x_eval)).to(device=dev, dtype=torch.float64)
    Xev_n = (Xev - mu) / sd
    eval_pred = (ymu + Xev_n @ w).detach().cpu().numpy()  # fit-space eval prediction (WITH bias)
    rmap = RidgeMap(
        mu=mu.to(torch.float32),
        sd=sd.to(torch.float32),
        w=w.to(torch.float32),
        bias=ymu.to(torch.float32),
        best_lam=float(best_lam),
        sigma=float(sigma),
    )
    return eval_pred, rmap


def fit_direct_hop_ridge(
    h_source_train: np.ndarray,
    h_target_train: np.ndarray,
    h_source_eval: np.ndarray,
    *,
    device: str,
    lambdas: list[float] = RIDGE_LAMBDAS,
    n: int | None = None,
    dual_max: int = 10000,
) -> RidgeMap:
    """Direct-hop ridge ℓ→ℓ* (JTC-style): predict the total displacement.

    Fits ``h_ℓ → (h_{ℓ*} − h_ℓ)`` on train, so ``apply`` gives the raw total
    displacement and ``ĥ_{ℓ*} = h_ℓ + apply(h_ℓ)``. Returns the RidgeMap (sigma=1,
    raw target). Answers "is composing one-step maps worse than one long hop?".

    Solver dispatch (additive; default preserves the parent's dual behavior): when
    ``n`` is given and ``n > dual_max`` the fit uses the primal path
    (``fit_ridge_primal``) — the dual m×m Gram is n×n fp64 (~80 GB at n=100k),
    infeasible for the scaling curve. ``n=None`` (parent callers) keeps the dual.
    """
    delta_direct = h_target_train - h_source_train
    if n is not None and n > dual_max:
        _, rmap = fit_ridge_primal(
            h_source_train, delta_direct, h_source_eval, sigma=1.0, device=device, lambdas=lambdas
        )
    else:
        _, rmap = fit_ridge_split(
            h_source_train, delta_direct, h_source_eval, sigma=1.0, device=device, lambdas=lambdas
        )
    return rmap


# ── MLP fits (batched split ensemble) ─────────────────────────────────────────


def fit_split_mlps(
    groups: list[SplitMLPGroup],
    *,
    device: str,
    seed: int = 658,
    chunk_size: int = 8,
    num_threads: int | None = None,
    max_epochs: int | None = None,
) -> tuple[dict, dict]:
    """Fit a batch of split-MLP Δ-maps; return (eval_preds_by_key, MLPMap params).

    Thin wrapper over ``fit_batched_split_mlp`` (SmoothL1, inner-val early stop).
    Returns ``(preds_by_key, params_by_key)`` — preds feed the atlas R², params
    are wrapped into ``MLPMap`` for transport. ``sigma`` is threaded per-map by
    the caller (via ``MLPMap.from_params``) since it depends on the group's
    target space. ``max_epochs=None`` uses the production default
    (``MLP_MAX_EPOCHS=300``); a smoke passes a small value to keep the wiring
    exercise fast.
    """
    kw = {} if max_epochs is None else {"max_epochs": max_epochs}
    res = fit_batched_split_mlp(
        groups,
        seed=seed,
        device=device,
        chunk_size=chunk_size,
        num_threads=num_threads,
        **kw,
    )
    return res.preds_by_key, res.params_by_key


# ── GRU fit (teacher-forced, batched over contexts) ───────────────────────────


def fit_depth_gru(
    traj_fit: np.ndarray,
    traj_val: np.ndarray,
    sigma_per_transition: np.ndarray,
    *,
    device: str,
    gru_hidden: int = 1024,
    emb_dim: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 300,
    batch_size: int = 512,
    seed: int = 658,
    smooth_l1_beta: float = 1.0,
) -> DepthGRU:
    """Teacher-forced depth-GRU fit on the RMS-normalized Δ target (best-val kept).

    traj_fit / traj_val: (N, 28, d) real trajectories. The target is the
    RMS-normalized Δ (Δ_ℓ / σ_m) so no single high-norm early layer dominates the
    shared-depth loss. Returns the trained ``DepthGRU`` on ``device``.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    n_trans = traj_fit.shape[1] - 1
    gru = DepthGRU(
        d_state=traj_fit.shape[2], gru_hidden=gru_hidden, emb_dim=emb_dim, n_transitions=n_trans
    ).to(dev)
    sig = torch.from_numpy(np.asarray(sigma_per_transition, dtype=np.float32)).to(dev)  # (T,)

    def _inputs_targets(traj_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.from_numpy(np.ascontiguousarray(traj_np)).to(device=dev, dtype=torch.float32)
        inputs = t[:, :n_trans, :]  # h_0..h_{T-1}
        delta = t[:, 1:, :] - t[:, :n_trans, :]  # Δ_0..Δ_{T-1}
        target = delta / sig.view(1, n_trans, 1)  # RMS-normalized
        return inputs, target

    x_fit, y_fit = _inputs_targets(traj_fit)
    x_val, y_val = _inputs_targets(traj_val)
    opt = torch.optim.AdamW(gru.parameters(), lr=lr)
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in gru.state_dict().items()}
    n_fit = x_fit.shape[0]
    rng = np.random.default_rng(seed)
    for _epoch in range(max_epochs):
        gru.train()
        order = rng.permutation(n_fit)
        for lo in range(0, n_fit, batch_size):
            idx = torch.from_numpy(order[lo : lo + batch_size]).to(dev)
            opt.zero_grad(set_to_none=True)
            pred = gru(x_fit[idx])
            loss = torch.nn.functional.smooth_l1_loss(pred, y_fit[idx], beta=smooth_l1_beta)
            loss.backward()
            opt.step()
        gru.eval()
        with torch.no_grad():
            vloss = float(
                torch.nn.functional.smooth_l1_loss(gru(x_val), y_val, beta=smooth_l1_beta).item()
            )
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in gru.state_dict().items()}
    gru.load_state_dict(best_state)
    gru.eval()
    return gru


def fit_depth_gru_source_only(
    traj_fit: np.ndarray,
    traj_val: np.ndarray,
    sigma_per_transition: np.ndarray,
    *,
    device: str,
    gru_hidden: int = 1024,
    emb_dim: int = 32,
    lr: float = 1e-3,
    max_epochs: int = 300,
    batch_size: int = 512,
    seed: int = 658,
    smooth_l1_beta: float = 1.0,
    transitions: list[int] | None = None,
) -> tuple[DepthGRU, dict]:
    """Single-state depth-GRU fit — the information-MATCHED analogue of ``fit_depth_gru``.

    Same ``DepthGRU`` architecture (``Embedding(27,32)`` + ``GRU(d+32,1024,1)`` +
    ``Linear(1024,d)``), same optimizer / loss / target spaces — the ONLY change vs
    ``fit_depth_gru`` is the INPUT the recurrence sees: instead of teacher-forcing the
    full ``(N,28,d)`` trajectory (prefix-informed), this builds INDEPENDENT
    ``(context × transition)`` single-state examples — input ``[h_ℓ, emb(ℓ)]``, target
    ``Δ_ℓ / σ_m`` (raw space: σ≡1) — 4000 contexts × 27 transitions = 108k examples per
    space, and trains via ``DepthGRU.forward_single`` (the length-1 unroll = a GRUCell
    step, zero init hidden), so the recurrence never consumes the h_0..h_ℓ prefix. This
    is the per-transition-map information set the affine ridge / MLP see.

    Batched AdamW over the flattened examples (batch ``batch_size``), SmoothL1, best-val
    kept on ``traj_val``'s single-state examples (same early-stopping convention as
    ``fit_depth_gru``). ``transitions`` (default = all n_trans) restricts the example set
    to a subset (the smoke passes ``[13]``); the SAME code path runs either way. Returns
    ``(trained DepthGRU on device, diagnostics)`` — diagnostics carries
    ``epochs_to_best_val`` / ``cap_hit`` / the val-loss curve + its last-5-epoch slope
    (plan §4.2: persist convergence diagnostics so the prefix-vs-source read can be gated
    on convergence parity).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    d = traj_fit.shape[2]
    n_trans = traj_fit.shape[1] - 1
    trans = list(range(n_trans)) if transitions is None else sorted(transitions)
    assert trans and all(0 <= m < n_trans for m in trans), (trans, n_trans)
    gru = DepthGRU(d_state=d, gru_hidden=gru_hidden, emb_dim=emb_dim, n_transitions=n_trans).to(dev)
    sig = torch.from_numpy(np.asarray(sigma_per_transition, dtype=np.float32)).to(dev)  # (n_trans,)

    def _examples(traj_np: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.from_numpy(np.ascontiguousarray(traj_np)).to(device=dev, dtype=torch.float32)
        hs, ys, idxs = [], [], []
        for m in trans:
            h_m = t[:, m, :]  # (N, d) — the source state h_ℓ (single state, no prefix)
            delta_m = t[:, m + 1, :] - t[:, m, :]  # (N, d) Δ_ℓ
            ys.append(delta_m / sig[m])  # RMS-normalized (σ≡1 ⇒ raw target)
            hs.append(h_m)
            idxs.append(torch.full((h_m.shape[0],), m, device=dev, dtype=torch.long))
        return torch.cat(hs, 0), torch.cat(ys, 0), torch.cat(idxs, 0)

    x_fit, y_fit, idx_fit = _examples(traj_fit)
    x_val, y_val, idx_val = _examples(traj_val)
    opt = torch.optim.AdamW(gru.parameters(), lr=lr)
    best_val = float("inf")
    best_epoch = -1
    best_state = {k: v.detach().clone() for k, v in gru.state_dict().items()}
    val_curve: list[float] = []
    n_fit = x_fit.shape[0]
    rng = np.random.default_rng(seed)
    for epoch in range(max_epochs):
        gru.train()
        order = rng.permutation(n_fit)
        for lo in range(0, n_fit, batch_size):
            sel = torch.from_numpy(order[lo : lo + batch_size]).to(dev)
            opt.zero_grad(set_to_none=True)
            pred = gru.forward_single(x_fit[sel], idx_fit[sel])
            loss = torch.nn.functional.smooth_l1_loss(pred, y_fit[sel], beta=smooth_l1_beta)
            loss.backward()
            opt.step()
        gru.eval()
        with torch.no_grad():
            vloss = float(
                torch.nn.functional.smooth_l1_loss(
                    gru.forward_single(x_val, idx_val), y_val, beta=smooth_l1_beta
                ).item()
            )
        val_curve.append(vloss)
        if vloss < best_val:
            best_val = vloss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in gru.state_dict().items()}
    gru.load_state_dict(best_state)
    gru.eval()
    last5 = val_curve[-5:]
    slope = (
        float(np.polyfit(np.arange(len(last5)), np.asarray(last5), 1)[0])
        if len(last5) >= 2
        else float("nan")
    )
    diagnostics = {
        "epochs_to_best_val": int(best_epoch),
        "epochs_run": len(val_curve),
        "max_epochs": int(max_epochs),
        # cap_hit ⇒ best-val was still improving at the last epoch (never plateaued);
        # a True here means the fit may be under-trained (didn't converge within the cap).
        "cap_hit": bool(best_epoch >= max_epochs - 1),
        "best_val_loss": float(best_val),
        "last5_epoch_slope": slope,
        "val_curve": [float(v) for v in val_curve],
        "n_examples_fit": int(n_fit),
        "transitions_trained": trans,
    }
    return gru, diagnostics


# ── transport (Stage 1) ────────────────────────────────────────────────────────


def transport_iterated(
    maps_by_transition: dict[int, object],
    h_source: torch.Tensor,
    source_layer: int,
    target_layer: int,
) -> torch.Tensor:
    """Iterated one-step composition ``ĥ_{ℓ*} = h_ℓ + Σ apply(ĥ)`` (ridge/MLP/id).

    ``maps_by_transition[m].apply(ĥ)`` returns the raw Δ̂ at transition m.
    ``h_source`` (N,d) is the REAL h_ℓ (torch). Works for RidgeMap / MLPMap /
    IdentityMap. Returns ĥ_{target} (N,d).
    """
    h = h_source
    for m in range(source_layer, target_layer):
        h = h + maps_by_transition[m].apply(h)
    return h


def gru_roll(
    gru: DepthGRU,
    traj_true: torch.Tensor,
    sigma_per_transition: torch.Tensor,
    source_layer: int,
    target_layer: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Roll the depth-GRU from ``source`` to ``target`` feeding its own predictions.

    ``traj_true`` (N,28,d) real trajectory (torch). Warms the recurrent state on
    the true prefix h_0..h_{source-1}, then from ``source`` feeds the CURRENT
    (predicted) state as the next input. Returns ``(ĥ_target (N,d),
    divergence (N, n_steps))`` where divergence[:,k] = ‖ĥ_{source+1+k} −
    h_true_{source+1+k}‖ (for the divergence-horizon metric). Prefix-informed
    (EXPLORATORY).
    """
    dev = traj_true.device
    n = traj_true.shape[0]
    h_state = None
    # Warm the state with the true prefix inputs x_0..x_{source-1}.
    for t in range(0, source_layer):
        x = torch.cat([traj_true[:, t, :], gru.emb(torch.full((n,), t, device=dev))], dim=-1)
        _out, h_state = gru.gru(x.unsqueeze(1), h_state)
    cur = traj_true[:, source_layer, :]  # ĥ_source = real h_source
    divergence = []
    for t in range(source_layer, target_layer):
        x = torch.cat([cur, gru.emb(torch.full((n,), t, device=dev))], dim=-1)
        out, h_state = gru.gru(x.unsqueeze(1), h_state)
        delta_hat = gru.head(out.squeeze(1)) * sigma_per_transition[t]  # raw Δ̂_t
        cur = cur + delta_hat  # ĥ_{t+1}
        div = torch.norm(cur - traj_true[:, t + 1, :], dim=1)  # (N,)
        divergence.append(div.detach().cpu().numpy())
    div_arr = np.stack(divergence, axis=1) if divergence else np.zeros((n, 0))
    return cur, div_arr


@dataclass
class GruSourceOnlyMap:
    """Memoryless single-state GRU one-step map (matched-information transport).

    ``.apply(H)`` returns the RAW Δ̂ at ``transition`` for a batch of states H —
    ``head(forward_single(H, transition)) * sigma_m`` — with a ZERO initial hidden
    state EVERY call (no carried recurrent state), so dropping it into the parent's
    ``transport_iterated`` (which threads only the current predicted state, no hidden)
    rolls the source-only GRU MEMORYLESS: each step reads only the current state,
    exactly matching the ridge/MLP transport regime and preserving matched information
    at inference. This is NOT ``gru_roll`` (which warms + carries the recurrent state on
    the true prefix — the prefix-informed regime). ``sigma_m`` un-scales the fit-space
    Δ̂ to raw (σ_m for the RMS-norm-trained GRU, 1.0 for a raw-trained GRU), mirroring
    the per-step ``* sigma_per_transition[t]`` in ``gru_roll``.
    """

    gru: DepthGRU
    transition: int
    sigma_m: float

    def apply(self, H: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.gru.forward_single(H, self.transition) * self.sigma_m
