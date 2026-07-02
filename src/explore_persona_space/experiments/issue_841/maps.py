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

_REPO_ROOT = Path(__file__).resolve().parents[3]
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
    "IdentityMap",
    "MLPMap",
    "RidgeMap",
    "delta_error_percentiles",
    "deltas_at",
    "fit_depth_gru",
    "fit_direct_hop_ridge",
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
    """Affine one-step map Δ̂ = ((H−μ)/σ_std) @ w, scaled to raw by ``sigma``."""

    mu: torch.Tensor  # (d,)
    sd: torch.Tensor  # (d,) input standardization std
    w: torch.Tensor  # (d, p) dual weights (fit-space)
    best_lam: float
    sigma: float  # target-space scale (σ_m if norm-space, else 1.0) → raw Δ̂

    def apply(self, H: torch.Tensor) -> torch.Tensor:
        hn = (H - self.mu) / self.sd
        return (hn @ self.w) * self.sigma

    def to(self, device: str) -> RidgeMap:
        return RidgeMap(
            self.mu.to(device), self.sd.to(device), self.w.to(device), self.best_lam, self.sigma
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

    Standardize on train (ddof=0, matching #658), select λ by PRESS/LOO over the
    train design (train-internal GCV), dual-solve the weights, predict on the
    eval slice. ``y_train_fit`` is the Δ target in the chosen fit space (raw or
    RMS-normalized). Returns ``(eval_pred_fitspace (n_eval,p), RidgeMap)`` — the
    eval preds feed the atlas R²; the RidgeMap is applied to eval-context
    activations in Stage-1 transport (its ``apply`` un-scales by ``sigma`` to raw).
    """
    _i658.DEVICE = device
    dev = torch.device(device)
    Xt = torch.from_numpy(np.ascontiguousarray(x_train)).to(device=dev, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(y_train_fit)).to(device=dev, dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9  # #658 numpy ddof=0 convention
    Xtr_n = (Xt - mu) / sd
    mse = _press_loo_mse_per_lambda(Xtr_n, Yt, lambdas)
    best_lam = lambdas[int(torch.argmin(mse).item())]
    w = _ridge_dual_weights(Xtr_n, Yt, best_lam)  # (d, p) fp64
    Xev = torch.from_numpy(np.ascontiguousarray(x_eval)).to(device=dev, dtype=torch.float64)
    Xev_n = (Xev - mu) / sd
    eval_pred = (Xev_n @ w).detach().cpu().numpy()  # fit-space eval prediction
    rmap = RidgeMap(
        mu=mu.to(torch.float32),
        sd=sd.to(torch.float32),
        w=w.to(torch.float32),
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
) -> RidgeMap:
    """Direct-hop ridge ℓ→ℓ* (JTC-style): predict the total displacement.

    Fits ``h_ℓ → (h_{ℓ*} − h_ℓ)`` on train, so ``apply`` gives the raw total
    displacement and ``ĥ_{ℓ*} = h_ℓ + apply(h_ℓ)``. Returns the RidgeMap (sigma=1,
    raw target). Answers "is composing one-step maps worse than one long hop?".
    """
    delta_direct = h_target_train - h_source_train
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
) -> tuple[dict, dict]:
    """Fit a batch of split-MLP Δ-maps; return (eval_preds_by_key, MLPMap params).

    Thin wrapper over ``fit_batched_split_mlp`` (SmoothL1, inner-val early stop).
    Returns ``(preds_by_key, params_by_key)`` — preds feed the atlas R², params
    are wrapped into ``MLPMap`` for transport. ``sigma`` is threaded per-map by
    the caller (via ``MLPMap.from_params``) since it depends on the group's
    target space.
    """
    res = fit_batched_split_mlp(
        groups,
        seed=seed,
        device=device,
        chunk_size=chunk_size,
        num_threads=num_threads,
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
