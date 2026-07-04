# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ℓ, σ, →, ‖·‖, ĥ, λ, μ) in scientific docstrings.
"""Issue #922 position-axis maps: Gram/eigh GCV ridge, batched MLP wrapper,
position-GRU, and the autoregressive rollout engine.

The per-layer next-POSITION update maps ``Δ_{l,t} = h_{l,t+1} − h_{l,t}`` (the
token-position sibling of #841's depth maps). Design contracts (plan §4.3):

- **Ridge = ONE Gram assembly + ONE ``torch.linalg.eigh`` per (layer, arm)**,
  with λ selected by GCV over ``np.logspace(-2, 3, 25)`` as pure eigenvalue
  arithmetic (no per-λ refit, no per-cell re-factorization — the #823 serial
  dense-factorization class is the named failure mode). The raw and RMS-norm
  target spaces share the SAME eigendecomposition (a per-layer SCALAR σ only
  rescales the targets: weights scale by 1/σ exactly, the GCV argmin is
  unchanged, and every reported R² is identical — asserted in the verify gate).
- **MLP fits ride the ported ``fit_batched_split_mlp``** (the #841 batched
  fixed-split multi-output trainer, ported verbatim from ``origin/issue-841``)
  through a store-fed chunking wrapper: the full (29, n≈150k, d) group stack
  never materializes; each resolve-chunk-cap-sized chunk of layers is gathered
  from the fp16 position store, fit, and released.
- **Standardization conventions differ by class and are inherited, not
  invented**: ridge standardizes X with ddof=0 + 1e-9 (the #658/#841 ridge
  convention); the MLP standardizes with ddof=1 + 1e-6 (the
  ``fit_batched_split_mlp`` convention). Both center the target on the train
  mean and add the intercept back at prediction.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# parents: [0]=issue_922 [1]=experiments [2]=explore_persona_space [3]=src [4]=repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
assert (_REPO_ROOT / "scripts" / "issue658_fit_predictors.py").is_file(), (
    f"maps922.py repo-root anchor wrong: {_REPO_ROOT} has no scripts/issue658_fit_predictors.py"
)
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from issue658_fit_predictors import _ridge_dual_weights  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    MLP_HIDDEN,
    MLP_LR,
    MLP_MAX_EPOCHS,
    MLP_WD,
    SplitMLPGroup,
    _split_mlp_eval,
    fit_batched_split_mlp,
)
from explore_persona_space.experiments.issue_841.maps import (  # noqa: E402
    RidgeMap,
    delta_error_percentiles,
    identity_relative_r2,
    mean_centered_r2,
)

logger = logging.getLogger("issue922_maps")

# The plan-registered λ grid (§4.3: GCV over [1e-2, 1e3], 25 points) — a DENSER
# grid over the SAME range as #658/#841's 6-point RIDGE_LAMBDAS.
RIDGE_LAMBDAS_922: list[float] = [float(x) for x in np.logspace(-2.0, 3.0, 25)]

__all__ = [
    "CONDITIONED_FORMS",
    "CONDITIONED_RECIPE_922",
    "RIDGE_LAMBDAS_922",
    "GramStats",
    "RidgeMap",
    "accumulate_grams",
    "accumulate_grams_b1",
    "apply_conditioned_delta",
    "apply_mlp_params",
    "apply_ridge_maps_batched",
    "assemble_b1_stats",
    "conditioned_param_count",
    "conditioned_predict_row",
    "delta_error_percentiles",
    "fit_conditioned_linear",
    "fit_direct_horizon_maps",
    "fit_phase_gpu_budget_bytes",
    "fit_position_gru",
    "fit_position_mlps",
    "gru_roll_states",
    "identity_relative_r2",
    "init_conditioned_params",
    "mean_centered_r2",
    "predict_direct_horizons_row",
    "ridge_gcv_from_grams",
    "ridge_predict",
    "roll_states_b1_ridge",
    "roll_states_conditioned",
    "roll_states_ridge",
    "stack_conditioned_params",
    "verify_conditioned_forms",
    "verify_direct_horizon_gcv",
    "verify_ridge_gcv_against_dual",
]


# ── Gram accumulation (one streaming pass over the store) ─────────────────────


@dataclass
class GramStats:
    """fp64 sufficient statistics of one (X, Y) regression problem.

    ``sxx = Σ x xᵀ`` (d,d), ``sx = Σ x`` (d,), ``sxy = Σ x yᵀ`` (d,p),
    ``sy = Σ y`` (p,), ``syy = Σ ‖y‖²`` (scalar), ``n`` rows. Centered /
    standardized Grams are derived EXACTLY from these (no second data pass):
    ``XnᵀXn = D⁻¹(sxx − n μμᵀ)D⁻¹`` and ``XnᵀYc = D⁻¹(sxy − μ syᵀ)`` (the
    cross term is exact because ``Σ(x−μ) = 0``).
    """

    n: int
    sx: torch.Tensor
    sxx: torch.Tensor
    sxy: torch.Tensor
    sy: torch.Tensor
    syy: float

    @classmethod
    def zeros(cls, d: int, p: int, device) -> GramStats:
        z = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=device)  # noqa: E731
        return cls(0, z(d), z(d, d), z(d, p), z(p), 0.0)

    def add_chunk(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Accumulate one fp64 row-chunk (c,d) x, (c,p) y."""
        assert x.dtype == torch.float64 and y.dtype == torch.float64, (x.dtype, y.dtype)
        self.n += int(x.shape[0])
        self.sx += x.sum(0)
        self.sxx += x.t() @ x
        self.sxy += x.t() @ y
        self.sy += y.sum(0)
        self.syy += float((y * y).sum().item())


def accumulate_grams(
    h_store: torch.Tensor,
    src_idx: torch.Tensor,
    rows: list[int],
    *,
    emb_row: int = 0,
    chunk: int = 4096,
    device: str = "cpu",
) -> dict:
    """One streaming pass building per-(row, arm) Gram stats for the ridge fits.

    ``h_store`` is the (R, P, H) fp16 position store (any device); ``src_idx``
    (n,) long tensor of SOURCE position indices (each transition is
    ``src → src+1`` within one context window — contiguity guaranteed by the
    capture). ``rows`` are the store-row indices to fit (block rows; the
    embedding row ``emb_row`` supplies the injected-token feature e_{t+1} =
    h_{emb,t+1} shared across layers).

    Returns ``{"ctx": {row: GramStats(d=H)}, "emb": {row: GramStats}, "tok":
    {row: GramStats(d=2H)}, "n": n}``. The tok Gram is assembled from the
    SHARED blocks (h-h, h-e, e-e) so the pass stays one-shot; the e-e block +
    e-mean are computed once (identical for every layer).
    """
    dev = torch.device(device)
    _R, _P, H = h_store.shape
    n = int(src_idx.numel())
    # Per-row blocks: Shh (H,H), She (H,H), Shy (H,H), Sey (H,H), Sh (H), Sy (H), syy.
    per_row = {
        r: {
            "Shh": torch.zeros(H, H, dtype=torch.float64, device=dev),
            "She": torch.zeros(H, H, dtype=torch.float64, device=dev),
            "Shy": torch.zeros(H, H, dtype=torch.float64, device=dev),
            "Sey": torch.zeros(H, H, dtype=torch.float64, device=dev),
            "Sh": torch.zeros(H, dtype=torch.float64, device=dev),
            "Sy": torch.zeros(H, dtype=torch.float64, device=dev),
            "syy": 0.0,
        }
        for r in rows
    }
    See = torch.zeros(H, H, dtype=torch.float64, device=dev)
    Se = torch.zeros(H, dtype=torch.float64, device=dev)
    t0 = time.time()
    for lo in range(0, n, chunk):
        sel = src_idx[lo : lo + chunk]
        e = h_store[emb_row, sel + 1, :].to(device=dev, dtype=torch.float64)  # (c,H)
        See += e.t() @ e
        Se += e.sum(0)
        for r in rows:
            h = h_store[r, sel, :].to(device=dev, dtype=torch.float64)
            y = h_store[r, sel + 1, :].to(device=dev, dtype=torch.float64) - h  # Δ
            blk = per_row[r]
            blk["Shh"] += h.t() @ h
            blk["She"] += h.t() @ e
            blk["Shy"] += h.t() @ y
            blk["Sey"] += e.t() @ y
            blk["Sh"] += h.sum(0)
            blk["Sy"] += y.sum(0)
            blk["syy"] += float((y * y).sum().item())
    logger.info(
        "[grams] accumulated %d rows x %d layers (+shared emb) in %.1fs",
        n,
        len(rows),
        time.time() - t0,
    )
    out: dict = {"ctx": {}, "emb": {}, "tok": {}, "n": n}
    for r in rows:
        blk = per_row[r]
        out["ctx"][r] = GramStats(n, blk["Sh"], blk["Shh"], blk["Shy"], blk["Sy"], blk["syy"])
        out["emb"][r] = GramStats(n, Se.clone(), See.clone(), blk["Sey"], blk["Sy"], blk["syy"])
        sx_tok = torch.cat([blk["Sh"], Se])
        sxx_tok = torch.cat(
            [torch.cat([blk["Shh"], blk["She"]], 1), torch.cat([blk["She"].t(), See], 1)], 0
        )
        sxy_tok = torch.cat([blk["Shy"], blk["Sey"]], 0)
        out["tok"][r] = GramStats(n, sx_tok, sxx_tok, sxy_tok, blk["Sy"], blk["syy"])
    return out


# ── GCV ridge in the eigenbasis (the plan §4.3 recipe) ────────────────────────


def ridge_gcv_from_grams(
    stats: GramStats,
    *,
    lambdas: list[float] = RIDGE_LAMBDAS_922,
    sigma: float = 1.0,
    sd_eps: float = 1e-9,
    eig: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[RidgeMap, dict]:
    """Closed-form affine ridge with GCV λ-selection from Gram statistics.

    ONE ``eigh`` of the standardized centered Gram; GCV across the λ grid is
    then eigenvalue arithmetic (no data pass, no per-λ refit)::

        s, V = eigh(XnᵀXn);  U = Vᵀ (XnᵀYc);  u_j = ‖U_j‖²
        df(λ)  = 1 + Σ_j s_j/(s_j+λ)                # +1 = unpenalized intercept
        SSE(λ) = ‖Yc‖² − 2 Σ_j f_j u_j + Σ_j f_j² s_j u_j,  f = 1/(s+λ)
        GCV(λ) = (SSE/n) / (1 − df/n)²              # df ≥ n ⇒ GCV := +inf

    X is standardized on train (ddof=0, +1e-9 — the #658/#841 convention); the
    target is train-mean-centered and the intercept ``ymu`` added back at
    prediction (identical contract to ``issue_841.maps.fit_ridge_split`` /
    ``fit_ridge_primal``, verified by ``verify_ridge_gcv_against_dual``).
    Returns ``(RidgeMap, diag)`` — the map is RAW-target-space (``sigma``
    rescales at apply time exactly as in #841); ``diag`` carries the GCV curve,
    selected λ, eigh wall-seconds, the fit-space train mean, and the
    eigendecomposition under ``"eig"``. When the DESIGN is shared across
    solves (the embedding arm — layer-independent X), pass the first solve's
    ``diag["eig"]`` back via ``eig=`` to skip the redundant eigh.
    """
    n, d = stats.n, int(stats.sx.shape[0])
    mu = stats.sx / n
    ymu = stats.sy / n
    A = stats.sxx - n * torch.outer(mu, mu)
    var = torch.clamp(torch.diagonal(A) / n, min=0.0)
    sd = var.sqrt() + sd_eps
    Astd = A / torch.outer(sd, sd)
    B = (stats.sxy - torch.outer(mu, stats.sy)) / sd.unsqueeze(1)  # XnᵀYc
    cyy = max(stats.syy - n * float((ymu * ymu).sum().item()), 0.0)
    if eig is not None:
        s, V = eig
        eigh_s = 0.0
    else:
        t0 = time.time()
        s, V = torch.linalg.eigh(Astd)
        eigh_s = time.time() - t0
        s = torch.clamp(s, min=0.0)
    U = V.t() @ B
    u = (U * U).sum(1)  # (d,)
    gcv_curve, sse_curve = [], []
    for lam in lambdas:
        f = 1.0 / (s + lam)
        df = 1.0 + float((s * f).sum().item())
        sse = max(
            cyy - 2.0 * float((f * u).sum().item()) + float((f * f * s * u).sum().item()), 0.0
        )
        denom = 1.0 - df / n
        gcv = float("inf") if denom <= 0 else (sse / n) / (denom * denom)
        gcv_curve.append(gcv)
        sse_curve.append(sse)
    best_i = int(np.argmin(gcv_curve))
    best_lam = float(lambdas[best_i])
    f = 1.0 / (s + best_lam)
    w = V @ (f.unsqueeze(1) * U)  # (d, p) fp64, centered-target weights
    rmap = RidgeMap(
        mu=mu.to(torch.float32),
        sd=sd.to(torch.float32),
        w=w.to(torch.float32),
        bias=ymu.to(torch.float32),
        best_lam=best_lam,
        sigma=float(sigma),
    )
    diag = {
        "best_lam": best_lam,
        "gcv_curve": gcv_curve,
        "sse_curve": sse_curve,
        "eigh_seconds": eigh_s,
        "n": n,
        "d": d,
        "ymu_fitspace": ymu.to(torch.float32).cpu(),
        "eig": (s, V),
    }
    return rmap, diag


def ridge_predict(rmap: RidgeMap, X: torch.Tensor) -> torch.Tensor:
    """Fit-space Δ̂ prediction (WITHOUT the raw-space sigma rescale): ȳ + Xn w."""
    xn = (X - rmap.mu) / rmap.sd
    return rmap.bias + xn @ rmap.w


def verify_ridge_gcv_against_dual(
    seed: int = 0, n: int = 200, d: int = 24, p: int = 6, lam: float = 10.0
) -> dict:
    """Equivalence gate: the Gram/eigh path matches #658's dual solver at fixed λ.

    Builds a toy (X, Y), fits (a) via ``accumulate_grams``-shaped stats +
    ``ridge_gcv_from_grams``'s eigenbasis weights AT THE FIXED λ, and (b) via
    the ported ``_ridge_dual_weights`` on the identically standardized/centered
    data. Asserts max|Δpred| ≤ 1e-8 on an eval slice (both fp64 — exact-math
    identity, only round-off differs). The gate exercises the EXACT callables
    the fit entrypoint dispatches (``ridge_gcv_from_grams`` via
    ``GramStats.add_chunk``), per the hollow-gate rule.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, p))
    Y = X @ W * 0.1 + rng.standard_normal((n, p)) * 0.05
    Xe = rng.standard_normal((40, d))
    Xt = torch.from_numpy(X).to(torch.float64)
    Yt = torch.from_numpy(Y).to(torch.float64)
    stats = GramStats.zeros(d, p, "cpu")
    for lo in range(0, n, 64):  # chunked, same code path as production
        stats.add_chunk(Xt[lo : lo + 64], Yt[lo : lo + 64])
    rmap, _ = ridge_gcv_from_grams(stats, lambdas=[lam])
    pred_gram = ridge_predict(rmap.to("cpu"), torch.from_numpy(Xe).to(torch.float32))

    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    Xn = (Xt - mu) / sd
    ymu = Yt.mean(0)
    w_dual = _ridge_dual_weights(Xn, Yt - ymu, lam)  # (d,p) fp64
    Xen = (torch.from_numpy(Xe).to(torch.float64) - mu) / sd
    pred_dual = ymu + Xen @ w_dual
    max_abs = float((pred_gram.to(torch.float64) - pred_dual).abs().max().item())
    tol = 1e-4  # fp32 RidgeMap storage rounds the fp64 solve; exact-math identity otherwise
    assert max_abs <= tol, f"ridge Gram/eigh vs dual parity FAILED: {max_abs:.3e} > {tol}"
    return {"max_abs_delta": max_abs, "tol": tol, "lam": lam}


# ── store-fed batched MLP wrapper ─────────────────────────────────────────────


def _gather_arm(
    h_store: torch.Tensor, row: int, idx: torch.Tensor, arm: str, emb_row: int = 0
) -> np.ndarray:
    """(n, d_arm) fp32 numpy X for one (row, arm) from the fp16 store."""
    h = h_store[row, idx, :].to(torch.float32)
    if arm == "ctx":
        return h.cpu().numpy()
    e = h_store[emb_row, idx + 1, :].to(torch.float32)
    if arm == "emb":
        return e.cpu().numpy()
    assert arm == "tok", arm
    return torch.cat([h, e], dim=1).cpu().numpy()


def _gather_delta(h_store: torch.Tensor, row: int, idx: torch.Tensor) -> np.ndarray:
    d = h_store[row, idx + 1, :].to(torch.float32) - h_store[row, idx, :].to(torch.float32)
    return d.cpu().numpy()


def fit_position_mlps(
    h_store: torch.Tensor,
    rows: list[int],
    fit_idx: torch.Tensor,
    val_idx: torch.Tensor,
    *,
    arm: str,
    space: str,
    sigma_by_row: dict[int, float],
    device: str,
    seed: int = 658,
    hidden: int = MLP_HIDDEN,
    lr: float = MLP_LR,
    wd: float = MLP_WD,
    max_epochs: int = MLP_MAX_EPOCHS,
    layer_chunk: int = 1,
    emb_row: int = 0,
) -> dict:
    """ONE batched multi-layer MLP fit per (arm, space), store-fed and chunked.

    The vectorize-many-cell-fits shape: all ``rows`` (layers) form one batched
    ensemble dispatched to the ported ``fit_batched_split_mlp``; because the
    full (29, n≈150k, d) fp32 group stack cannot materialize (≈62-125 GB),
    groups are gathered from the fp16 store ``layer_chunk`` layers at a time
    and fed through the SAME trainer (its own chunk loop then runs the batch).
    Per-chunk reseeding means every chunk's groups draw inits from the same
    seed-``seed`` sequence — deterministic and reproducible; init identity
    across layers is immaterial (independent problems). Loss = SmoothL1(β=1)
    on Δ/σ (``space``: raw ⇒ σ≡1); early stop = per-member best-val snapshot
    (the fit_batched_split_mlp contract). Returns
    ``{row: {"params": {...}, "best_val_epoch": int}}``.
    """
    out: dict = {}
    for lo in range(0, len(rows), max(1, layer_chunk)):
        chunk_rows = rows[lo : lo + max(1, layer_chunk)]
        groups = []
        for r in chunk_rows:
            sig = 1.0 if space == "raw" else float(sigma_by_row[r])
            Xtr = _gather_arm(h_store, r, fit_idx, arm, emb_row)
            Ytr = _gather_delta(h_store, r, fit_idx) / sig
            Xval = _gather_arm(h_store, r, val_idx, arm, emb_row)
            Yval = _gather_delta(h_store, r, val_idx) / sig
            groups.append(
                SplitMLPGroup(
                    key=(r,), X_train=Xtr, Y_train=Ytr, X_eval=Xval, X_val=Xval, Y_val=Yval
                )
            )
        t0 = time.time()
        res = fit_batched_split_mlp(
            groups,
            seed=seed,
            hidden=hidden,
            lr=lr,
            wd=wd,
            max_epochs=max_epochs,
            device=device,
            chunk_size=len(groups),
            smooth_l1_beta=1.0,
        )
        logger.info(
            "[mlp] arm=%s space=%s rows=%s fit in %.1fs (n=%d, epochs<=%d)",
            arm,
            space,
            chunk_rows,
            time.time() - t0,
            len(fit_idx),
            max_epochs,
        )
        for r in chunk_rows:
            out[r] = {
                "params": res.params_by_key[(r,)],
                "best_val_epoch": res.best_val_epoch_by_key[(r,)],
            }
        del groups, res
    return out


def apply_mlp_params(params: dict, X: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Fit-space Δ̂ from stored MLP params on (n, d_in) X — via ``_split_mlp_eval``."""
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    pred = _split_mlp_eval(
        X.to(device).to(torch.float32).unsqueeze(0),
        t(params["W1"]).unsqueeze(0),
        t(params["b1"]).unsqueeze(0),
        t(params["W2"]).unsqueeze(0),
        t(params["b2"]).unsqueeze(0),
        t(params["mu"]).unsqueeze(0),
        t(params["sd"]).unsqueeze(0),
    )
    return pred[0]


# ── batched rollout engine (ridge / MLP maps, all layers at once) ─────────────


def apply_ridge_maps_batched(
    H: torch.Tensor, mus: torch.Tensor, sds: torch.Tensor, ws: torch.Tensor, biases: torch.Tensor
) -> torch.Tensor:
    """Raw Δ̂ for a stack of per-layer ridge maps: (L,N,d) → (L,N,p) via bmm.

    ``mus/sds`` (L,d), ``ws`` (L,d,p), ``biases`` (L,p). Raw-space maps only
    (sigma=1 — the rollout composes raw Δ̂, the #841 transport convention).
    """
    hn = (H - mus.unsqueeze(1)) / sds.unsqueeze(1)
    return torch.baddbmm(biases.unsqueeze(1), hn, ws)


def roll_states_ridge(
    seed_h: torch.Tensor,
    boundary_stack: dict,
    answer_stack: dict,
    k_max: int,
    *,
    emb_next: torch.Tensor | None = None,
    use_boundary_first: bool = True,
) -> list[torch.Tensor]:
    """Autoregressive position roll for L layers at once.

    ``seed_h`` (L, N, H) fp32 = true h_{l,T}. Step 1 applies the BOUNDARY maps
    (fit on the prompt→answer transition), steps ≥2 the ANSWER maps (plan
    §4.3 primary path); ``use_boundary_first=False`` gives the naive
    all-answer-map diagnostic roll. Stacks are ``{"mus","sds","ws","biases"}``
    with (L,·) leading dims; for token-informed stacks ``emb_next`` (L→shared)
    must supply (N, k_max, H) TRUE next-token embeddings (h_{0,T+k}); the ctx
    stacks take ``emb_next=None``. Returns ``[ĥ_{T+1}, ..., ĥ_{T+k_max}]``
    each (L, N, H) fp32 — the caller scores/streams them.
    """
    out = []
    h = seed_h
    for k in range(1, k_max + 1):
        stack = boundary_stack if (k == 1 and use_boundary_first) else answer_stack
        if emb_next is not None:
            e = emb_next[:, k - 1, :].unsqueeze(0).expand(h.shape[0], -1, -1)
            x = torch.cat([h, e], dim=2)
        else:
            x = h
        delta = apply_ridge_maps_batched(
            x, stack["mus"], stack["sds"], stack["ws"], stack["biases"]
        )
        h = h + delta
        out.append(h)
    return out


# ── exploratory position-GRU (per read-out layer) ─────────────────────────────


class PositionGRU(torch.nn.Module):
    """1-layer GRU over token positions predicting the raw update Δ_{l,t}.

    Input h_{l,t} standardized by the fit-set per-dim moments; head maps the
    hidden state to Δ̂ (raw space — rollout composes raw). Exploratory only
    (plan §4.3): at rollout it is warmed on the observed prompt-window
    positions (legal information at forecast time) then rolled, so it carries
    strictly more information than the memoryless maps and is reported beside,
    never inside, the context-only headline.
    """

    def __init__(self, d: int, hidden: int = 1024):
        super().__init__()
        self.gru = torch.nn.GRU(d, hidden, num_layers=1, batch_first=True)
        self.head = torch.nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        out, hn = self.gru(x, h0)
        return self.head(out), hn


def fit_position_gru(
    seqs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    val_sel: torch.Tensor,
    *,
    mu: torch.Tensor,
    sd: torch.Tensor,
    device: str,
    hidden: int = 1024,
    lr: float = MLP_LR,
    wd: float = MLP_WD,
    max_epochs: int = 40,
    batch_size: int = 256,
    seed: int = 658,
) -> tuple[PositionGRU, dict]:
    """Teacher-forced GRU fit on padded window sequences (one read-out layer).

    ``seqs`` (N, T, H) fp32 inputs h_{l,t}; ``targets`` (N, T, H) raw Δ;
    ``mask`` (N, T) bool valid steps; ``val_sel`` (N,) bool marks inner-val
    sequences (train = ~val_sel). SmoothL1(β=1) on masked steps; best-val
    snapshot early stopping. Returns the best-val model (on ``device``) + diag.
    """
    dev = torch.device(device)
    torch.manual_seed(seed)
    net = PositionGRU(seqs.shape[-1], hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    tr_idx = torch.nonzero(~val_sel, as_tuple=True)[0]
    va_idx = torch.nonzero(val_sel, as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    best_val, best_state, best_epoch = math.inf, None, -1

    def _loss_on(idx: torch.Tensor, train: bool) -> float:
        total, count = 0.0, 0
        for lo in range(0, len(idx), batch_size):
            sel = idx[lo : lo + batch_size]
            x = ((seqs[sel].to(dev) - mu) / sd).to(torch.float32)
            y = targets[sel].to(dev)
            m = mask[sel].to(dev)
            pred, _ = net(x)
            per = torch.nn.functional.smooth_l1_loss(pred, y, reduction="none", beta=1.0)
            per = (per.mean(-1) * m).sum() / m.sum().clamp(min=1)
            if train:
                opt.zero_grad(set_to_none=True)
                per.backward()
                opt.step()
            total += float(per.item()) * int(m.sum().item())
            count += int(m.sum().item())
        return total / max(count, 1)

    for epoch in range(max_epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=g)]
        _loss_on(perm, train=True)
        with torch.no_grad():
            vloss = _loss_on(va_idx, train=False)
        if vloss < best_val:
            best_val, best_epoch = vloss, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"best_val_loss": best_val, "best_val_epoch": best_epoch}


@torch.no_grad()
def gru_roll_states(
    net: PositionGRU,
    warm_seq: torch.Tensor,
    seed_h: torch.Tensor,
    k_max: int,
    *,
    mu: torch.Tensor,
    sd: torch.Tensor,
) -> list[torch.Tensor]:
    """Warm the GRU on the observed prompt-window states, then roll k steps.

    ``warm_seq`` (N, T_warm, H) — the prompt-window states up to (and
    including) h_{T}; ``seed_h`` (N, H) = h_{T}. Returns [ĥ_{T+1}..ĥ_{T+k}].
    """
    xw = (warm_seq - mu) / sd
    out, hn = net(xw.to(torch.float32))
    h = seed_h + out[:, -1, :]  # first step: Δ̂ predicted at the T position
    states = [h]
    for _ in range(k_max - 1):
        x = ((h - mu) / sd).to(torch.float32).unsqueeze(1)
        o, hn = net(x, hn)
        h = h + o[:, -1, :]
        states.append(h)
    return states


# ═══════════════════════════════════════════════════════════════════════════════
# v6 AMENDMENT (plan §4.3b): context-conditioned arms (b1 additive, b2
# operator-valued in three capacity-matched structured linear forms) + direct
# per-horizon maps (arm c) + the fit-phase device-budget arithmetic (the r1
# code-review CRITICAL fix). c := h_{l,T} (same-layer last formatted-prompt
# state) everywhere.
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONED_FORMS = ("b1_grad", "film", "lowrank", "mixture")
# The v3 MLP recipe verbatim (plan §11): AdamW lr 1e-3 / wd 1e-4, SmoothL1 β=1
# on raw Δ, ≤300 epochs, best-inner-val snapshot, init seed 658. ``patience``
# stops the epoch loop once NO row has improved for that many epochs (a
# wall-time measure on top of the best-val snapshot; the snapshot semantics
# are unchanged — fit_batched_split_mlp's contract).
CONDITIONED_RECIPE_922 = {
    "lr": MLP_LR,
    "wd": MLP_WD,
    "max_epochs": MLP_MAX_EPOCHS,
    "beta": 1.0,
    "batch_size": 8192,
    "patience": 25,
}
LOWRANK_RANK_922 = 1195  # r = round(d/3): d² + 3dr = 2d² (capacity-pinned, §4.3b)
MIXTURE_K_922 = 2  # K·d² = 2d² (capacity-pinned)


def fit_phase_gpu_budget_bytes(
    store_bytes: int, h_dim: int, row_chunk: int, *, extra_gib: float = 4.0
) -> tuple[int, int]:
    """(need_with_store, need_grams_only) device bytes for the fit phase.

    The r1 CRITICAL fix: the store-to-GPU decision must budget the Gram
    ASSEMBLY footprint, not just the store. Per row-chunk the answer-segment
    pass holds, in fp64: 4 per-row blocks (Shh/She/Shy/Sey = 4·H²), the tok
    assembly ((2H)² + 2H·H = 6·H²), and the emb ``See.clone()`` (1·H²) — 11·H²
    per row. The b1 second pass (3 new blocks + ONE row's assembled 2H stats)
    peaks lower. Add the d=2H fp64 eigh workspace (input + V + work ≈ 6 copies
    of (2H)²... conservatively) and a fixed margin.
    """
    hh = h_dim * h_dim * 8
    answer_leg = 11 * hh * row_chunk
    b1_leg = (2 + 3) * hh * row_chunk + 6 * hh  # kept ctx stats + new blocks + 1-row assembly
    eigh_ws = 6 * (2 * h_dim) * (2 * h_dim) * 8
    grams_only = max(answer_leg, b1_leg) + eigh_ws + int(extra_gib * (1 << 30))
    return store_bytes + grams_only, grams_only


# ── b1 closed-form: second streaming Gram pass over [h_t, c] ─────────────────


def accumulate_grams_b1(
    h_store: torch.Tensor,
    src_idx: torch.Tensor,
    src_T_idx: torch.Tensor,
    rows: list[int],
    *,
    chunk: int = 4096,
    device: str = "cpu",
) -> dict:
    """SECOND streaming pass: the c-blocks of the b1 design X = [h_t, c].

    ``src_T_idx`` (n,) maps each transition to its context's T-row (global
    store position of the last formatted-prompt token), so c = h_{l,T} gathers
    per chunk at the SAME layer (c is per-layer — no shared block). Returns
    ``{row: {"Shc","Scc","Scy","Sc"}}`` fp64 on ``device``; assemble the 2H
    GramStats per row with :func:`assemble_b1_stats` (fit-and-free — never
    hold all rows' assembled 2H stats at once).
    """
    dev = torch.device(device)
    _R, _P, H = h_store.shape
    n = int(src_idx.numel())
    assert src_T_idx.numel() == n, (src_T_idx.numel(), n)
    per_row = {
        r: {
            "Shc": torch.zeros(H, H, dtype=torch.float64, device=dev),
            "Scc": torch.zeros(H, H, dtype=torch.float64, device=dev),
            "Scy": torch.zeros(H, H, dtype=torch.float64, device=dev),
            "Sc": torch.zeros(H, dtype=torch.float64, device=dev),
        }
        for r in rows
    }
    t0 = time.time()
    for lo in range(0, n, chunk):
        sel = src_idx[lo : lo + chunk]
        selT = src_T_idx[lo : lo + chunk]
        for r in rows:
            hh = h_store[r, sel, :].to(device=dev, dtype=torch.float64)
            cc = h_store[r, selT, :].to(device=dev, dtype=torch.float64)
            y = h_store[r, sel + 1, :].to(device=dev, dtype=torch.float64) - hh
            blk = per_row[r]
            blk["Shc"] += hh.t() @ cc
            blk["Scc"] += cc.t() @ cc
            blk["Scy"] += cc.t() @ y
            blk["Sc"] += cc.sum(0)
    logger.info(
        "[grams-b1] accumulated %d rows x %d layers (c-blocks) in %.1fs",
        n,
        len(rows),
        time.time() - t0,
    )
    return per_row


def assemble_b1_stats(ctx_stats: GramStats, blk: dict) -> GramStats:
    """Assemble one row's d=2H GramStats for X=[h_t, c] (the r1 tok-assembly shape)."""
    sx = torch.cat([ctx_stats.sx, blk["Sc"]])
    sxx = torch.cat(
        [
            torch.cat([ctx_stats.sxx, blk["Shc"]], 1),
            torch.cat([blk["Shc"].t(), blk["Scc"]], 1),
        ],
        0,
    )
    sxy = torch.cat([ctx_stats.sxy, blk["Scy"]], 0)
    return GramStats(ctx_stats.n, sx, sxx, sxy, ctx_stats.sy, ctx_stats.syy)


# ── conditioned forms: params, forward, batched gradient trainer ──────────────


def conditioned_param_count(
    form: str, d: int, *, rank: int = LOWRANK_RANK_922, n_mix: int = MIXTURE_K_922
) -> int:
    """WEIGHT parameter count per row (biases ≈ d excluded — the §4.3b table)."""
    if form in ("b1_grad", "film"):
        return 2 * d * d
    if form == "lowrank":
        return d * d + 3 * d * rank
    if form == "mixture":
        return n_mix * d * d + n_mix * d
    raise ValueError(f"unknown conditioned form {form!r}")


def init_conditioned_params(
    form: str,
    d: int,
    *,
    rank: int = LOWRANK_RANK_922,
    n_mix: int = MIXTURE_K_922,
    seed: int = 658,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Seeded fp32 init (scale d^-1/2 weights, zero bias) for one row's params.

    Deterministic on CPU then moved (device-independent inits). ``lowrank``
    needs non-zero U/V/Ws (the gated core is multiplicative); ``mixture``
    starts near the uniform gate (small Ww).
    """
    g = torch.Generator().manual_seed(seed)
    s = d**-0.5

    def _rnd(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * s).to(device)

    if form == "b1_grad":
        w = {"Wh": _rnd(d, d), "Wc": _rnd(d, d)}
    elif form == "film":
        w = {"A": _rnd(d, d), "Wg": _rnd(d, d)}
    elif form == "lowrank":
        w = {"A": _rnd(d, d), "U": _rnd(d, rank), "V": _rnd(d, rank), "Ws": _rnd(d, rank)}
    elif form == "mixture":
        w = {f"Am{m}": _rnd(d, d) for m in range(n_mix)}
        w["Ww"] = _rnd(d, n_mix)
    else:
        raise ValueError(form)
    w["b"] = torch.zeros(d, dtype=torch.float32, device=device)
    return w


def apply_conditioned_delta(
    form: str, w: dict[str, torch.Tensor], hn: torch.Tensor, cn: torch.Tensor
) -> torch.Tensor:
    """Fit-space Δ̂ for STANDARDIZED inputs; broadcasts over leading dims.

    Works both unstacked (``hn`` (n,d), weights (d,d), bias (d,)) and
    row-stacked (``hn`` (L,n,d), weights (L,d,d), bias (L,d) — matmul
    broadcasting = bmm; the bias gains the broadcast dim here). This is the
    ONE forward the trainer, the single-step eval, and the conditioned rollout
    all dispatch (hollow-gate rule).
    """
    b = w["b"]
    if b.dim() == 2 and hn.dim() == 3:  # row-stacked (L,d) bias vs (L,n,d) inputs
        b = b.unsqueeze(1)
    if form == "b1_grad":
        return hn @ w["Wh"] + cn @ w["Wc"] + b
    if form == "film":
        return hn @ w["A"] + (cn @ w["Wg"]) * hn + b
    if form == "lowrank":
        core = (cn @ w["Ws"]) * (hn @ w["V"])  # (..., r), c-gated
        return hn @ w["A"] + core @ w["U"].transpose(-2, -1) + b
    if form == "mixture":
        alpha = torch.softmax(cn @ w["Ww"], dim=-1)  # (..., K)
        out = torch.zeros_like(hn)
        m = 0
        while f"Am{m}" in w:
            out = out + alpha[..., m : m + 1] * (hn @ w[f"Am{m}"])
            m += 1
        return out + b
    raise ValueError(f"unknown conditioned form {form!r}")


def _moments_fp64(x_fp16: torch.Tensor, chunk: int = 65536) -> tuple[torch.Tensor, torch.Tensor]:
    """(mu, sd) fp32 with the split-MLP convention (ddof=1, +1e-6), fp64 sums."""
    n, d = x_fp16.shape
    s = torch.zeros(d, dtype=torch.float64, device=x_fp16.device)
    s2 = torch.zeros(d, dtype=torch.float64, device=x_fp16.device)
    for lo in range(0, n, chunk):
        xx = x_fp16[lo : lo + chunk].to(torch.float64)
        s += xx.sum(0)
        s2 += (xx * xx).sum(0)
    mu = s / n
    var = torch.clamp((s2 - n * mu * mu) / max(n - 1, 1), min=0.0)
    return mu.to(torch.float32), (var.sqrt() + 1e-6).to(torch.float32)


def fit_conditioned_linear(
    h_store: torch.Tensor,
    rows: list[int],
    fit_idx: torch.Tensor,
    val_idx: torch.Tensor,
    fit_T_idx: torch.Tensor,
    val_T_idx: torch.Tensor,
    *,
    form: str,
    rank: int = LOWRANK_RANK_922,
    n_mix: int = MIXTURE_K_922,
    device: str = "cpu",
    seed: int = 658,
    recipe: dict | None = None,
    layer_chunk: int = 1,
) -> dict[int, dict]:
    """ONE shared gradient entry point for all four conditioned linear forms.

    The §4.3b contract: b1_grad / film / lowrank / mixture are recipe-identical
    BY CONSTRUCTION (same optimizer, loss, standardization, early stop, seed,
    minibatching) because they all run through this trainer, differing only in
    ``apply_conditioned_delta``'s form branch. Rows are fit BATCHED per
    ``layer_chunk`` (stacked leading-L tensors, matmul-broadcast bmm — the
    vectorize-many-cell-fits shape; the full 9-row stack of 25.7M-param forms
    + AdamW state does not fit HBM beside the store, hence the chunking, the
    same accepted pattern as ``fit_position_mlps``). Loss = SmoothL1(β) on raw
    Δ (σ≡1 — the v4 arms are raw-space only). Inputs standardized per row with
    the split-MLP convention (ddof=1, +1e-6); c = h_{l,T} via ``fit_T_idx``.
    Returns ``{row: {"form","weights","mu_h","sd_h","mu_c","sd_c",
    "best_val_epoch","best_val_loss","n_epochs_run","n_params_weights",
    "val_curve"}}`` (weights = best-val snapshot, fp32 CPU).
    """
    rec = {**CONDITIONED_RECIPE_922, **(recipe or {})}
    dev = torch.device(device)
    n_fit, n_val = int(fit_idx.numel()), int(val_idx.numel())
    out: dict[int, dict] = {}
    form_ix = CONDITIONED_FORMS.index(form)
    for lo in range(0, len(rows), max(1, layer_chunk)):
        chunk_rows = rows[lo : lo + max(1, layer_chunk)]
        L = len(chunk_rows)
        # gather fp16 inputs on device (h_t, c, raw Δ target), stacked (L,n,d)
        Xh = torch.stack([h_store[r, fit_idx, :] for r in chunk_rows]).to(dev)
        Xc = torch.stack([h_store[r, fit_T_idx, :] for r in chunk_rows]).to(dev)
        Y = torch.stack(
            [
                (
                    h_store[r, fit_idx + 1, :].to(torch.float32)
                    - h_store[r, fit_idx, :].to(torch.float32)
                ).to(torch.float16)
                for r in chunk_rows
            ]
        ).to(dev)
        Vh = torch.stack([h_store[r, val_idx, :] for r in chunk_rows]).to(dev)
        Vc = torch.stack([h_store[r, val_T_idx, :] for r in chunk_rows]).to(dev)
        Vy = torch.stack(
            [
                (
                    h_store[r, val_idx + 1, :].to(torch.float32)
                    - h_store[r, val_idx, :].to(torch.float32)
                ).to(torch.float16)
                for r in chunk_rows
            ]
        ).to(dev)
        moms = [(_moments_fp64(Xh[li]), _moments_fp64(Xc[li])) for li in range(L)]
        mu_h = torch.stack([m[0][0] for m in moms]).unsqueeze(1).to(dev)  # (L,1,d)
        sd_h = torch.stack([m[0][1] for m in moms]).unsqueeze(1).to(dev)
        mu_c = torch.stack([m[1][0] for m in moms]).unsqueeze(1).to(dev)
        sd_c = torch.stack([m[1][1] for m in moms]).unsqueeze(1).to(dev)
        # stacked params: (L, *shape); per-row deterministic seeded init
        per_row_init = [
            init_conditioned_params(
                form, Xh.shape[-1], rank=rank, n_mix=n_mix, seed=seed + 131 * r + form_ix
            )
            for r in chunk_rows
        ]
        W = {
            k: torch.stack([p[k] for p in per_row_init]).to(dev).requires_grad_(True)
            for k in per_row_init[0]
        }
        opt = torch.optim.AdamW(W.values(), lr=rec["lr"], weight_decay=rec["wd"])
        gen = torch.Generator().manual_seed(seed)
        bs = int(rec["batch_size"])
        best_val = [math.inf] * L
        best_epoch = [-1] * L
        best_snap: list[dict | None] = [None] * L
        val_curve: list[list[float]] = []
        n_epochs_run = 0
        for epoch in range(int(rec["max_epochs"])):
            perm = torch.randperm(n_fit, generator=gen)
            for blo in range(0, n_fit, bs):
                sel = perm[blo : blo + bs].to(dev)
                hn = (Xh[:, sel].to(torch.float32) - mu_h) / sd_h
                cn = (Xc[:, sel].to(torch.float32) - mu_c) / sd_c
                pred = apply_conditioned_delta(form, W, hn, cn)
                loss = torch.nn.functional.smooth_l1_loss(
                    pred, Y[:, sel].to(torch.float32), beta=rec["beta"]
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                vtot = torch.zeros(L, dtype=torch.float64, device=dev)
                for blo in range(0, n_val, bs):
                    hn = (Vh[:, blo : blo + bs].to(torch.float32) - mu_h) / sd_h
                    cn = (Vc[:, blo : blo + bs].to(torch.float32) - mu_c) / sd_c
                    pred = apply_conditioned_delta(form, W, hn, cn)
                    per = torch.nn.functional.smooth_l1_loss(
                        pred,
                        Vy[:, blo : blo + bs].to(torch.float32),
                        beta=rec["beta"],
                        reduction="none",
                    )
                    vtot += per.mean(dim=-1).sum(dim=-1).to(torch.float64)
                vloss = (vtot / max(n_val, 1)).cpu().tolist()
            assert all(np.isfinite(v) for v in vloss), (form, chunk_rows, epoch, vloss)
            val_curve.append([float(v) for v in vloss])
            for li in range(L):
                if vloss[li] < best_val[li] - 1e-9:
                    best_val[li] = vloss[li]
                    best_epoch[li] = epoch
                    best_snap[li] = {
                        k: v[li].detach().to("cpu", torch.float32).clone() for k, v in W.items()
                    }
            n_epochs_run = epoch + 1
            if all(epoch - be >= rec["patience"] for be in best_epoch):
                break
        for li, r in enumerate(chunk_rows):
            assert best_snap[li] is not None, (form, r)
            out[r] = {
                "form": form,
                "weights": best_snap[li],
                "mu_h": mu_h[li, 0].cpu(),
                "sd_h": sd_h[li, 0].cpu(),
                "mu_c": mu_c[li, 0].cpu(),
                "sd_c": sd_c[li, 0].cpu(),
                "best_val_epoch": best_epoch[li],
                "best_val_loss": best_val[li],
                "n_epochs_run": n_epochs_run,
                "n_params_weights": conditioned_param_count(
                    form, Xh.shape[-1], rank=rank, n_mix=n_mix
                ),
                "val_curve": [vc[li] for vc in val_curve],
            }
        logger.info(
            "[cond] form=%s rows=%s best_val_epochs=%s (%d epochs run)",
            form,
            chunk_rows,
            [out[r]["best_val_epoch"] for r in chunk_rows],
            n_epochs_run,
        )
        del Xh, Xc, Y, Vh, Vc, Vy, W, opt
    return out


def conditioned_predict_row(
    pblob: dict, h: torch.Tensor, c: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """Raw Δ̂ for ONE row's fitted conditioned params on (n,d) h and c."""
    dev = torch.device(device)
    w = {k: v.to(dev) for k, v in pblob["weights"].items()}
    hn = (h.to(dev, torch.float32) - pblob["mu_h"].to(dev)) / pblob["sd_h"].to(dev)
    cn = (c.to(dev, torch.float32) - pblob["mu_c"].to(dev)) / pblob["sd_c"].to(dev)
    return apply_conditioned_delta(pblob["form"], w, hn, cn)


# ── conditioned + b1-ridge rollout engines (batched over rows) ────────────────


def stack_conditioned_params(per_row: dict[int, dict], rows: list[int], device: str) -> dict:
    """Stack per-row conditioned params into leading-L tensors for the roll."""
    dev = torch.device(device)
    keys = per_row[rows[0]]["weights"].keys()
    return {
        "form": per_row[rows[0]]["form"],
        "weights": {k: torch.stack([per_row[r]["weights"][k] for r in rows]).to(dev) for k in keys},
        "mu_h": torch.stack([per_row[r]["mu_h"] for r in rows]).to(dev),
        "sd_h": torch.stack([per_row[r]["sd_h"] for r in rows]).to(dev),
        "mu_c": torch.stack([per_row[r]["mu_c"] for r in rows]).to(dev),
        "sd_c": torch.stack([per_row[r]["sd_c"] for r in rows]).to(dev),
    }


def roll_states_conditioned(
    seed_h: torch.Tensor,
    boundary_stack: dict,
    cond_stack: dict,
    k_max: int,
) -> list[torch.Tensor]:
    """Conditioned autoregressive roll: v3 boundary ctx map at k=1, then A(c)-form.

    ``seed_h`` (L, N, H) fp32 = true h_{l,T} ≡ c (the boundary identity — plan
    §4.3b's rollout discipline: a conditioned boundary fit would be a
    degenerate re-parameterization, so step 1 uses the v3 boundary ridge
    verbatim); c is re-injected every step k ≥ 2. Batched over rows via
    matmul broadcasting; no per-step Python beyond the k loop.
    """
    cn = (seed_h - cond_stack["mu_c"].unsqueeze(1)) / cond_stack["sd_c"].unsqueeze(1)
    out = []
    h = seed_h
    for k in range(1, k_max + 1):
        if k == 1:
            delta = apply_ridge_maps_batched(
                h,
                boundary_stack["mus"],
                boundary_stack["sds"],
                boundary_stack["ws"],
                boundary_stack["biases"],
            )
        else:
            hn = (h - cond_stack["mu_h"].unsqueeze(1)) / cond_stack["sd_h"].unsqueeze(1)
            delta = apply_conditioned_delta(cond_stack["form"], cond_stack["weights"], hn, cn)
        h = h + delta
        out.append(h)
    return out


def roll_states_b1_ridge(
    seed_h: torch.Tensor,
    boundary_stack: dict,
    b1_stack: dict,
    k_max: int,
) -> list[torch.Tensor]:
    """Closed-form b1 roll: boundary ctx map at k=1, then the [h, c] ridge.

    c ≡ seed_h is concatenated onto the current state each step (the additive
    conditioning re-injection); the b1 stack's maps have d=2H designs.
    """
    out = []
    h = seed_h
    for k in range(1, k_max + 1):
        if k == 1:
            x, stack = h, boundary_stack
        else:
            x, stack = torch.cat([h, seed_h], dim=2), b1_stack
        delta = apply_ridge_maps_batched(
            x, stack["mus"], stack["sds"], stack["ws"], stack["biases"]
        )
        h = h + delta
        out.append(h)
    return out


# ── arm c: direct per-horizon GCV ridge (exact per-k designs, batched eigh) ───


def fit_direct_horizon_maps(
    h_store: torch.Tensor,
    row: int,
    T_pos: torch.Tensor,
    kcap: np.ndarray,
    *,
    k_max: int = 40,
    lambdas: list[float] = RIDGE_LAMBDAS_922,
    device: str = "cpu",
    k_chunk: int = 8,
    min_n: int = 8,
    sd_eps: float = 1e-9,
) -> dict:
    """Per-horizon GCV ridge c → D_k := h_{l,T+k} − h_{l,T} for ONE row.

    Exact per-k designs: the valid context set at horizon k is
    ``{i: kcap_i ≥ k}`` (nested — contexts sorted by kcap descending give
    prefix designs), so the plan's "one eigh shared across horizons" holds
    ONLY between horizons whose valid sets coincide; where the set thins the
    Gram genuinely changes and the eigendecomposition is recomputed EXACTLY —
    as ONE BATCHED ``torch.linalg.eigh`` call per k-chunk (no serial per-k
    Python eigh loop; the ``eig=``-style reuse fires for equal consecutive
    designs via the m_k dedupe). Prefix Grams are built incrementally (one
    pass of GEMMs over the sorted context matrix). ``k=1``'s design = the fit
    contexts with A ≥ 1 = EXACTLY the v3 boundary-map fit set (the §4.3b
    coherence identity). Returns ``{"maps": {k: RidgeMap}, "diag": {k:
    {"n","best_lam","eigh_batched"}}}``; horizons with n < ``min_n`` are
    skipped (recorded in diag).
    """
    dev = torch.device(device)
    kcap = np.minimum(np.asarray(kcap, dtype=np.int64), k_max)
    order = np.argsort(-kcap, kind="stable")
    Ts = torch.as_tensor(np.asarray(T_pos)[order], dtype=torch.long)
    kc = kcap[order]
    n0, H = int(Ts.numel()), h_store.shape[-1]
    m_of_k = {k: int((kc >= k).sum()) for k in range(1, k_max + 1)}
    C = h_store[row, Ts, :].to(device=dev, dtype=torch.float64)  # (n0, H) sorted by kcap desc
    # incremental prefix Grams at every distinct m (nested designs)
    distinct_m = sorted({m for m in m_of_k.values() if m >= min_n})
    G = torch.zeros(H, H, dtype=torch.float64, device=dev)
    s = torch.zeros(H, dtype=torch.float64, device=dev)
    prefix: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    prev = 0
    for m in distinct_m:
        blk = C[prev:m]
        G = G + blk.t() @ blk
        s = s + blk.sum(0)
        prefix[m] = (G.clone(), s.clone())
        prev = m
    maps: dict[int, RidgeMap] = {}
    diag: dict[int, dict] = {}
    lam_t = torch.tensor(lambdas, dtype=torch.float64, device=dev)  # (Λ,)
    eig_cache: dict[int, tuple] = {}
    for k_lo in range(1, k_max + 1, k_chunk):
        ks = [k for k in range(k_lo, min(k_lo + k_chunk, k_max + 1))]
        live = [k for k in ks if m_of_k[k] >= min_n]
        for k in ks:
            if k not in live:
                diag[k] = {"n": m_of_k[k], "skipped": f"n < min_n={min_n}"}
        if not live:
            continue
        # batched eigh over the chunk's NEW distinct designs
        new_ms = sorted({m_of_k[k] for k in live} - set(eig_cache))
        if new_ms:
            stack = []
            for m in new_ms:
                Gm, sm = prefix[m]
                mu = sm / m
                A = Gm - m * torch.outer(mu, mu)
                var = torch.clamp(torch.diagonal(A) / m, min=0.0)
                sd = var.sqrt() + sd_eps
                stack.append(A / torch.outer(sd, sd))
            t0 = time.time()
            evals, evecs = torch.linalg.eigh(torch.stack(stack))
            eigh_s = time.time() - t0
            for bi, m in enumerate(new_ms):
                Gm, sm = prefix[m]
                mu = sm / m
                A = Gm - m * torch.outer(mu, mu)
                var = torch.clamp(torch.diagonal(A) / m, min=0.0)
                sd = var.sqrt() + sd_eps
                eig_cache[m] = (
                    torch.clamp(evals[bi], min=0.0),
                    evecs[bi],
                    mu,
                    sd,
                    eigh_s / len(new_ms),
                )
            del stack, evals, evecs
        for k in live:
            m = m_of_k[k]
            se, V, mu, sd, eigh_s = eig_cache[m]
            tgt = h_store[row, Ts[:m] + k, :].to(device=dev, dtype=torch.float64) - C[:m]
            sxy = C[:m].t() @ tgt
            sy = tgt.sum(0)
            syy = float((tgt * tgt).sum().item())
            ymu = sy / m
            B = (sxy - torch.outer(mu, sy)) / sd.unsqueeze(1)
            cyy = max(syy - m * float((ymu * ymu).sum().item()), 0.0)
            U = V.t() @ B
            u = (U * U).sum(1)  # (d,)
            f = 1.0 / (se.unsqueeze(0) + lam_t.unsqueeze(1))  # (Λ, d)
            df = 1.0 + (se.unsqueeze(0) * f).sum(1)
            sse = torch.clamp(cyy - 2.0 * (f @ u) + ((f * f) @ (se * u)), min=0.0)
            denom = 1.0 - df / m
            gcv = torch.where(
                denom > 0, (sse / m) / (denom * denom), torch.full_like(sse, float("inf"))
            )
            best_i = int(torch.argmin(gcv).item())
            best_lam = float(lambdas[best_i])
            fbest = 1.0 / (se + best_lam)
            w = V @ (fbest.unsqueeze(1) * U)
            maps[k] = RidgeMap(
                mu=mu.to(torch.float32).cpu(),
                sd=sd.to(torch.float32).cpu(),
                w=w.to(torch.float32).cpu(),
                bias=ymu.to(torch.float32).cpu(),
                best_lam=best_lam,
                sigma=1.0,
            )
            diag[k] = {"n": m, "best_lam": best_lam, "eigh_seconds_share": eigh_s}
            del tgt, sxy, B, U
        # evict designs no longer reachable (m_k is non-increasing in k)
        min_m_left = min(
            (m_of_k[k] for k in range(k_lo + k_chunk, k_max + 1) if m_of_k[k] >= min_n),
            default=None,
        )
        if min_m_left is not None:
            for m in [
                m
                for m in eig_cache
                if m > max(m_of_k.get(kk, 0) for kk in range(k_lo + k_chunk, k_max + 1))
            ]:
                eig_cache.pop(m, None)
        else:
            eig_cache.clear()
    return {"maps": maps, "diag": diag, "n0": n0, "k_max": k_max}


@torch.no_grad()
def predict_direct_horizons_row(
    direct_row: dict, c: torch.Tensor, seed: torch.Tensor, k_max: int, device: str = "cpu"
) -> list[torch.Tensor]:
    """[ĥ_{T+1}..ĥ_{T+k_max}] for ONE row: ĥ_{T+k} = seed + D̂_k(c), batched over k.

    Missing-k maps (skipped fits) yield NaN states so downstream scoring
    excludes them. One bmm over the available horizons.
    """
    dev = torch.device(device)
    c32 = c.to(dev, torch.float32)
    seed32 = seed.to(dev, torch.float32)
    avail = [k for k in range(1, k_max + 1) if k in direct_row["maps"]]
    out: list[torch.Tensor] = [torch.full_like(seed32, float("nan")) for _ in range(k_max)]
    if not avail:
        return out
    mus = torch.stack([direct_row["maps"][k].mu for k in avail]).to(dev)  # (K,d)
    sds = torch.stack([direct_row["maps"][k].sd for k in avail]).to(dev)
    ws = torch.stack([direct_row["maps"][k].w for k in avail]).to(dev)  # (K,d,H)
    biases = torch.stack([direct_row["maps"][k].bias for k in avail]).to(dev)
    xn = (c32.unsqueeze(0) - mus.unsqueeze(1)) / sds.unsqueeze(1)  # (K,N,d)
    dhat = torch.baddbmm(biases.unsqueeze(1), xn, ws)  # (K,N,H)
    for ki, k in enumerate(avail):
        out[k - 1] = seed32 + dhat[ki]
    return out


# ── equivalence gates for the v6 paths (the --verify-fits extension) ──────────


def verify_conditioned_forms(seed: int = 0, n: int = 40, d: int = 8, rank: int = 3) -> dict:
    """Per-form gate: the DISPATCHED ``apply_conditioned_delta`` vs an analytic
    per-sample loop implementing the §4.3b equations directly. Also asserts
    the capacity table's exact param counts at d=3584."""
    g = torch.Generator().manual_seed(seed)
    hn = torch.randn(n, d, generator=g, dtype=torch.float64).float()
    cn = torch.randn(n, d, generator=g, dtype=torch.float64).float()
    res = {}
    for form in CONDITIONED_FORMS:
        w = init_conditioned_params(form, d, rank=rank, n_mix=2, seed=seed + 1)
        got = apply_conditioned_delta(form, w, hn, cn)
        want = torch.empty_like(got)
        for i in range(n):
            h_i, c_i = hn[i], cn[i]
            if form == "b1_grad":
                want[i] = w["Wh"].t() @ h_i + w["Wc"].t() @ c_i + w["b"]
            elif form == "film":
                want[i] = w["A"].t() @ h_i + (w["Wg"].t() @ c_i) * h_i + w["b"]
            elif form == "lowrank":
                core = (w["Ws"].t() @ c_i) * (w["V"].t() @ h_i)
                want[i] = w["A"].t() @ h_i + w["U"] @ core + w["b"]
            else:  # mixture
                z = w["Ww"].t() @ c_i
                a = torch.softmax(z, dim=0)
                want[i] = a[0] * (w["Am0"].t() @ h_i) + a[1] * (w["Am1"].t() @ h_i) + w["b"]
        max_abs = float((got - want).abs().max().item())
        assert max_abs <= 1e-4, (form, max_abs)
        res[form] = {"max_abs_delta": max_abs}
    # the §4.3b capacity table, exact
    assert conditioned_param_count("b1_grad", 3584) == 25_690_112
    assert conditioned_param_count("film", 3584) == 25_690_112
    assert conditioned_param_count("lowrank", 3584, rank=1195) == 25_693_696
    assert conditioned_param_count("mixture", 3584, n_mix=2) == 25_697_280
    res["capacity_table"] = "exact"
    return res


def verify_b1_gram_assembly(seed: int = 0, n: int = 120, d: int = 6) -> dict:
    """Gate: accumulate_grams(ctx) + accumulate_grams_b1 + assemble_b1_stats
    reproduces a direct GramStats pass over the concatenated design [h, c]."""
    rng = np.random.default_rng(seed)
    R, P = 2, n + 20
    h_store = torch.from_numpy(rng.standard_normal((R, P, d))).to(torch.float16)
    src = torch.arange(0, n, dtype=torch.long)
    src_T = torch.full((n,), P - 2, dtype=torch.long)
    grams = accumulate_grams(h_store, src, [1], emb_row=0, chunk=32, device="cpu")
    blk = accumulate_grams_b1(h_store, src, src_T, [1], chunk=32, device="cpu")[1]
    got = assemble_b1_stats(grams["ctx"][1], blk)
    ref = GramStats.zeros(2 * d, d, "cpu")
    hh = h_store[1, src, :].to(torch.float64)
    cc = h_store[1, src_T, :].to(torch.float64)
    yy = h_store[1, src + 1, :].to(torch.float64) - hh
    ref.add_chunk(torch.cat([hh, cc], 1), yy)
    for a, b in ((got.sxx, ref.sxx), (got.sxy, ref.sxy), (got.sx, ref.sx), (got.sy, ref.sy)):
        assert float((a - b).abs().max()) <= 1e-8
    assert abs(got.syy - ref.syy) <= 1e-6
    return {"max_abs_delta": 0.0, "n": n, "d": d}


def verify_direct_horizon_gcv(seed: int = 0, n_ctx: int = 30, d: int = 10, k_max: int = 3) -> dict:
    """Gate: fit_direct_horizon_maps' batched per-k GCV path matches the
    serial ``ridge_gcv_from_grams`` on each horizon's exact subset design."""
    rng = np.random.default_rng(seed)
    win = k_max + 2
    P = n_ctx * win
    h_store = torch.from_numpy(rng.standard_normal((1, P, d))).to(torch.float16)
    T_pos = torch.arange(0, P, win, dtype=torch.long)
    kcap = rng.integers(1, k_max + 1, size=n_ctx)
    got = fit_direct_horizon_maps(
        h_store, 0, T_pos, kcap, k_max=k_max, device="cpu", k_chunk=2, min_n=2
    )
    max_abs = 0.0
    for k, rmap in got["maps"].items():
        sel = np.where(np.minimum(kcap, k_max) >= k)[0]
        # serial reference: GramStats over the SAME subset, ridge_gcv_from_grams
        stats = GramStats.zeros(d, d, "cpu")
        cs = h_store[0, T_pos[sel], :].to(torch.float64)
        ts = h_store[0, T_pos[sel] + k, :].to(torch.float64) - cs
        stats.add_chunk(cs, ts)
        ref, _ = ridge_gcv_from_grams(stats)
        assert ref.best_lam == rmap.best_lam, (k, ref.best_lam, rmap.best_lam)
        max_abs = max(max_abs, float((ref.w - rmap.w).abs().max().item()))
    assert max_abs <= 1e-4, max_abs
    return {"max_abs_delta": max_abs, "n_horizons": len(got["maps"])}
