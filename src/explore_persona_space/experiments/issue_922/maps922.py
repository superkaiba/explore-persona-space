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
    "RIDGE_LAMBDAS_922",
    "GramStats",
    "RidgeMap",
    "accumulate_grams",
    "apply_mlp_params",
    "apply_ridge_maps_batched",
    "delta_error_percentiles",
    "fit_position_gru",
    "fit_position_mlps",
    "gru_roll_states",
    "identity_relative_r2",
    "mean_centered_r2",
    "ridge_gcv_from_grams",
    "ridge_predict",
    "roll_states_ridge",
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
