# ruff: noqa: RUF002, RUF003
"""Phase-3 fit engine for issue #1739 (round C1): matched budgets, whitening, maps.

Everything here is layer-leading ``(Ly, n, d)`` and BATCHED (vectorize-first):

- Matched-budget protocol: per (behavior, regime, U rung, L rung, draw, seed)
  the labeled subset + GROUP-level folds are realized ONCE
  (:func:`realize_budget_cell`) and shared verbatim by every arm.
- Whitening: shrinkage-regularized ``Σ_γ^{-1/2}`` fit on the U pool only
  (γ per layer by held-out Gaussian NLL over a fixed grid), applied unchanged
  to every eval rung — no transductive refit.
- Context→answer maps: ``issue_779.fit_h.ridge_fit_predict_fast_layer_batched``
  (one batched Gram-eigh over the layer axis, ``return_weights=True``) with
  the standardization params replicated so the frozen map applies to new x.
- PV directions: E1 diff-of-means, E2 within-context matched-pair,
  E2p pooled — all mask-GEMM vectorized (no per-context python loop).
- Map diagnostics: held-out R² + identity+learned-bias baseline + kNN
  retrieval (``analysis.mapping_baselines``) per layer, reusable per eval rung.

No network, no GPU requirement (``device`` parametrized; CPU default).
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from explore_persona_space.experiments.issue_1739.constants import (
    KNN_KS,
    N_FOLDS,
    RIDGE_LAMBDAS,
    WHITEN_HOLDOUT_FRAC,
    WHITEN_SHRINKAGE_GRID,
)

logger = logging.getLogger(__name__)

# Nonlinear context->answer map kinds (#1739 nonlinear-map round). Both reuse
# the #779 N1M fitters verbatim (``scripts/issue779_ffc_n1m_fits.py``:
# ``fit_mlp`` / ``fit_krr_nystrom`` + ``apply_map``) — see
# :func:`fit_nonlinear_map`. "linear" stays the default everywhere.
NONLINEAR_MAP_KINDS = ("mlp", "kernel")

# Per-layer nonlinear-map hyperparameters. The MLP width is the plan-named
# 3584->512->3584 house recipe; lr/wd/patience/batch/max_epochs are the #779
# constants the reused fitter itself reads (F.MLP_WD / F.MLP_PATIENCE), so the
# only values named here are the ones the fitter takes as arguments.
MLP_MAP_WIDTH = 512
MLP_MAP_LR = 1e-3  # Source: #779 fitter_fair_comparison MLP_LR (same fitter, same d)
MLP_MAP_MAX_EPOCHS = 300  # Source: #779 MLP_MAX_EPOCHS (early-stop patience 20)
MLP_MAP_BATCH = 4096  # Source: #779 MLP_BATCH
KRR_MAP_M_CENTERS = 4096  # Nystrom landmarks; clamped to n_train by the fitter
KRR_MAP_GAMMA_MULT = (1.0,)  # Source: #779 KRR_GAMMA_MULT (median-heuristic gamma)
KRR_MAP_LAMBDAS = (1e-1, 1e1)  # Source: #779 KRR_LAMBDAS
KRR_MAP_BLOCK = 4096  # streaming Phi block


# ---------------------------------------------------------------------------
# matched-budget protocol: labeled draws + shared group-level folds
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BudgetCell:
    """One realized (L, draw, seed) labeled budget, shared by EVERY arm.

    ``row_idx`` indexes the labeled table; ``fold_ids`` assigns each selected
    row its group-level fold. Fold assignment is computed ONCE here — arms
    never re-draw folds (the matched-budget protocol).
    """

    row_idx: np.ndarray
    fold_ids: np.ndarray
    n_folds: int
    budget_l: int
    draw: int
    seed: int
    fold_scheme: str


def realize_budget_cell(
    group_keys: list[str] | np.ndarray,
    *,
    budget_l: int,
    draw: int,
    seed: int,
    n_folds: int = N_FOLDS,
) -> BudgetCell:
    """Draw a group-respecting labeled subset of ~``budget_l`` rows + folds.

    Whole groups are taken in a seeded permutation order until the budget is
    reached (the final group is row-truncated to hit ``budget_l`` exactly);
    folds are assigned round-robin over the selected groups in the same
    permuted order, so a group's rows never straddle folds. Deterministic in
    ``(seed, draw, budget_l)``; asserts every fold non-empty.
    """
    keys = np.asarray(group_keys)
    n = len(keys)
    if budget_l < 1:
        raise ValueError(f"budget_l must be >= 1, got {budget_l}")
    uniq = np.unique(keys)  # sorted — deterministic base order
    rng = np.random.default_rng([1739, int(seed), int(draw), int(budget_l)])
    order = rng.permutation(len(uniq))

    rows_by_group = {g: np.flatnonzero(keys == g) for g in uniq}
    selected: list[np.ndarray] = []
    group_of_selected: list[int] = []
    total = 0
    for gi in order:
        g = uniq[gi]
        rows = rows_by_group[g]
        rows = rows[rng.permutation(len(rows))]
        take = min(len(rows), budget_l - total)
        if take <= 0:
            break
        selected.append(rows[:take])
        group_of_selected.append(int(gi))
        total += take
        if total >= budget_l:
            break
    if total < min(budget_l, n):
        raise RuntimeError(f"budget draw under-filled: {total} < min({budget_l}, {n})")

    n_groups_sel = len(selected)
    n_folds_eff = min(n_folds, n_groups_sel)
    fold_of_group = {group_of_selected[j]: j % n_folds_eff for j in range(n_groups_sel)}
    row_idx = np.concatenate(selected)
    fold_ids = np.concatenate(
        [
            np.full(len(rows), fold_of_group[g], dtype=np.int64)
            for rows, g in zip(selected, group_of_selected, strict=True)
        ]
    )
    sort = np.argsort(row_idx, kind="stable")
    row_idx, fold_ids = row_idx[sort], fold_ids[sort]
    counts = np.bincount(fold_ids, minlength=n_folds_eff)
    assert (counts > 0).all(), f"empty fold in budget cell: {counts.tolist()}"
    if n_folds_eff < n_folds:
        logger.warning(
            "[fits] budget cell L=%d draw=%d: only %d groups -> %d folds",
            budget_l,
            draw,
            n_groups_sel,
            n_folds_eff,
        )
    return BudgetCell(
        row_idx=row_idx,
        fold_ids=fold_ids,
        n_folds=n_folds_eff,
        budget_l=int(budget_l),
        draw=int(draw),
        seed=int(seed),
        fold_scheme=f"group-roundrobin-k{n_folds_eff}",
    )


def compose_u_pool(
    n_generic: int, n_eliciting: int, *, f_u: float, size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Composition-factor U pool: index selections (generic_idx, eliciting_idx).

    ``f_u`` is the fraction of the U pool drawn from the behavior-eliciting
    pool (plan §4b; Config A evil only). Seeded, without replacement; fails
    loud when either side cannot supply its quota.
    """
    if not 0.0 <= f_u <= 1.0:
        raise ValueError(f"f_u out of [0,1]: {f_u}")
    n_elic = round(f_u * size)
    n_gen = size - n_elic
    if n_gen > n_generic or n_elic > n_eliciting:
        raise ValueError(
            f"compose_u_pool: need {n_gen} generic (have {n_generic}) + "
            f"{n_elic} eliciting (have {n_eliciting})"
        )
    rng = np.random.default_rng([1739, 2, int(seed), round(f_u * 1000), size])
    gen_idx = np.sort(rng.choice(n_generic, size=n_gen, replace=False))
    elic_idx = (
        np.sort(rng.choice(n_eliciting, size=n_elic, replace=False))
        if n_elic
        else np.empty(0, dtype=np.int64)
    )
    return gen_idx.astype(np.int64), elic_idx.astype(np.int64)


# ---------------------------------------------------------------------------
# whitening (fit on U only; frozen for every eval rung)
# ---------------------------------------------------------------------------


def _eigh_robust(mat):
    """Batched eigh with the cuSOLVER->CPU LAPACK fallback (gotchas.md #1335)."""
    import torch

    try:
        return torch.linalg.eigh(mat)
    except torch.linalg.LinAlgError:
        logger.warning("[fits] cuda eigh non-convergence; CPU fallback (n=%s)", tuple(mat.shape))
        w, v = torch.linalg.eigh(mat.cpu())
        return w.to(mat.device), v.to(mat.device)


@dataclasses.dataclass
class Whitening:
    """Frozen per-layer whitening transform ``z = Σ_γ^{-1/2}(x - μ)``."""

    mu: np.ndarray  # (Ly, d)
    w: np.ndarray  # (Ly, d, d)
    gamma: np.ndarray  # (Ly,) chosen shrinkage per layer


def fit_whitening(
    x_u: np.ndarray,
    *,
    gammas: tuple[float, ...] = WHITEN_SHRINKAGE_GRID,
    holdout_frac: float = WHITEN_HOLDOUT_FRAC,
    seed: int = 0,
    device: str = "cpu",
    layer_chunk: int = 8,
) -> Whitening:
    """Shrinkage whitening fit on the U pool only, γ per layer by held-out NLL.

    ``x_u`` is (Ly, n, d). ``Σ_γ = (1-γ)Σ + γ(trΣ/d)I`` shares eigenvectors
    with Σ, so the per-γ held-out Gaussian NLL reduces to eigenvalue algebra —
    the whole grid is evaluated with ONE batched eigh per layer chunk.
    """
    import torch

    x = np.asarray(x_u, dtype=np.float64)
    n_layers, n, d = x.shape
    rng = np.random.default_rng([1739, 3, int(seed)])
    perm = rng.permutation(n)
    n_hold = round(holdout_frac * n) if n >= 5 else 0
    hold, tr = perm[:n_hold], perm[n_hold:]
    dev = torch.device(device)

    mu = np.empty((n_layers, d))
    w_out = np.empty((n_layers, d, d))
    gamma_out = np.empty(n_layers)
    for lo in range(0, n_layers, layer_chunk):
        sl = slice(lo, min(lo + layer_chunk, n_layers))
        xt = torch.as_tensor(x[sl][:, tr], device=dev)  # (c, n_tr, d)
        m = xt.mean(dim=1, keepdim=True)  # (c, 1, d)
        xc = xt - m
        cov = xc.transpose(1, 2) @ xc / max(len(tr), 1)  # (c, d, d)
        evals, evecs = _eigh_robust(cov)
        evals = torch.clamp(evals, min=0.0)  # (c, d)
        tr_mean = evals.mean(dim=1, keepdim=True)  # trΣ/d
        if n_hold:
            xh = torch.as_tensor(x[sl][:, hold], device=dev) - m
            eh = xh @ evecs  # (c, n_hold, d) holdout in eigenbasis
            diag_hold = (eh**2).mean(dim=1)  # (c, d)
        else:
            diag_hold = evals  # degenerate: NLL over train evals (γ pick still defined)
        nlls = []
        for g in gammas:
            lam = (1.0 - g) * evals + g * tr_mean  # (c, d)
            nlls.append(torch.log(lam).sum(dim=1) + (diag_hold / lam).sum(dim=1))
        gi = torch.stack(nlls, dim=1).argmin(dim=1)  # (c,)
        g_best = torch.as_tensor([float(gammas[int(i)]) for i in gi], device=dev)
        lam_best = (1.0 - g_best[:, None]) * evals + g_best[:, None] * tr_mean
        inv_sqrt = evecs @ (lam_best.clamp(min=1e-12).rsqrt()[:, :, None] * evecs.transpose(1, 2))
        mu[sl] = m.squeeze(1).cpu().numpy()
        w_out[sl] = inv_sqrt.cpu().numpy()
        gamma_out[sl] = g_best.cpu().numpy()
    logger.info(
        "[fits] whitening fit: Ly=%d n=%d d=%d gammas=%s", n_layers, n, d, gamma_out.tolist()
    )
    return Whitening(mu=mu, w=w_out, gamma=gamma_out)


def apply_whitening(x: np.ndarray, wh: Whitening) -> np.ndarray:
    """Apply the frozen transform to (Ly, n, d) activations (batched matmul)."""
    x = np.asarray(x, dtype=np.float64)
    return (x - wh.mu[:, None, :]) @ wh.w


# ---------------------------------------------------------------------------
# PV directions (E1 / E2 / E2p) — mask-GEMM vectorized
# ---------------------------------------------------------------------------


def extract_rb_e1(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """E1 diff-of-means direction: (n_pos, Ly, d), (n_neg, Ly, d) -> (Ly, d)."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    assert pos.ndim == 3 and neg.ndim == 3 and pos.shape[1:] == neg.shape[1:], (
        pos.shape,
        neg.shape,
    )
    return pos.mean(axis=0) - neg.mean(axis=0)


def matched_pair_split_weights(
    scores: np.ndarray,
    *,
    spread_min: float,
    pooled: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Row weights realizing the E2 / E2p contrast as ONE weighted sum.

    ``scores`` (n_ctx, K) with NaN = dropped draw. Returns
    ``(w_hi, w_lo, n_qualifying)`` with shapes (n_ctx, K) such that
    ``rb = sum_ck w_hi[c,k] * acts[c,k] - sum_ck w_lo[c,k] * acts[c,k]``
    reproduces :func:`extract_rb_matched` exactly (weights fold the
    per-context normalization + the mean over qualifying contexts), which
    lets the fits CLI compute E2/E2p directions from the per-ROLLOUT store
    rows via one mask-GEMM per layer — no (n_ctx, K, Ly, d) materialization.

    E2 (``pooled=False``): a context QUALIFIES when it has >= 2 kept draws
    AND its within-context score spread (max - min of the KEPT per-rollout
    scores) >= ``spread_min`` (the plan §4 selection is on the per-rollout
    K-sample scores; the split point is the context's own midpoint). E2p
    (``pooled=True``): one global midpoint split over ALL kept answers
    (topic-confounded by design — plan §5 ``pv_e2p``).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n_ctx, _k = scores.shape
    kept = np.isfinite(scores)
    if pooled:
        flat = scores[kept]
        if flat.size < 2:
            raise ValueError("E2p: fewer than 2 kept answers")
        mid = 0.5 * (np.nanmax(scores) + np.nanmin(scores))
        hi = kept & (scores >= mid)
        lo = kept & (scores < mid)
        if hi.sum() == 0 or lo.sum() == 0:
            raise ValueError("E2p: degenerate global split (no spread)")
        return hi / hi.sum(), lo / lo.sum(), int(n_ctx)

    smax = np.where(kept, scores, -np.inf).max(axis=1)
    smin = np.where(kept, scores, np.inf).min(axis=1)
    qual = (kept.sum(axis=1) >= 2) & ((smax - smin) >= spread_min)
    if not qual.any():
        raise ValueError(f"E2: zero qualifying contexts at spread_min={spread_min}")
    mid = 0.5 * (smax + smin)  # (n_ctx,)
    hi = kept & (scores >= mid[:, None]) & qual[:, None]
    lo = kept & (scores < mid[:, None]) & qual[:, None]
    n_qual = int(qual.sum())
    hi_n = np.maximum(hi.sum(axis=1), 1)[:, None]
    lo_n = np.maximum(lo.sum(axis=1), 1)[:, None]
    w_hi = hi / (hi_n * n_qual)
    w_lo = lo / (lo_n * n_qual)
    return w_hi, w_lo, n_qual


def extract_rb_matched(
    acts: np.ndarray,
    scores: np.ndarray,
    *,
    spread_min: float,
    pooled: bool = False,
) -> tuple[np.ndarray, int]:
    """E2 / E2p direction from per-context K-answer activations.

    ``acts`` (n_ctx, K, Ly, d); ``scores`` (n_ctx, K) with NaN = dropped draw.
    E2 (``pooled=False``): within qualifying contexts (score spread >=
    ``spread_min``), split at the context's own score MIDPOINT, diff the
    hi/lo means, average over qualifying contexts. E2p (``pooled=True``):
    one global midpoint split over ALL kept answers (topic-confounded by
    design — plan §5 ``pv_e2p``). Returns ((Ly, d), n_qualifying_contexts).
    The split weights live in :func:`matched_pair_split_weights` so the
    fits CLI can apply the same contrast to flat per-rollout store rows.
    """
    acts = np.asarray(acts, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_ctx, k, _n_layers, _d = acts.shape
    assert scores.shape == (n_ctx, k), (scores.shape, (n_ctx, k))
    w_hi, w_lo, n_qual = matched_pair_split_weights(scores, spread_min=spread_min, pooled=pooled)
    rb = np.einsum("ck,ckld->ld", w_hi - w_lo, acts, optimize=True)
    return rb, n_qual


# ---------------------------------------------------------------------------
# context->answer maps + diagnostics
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MapFit:
    """Frozen context->answer map (standardized-input-space weights).

    LINEAR (``kind == "linear"``, the default): application contract mirrors
    ``ridge_fit_predict_fast_layer_batched`` —
    ``pred = ((x - x_mu)/x_sd) @ w + y_mu`` (float64, layer-leading).

    NONLINEAR (``kind in {"mlp", "kernel"}``, #1739 nonlinear-map round): ``w``
    / ``x_mu`` / ``x_sd`` / ``y_mu`` are ``None`` and the map lives in
    ``nl_payloads`` — one per-layer payload in the #779 N1M ``apply_map``
    format (``issue779_ffc_n1m_fits.apply_map``), which carries its OWN
    standardizer per layer. :func:`apply_map` dispatches on ``kind``, so every
    downstream arm that consumes a map (arms 6/7/8) works unchanged.
    """

    w: np.ndarray | None  # (Ly, d, d) — None for a nonlinear kind
    x_mu: np.ndarray | None  # (Ly, 1, d)
    x_sd: np.ndarray | None  # (Ly, 1, d)
    y_mu: np.ndarray | None  # (Ly, 1, d)
    diagnostics: dict
    kind: str = "linear"
    nl_payloads: tuple[dict, ...] = ()  # per-layer N1M apply_map payloads
    apply_device: str = "cpu"  # device the nonlinear per-layer apply runs on

    def __post_init__(self) -> None:
        if self.kind == "linear":
            if self.w is None:
                raise ValueError("MapFit(kind='linear') requires w")
        elif self.kind in NONLINEAR_MAP_KINDS:
            if not self.nl_payloads:
                raise ValueError(f"MapFit(kind={self.kind!r}) requires nl_payloads")
        else:
            raise ValueError(f"unknown MapFit kind {self.kind!r}")


def r2_pooled(pred: np.ndarray, true: np.ndarray) -> float:
    """Pooled held-out R² over all rows x dims (ss_tot around the true mean)."""
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def map_diagnostics(
    pred: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    knn_ks: tuple[int, ...] = KNN_KS,
) -> dict:
    """Per-layer map diagnostics: held-out R², identity+bias R², kNN retrieval.

    All arrays layer-leading (Ly, n, d). The identity+learned-bias baseline +
    kNN retrieval are the standing mapping-baselines pair (CLAUDE.md rule;
    ``analysis/mapping_baselines``); chance = k/n_pool is carried by the
    helper's own ``chance_at_k`` field.
    """
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    n_layers = pred.shape[0]
    per_layer = []
    for li in range(n_layers):
        ib_pred = identity_bias_predict(x_train[li], y_train[li], x_eval[li])
        row = {
            "layer_idx": li,
            "r2_map": r2_pooled(pred[li], y_eval[li]),
            "r2_identity_bias": r2_pooled(ib_pred, y_eval[li]),
            "knn": {
                metric: knn_retrieval(pred[li], y_eval[li], ks=knn_ks, metric=metric)
                for metric in ("euclidean", "cosine")
            },
        }
        per_layer.append(row)
    return {"per_layer": per_layer, "knn_ks": list(knn_ks)}


def ridge_fit_predict_primal_layer_batched(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    lambdas: np.ndarray | tuple[float, ...] = RIDGE_LAMBDAS,
    device: str = "cpu",
    return_weights: bool = False,
    layer_chunk: int = 4,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """PRIMAL (feature-space) twin of ``fit_h.ridge_fit_predict_fast_layer_batched``.

    Same recipe per slice — standardize-X on train stats (train mean,
    population std + 1e-9), center-Y, GCV lambda selected PER SLICE over
    ``lambdas``, un-centered predictions, float64 — solved in the d x d
    FEATURE Gram instead of the n_tr x n_tr dual Gram.

    Memory arithmetic (the M6 / `dual-gram-full-u-compute-shape` fix): at the
    production full-U regime (n_tr ~= 15,034 of the 18,793-row fit pool after
    the 20% diagnostics holdout; d = 3584; 28 layers) the DUAL Gram is
    28 x 15,034^2 x 8 B ~= 50.6 GB fp64 before eigh workspace — far over the
    plan §9 RAM row — while the PRIMAL Gram is 3584^2 x 8 B ~= 0.103 GB per
    layer (~2.9 GB for all 28), and this function keeps only ``layer_chunk``
    layers resident at once (Gram + eigenvectors + A ~= 3 x 0.103 GB per
    layer). The realized U ladder tops out at the store's 18,793 fit rows
    (the plan's nominal 50k rung exceeds the realized #1092 pool).

    GCV parity with the dual helper is exact algebra: the nonzero eigenvalues
    of Z Z^T and Z^T Z coincide; ``dof(lam) = sum_i s_i/(s_i+lam)`` is
    identical on both sides (zero eigenvalues contribute 0), and the dual rss
    identity ``rss = tot - sum_k (2 f_k - f_k^2) ||V^T Yc||_k^2`` maps to
    ``rss = tot - sum_i ||A_i||^2 (s_i + 2 lam)/(s_i + lam)^2`` with
    ``A = V_p^T Z^T Yc`` (division-free in s). Weights are the ridge normal
    equations ``W = V_p diag(1/(s+lam)) A`` — the same W the dual
    reconstructs. Parity is pinned by
    ``tests/test_issue1739_fits.py::test_primal_dual_ridge_parity``.
    """
    import torch

    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    xe = np.asarray(x_eval, dtype=np.float64)
    assert x.ndim == 3 and y.ndim == 3 and xe.ndim == 3, (x.shape, y.shape, xe.shape)
    n_slices, ntr, d = x.shape
    lam_grid = np.asarray(lambdas, dtype=np.float64)
    dev = torch.device(device)
    preds = np.empty((n_slices, xe.shape[1], y.shape[2]))
    w_out = np.empty((n_slices, d, y.shape[2])) if return_weights else None
    for lo in range(0, n_slices, layer_chunk):
        sl = slice(lo, min(lo + layer_chunk, n_slices))
        xtr = torch.as_tensor(x[sl], device=dev)
        ytr = torch.as_tensor(y[sl], device=dev)
        xev = torch.as_tensor(xe[sl], device=dev)
        xmu = xtr.mean(dim=1, keepdim=True)
        xsd = xtr.std(dim=1, unbiased=False, keepdim=True) + 1e-9  # population std (twin parity)
        xn = (xtr - xmu) / xsd
        xen = (xev - xmu) / xsd
        ymu = ytr.mean(dim=1, keepdim=True)
        yc = ytr - ymu
        gram = xn.transpose(1, 2) @ xn  # (c, d, d)
        s, v = _eigh_robust(gram)
        s = torch.clamp(s, min=0.0)  # (c, d)
        a = v.transpose(1, 2) @ (xn.transpose(1, 2) @ yc)  # (c, d, d_out)
        sq_a = (a**2).sum(dim=2)  # (c, d)
        tot = (yc**2).sum(dim=(1, 2))  # (c,)
        gcv = torch.empty((s.shape[0], len(lam_grid)), dtype=torch.float64, device=dev)
        for li, lam in enumerate(lam_grid):
            lam_f = float(lam)
            rss = tot - (sq_a * (s + 2.0 * lam_f) / (s + lam_f) ** 2).sum(dim=1)
            dof = (s / (s + lam_f)).sum(dim=1)
            denom = (ntr - dof) ** 2
            gcv[:, li] = torch.where(denom > 1e-12, rss / denom, torch.full_like(rss, float("inf")))
        best = gcv.argmin(dim=1)
        best_lam = torch.as_tensor(lam_grid, device=dev)[best]  # (c,)
        w = v @ (a / (s + best_lam[:, None])[:, :, None])  # (c, d, d_out)
        preds[sl] = (xen @ w + ymu).cpu().numpy()
        if w_out is not None:
            w_out[sl] = w.cpu().numpy()
    if return_weights:
        return preds, w_out
    return preds


def ridge_layer_batched_auto(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    lambdas: np.ndarray | tuple[float, ...] = RIDGE_LAMBDAS,
    device: str = "cpu",
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Route a batched ridge solve to the cheaper Gram: primal iff n_tr > d.

    n_tr <= d delegates to the parent dual helper
    (``fit_h.ridge_fit_predict_fast_layer_batched``, n_tr x n_tr Gram);
    n_tr > d runs the primal twin above (d x d Gram). Both implement the
    identical standardize/GCV/predict recipe (parity test-pinned), so the
    branch is a memory/throughput routing decision, never a semantic one.
    """
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched,
    )

    n_tr, d = np.asarray(x_train).shape[1], np.asarray(x_train).shape[2]
    if n_tr > d:
        return ridge_fit_predict_primal_layer_batched(
            x_train,
            y_train,
            x_eval,
            lambdas=lambdas,
            device=device,
            return_weights=return_weights,
        )
    return ridge_fit_predict_fast_layer_batched(
        x_train,
        y_train,
        x_eval,
        lambdas=np.asarray(lambdas, dtype=np.float64),
        device=device,
        return_weights=return_weights,
    )


def ridge_gcv_predict_per_target(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evals: list[np.ndarray],
    *,
    lambdas: np.ndarray | tuple[float, ...] = RIDGE_LAMBDAS,
    device: str = "cpu",
    layer_chunk: int = 4,
) -> list[np.ndarray]:
    """Batched ridge with PER-TARGET GCV lambda over ONE shared factorization.

    ``x_train`` (S, ntr, d); ``y_train`` (S, ntr, T) — T independent targets
    that SHARE the design matrix, so the expensive Gram + eigh is computed
    once per (slice, fold) and every target only adds cheap spectral algebra.
    ``x_evals`` is a list of eval matrices (S, nev_i, d); returns one
    (S, nev_i, T) prediction array per eval matrix.

    Semantics are the EXACT per-target twin of :func:`ridge_layer_batched_auto`
    at T=1 (same standardize-X train-stats recipe — population std + 1e-9 —
    center-Y, GCV per target over ``lambdas``, un-centered predictions,
    float64; primal d x d Gram when ntr > d else the dual n_tr x n_tr Gram);
    parity is test-pinned by
    ``tests/test_issue1739_fits.py::test_ridge_gcv_per_target_matches_auto``.
    This is the cross-arm/cross-regime dedup lever of the #1739 fits round:
    the old per-(arm, layer, fold) slice pool re-ran one eigh PER TARGET.
    """
    import torch

    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    assert x.ndim == 3 and y.ndim == 3, (x.shape, y.shape)
    n_slices, ntr, d = x.shape
    n_t = y.shape[2]
    assert y.shape[:2] == (n_slices, ntr), (y.shape, x.shape)
    evs = [np.asarray(e, dtype=np.float64) for e in x_evals]
    for e in evs:
        assert e.ndim == 3 and e.shape[0] == n_slices and e.shape[2] == d, (e.shape, x.shape)
    lam_grid = np.asarray(lambdas, dtype=np.float64)
    dev = torch.device(device)
    primal = ntr > d
    preds = [np.empty((n_slices, e.shape[1], n_t)) for e in evs]
    for lo in range(0, n_slices, layer_chunk):
        sl = slice(lo, min(lo + layer_chunk, n_slices))
        xtr = torch.as_tensor(x[sl], device=dev)
        ytr = torch.as_tensor(y[sl], device=dev)
        xmu = xtr.mean(dim=1, keepdim=True)
        xsd = xtr.std(dim=1, unbiased=False, keepdim=True) + 1e-9  # population std (twin parity)
        xn = (xtr - xmu) / xsd
        ymu = ytr.mean(dim=1, keepdim=True)  # (c, 1, T)
        yc = ytr - ymu
        tot = (yc**2).sum(dim=1)  # (c, T)
        if primal:
            gram = xn.transpose(1, 2) @ xn  # (c, d, d)
            s, v = _eigh_robust(gram)
            s = torch.clamp(s, min=0.0)  # (c, d)
            a = v.transpose(1, 2) @ (xn.transpose(1, 2) @ yc)  # (c, d, T)
            sq_a = a**2
        else:
            gram = xn @ xn.transpose(1, 2)  # (c, ntr, ntr) dual Gram
            s, v = _eigh_robust(gram)
            s = torch.clamp(s, min=0.0)  # (c, ntr)
            a = v.transpose(1, 2) @ yc  # (c, ntr, T)
            sq_a = a**2
        c = s.shape[0]
        gcv = torch.empty((c, len(lam_grid), n_t), dtype=torch.float64, device=dev)
        for li, lam in enumerate(lam_grid):
            lam_f = float(lam)
            if primal:
                # primal rss identity: rss_t = tot_t - sum_i sq_a[t,i] (s_i+2lam)/(s_i+lam)^2
                shrink = (s + 2.0 * lam_f) / (s + lam_f) ** 2  # (c, d)
                rss = tot - torch.einsum("cdt,cd->ct", sq_a, shrink)
                dof = (s / (s + lam_f)).sum(dim=1)  # (c,)
            else:
                # dual rss identity: rss_t = tot_t - sum_k (2 f_k - f_k^2) (V^T Yc)_kt^2
                filt = s / (s + lam_f)  # (c, ntr)
                rss = tot - torch.einsum("cnt,cn->ct", sq_a, 2.0 * filt - filt**2)
                dof = filt.sum(dim=1)
            denom = (ntr - dof) ** 2  # (c,)
            gcv[:, li, :] = torch.where(
                (denom > 1e-12)[:, None], rss / denom[:, None], torch.full_like(rss, float("inf"))
            )
        best = gcv.argmin(dim=1)  # (c, T)
        lam_sel = torch.as_tensor(lam_grid, device=dev)[best]  # (c, T)
        f_sel = 1.0 / (s[:, :, None] + lam_sel[:, None, :])  # (c, d|ntr, T)
        if primal:
            w = v @ (a * f_sel)  # (c, d, T) standardized-space weights
            for ei, e in enumerate(evs):
                xen = (torch.as_tensor(e[sl], device=dev) - xmu) / xsd
                preds[ei][sl] = (xen @ w + ymu).cpu().numpy()
        else:
            alpha = v @ (a * f_sel)  # (c, ntr, T) dual coefficients
            xnt = xn.transpose(1, 2)  # (c, d, ntr)
            w = xnt @ alpha  # (c, d, T)
            for ei, e in enumerate(evs):
                xen = (torch.as_tensor(e[sl], device=dev) - xmu) / xsd
                preds[ei][sl] = (xen @ w + ymu).cpu().numpy()
    return preds


def krr_scalar_fold_predict(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_ev: np.ndarray,
    *,
    seed: int = 0,
    device: str = "cpu",
    m_centers: int = KRR_MAP_M_CENTERS,
    gamma_mult: tuple[float, ...] = KRR_MAP_GAMMA_MULT,
    lambdas: tuple[float, ...] = KRR_MAP_LAMBDAS,
    layer_chunk: int | None = None,
    diag_out: dict | None = None,
) -> np.ndarray:
    """Layer-BATCHED Nystrom RBF KRR for a SCALAR target (the arm-18 oracle).

    Mirrors the nonlinear-map round's kernel recipe (:func:`fit_nonlinear_map`
    ``kind="kernel"`` -> ``n1m.fit_krr_nystrom``) with input = the whitened
    TRUE answer acts and target = the graded DV, batched over the LAYER axis
    (plan §4 item 2 — never a per-(fold, lambda) re-factorization):

    - raw X + per-layer MEDIAN-HEURISTIC gamma (1/median off-diagonal sq
      distance over a <=2000-row subsample — the ``median_heuristic_gamma``
      convention), scaled by ``gamma_mult``;
    - inner-val (gamma, lambda) selection carved from the fold-TRAIN rows with
      the SAME rng key family as the nlmap kernel arm (``[1739, 5, seed]``,
      ``n_val = max(2, round(0.1 * n_tr))``) — selection never touches the
      eval fold;
    - Nystrom landmarks ``m = min(m_centers, n_inner)`` drawn once (shared
      across layers, like the per-layer fitter's identical-seed draws), K_mm
      whitener via batched :func:`_eigh_robust` (eig floor 1e-10), feature
      ridge ``G = Phi^T Phi`` with ONE batched eigh per (layer, gamma) shared
      across the whole lambda grid (raw lambda, Y train-centered — the #779
      conventions);
    - eval predictions come from the model fit on the INNER rows at the
      selected pair (exactly the n1m diagnostics path — no inner+val refit).

    ``x_tr`` (Ly, n_tr, d), ``y_tr`` (n_tr,), ``x_ev`` (Ly, n_ev, d) ->
    (Ly, n_ev) fp64 predictions. ``diag_out`` (optional dict) receives the
    per-layer selected {gamma, lambda, val_mse} + shapes.
    """
    import torch

    x = np.asarray(x_tr, dtype=np.float64)
    y = np.asarray(y_tr, dtype=np.float64)
    xe = np.asarray(x_ev, dtype=np.float64)
    assert x.ndim == 3 and xe.ndim == 3 and x.shape[0] == xe.shape[0], (x.shape, xe.shape)
    n_layers, ntr, _d = x.shape
    assert y.shape == (ntr,), (y.shape, ntr)
    rng = np.random.default_rng([1739, 5, int(seed)])
    p = rng.permutation(ntr)
    n_val = max(2, round(0.1 * ntr))
    val_i, inner_i = p[:n_val], p[n_val:]
    if len(inner_i) < 2:
        raise ValueError(f"krr_scalar_fold_predict: too few inner rows ({len(inner_i)})")
    m = int(min(m_centers, len(inner_i)))
    lm_rows = inner_i[np.random.default_rng(int(seed)).choice(len(inner_i), size=m, replace=False)]
    g_sub = min(2000, len(inner_i))
    g_rows = inner_i[
        np.random.default_rng(int(seed) + 1).choice(len(inner_i), size=g_sub, replace=False)
    ]
    dev = torch.device(device)
    if layer_chunk is None:
        layer_chunk = 4 if ntr <= 3000 else (2 if ntr <= 8000 else 1)
    lam_grid = [float(lam) for lam in lambdas]
    preds = np.empty((n_layers, xe.shape[1]))
    ymu = float(y[inner_i].mean())
    per_layer_sel: list[dict] = [{} for _ in range(n_layers)]
    n_degenerate = 0
    for lo in range(0, n_layers, layer_chunk):
        sl = slice(lo, min(lo + layer_chunk, n_layers))
        xt = torch.as_tensor(x[sl], device=dev)  # (c, ntr, d)
        xev_t = torch.as_tensor(xe[sl], device=dev)
        z_lm = xt[:, lm_rows].contiguous()  # (c, m, d) landmarks
        d_g = torch.cdist(xt[:, g_rows], xt[:, g_rows]) ** 2  # (c, g, g)
        c, g = d_g.shape[0], d_g.shape[1]
        iu = torch.triu_indices(g, g, offset=1, device=dev)
        med = d_g[:, iu[0], iu[1]].median(dim=1).values  # (c,)
        # A degenerate median (duplicate-dominated subsample: median sq
        # distance 0) is a RECORDED per-layer skip, never a grid-killing
        # assert — one bad (layer, gamma) cell must not abort a whole fits
        # invocation (code-review r1 Minor 5; the #1739-r10 batched-solve
        # incident class). The layer computes under a placeholder gamma to
        # keep the batch shape, then its predictions are overwritten with
        # NaN and flagged in diag_out (downstream nanargmax/spearman treat
        # NaN layers as absent).
        degenerate = med <= 0  # (c,) bool
        if bool(degenerate.any()):
            logger.warning(
                "[fits] krr median-heuristic gamma degenerate (median sq distance 0) on "
                "%d/%d layer(s) in chunk [%d:%d) — NaN predictions recorded for those layers",
                int(degenerate.sum()),
                c,
                lo,
                lo + c,
            )
        med = torch.where(degenerate, torch.ones_like(med), med)
        base_gamma = 1.0 / med
        yc = torch.as_tensor(y[inner_i] - ymu, dtype=torch.float64, device=dev)  # (ni,)
        y_val = torch.as_tensor(y[val_i], dtype=torch.float64, device=dev)
        best_mse = torch.full((c,), float("inf"), dtype=torch.float64, device=dev)
        best_pred = torch.full((c, xev_t.shape[1]), float("nan"), dtype=torch.float64, device=dev)
        best_gl = torch.zeros((c, 2), dtype=torch.float64, device=dev)  # (gamma, lambda)
        for gm in gamma_mult:
            gamma = base_gamma * float(gm)  # (c,)
            gview = gamma[:, None, None]
            k_mm = torch.exp(-gview * torch.cdist(z_lm, z_lm) ** 2)  # (c, m, m)
            w_mm, v_mm = _eigh_robust(k_mm)
            inv_sqrt = v_mm @ (
                torch.clamp(w_mm, min=1e-10).rsqrt()[:, :, None] * v_mm.transpose(1, 2)
            )
            phi_in = torch.exp(-gview * torch.cdist(xt[:, inner_i], z_lm) ** 2) @ inv_sqrt
            gram = phi_in.transpose(1, 2) @ phi_in  # (c, m, m) — ONE eigh, shared over lambdas
            phi_y = torch.einsum("cnm,n->cm", phi_in, yc)
            a_eig, q_eig = _eigh_robust(gram)
            a_eig = torch.clamp(a_eig, min=0.0)
            qtb = torch.einsum("cmk,cm->ck", q_eig, phi_y)  # (c, m)
            phi_val = torch.exp(-gview * torch.cdist(xt[:, val_i], z_lm) ** 2) @ inv_sqrt
            phi_ev = torch.exp(-gview * torch.cdist(xev_t, z_lm) ** 2) @ inv_sqrt
            for lam in lam_grid:
                w_feat = torch.einsum("cmk,ck->cm", q_eig, qtb / (a_eig + lam))  # (c, m)
                pred_val = torch.einsum("cvm,cm->cv", phi_val, w_feat) + ymu
                mse = ((pred_val - y_val[None, :]) ** 2).mean(dim=1)  # (c,)
                improved = torch.isfinite(mse) & (mse < best_mse)
                if bool(improved.any()):
                    pred_ev = torch.einsum("cvm,cm->cv", phi_ev, w_feat) + ymu
                    best_pred = torch.where(improved[:, None], pred_ev, best_pred)
                    best_mse = torch.where(improved, mse, best_mse)
                    sel = torch.stack([gamma, torch.full_like(gamma, lam)], dim=1)  # (c, 2)
                    best_gl = torch.where(improved[:, None], sel, best_gl)
            del k_mm, inv_sqrt, phi_in, gram, phi_y, a_eig, q_eig, qtb, phi_val, phi_ev
        best_pred[degenerate] = float("nan")  # recorded skip, never a fabricated fit
        preds[sl] = best_pred.cpu().numpy()
        gl = best_gl.cpu().numpy()
        vm = best_mse.cpu().numpy()
        deg = degenerate.cpu().numpy()
        for li in range(c):
            per_layer_sel[lo + li] = {
                "gamma": float(gl[li, 0]),
                "lambda": float(gl[li, 1]),
                "val_mse": float(vm[li]),
            }
            if bool(deg[li]):
                per_layer_sel[lo + li]["degenerate_gamma"] = True
                n_degenerate += 1
        del xt, xev_t, z_lm, d_g, best_pred, best_mse, best_gl
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if n_degenerate == n_layers:
        # Every layer degenerate = pathological input (identical rows across
        # ALL layers), not a per-cell blip — fail loud, never all-NaN output.
        raise ValueError(
            f"krr_scalar_fold_predict: median-heuristic gamma degenerate on ALL "
            f"{n_layers} layers (duplicate-dominated inputs)"
        )
    if diag_out is not None:
        diag_out.update(
            {
                "kernel": "RBF Nystrom (layer-batched; one eigh per (layer, gamma))",
                "n_degenerate_gamma_layers": n_degenerate,
                "m_centers": m,
                "n_inner": len(inner_i),
                "n_val": int(n_val),
                "gamma_mult": [float(g) for g in gamma_mult],
                "lambdas": lam_grid,
                "per_layer": per_layer_sel,
            }
        )
    return preds


def fit_linear_map(
    x_u: np.ndarray,
    y_u: np.ndarray,
    *,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
    device: str = "cpu",
    holdout_frac: float = WHITEN_HOLDOUT_FRAC,
    seed: int = 0,
    knn_ks: tuple[int, ...] = KNN_KS,
) -> MapFit:
    """Fit the batched context->answer ridge map on the U pool + diagnostics.

    Diagnostics (held-out R^2 + mapping-baselines pair) come from an
    80/20-split fit; the FROZEN map weights are then REFIT on the FULL U pool
    (so a U-ladder rung's nominal budget is its effective fit size — the
    round-1 review minor: W trained on 0.8 x nominal distorted the ladder
    semantics). Both fits route through :func:`ridge_layer_batched_auto`
    (primal d x d Gram when n_tr > d, else the parent dual helper); GCV
    lambda per layer over ``lambdas``. Standardization params replicate the
    helper's own convention (train mean, population std + 1e-9, float64) so
    the frozen map applies to new x exactly as the helper would have.
    """
    x = np.asarray(x_u, dtype=np.float64)
    y = np.asarray(y_u, dtype=np.float64)
    assert x.shape == y.shape and x.ndim == 3, (x.shape, y.shape)
    n_layers, n, _d = x.shape
    rng = np.random.default_rng([1739, 4, int(seed)])
    perm = rng.permutation(n)
    n_hold = max(2, round(holdout_frac * n))
    hold, tr = perm[:n_hold], perm[n_hold:]
    if len(tr) < 2:
        raise ValueError(f"fit_linear_map: too few train rows ({len(tr)})")

    x_tr, y_tr, x_ho, y_ho = x[:, tr], y[:, tr], x[:, hold], y[:, hold]
    preds_hold = ridge_layer_batched_auto(x_tr, y_tr, x_ho, lambdas=lambdas, device=device)
    diagnostics = map_diagnostics(preds_hold, x_ho, y_ho, x_tr, y_tr, knn_ks=knn_ks)
    diagnostics["n_train"], diagnostics["n_holdout"] = len(tr), int(n_hold)
    # Frozen-map refit on the FULL U rung (diagnostics stay honest held-out
    # reads from the split fit above; the weights consume the whole budget).
    _preds_dummy, w = ridge_layer_batched_auto(
        x, y, x[:, :2], lambdas=lambdas, device=device, return_weights=True
    )
    x_mu = x.mean(axis=1, keepdims=True)
    x_sd = x.std(axis=1, keepdims=True) + 1e-9  # population std (helper parity)
    y_mu = y.mean(axis=1, keepdims=True)
    diagnostics["w_fit_rows"] = int(n)
    diagnostics["w_refit_on_full_u"] = True
    diagnostics["solver"] = "primal" if n > x.shape[2] else "dual"
    logger.info(
        "[fits] linear map fit: Ly=%d n_tr=%d n_hold=%d w_fit_rows=%d solver=%s mean r2_map=%.4f",
        n_layers,
        len(tr),
        n_hold,
        n,
        diagnostics["solver"],
        float(np.mean([r["r2_map"] for r in diagnostics["per_layer"]])),
    )
    return MapFit(w=w, x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, diagnostics=diagnostics)


def _nl_split(n: int, holdout_frac: float, seed: int):
    """The IDENTICAL holdout split :func:`fit_linear_map` draws.

    Same rng key ``[1739, 4, seed]`` and same permutation slice, so a
    nonlinear map's held-out R²/kNN diagnostics are computed on the SAME rows
    as the linear map's at the same (variant, U rung, seed) — the whole point
    of the round is a cell-for-cell linear-vs-nonlinear comparison, which a
    differently-drawn holdout would silently break.
    """
    rng = np.random.default_rng([1739, 4, int(seed)])
    perm = rng.permutation(n)
    n_hold = max(2, round(holdout_frac * n))
    return perm[:n_hold], perm[n_hold:]


def fit_nonlinear_map(
    x_u: np.ndarray,
    y_u: np.ndarray,
    *,
    kind: str,
    device: str = "cpu",
    holdout_frac: float = WHITEN_HOLDOUT_FRAC,
    seed: int = 0,
    knn_ks: tuple[int, ...] = KNN_KS,
    mlp_width: int = MLP_MAP_WIDTH,
    krr_m_centers: int = KRR_MAP_M_CENTERS,
    refit_full: bool = True,
) -> MapFit:
    """Fit a per-layer NONLINEAR context->answer map on the U pool + diagnostics.

    Reuses the #779 N1M fitters VERBATIM (no re-implemented fit math):
    ``kind="mlp"`` -> ``fit_mlp`` (3584->``mlp_width``->3584 GELU, AdamW,
    minibatched, internal-val early stop); ``kind="kernel"`` ->
    ``fit_krr_nystrom`` (RBF Nystrom KRR, median-heuristic gamma,
    (gamma, lambda) val-selected, streaming Phi^TPhi). Each fitter's
    ``capture_out`` payload is the frozen map, applied by
    :func:`apply_nl_map` through the SAME ``apply_map`` predict path #779
    roundtrip-gates.

    Two-stage, mirroring :func:`fit_linear_map` so the U-ladder semantics
    MATCH the linear arm: diagnostics come from an 80/20 split fit (held-out
    R² + identity+bias baseline + kNN retrieval via :func:`map_diagnostics`),
    then the FROZEN payload is REFIT on the FULL U pool (``refit_full=True``)
    so a rung's nominal budget is its effective fit size. ``refit_full=False``
    keeps the split fit as the frozen map (halves the cost; makes the rung's
    effective fit size 0.8x nominal — a stated deviation, not the default).

    Serial over the LAYER axis by construction, and that is the correct shape
    here rather than a vectorize-first violation: each per-layer fit is a
    large-n FLOP-bound GPU job (the reused fitters' own docstring: "FLOP-bound
    single large-n fit — NOT a many-cell loop"), not the tiny-op
    overhead-bound regime ``vectorize-many-cell-fits.md`` targets.
    """
    if kind not in NONLINEAR_MAP_KINDS:
        raise ValueError(f"kind must be one of {NONLINEAR_MAP_KINDS}, got {kind!r}")
    import time as _time

    import torch

    n1m = _n1m()
    x = np.asarray(x_u, dtype=np.float64)
    y = np.asarray(y_u, dtype=np.float64)
    assert x.shape == y.shape and x.ndim == 3, (x.shape, y.shape)
    n_layers, n, _d = x.shape
    hold, tr = _nl_split(n, holdout_frac, seed)
    if len(tr) < 4:
        raise ValueError(f"fit_nonlinear_map: too few train rows ({len(tr)})")
    dev = torch.device(device)

    def _fit_one(xl, yl, tr_idx, te_idx, payload):
        """One layer's fit; returns pred at ``te_idx``. Payload filled in place."""
        if kind == "mlp":
            return n1m.fit_mlp(
                xl,
                yl,
                tr_idx,
                te_idx,
                mlp_width,
                MLP_MAP_LR,
                MLP_MAP_MAX_EPOCHS,
                MLP_MAP_BATCH,
                int(seed),
                dev,
                capture_out=payload,
            )
        # kernel: carve an inner val out of tr for (gamma, lambda) selection so
        # selection never touches the diagnostics holdout.
        rng = np.random.default_rng([1739, 5, int(seed)])
        p = rng.permutation(len(tr_idx))
        n_val = max(2, round(0.1 * len(tr_idx)))
        val_idx, inner_idx = tr_idx[p[:n_val]], tr_idx[p[n_val:]]
        return n1m.fit_krr_nystrom(
            xl,
            yl,
            inner_idx,
            val_idx,
            te_idx,
            m_centers=krr_m_centers,
            gamma_mult=KRR_MAP_GAMMA_MULT,
            lambdas=KRR_MAP_LAMBDAS,
            seed=int(seed),
            dev=dev,
            block=KRR_MAP_BLOCK,
            capture_out=payload,
        )

    # ---- stage 1: split fit -> honest held-out diagnostics -----------------
    t0 = _time.time()
    preds_hold, fit_meta = [], []
    for li in range(n_layers):
        pred, meta = _fit_one(x[li], y[li], tr, hold, {})
        preds_hold.append(np.asarray(pred, dtype=np.float64))
        fit_meta.append(meta)
        print(
            f"[nlmap] {kind} diag layer {li + 1}/{n_layers} "
            f"n_tr={len(tr)} elapsed={_time.time() - t0:.1f}s",
            flush=True,
        )
    diagnostics = map_diagnostics(
        np.stack(preds_hold), x[:, hold], y[:, hold], x[:, tr], y[:, tr], knn_ks=knn_ks
    )
    diagnostics.update(
        {
            "n_train": len(tr),
            "n_holdout": len(hold),
            "map_kind": kind,
            "fit_meta_per_layer": fit_meta,
            "apply_device": str(device),
            "diag_fit_s": round(_time.time() - t0, 1),
        }
    )

    # ---- stage 2: frozen payload (full-pool refit by default) --------------
    t1 = _time.time()
    payloads: list[dict] = []
    if refit_full:
        all_rows = np.arange(n, dtype=np.int64)
        for li in range(n_layers):
            payload: dict = {}
            _fit_one(x[li], y[li], all_rows, all_rows[:2], payload)
            if not payload:
                raise RuntimeError(f"{kind} fitter returned no capture payload (layer {li})")
            payloads.append(payload)
            print(
                f"[nlmap] {kind} refit layer {li + 1}/{n_layers} "
                f"n_fit={n} elapsed={_time.time() - t1:.1f}s",
                flush=True,
            )
    else:
        for li in range(n_layers):
            payload = {}
            _fit_one(x[li], y[li], tr, hold, payload)
            payloads.append(payload)
    diagnostics.update(
        {
            "w_fit_rows": int(n if refit_full else len(tr)),
            "w_refit_on_full_u": bool(refit_full),
            "solver": kind,
            "refit_s": round(_time.time() - t1, 1),
        }
    )
    logger.info(
        "[fits] %s map fit: Ly=%d n_tr=%d n_hold=%d w_fit_rows=%d mean r2_map=%.4f",
        kind,
        n_layers,
        len(tr),
        len(hold),
        diagnostics["w_fit_rows"],
        float(np.mean([r["r2_map"] for r in diagnostics["per_layer"]])),
    )
    return MapFit(
        w=None,
        x_mu=None,
        x_sd=None,
        y_mu=None,
        diagnostics=diagnostics,
        kind=kind,
        nl_payloads=tuple(payloads),
        apply_device=str(device),
    )


def apply_map(x: np.ndarray, m: MapFit, *, w: np.ndarray | None = None) -> np.ndarray:
    """Apply the frozen map to (Ly, n, d): ``((x-x_mu)/x_sd) @ W + y_mu``.

    ``w`` overrides the weight tensor (the shuffled-map control applies the
    SAME standardization with row-permuted weights).

    NONLINEAR kinds dispatch to :func:`apply_nl_map` (per-layer N1M payloads);
    ``w`` is then meaningless and raises rather than being silently ignored —
    a shuffled-weight control has no nonlinear analogue in this round.
    """
    if m.kind != "linear":
        if w is not None:
            raise ValueError(
                f"apply_map(w=...) is linear-only; MapFit kind={m.kind!r} has no weight tensor"
            )
        return apply_nl_map(x, m)
    x = np.asarray(x, dtype=np.float64)
    weights = m.w if w is None else w
    return ((x - m.x_mu) / m.x_sd) @ weights + m.y_mu


def _n1m():
    """Import the #779 N1M fitter module (repo-root syspath-guarded).

    ``scripts/`` is not on ``sys.path`` under script mode (#823 trap), so the
    guard is mandatory for this ``src/``-side import — mirrors the
    ``_ensure_repo_root_on_syspath`` pattern the #1739 scripts already use.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    sentinel = root / "scripts" / "issue779_ffc_n1m_fits.py"
    if not sentinel.is_file():  # wrong depth => fail loud, never a silent miss
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # The module lives in scripts/, so ROOT alone does not make it importable —
    # under script mode sys.path[0] happens to be scripts/ already, which is why
    # root-only worked there and left in-process callers (pytest) depending on
    # import-order luck from a sibling test. Insert the dir the sentinel proves.
    if str(sentinel.parent) not in sys.path:
        sys.path.insert(0, str(sentinel.parent))
    import issue779_ffc_n1m_fits as n1m

    return n1m


def apply_nl_map(x: np.ndarray, m: MapFit) -> np.ndarray:
    """Apply a per-layer nonlinear map to (Ly, n, d) -> (Ly, n, d) float64.

    Each layer's payload carries its own standardizer, so this is exactly the
    #779 ``apply_map`` predict path per layer (the roundtrip-gated one) — no
    re-derivation here. Serial over the LAYER axis by construction: the
    payload format is per-layer and each apply is one large GEMM / cdist
    (FLOP-bound), not a tiny-op loop.
    """
    import torch

    n1m = _n1m()
    x = np.asarray(x, dtype=np.float64)
    if len(m.nl_payloads) != x.shape[0]:
        raise ValueError(f"nl_payloads {len(m.nl_payloads)} != n_layers {x.shape[0]}")
    dev = torch.device(m.apply_device)
    return np.stack([n1m.apply_map(m.nl_payloads[li], x[li], dev) for li in range(x.shape[0])])


def shuffled_map_weights(w: np.ndarray, *, seed: int) -> np.ndarray:
    """Row-permuted (input-dim) map weights per layer — the arm-13 control.

    A row permutation preserves the per-layer Frobenius norm EXACTLY; the
    equality is asserted and logged (the plan's norm-preservation check).
    """
    w = np.asarray(w)
    rng = np.random.default_rng([1739, 5, int(seed)])
    out = np.empty_like(w)
    for li in range(w.shape[0]):  # Ly is small; permutation per layer
        out[li] = w[li][rng.permutation(w.shape[1])]
    norms = np.linalg.norm(w.reshape(w.shape[0], -1), axis=1)
    norms_shuf = np.linalg.norm(out.reshape(w.shape[0], -1), axis=1)
    assert np.allclose(norms, norms_shuf, rtol=1e-12), (norms, norms_shuf)
    logger.info(
        "[fits] shuffled-map control: per-layer Frobenius preserved (max rel diff %.2e)",
        float(np.max(np.abs(norms - norms_shuf) / np.maximum(norms, 1e-30))),
    )
    return out
