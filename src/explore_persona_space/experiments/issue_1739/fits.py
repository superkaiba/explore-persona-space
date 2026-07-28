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
    """Frozen linear context->answer map (standardized-input-space weights).

    Application contract mirrors ``ridge_fit_predict_fast_layer_batched``:
    ``pred = ((x - x_mu)/x_sd) @ w + y_mu`` (float64, layer-leading).
    """

    w: np.ndarray  # (Ly, d, d)
    x_mu: np.ndarray  # (Ly, 1, d)
    x_sd: np.ndarray  # (Ly, 1, d)
    y_mu: np.ndarray  # (Ly, 1, d)
    diagnostics: dict


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


def apply_map(x: np.ndarray, m: MapFit, *, w: np.ndarray | None = None) -> np.ndarray:
    """Apply the frozen map to (Ly, n, d): ``((x-x_mu)/x_sd) @ W + y_mu``.

    ``w`` overrides the weight tensor (the shuffled-map control applies the
    SAME standardization with row-permuted weights).
    """
    x = np.asarray(x, dtype=np.float64)
    weights = m.w if w is None else w
    return ((x - m.x_mu) / m.x_sd) @ weights + m.y_mu


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
