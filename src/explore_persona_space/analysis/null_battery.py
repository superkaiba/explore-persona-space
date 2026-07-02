"""Null battery for the Persona Vectors prediction-specificity test (issue #778).

The paper (arXiv 2507.21509) tests whether its persona-vector direction ``r_B``
predicts trait expression, but its ONLY specificity control is cross-trait
directions (which are weak because the traits are correlated). This module adds a
rigorous null battery so the claim "``r_B`` carries trait-SPECIFIC signal" is
tested against what an arbitrary direction — or the extraction pipeline run on
shuffled labels — would produce.

Everything here is closed-form / sampling statistics over cached activation
tensors (Pearson r, PCA, covariance, permutation, bootstrap) — no model calls,
no gradient descent. It runs CPU-only, off-pod (plan v2 §9).

Four nulls (plan v2 §5):
  - ``perm``:      re-run diff-of-means on SHUFFLED pos/neg labels of the cached
                   extraction activation pools (≥200 draws) — the gold-standard
                   null (destroys the trait signal, keeps the pipeline structure).
  - ``randnorm``:  random directions ~ N(0, Σ_activations) with diagonal
                   shrinkage, renormalized to ‖r_B‖ (≥200 draws; NOT isotropic).
  - ``crosstrait``: the OTHER traits' ``r_B`` (fixed, 2 dirs/trait) — the paper's
                   own control, recomputed for direct comparison.
  - ``pca_topk``:  top-k principal components of the (pos-neg) per-pair activation
                   differences (fixed, k dirs).

Selection-symmetric null policy (Must-Fix #1): the observed matched-trait
statistic is ``max_over_28_layers(|r|)``. To avoid the 28-chances-vs-1 asymmetry,
EVERY null draw AND every fixed null direction computes its OWN
``max_over_28_layers(|r|)`` before contributing to the null band. The per-draw x
per-layer |r| matrices are persisted so the analyzer can recompute the honest
band post-hoc.

Monitoring setting reports BOTH ``overall_r`` AND ``within_condition_r``
(Must-Fix #2): the within-condition r is the honest "controls for prompt type"
read (Fisher-z-averaged within-system-prompt r).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger("issue778.null_battery")

N_LAYERS = 28
DEFAULT_N_DRAWS = 200
DEFAULT_PCA_K = 5
DEFAULT_SHRINKAGE_LAMBDAS: tuple[float, ...] = (0.05, 0.1, 0.2)
PRIMARY_LAMBDA = 0.1
DEFAULT_BOOTSTRAP = 10_000


# ── Core correlation primitives ─────────────────────────────────────────────────


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between 1-D arrays; NaN if either is constant (undefined)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"_pearson expects matching 1-D arrays, got {x.shape} {y.shape}")
    if x.size < 3:
        return float("nan")
    xs = x - x.mean()
    ys = y - y.mean()
    denom = np.sqrt((xs * xs).sum() * (ys * ys).sum())
    if denom == 0:
        return float("nan")
    return float((xs * ys).sum() / denom)


def project(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """a_proj_b: scalar projection of each row of ``activations`` onto ``direction``.

    Faithful to the paper's ``a_proj_b(a, b) = (a·b) / ‖b‖`` (cal_projection.py).

    Args:
        activations: ``(n, D)`` per-example activation at ONE layer.
        direction:   ``(D,)`` direction at the SAME layer.
    Returns:
        ``(n,)`` scalar projections.
    """
    activations = np.asarray(activations, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if activations.ndim != 2 or direction.ndim != 1:
        raise ValueError(f"shapes: activations {activations.shape}, direction {direction.shape}")
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros(activations.shape[0], dtype=np.float64)
    return (activations @ direction) / norm


def r_per_layer(
    activations: np.ndarray,
    direction_per_layer: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Pearson r between projection and ``target`` at EVERY layer.

    Args:
        activations: ``(n, L, D)`` per-example, per-layer predictor activation
            (last-prompt-token for monitoring; finetuned-base shift for finetune).
        direction_per_layer: ``(L, D)`` direction per layer (``r_B`` or a null).
        target: ``(n,)`` trait-expression DV per example.
    Returns:
        ``(L,)`` per-layer Pearson r.
    """
    activations = np.asarray(activations, dtype=np.float64)
    direction_per_layer = np.asarray(direction_per_layer, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n, L, D = activations.shape
    if direction_per_layer.shape != (L, D):
        raise ValueError(f"direction {direction_per_layer.shape} != (L,D)=({L},{D})")
    if target.shape != (n,):
        raise ValueError(f"target {target.shape} != (n,)=({n},)")
    out = np.empty(L, dtype=np.float64)
    for layer in range(L):
        proj = project(activations[:, layer, :], direction_per_layer[layer])
        out[layer] = _pearson(proj, target)
    return out


def max_abs_over_layers(r_layers: np.ndarray) -> float:
    """max_over_layers(|r|), ignoring NaN layers. NaN only if ALL layers NaN."""
    r_layers = np.asarray(r_layers, dtype=np.float64)
    absr = np.abs(r_layers)
    if np.all(np.isnan(absr)):
        return float("nan")
    return float(np.nanmax(absr))


def argmax_abs_layer(r_layers: np.ndarray) -> int:
    """Layer index of max |r| (0-based, ignoring NaN)."""
    absr = np.abs(np.asarray(r_layers, dtype=np.float64))
    absr = np.where(np.isnan(absr), -np.inf, absr)
    return int(np.argmax(absr))


# ── Within-condition r (monitoring only) ────────────────────────────────────────


def within_condition_r_per_layer(
    activations: np.ndarray,
    direction_per_layer: np.ndarray,
    target: np.ndarray,
    condition_ids: np.ndarray,
) -> np.ndarray:
    """Fisher-z-averaged within-condition Pearson r at every layer.

    Groups examples by ``condition_ids`` (the system-prompt id in monitoring),
    computes Pearson r WITHIN each group, Fisher-z averages across groups
    (weighted by n-3, the standard weighting), and inverts back to r. This is
    the honest "controls for prompt type" read — the paper cautions the overall r
    is "driven primarily from distinguishing between different prompt types".

    Groups with < 4 members (r undefined / near-degenerate) are dropped from the
    average; a layer with no usable group yields NaN.
    """
    activations = np.asarray(activations, dtype=np.float64)
    direction_per_layer = np.asarray(direction_per_layer, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    condition_ids = np.asarray(condition_ids)
    n, L, _ = activations.shape
    if condition_ids.shape != (n,):
        raise ValueError(f"condition_ids {condition_ids.shape} != (n,)=({n},)")
    uniq = np.unique(condition_ids)
    out = np.empty(L, dtype=np.float64)
    for layer in range(L):
        proj = project(activations[:, layer, :], direction_per_layer[layer])
        z_sum = 0.0
        w_sum = 0.0
        for c in uniq:
            mask = condition_ids == c
            if mask.sum() < 4:
                continue
            r = _pearson(proj[mask], target[mask])
            if np.isnan(r):
                continue
            r = float(np.clip(r, -0.999999, 0.999999))
            z = np.arctanh(r)
            w = mask.sum() - 3  # Fisher-z variance weighting
            z_sum += w * z
            w_sum += w
        out[layer] = np.tanh(z_sum / w_sum) if w_sum > 0 else float("nan")
    return out


# ── Batched correlation core (draw-axis vectorization, #778 r4 perf) ───────────


def _batched_r_per_layer(
    predictor_acts: np.ndarray,
    directions: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Signed Pearson r for a STACK of directions: ``(B, L, D)`` -> ``(B, L)``.

    Pearson r is invariant under positive rescaling of the projection, so the
    per-direction norm division in :func:`project` is skipped; a zero direction
    (or any constant projection) yields NaN, matching ``_pearson``'s
    ``denom == 0`` / ``size < 3`` behavior.
    """
    predictor_acts = np.asarray(predictor_acts, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    m, L, D = predictor_acts.shape
    B = directions.shape[0]
    if directions.shape != (B, L, D):
        raise ValueError(f"directions {directions.shape} != (B,L,D)=({B},{L},{D})")
    out = np.full((B, L), np.nan, dtype=np.float64)
    if m < 3:
        return out
    tm = target - target.mean()
    t_ss = float((tm * tm).sum())
    for layer in range(L):
        proj = predictor_acts[:, layer, :] @ directions[:, layer, :].T  # (m, B)
        pm = proj - proj.mean(axis=0, keepdims=True)
        num = tm @ pm  # (B,)
        den = np.sqrt((pm * pm).sum(axis=0) * t_ss)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num / den
        r[den == 0] = np.nan
        out[:, layer] = r
    return out


def _batched_within_condition_r_per_layer(
    predictor_acts: np.ndarray,
    directions: np.ndarray,
    target: np.ndarray,
    condition_ids: np.ndarray,
) -> np.ndarray:
    """Batched Fisher-z within-condition r: ``(B, L, D)`` -> ``(B, L)``.

    Per (direction, layer): Pearson r WITHIN each condition group, Fisher-z
    averaged with n-3 weights, inverted back to r — element-wise identical
    semantics to :func:`within_condition_r_per_layer` (groups with < 4 members
    or NaN r are dropped PER ELEMENT; all-dropped yields NaN).
    """
    predictor_acts = np.asarray(predictor_acts, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    condition_ids = np.asarray(condition_ids)
    m, L, D = predictor_acts.shape
    B = directions.shape[0]
    if directions.shape != (B, L, D):
        raise ValueError(f"directions {directions.shape} != (B,L,D)=({B},{L},{D})")
    if condition_ids.shape != (m,):
        raise ValueError(f"condition_ids {condition_ids.shape} != (n,)=({m},)")
    uniq = np.unique(condition_ids)
    out = np.empty((B, L), dtype=np.float64)
    for layer in range(L):
        proj = predictor_acts[:, layer, :] @ directions[:, layer, :].T  # (m, B)
        z_sum = np.zeros(B, dtype=np.float64)
        w_sum = np.zeros(B, dtype=np.float64)
        for c in uniq:
            mask = condition_ids == c
            nc = int(mask.sum())
            if nc < 4:
                continue
            pj = proj[mask]  # (nc, B)
            tg = target[mask]
            tgm = tg - tg.mean()
            t_ss = float((tgm * tgm).sum())
            pm = pj - pj.mean(axis=0, keepdims=True)
            num = tgm @ pm
            den = np.sqrt((pm * pm).sum(axis=0) * t_ss)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = num / den
            r[den == 0] = np.nan
            good = np.isfinite(r)
            z = np.arctanh(np.clip(np.where(good, r, 0.0), -0.999999, 0.999999))
            w = float(nc - 3)  # Fisher-z variance weighting
            z_sum += np.where(good, w * z, 0.0)
            w_sum += np.where(good, w, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            layer_r = np.tanh(z_sum / w_sum)
        layer_r[w_sum <= 0] = np.nan
        out[:, layer] = layer_r
    return out


def _batched_abs_r(
    predictor_acts: np.ndarray,
    directions: np.ndarray,
    target: np.ndarray,
    *,
    within: bool,
    condition_ids: np.ndarray | None,
) -> np.ndarray:
    """|r| for a ``(B, L, D)`` direction stack -> ``(B, L)`` (routing helper)."""
    if within:
        if condition_ids is None:
            raise ValueError("within=True requires condition_ids")
        r = _batched_within_condition_r_per_layer(predictor_acts, directions, target, condition_ids)
    else:
        r = _batched_r_per_layer(predictor_acts, directions, target)
    return np.abs(r)


# ── Null 1: shuffled-label permutation (gold standard) ─────────────────────────


def perm_null_draws(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    predictor_acts: np.ndarray,
    target: np.ndarray,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = 0,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Per-draw x per-layer |r| for the shuffled-label permutation null.

    Pools the extraction pos+neg activation POOLS, shuffles the pos/neg labels,
    recomputes the diff-of-means direction per layer, projects the SAME predictor
    activations onto it, and correlates against the SAME target. Repeats
    ``n_draws`` times.

    Args:
        pos_acts / neg_acts: ``(n_pos, L, D)`` / ``(n_neg, L, D)`` extraction
            response-avg activation pools (kept rollouts).
        predictor_acts: ``(m, L, D)`` prediction-setting predictor activations
            (last-prompt-token or finetuned-base shift).
        target: ``(m,)`` trait-expression DV.
        within: if True, use the within-condition r (monitoring only).
        condition_ids: ``(m,)`` group ids (required when within=True).
    Returns:
        ``(n_draws, L)`` |r| matrix.
    """
    pos_acts = np.asarray(pos_acts, dtype=np.float64)
    neg_acts = np.asarray(neg_acts, dtype=np.float64)
    n_pos, L, D = pos_acts.shape
    n_neg = neg_acts.shape[0]
    pool = np.concatenate([pos_acts, neg_acts], axis=0)  # (n_pos+n_neg, L, D)
    n_total = n_pos + n_neg
    rng = np.random.default_rng(seed)
    # Same rng consumption order as the serial reference: one permutation per
    # draw, sequentially — only the arithmetic below is batched (#778 r4 perf).
    pos_idx = np.empty((n_draws, n_pos), dtype=np.intp)
    for d in range(n_draws):
        pos_idx[d] = rng.permutation(n_total)[:n_pos]
    # Subset-sum identity: with pool_sum fixed,
    #   direction_d = S_pos_d/n_pos - (pool_sum - S_pos_d)/n_neg,
    # so ALL draws' directions come from ONE selection-matrix GEMM per chunk
    # instead of two full pool-mean passes per draw (the ~4 s/draw serial trap).
    pool2d = pool.reshape(n_total, L * D)
    pool_sum = pool2d.sum(axis=0)  # (L*D,)
    out = np.empty((n_draws, L), dtype=np.float64)
    chunk = 250  # bounds the (chunk, L*D) intermediates to ~200 MB at 28x3584
    for s in range(0, n_draws, chunk):
        idx = pos_idx[s : s + chunk]  # (b, n_pos)
        b = idx.shape[0]
        sel = np.zeros((b, n_total), dtype=np.float64)
        sel[np.arange(b)[:, None], idx] = 1.0
        s_pos = sel @ pool2d  # (b, L*D)
        dirs = (s_pos / n_pos - (pool_sum - s_pos) / n_neg).reshape(b, L, D)
        out[s : s + b] = _batched_abs_r(
            predictor_acts, dirs, target, within=within, condition_ids=condition_ids
        )
    return out


def _perm_null_draws_serial(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    predictor_acts: np.ndarray,
    target: np.ndarray,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = 0,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Serial reference for :func:`perm_null_draws` — equivalence tests ONLY.

    Kept as the ground-truth implementation the vectorized path is verified
    against (`tests/test_null_battery_vectorized.py`); never called in
    production.
    """
    pos_acts = np.asarray(pos_acts, dtype=np.float64)
    neg_acts = np.asarray(neg_acts, dtype=np.float64)
    n_pos, L, _D = pos_acts.shape
    n_neg = neg_acts.shape[0]
    pool = np.concatenate([pos_acts, neg_acts], axis=0)  # (n_pos+n_neg, L, D)
    n_total = n_pos + n_neg
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, L), dtype=np.float64)
    for d in range(n_draws):
        perm = rng.permutation(n_total)
        fake_pos = pool[perm[:n_pos]]
        fake_neg = pool[perm[n_pos:]]
        direction = fake_pos.mean(axis=0) - fake_neg.mean(axis=0)  # (L, D)
        if within:
            r_layers = within_condition_r_per_layer(
                predictor_acts, direction, target, condition_ids
            )
        else:
            r_layers = r_per_layer(predictor_acts, direction, target)
        out[d] = np.abs(r_layers)
    return out


# ── Null 2: norm-matched random direction (covariance-realistic) ───────────────


def _shrunk_cholesky(acts_2d: np.ndarray, lam: float) -> np.ndarray:
    """Cholesky factor of the shrunk covariance Σ = (1-λ)Σ_emp + λ·diag(Σ_emp).

    ``acts_2d`` is ``(n, D)`` at ONE layer. Returns the ``(D, D)`` lower-triangular
    Cholesky factor for sampling N(0, Σ). A tiny jitter is added if the factor is
    not PD after shrinkage (numerical safety in 3584-dim).
    """
    acts_2d = np.asarray(acts_2d, dtype=np.float64)
    cov = np.cov(acts_2d, rowvar=False)  # (D, D)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - lam) * cov + lam * diag
    for jitter in (0.0, 1e-6, 1e-4, 1e-2):
        try:
            return np.linalg.cholesky(shrunk + jitter * np.eye(shrunk.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("shrunk covariance not PD even after jitter")


def randnorm_null_draws(
    pool_acts_per_layer: dict[int, np.ndarray],
    rb_norm_per_layer: np.ndarray,
    predictor_acts: np.ndarray,
    target: np.ndarray,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    lam: float = PRIMARY_LAMBDA,
    seed: int = 0,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Per-draw x per-layer |r| for the norm-matched random-direction null.

    Each draw samples a random direction per layer from N(0, Σ_activations) with
    diagonal shrinkage λ, renormalized to ‖r_B[layer]‖ (norm-matched, NOT
    isotropic — a plausible activation-space direction).

    Args:
        pool_acts_per_layer: {layer_idx: (n, D)} — the activation pool used to
            estimate Σ per layer (the stacked extraction pos+neg pool).
        rb_norm_per_layer: ``(L,)`` ‖r_B[layer]‖ target norms.
        predictor_acts / target / within / condition_ids: as in perm_null_draws.
    Returns:
        ``(n_draws, L)`` |r| matrix.
    """
    L = predictor_acts.shape[1]
    rng = np.random.default_rng(seed)
    # Precompute Cholesky per layer (expensive; do once).
    chols: dict[int, np.ndarray] = {}
    for layer in range(L):
        chols[layer] = _shrunk_cholesky(pool_acts_per_layer[layer], lam)
    out = np.empty((n_draws, L), dtype=np.float64)
    D = predictor_acts.shape[2]
    # |r| is invariant under positive per-direction rescaling, so the
    # norm-match to ‖r_B‖ (and the unit-normalization inside project()) is a
    # no-op for this return value — sample v = chol @ z and correlate directly.
    # A zero-norm v yields a constant projection -> NaN, matching the serial
    # reference. rng order matches the serial nested loop: standard_normal
    # fills a C-order (b, L, D) block draw-major, layer-minor (#778 r4 perf).
    chunk = 250  # bounds the (chunk, L, D) gaussian/direction blocks to ~200 MB
    for s in range(0, n_draws, chunk):
        b = min(chunk, n_draws - s)
        z_block = rng.standard_normal((b, L, D))
        dirs = np.empty((b, L, D), dtype=np.float64)
        for layer in range(L):
            dirs[:, layer, :] = z_block[:, layer, :] @ chols[layer].T
        out[s : s + b] = _batched_abs_r(
            predictor_acts, dirs, target, within=within, condition_ids=condition_ids
        )
    return out


def _randnorm_null_draws_serial(
    pool_acts_per_layer: dict[int, np.ndarray],
    rb_norm_per_layer: np.ndarray,
    predictor_acts: np.ndarray,
    target: np.ndarray,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    lam: float = PRIMARY_LAMBDA,
    seed: int = 0,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Serial reference for :func:`randnorm_null_draws` — equivalence tests ONLY."""
    L = predictor_acts.shape[1]
    rng = np.random.default_rng(seed)
    chols: dict[int, np.ndarray] = {}
    for layer in range(L):
        chols[layer] = _shrunk_cholesky(pool_acts_per_layer[layer], lam)
    out = np.empty((n_draws, L), dtype=np.float64)
    D = predictor_acts.shape[2]
    for d in range(n_draws):
        direction = np.empty((L, D), dtype=np.float64)
        for layer in range(L):
            z = rng.standard_normal(D)
            v = chols[layer] @ z
            vn = np.linalg.norm(v)
            if vn == 0:
                direction[layer] = v
            else:
                direction[layer] = v / vn * rb_norm_per_layer[layer]
        if within:
            r_layers = within_condition_r_per_layer(
                predictor_acts, direction, target, condition_ids
            )
        else:
            r_layers = r_per_layer(predictor_acts, direction, target)
        out[d] = np.abs(r_layers)
    return out


# ── Null 3: cross-trait directions (fixed) ─────────────────────────────────────


def crosstrait_null(
    other_rbs: dict[str, np.ndarray],
    predictor_acts: np.ndarray,
    target: np.ndarray,
    *,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Per-direction x per-layer |r| for the cross-trait null (fixed directions).

    Args:
        other_rbs: {other_trait: (L, D)} — the OTHER traits' r_B directions.
    Returns:
        ``(n_directions, L)`` |r| matrix (rows ordered by sorted trait name).
    """
    names = sorted(other_rbs)
    L = predictor_acts.shape[1]
    out = np.empty((len(names), L), dtype=np.float64)
    for i, name in enumerate(names):
        direction = np.asarray(other_rbs[name], dtype=np.float64)
        if within:
            r_layers = within_condition_r_per_layer(
                predictor_acts, direction, target, condition_ids
            )
        else:
            r_layers = r_per_layer(predictor_acts, direction, target)
        out[i] = np.abs(r_layers)
    return out


# ── Null 4: PCA top-k of pos-neg per-pair differences (fixed) ──────────────────


def pca_topk_directions(
    diff_acts: np.ndarray,
    k: int = DEFAULT_PCA_K,
) -> np.ndarray:
    """Top-k principal components of the (pos-neg) per-pair activation diffs.

    Args:
        diff_acts: ``(n_pairs, L, D)`` per-pair (pos-neg) activation differences.
    Returns:
        ``(k, L, D)`` — the top-k PC directions PER LAYER (one PCA per layer).
    """
    diff_acts = np.asarray(diff_acts, dtype=np.float64)
    n_pairs, L, D = diff_acts.shape
    k_eff = min(k, n_pairs, D)
    out = np.zeros((k, L, D), dtype=np.float64)
    for layer in range(L):
        X = diff_acts[:, layer, :]  # (n_pairs, D)
        Xc = X - X.mean(axis=0, keepdims=True)
        # Economy SVD: right singular vectors are the PCs.
        try:
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            # gesdd non-convergence fallback (memory: numpy SVD non-convergence).
            from scipy.linalg import svd as scipy_svd

            _, _, vt = scipy_svd(Xc, full_matrices=False, lapack_driver="gesvd")
        for j in range(k_eff):
            out[j, layer] = vt[j]
    return out


def pca_topk_null(
    diff_acts: np.ndarray,
    predictor_acts: np.ndarray,
    target: np.ndarray,
    *,
    k: int = DEFAULT_PCA_K,
    within: bool = False,
    condition_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Per-direction x per-layer |r| for the PCA top-k null (fixed directions).

    Returns:
        ``(k, L)`` |r| matrix (one row per PC).
    """
    dirs = pca_topk_directions(diff_acts, k)  # (k, L, D)
    K, L, _ = dirs.shape
    out = np.empty((K, L), dtype=np.float64)
    for j in range(K):
        if within:
            r_layers = within_condition_r_per_layer(predictor_acts, dirs[j], target, condition_ids)
        else:
            r_layers = r_per_layer(predictor_acts, dirs[j], target)
        out[j] = np.abs(r_layers)
    return out


# ── Bootstrap CI on the observed matched-trait r ───────────────────────────────


def bootstrap_ci_matched_r(
    predictor_acts: np.ndarray,
    rb_per_layer: np.ndarray,
    target: np.ndarray,
    selected_layer: int,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float]:
    """95% bootstrap CI on the observed matched-trait r at ``selected_layer``.

    Resamples the (predictor, target) rows with replacement ``n_boot`` times,
    recomputes r at the FIXED selected layer each time.
    """
    predictor_acts = np.asarray(predictor_acts, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = predictor_acts.shape[0]
    proj = project(predictor_acts[:, selected_layer, :], rb_per_layer[selected_layer])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = _pearson(proj[idx], target[idx])
    valid = boots[~np.isnan(boots)]
    if valid.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))


# ── Empirical p + band ─────────────────────────────────────────────────────────


def empirical_p_one_sided(observed: float, null_max_abs: np.ndarray) -> float:
    """One-sided empirical p: P(null max|r| >= observed), with the +1 correction.

    ``null_max_abs`` is the per-draw / per-direction max-over-layers |r| array.
    NaN null entries are dropped (undefined draws).
    """
    null_max_abs = np.asarray(null_max_abs, dtype=np.float64)
    valid = null_max_abs[~np.isnan(null_max_abs)]
    if valid.size == 0:
        return float("nan")
    n_ge = int((valid >= observed).sum())
    return (n_ge + 1) / (valid.size + 1)


def null_band(null_max_abs: np.ndarray) -> tuple[float, float]:
    """[p2.5, p97.5] band of the per-draw / per-direction max-over-layers |r|."""
    null_max_abs = np.asarray(null_max_abs, dtype=np.float64)
    valid = null_max_abs[~np.isnan(null_max_abs)]
    if valid.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH-adjusted p-values (FDR). NaN p-values pass through as NaN."""
    arr = np.asarray(pvals, dtype=np.float64)
    finite_mask = ~np.isnan(arr)
    finite = arr[finite_mask]
    m = finite.size
    adj = np.full_like(arr, np.nan)
    if m == 0:
        return adj.tolist()
    order = np.argsort(finite)
    ranked = finite[order]
    bh = ranked * m / (np.arange(1, m + 1))
    # Enforce monotonicity (step-up).
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    out_finite = np.empty(m, dtype=np.float64)
    out_finite[order] = bh
    adj[finite_mask] = out_finite
    return adj.tolist()


# ── Top-level per-(trait, setting) result ──────────────────────────────────────


@dataclass
class NullResult:
    """One null's summary for a (trait x setting) cell."""

    n_draws: int
    r_p2_5: float
    r_p97_5: float
    empirical_p_one_sided: float
    draws_max_abs: list[float] = field(default_factory=list)  # per-draw max|r|


@dataclass
class SettingResult:
    """Full null-battery result for one (trait x setting) cell.

    ``setting`` in {monitoring_overall, monitoring_within, finetune}.
    """

    trait: str
    setting: str
    n_points: int
    matched_r: float  # observed matched-trait max-over-layers |r| (signed r at selected layer)
    matched_max_abs: float  # observed max-over-layers |r|
    matched_selected_layer: int  # 0-based layer index of max |r|
    matched_r_bootstrap_ci_95: tuple[float, float]
    nulls: dict[str, NullResult]
    reproducibility: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["nulls"] = {k: asdict(v) for k, v in self.nulls.items()}
        return d


def compute_setting(
    trait: str,
    setting: str,
    *,
    predictor_acts: np.ndarray,
    rb_per_layer: np.ndarray,
    target: np.ndarray,
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    other_rbs: dict[str, np.ndarray],
    condition_ids: np.ndarray | None = None,
    n_draws: int = DEFAULT_N_DRAWS,
    lam: float = PRIMARY_LAMBDA,
    pca_k: int = DEFAULT_PCA_K,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> tuple[SettingResult, dict[str, np.ndarray]]:
    """Compute the full null battery for one (trait x setting) cell.

    Returns (result, draw_matrices) where draw_matrices maps
    ``<null_kind>`` -> ``(n_draws|n_dirs, L)`` |r| matrix (the persisted
    downstream inputs, plan v2 §5/§6.5).

    ``within`` is inferred from ``setting`` (``monitoring_within`` uses the
    Fisher-z within-condition r; requires ``condition_ids``).
    """
    within = setting == "monitoring_within"
    if within and condition_ids is None:
        raise ValueError("monitoring_within requires condition_ids")

    # Observed matched-trait r per layer -> max-over-layers |r| + selected layer.
    if within:
        matched_r_layers = within_condition_r_per_layer(
            predictor_acts, rb_per_layer, target, condition_ids
        )
    else:
        matched_r_layers = r_per_layer(predictor_acts, rb_per_layer, target)
    matched_max = max_abs_over_layers(matched_r_layers)
    sel_layer = argmax_abs_layer(matched_r_layers)
    matched_r_signed = float(matched_r_layers[sel_layer])
    ci = bootstrap_ci_matched_r(
        predictor_acts, rb_per_layer, target, sel_layer, n_boot=n_boot, seed=seed
    )

    # Pool for randnorm Σ estimation (stacked pos+neg per layer).
    pool = np.concatenate([pos_acts, neg_acts], axis=0)  # (n_pool, L, D)
    L = predictor_acts.shape[1]
    pool_by_layer = {layer: pool[:, layer, :] for layer in range(L)}
    rb_norms = np.linalg.norm(np.asarray(rb_per_layer, dtype=np.float64), axis=1)  # (L,)

    # per-pair diffs for PCA: pair pos_i with neg_i up to min pool size.
    n_pair = min(pos_acts.shape[0], neg_acts.shape[0])
    diff_acts = pos_acts[:n_pair] - neg_acts[:n_pair]  # (n_pair, L, D)

    draw_matrices: dict[str, np.ndarray] = {}
    draw_matrices["perm"] = perm_null_draws(
        pos_acts,
        neg_acts,
        predictor_acts,
        target,
        n_draws=n_draws,
        seed=seed,
        within=within,
        condition_ids=condition_ids,
    )
    draw_matrices["randnorm"] = randnorm_null_draws(
        pool_by_layer,
        rb_norms,
        predictor_acts,
        target,
        n_draws=n_draws,
        lam=lam,
        seed=seed,
        within=within,
        condition_ids=condition_ids,
    )
    draw_matrices["crosstrait"] = crosstrait_null(
        other_rbs, predictor_acts, target, within=within, condition_ids=condition_ids
    )
    draw_matrices["pca_topk"] = pca_topk_null(
        diff_acts, predictor_acts, target, k=pca_k, within=within, condition_ids=condition_ids
    )

    nulls: dict[str, NullResult] = {}
    for kind, mat in draw_matrices.items():
        per_draw_max = np.array([max_abs_over_layers(mat[i]) for i in range(mat.shape[0])])
        lo, hi = null_band(per_draw_max)
        p = empirical_p_one_sided(matched_max, per_draw_max)
        nulls[kind] = NullResult(
            n_draws=int(mat.shape[0]),
            r_p2_5=lo,
            r_p97_5=hi,
            empirical_p_one_sided=p,
            draws_max_abs=[float(v) for v in per_draw_max],
        )

    result = SettingResult(
        trait=trait,
        setting=setting,
        n_points=int(predictor_acts.shape[0]),
        matched_r=matched_r_signed,
        matched_max_abs=matched_max,
        matched_selected_layer=sel_layer,
        matched_r_bootstrap_ci_95=ci,
        nulls=nulls,
    )
    return result, draw_matrices
