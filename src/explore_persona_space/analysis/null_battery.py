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
    return _batched_r_overall(activations, direction_per_layer[None], target)[:, 0]


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


# ── Batched primitives (issue #834 vectorization) ───────────────────────────────
#
# The draw/layer Python loops below were the #722-class overhead-bound hotspot
# (2+ h at 1000 draws). These helpers batch the ARITHMETIC only; every random
# quantity is still generated in the exact original loop order (bit-identical
# rng streams), so seed semantics are frozen (plan v2 §3.1).

_MAX_BATCH_BYTES = 2 * 1024**3  # cap on NEW per-chunk batch buffers (plan v2 §3.4)


def _k_chunks(n_draws: int, bytes_per_draw: int):
    """Yield contiguous ``(start, stop)`` slices of the draw axis.

    Chunk size is set so the projected NEW-buffer footprint stays under
    ``_MAX_BATCH_BYTES``. Chunking touches arithmetic only — rng generation
    always runs over the FULL draw range first, so chunk boundaries cannot
    perturb the stream.
    """
    if n_draws <= 0:
        return
    per = max(1, int(_MAX_BATCH_BYTES // max(1, bytes_per_draw)))
    for start in range(0, n_draws, per):
        yield start, min(start + per, n_draws)


def _batched_project(activations: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Batched ``project``: ``(n, L, D)`` x ``(K, L, D)`` -> ``(n, L, K)``.

    proj[i, l, k] = (activations[i, l] @ directions[k, l]) / ‖directions[k, l]‖,
    with proj[:, l, k] = 0 where the norm is 0 (matches ``project``).
    Per-layer GEMM (L BLAS-3 calls) rather than bare einsum (plan v2 §3.2).
    """
    activations = np.asarray(activations, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if activations.ndim != 3 or directions.ndim != 3:
        raise ValueError(f"shapes: activations {activations.shape}, directions {directions.shape}")
    n, L, D = activations.shape
    K = directions.shape[0]
    if directions.shape != (K, L, D):
        raise ValueError(f"directions {directions.shape} != (K,L,D)=({K},{L},{D})")
    norms = np.linalg.norm(directions, axis=2)  # (K, L)
    proj = np.empty((n, L, K), dtype=np.float64)
    for layer in range(L):
        proj[:, layer, :] = activations[:, layer, :] @ directions[:, layer, :].T
    safe = np.where(norms == 0, 1.0, norms)  # (K, L)
    proj /= safe.T[None, :, :]
    zero_lk = (norms == 0).T  # (L, K)
    if zero_lk.any():
        proj[:, zero_lk] = 0.0
    return proj


def _batched_pearson(proj: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Batched ``_pearson``: ``(n, L, K)`` x ``(n,)`` -> r ``(L, K)``.

    Same algebra as ``_pearson`` (centered sums of products over n); NaN where
    n < 3 or either operand's centered sum-of-squares is 0. The reductions run
    over the LAST (contiguous) axis of a ``(L, K, n)`` transpose so numpy uses
    the same pairwise summation as the scalar 1-D path — a constant column
    centers to EXACTLY zero (denom == 0 -> NaN), never to ~1e-18 residue (a
    strided axis-0 mean falls back to sequential summation and misses the NaN
    branch).
    """
    proj = np.asarray(proj, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if proj.ndim != 3:
        raise ValueError(f"proj {proj.shape} is not (n, L, K)")
    n, L, K = proj.shape
    if target.shape != (n,):
        raise ValueError(f"target {target.shape} != (n,)=({n},)")
    if n < 3:
        return np.full((L, K), np.nan, dtype=np.float64)
    tc = target - target.mean()
    t_ss = float((tc * tc).sum())
    pt = np.ascontiguousarray(np.moveaxis(proj, 0, -1))  # (L, K, n)
    pc = pt - pt.mean(axis=-1, keepdims=True)
    p_ss = (pc * pc).sum(axis=-1)  # (L, K)
    cov = (pc * tc).sum(axis=-1)  # (L, K); tc broadcasts along n
    denom = np.sqrt(p_ss * t_ss)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, np.nan, cov / denom)


def _batched_r_overall(
    activations: np.ndarray,
    directions: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Batched overall r: ``(n, L, D)`` x ``(K, L, D)`` x ``(n,)`` -> r ``(L, K)``.

    Projects REFERENCE-CENTERED activations (``acts - acts[0]``): Pearson r is
    exactly shift-invariant, and centering makes identical activation rows
    EXACT zero rows, so a constant-projection layer yields denom == 0 -> NaN
    under ANY BLAS kernel (GEMM row-blocking otherwise leaves ~1e-17 residue
    on identical rows, silently missing the loop path's NaN branch).
    """
    activations = np.asarray(activations, dtype=np.float64)
    if activations.shape[0]:  # n == 0 has no reference row; _batched_pearson NaNs n < 3
        activations = activations - activations[0]
    proj = _batched_project(activations, directions)  # (n, L, K)
    return _batched_pearson(proj, target)


def _batched_within_r(
    activations: np.ndarray,
    directions: np.ndarray,
    target: np.ndarray,
    condition_ids: np.ndarray,
) -> np.ndarray:
    """Batched ``within_condition_r_per_layer``: -> r ``(L, K)``.

    Per-(l, k)-ELEMENT NaN-masked Fisher-z weighting: the loop's
    ``if np.isnan(r): continue`` drops a NaN group from that (layer[, draw])
    cell's z_sum AND w_sum while the group still contributes at every other
    cell — so weights are accumulated with a per-element finite mask, never a
    wholesale per-group drop (plan v2 §3.2, ensemble Must-Fix). Each group's
    activations are centered on the group's own first row before projecting
    (same exact-NaN rationale as ``_batched_r_overall``, at group scope).
    """
    activations = np.asarray(activations, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    condition_ids = np.asarray(condition_ids)
    n, L, _D = activations.shape
    K = directions.shape[0]
    if condition_ids.shape != (n,):
        raise ValueError(f"condition_ids {condition_ids.shape} != (n,)=({n},)")
    z_sum = np.zeros((L, K), dtype=np.float64)
    w_sum = np.zeros((L, K), dtype=np.float64)
    for c in np.unique(condition_ids):
        mask = condition_ids == c
        n_c = int(mask.sum())
        if n_c < 4:
            continue
        acts_g = activations[mask]
        proj_g = _batched_project(acts_g - acts_g[0], directions)  # (n_c, L, K)
        r_c = _batched_pearson(proj_g, target[mask])  # (L, K); NaN where undefined
        finite = np.isfinite(r_c)
        z = np.arctanh(np.clip(r_c, -0.999999, 0.999999))
        w = float(n_c - 3)  # Fisher-z variance weighting
        z_sum += np.where(finite, w * z, 0.0)
        w_sum += np.where(finite, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w_sum > 0, np.tanh(z_sum / w_sum), np.nan)


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
    n = activations.shape[0]
    if condition_ids.shape != (n,):
        raise ValueError(f"condition_ids {condition_ids.shape} != (n,)=({n},)")
    return _batched_within_r(activations, direction_per_layer[None], target, condition_ids)[:, 0]


# ── Null 1: shuffled-label permutation (gold standard) ─────────────────────────


def _perm_directions(pool: np.ndarray, perms: np.ndarray, n_pos: int) -> np.ndarray:
    """Diff-of-means directions for K label shuffles as ONE GEMM.

    ``pool (n_total, L, D)``, ``perms (K, n_total)`` int -> dirs ``(K, L, D)``.
    Builds W ``(K, n_total)`` with W[k, perms[k, :n_pos]] = 1/n_pos and
    W[k, perms[k, n_pos:]] = -1/n_neg, so ``W @ pool.reshape(n_total, L*D)``
    is ``fake_pos.mean(0) - fake_neg.mean(0)`` for every shuffle at once.
    """
    pool = np.asarray(pool, dtype=np.float64)
    n_total, L, D = pool.shape
    K = perms.shape[0]
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        # Degenerate empty side -> NaN directions (the loop's mean() of an empty
        # slice is NaN; a weight trick cannot express this — an empty side has
        # no indices to scatter into, leaving a finite one-sided mean).
        return np.full((K, L, D), np.nan, dtype=np.float64)
    W = np.zeros((K, n_total), dtype=np.float64)
    np.put_along_axis(W, perms[:, :n_pos], 1.0 / n_pos, axis=1)
    np.put_along_axis(W, perms[:, n_pos:], -1.0 / n_neg, axis=1)
    return (W @ pool.reshape(n_total, L * D)).reshape(K, L, D)


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
    predictor_acts = np.asarray(predictor_acts, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n_pos, L, D = pos_acts.shape
    pool = np.concatenate([pos_acts, neg_acts], axis=0)  # (n_pos+n_neg, L, D)
    n_total = pool.shape[0]
    rng = np.random.default_rng(seed)
    # RNG in the EXACT original loop order over the FULL draw range (plan v2
    # §3.1): one rng.permutation(n_total) per draw, before any chunked
    # arithmetic — bit-identical stream to the loop version.
    if n_draws == 0:
        return np.empty((0, L), dtype=np.float64)
    perms = np.stack([rng.permutation(n_total) for _ in range(n_draws)])  # (K, n_total)
    out = np.empty((n_draws, L), dtype=np.float64)
    n_pred = predictor_acts.shape[0]
    bytes_per_draw = L * D * 8 + 4 * n_pred * L * 8  # dirs + proj + pearson temps (§3.4)
    for start, stop in _k_chunks(n_draws, bytes_per_draw):
        dirs = _perm_directions(pool, perms[start:stop], n_pos)  # (k, L, D)
        if within:
            r = _batched_within_r(predictor_acts, dirs, target, condition_ids)  # (L, k)
        else:
            r = _batched_r_overall(predictor_acts, dirs, target)  # (L, k)
        out[start:stop] = np.abs(r).T
    return out


# ── Null 2: norm-matched random direction (covariance-realistic) ───────────────


def shrunk_cholesky_from_cov(cov: np.ndarray, lam: float) -> np.ndarray:
    """Shrink an ALREADY-COMPUTED covariance and return its Cholesky factor.

    Σ = (1-λ)·cov + λ·diag(cov); jitter ladder (0, 1e-6, 1e-4, 1e-2) for PD
    safety in high dim. Extracted from :func:`_shrunk_cholesky` (which keeps
    its exact behavior by delegating here) so callers that must accumulate the
    covariance CHUNKED in fp64 (e.g. an 88k-row train-answer covariance
    streamed off a memmap — issue #2202 P0.5) reuse the identical shrink +
    jitter logic instead of re-implementing it.
    """
    cov = np.asarray(cov, dtype=np.float64)
    diag = np.diag(np.diag(cov))
    shrunk = (1.0 - lam) * cov + lam * diag
    for jitter in (0.0, 1e-6, 1e-4, 1e-2):
        try:
            return np.linalg.cholesky(shrunk + jitter * np.eye(shrunk.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("shrunk covariance not PD even after jitter")


def _shrunk_cholesky(acts_2d: np.ndarray, lam: float) -> np.ndarray:
    """Cholesky factor of the shrunk covariance Σ = (1-λ)Σ_emp + λ·diag(Σ_emp).

    ``acts_2d`` is ``(n, D)`` at ONE layer. Returns the ``(D, D)`` lower-triangular
    Cholesky factor for sampling N(0, Σ). A tiny jitter is added if the factor is
    not PD after shrinkage (numerical safety in 3584-dim). Delegates the shrink +
    jitter logic to :func:`shrunk_cholesky_from_cov`.
    """
    acts_2d = np.asarray(acts_2d, dtype=np.float64)
    return shrunk_cholesky_from_cov(np.cov(acts_2d, rowvar=False), lam)


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
    predictor_acts = np.asarray(predictor_acts, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rb_norm_per_layer = np.asarray(rb_norm_per_layer, dtype=np.float64)
    L = predictor_acts.shape[1]
    rng = np.random.default_rng(seed)
    # Precompute Cholesky per layer (expensive; do once — rng-free, so the
    # stream position at the first draw is unchanged).
    chols: dict[int, np.ndarray] = {}
    for layer in range(L):
        chols[layer] = _shrunk_cholesky(pool_acts_per_layer[layer], lam)
    D = predictor_acts.shape[2]
    if n_draws == 0:
        return np.empty((0, L), dtype=np.float64)
    # RNG in the EXACT original loop order over the FULL draw range (plan v2
    # §3.1): one rng.standard_normal(D) per (draw, layer), draw-major /
    # layer-minor — bit-identical stream to the loop version.
    z_stack = np.empty((n_draws, L, D), dtype=np.float64)
    for d in range(n_draws):
        for layer in range(L):
            z_stack[d, layer] = rng.standard_normal(D)
    out = np.empty((n_draws, L), dtype=np.float64)
    n_pred = predictor_acts.shape[0]
    bytes_per_draw = 2 * L * D * 8 + 4 * n_pred * L * 8  # Z + dirs + proj + pearson temps (§3.4)
    for start, stop in _k_chunks(n_draws, bytes_per_draw):
        k = stop - start
        dirs = np.empty((k, L, D), dtype=np.float64)
        for layer in range(L):
            v = z_stack[start:stop, layer, :] @ chols[layer].T  # rows == chols[layer] @ z
            vn = np.linalg.norm(v, axis=1)  # (k,)
            # vn == 0 rows keep v unscaled (v is the zero vector) — matches the loop.
            scale = np.where(vn == 0, 1.0, rb_norm_per_layer[layer] / np.where(vn == 0, 1.0, vn))
            dirs[:, layer, :] = v * scale[:, None]
        if within:
            r = _batched_within_r(predictor_acts, dirs, target, condition_ids)  # (L, k)
        else:
            r = _batched_r_overall(predictor_acts, dirs, target)  # (L, k)
        out[start:stop] = np.abs(r).T
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
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    # RNG in the EXACT original loop order over the FULL draw range (plan v2
    # §3.1): one rng.integers(0, n, size=n) per bootstrap draw.
    idx = np.stack([rng.integers(0, n, size=n) for _ in range(n_boot)])  # (n_boot, n)
    if n < 3:
        boots = np.full(n_boot, np.nan, dtype=np.float64)
    else:
        x = proj[idx]  # (n_boot, n)
        y = target[idx]  # (n_boot, n)
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        denom = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = np.where(denom == 0, np.nan, (xc * yc).sum(axis=1) / denom)
    valid = boots[~np.isnan(boots)]
    if valid.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))


def bootstrap_ci_within_r(
    predictor_acts: np.ndarray,
    rb_per_layer: np.ndarray,
    target: np.ndarray,
    condition_ids: np.ndarray,
    selected_layer: int,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float]:
    """95% bootstrap CI on the WITHIN-CONDITION Fisher-z r at ``selected_layer``.

    Propagates the WITHIN estimator's sampling variance via a STRATIFIED row
    bootstrap: each draw resamples rows with replacement INSIDE each condition
    group (preserving group sizes), then recomputes the Fisher-z-weighted
    within-condition r at the fixed layer. ``bootstrap_ci_matched_r`` (pooled
    row resampling of the overall Pearson r) is NOT a valid CI for the within
    setting — the #778 within-CI bug (audit marker v70): ``compute_setting``
    called ``bootstrap_ci_matched_r`` without ``condition_ids`` on the within
    branch and reported the POOLED CI as the within CI. Groups with < 4 rows
    are dropped from a draw's average (mirroring
    ``within_condition_r_per_layer``); a draw with no usable group is NaN and
    excluded from the percentiles.
    """
    predictor_acts = np.asarray(predictor_acts, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    condition_ids = np.asarray(condition_ids)
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    proj = project(predictor_acts[:, selected_layer, :], rb_per_layer[selected_layer])  # (n,)
    groups = [np.where(condition_ids == c)[0] for c in np.unique(condition_ids)]
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        z_sum = 0.0
        w_sum = 0.0
        for g in groups:
            if g.size < 4:
                continue
            idx = g[rng.integers(0, g.size, size=g.size)]
            r = _pearson(proj[idx], target[idx])
            if np.isnan(r):
                continue
            z_sum += (g.size - 3) * np.arctanh(np.clip(r, -0.999999, 0.999999))
            w_sum += g.size - 3
        if w_sum > 0:
            boots[b] = np.tanh(z_sum / w_sum)
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
    if within:
        # The #778 within-CI fix (audit marker v70): the within setting's CI must
        # resample WITHIN condition blocks — never the pooled row bootstrap.
        ci = bootstrap_ci_within_r(
            predictor_acts,
            rb_per_layer,
            target,
            condition_ids,
            sel_layer,
            n_boot=n_boot,
            seed=seed,
        )
    else:
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
