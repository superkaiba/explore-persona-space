"""Issue #2221 monitor math — fully vectorized correlation / bootstrap / null battery.

Design contract (plan §6): NO per-draw Python loop. The 10k-draw paired
bootstrap and the 1k-draw null ladder are rank-transform-once + batched
matrix ops (the ``analysis/null_battery.py`` / ``vectorized_mlp_skill.py``
pattern), chunked along the draw axis to bound RAM.

Statistical conventions:
- Spearman r per (arm, trait, layer) across the 24 fine-tunes.
- Bootstrap draws resample the 24 fine-tunes WITH replacement; per draw the
  read-out position is RE-SELECTED over the arm's full position axis
  (layer x {prefix, context} = 56 positions for the mapped arm c) — the
  selection-symmetric convention. Ranks are computed ONCE on the original
  sample and Pearson-correlated on the resampled rank rows (the registered
  batched approximation; ties/re-ranking inside a draw are ignored).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from . import constants as C

# ── rank / correlation primitives ─────────────────────────────────────────────


def rank_transform(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average-rank transform along ``axis`` (Spearman's rank convention).

    Uses scipy's native vectorized ``axis=`` kwarg — never a per-row Python
    loop (the bootstrap battery ranks ~1e5 rows per chunk).
    """
    return rankdata(np.asarray(x, dtype=np.float64), method="average", axis=axis)


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson r between ``a (..., n)`` and ``b (..., n)`` (broadcast)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    num = (ac * bc).sum(axis=-1)
    den = np.sqrt((ac**2).sum(axis=-1) * (bc**2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.where(den > 0, r, np.nan)


def spearman_by_position(x_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman r per position: ``x_pos (P, n)`` vs ``y (n,)`` -> ``(P,)``."""
    xr = rank_transform(x_pos, axis=-1)
    yr = rank_transform(y[None, :], axis=-1)[0]
    return pearson_rows(xr, np.broadcast_to(yr, xr.shape))


def select_position(r_by_pos: np.ndarray) -> tuple[int, float]:
    """Predictivity selection: argmax |r| over positions -> (index, signed r)."""
    r = np.asarray(r_by_pos, dtype=np.float64)
    finite = np.where(np.isfinite(r), np.abs(r), -np.inf)
    idx = int(np.argmax(finite))
    return idx, float(r[idx])


# ── vectorized bootstrap ──────────────────────────────────────────────────────


def bootstrap_indices(rng: np.random.Generator, n: int, n_draws: int) -> np.ndarray:
    """(n_draws, n) resample-with-replacement index matrix."""
    return rng.integers(0, n, size=(n_draws, n))


def bootstrap_pearson(
    x_pos: np.ndarray, y: np.ndarray, idx: np.ndarray, *, chunk: int = 2000
) -> np.ndarray:
    """Batched EXACT bootstrap Spearman: gather, re-rank per draw, Pearson.

    For each draw row of ``idx (B, n)`` the resampled values of every
    position row of ``x_pos (P, n)`` and of ``y (n,)`` are RE-RANKED within
    the draw (exact Spearman under resampling ties), then row-wise Pearson
    is computed batched — no per-draw Python loop. Returns ``(B, P)``.
    """
    x = np.asarray(x_pos, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    n_draws = idx.shape[0]
    out = np.empty((n_draws, x.shape[0]), dtype=np.float64)
    for lo in range(0, n_draws, chunk):
        hi = min(lo + chunk, n_draws)
        sl = idx[lo:hi]  # (b, n)
        xg = rank_transform(x[:, sl], axis=-1)  # (P, b, n) re-ranked per draw
        yg = rank_transform(yv[sl], axis=-1)  # (b, n)
        r = pearson_rows(xg, yg[None, :, :])  # (P, b)
        out[lo:hi] = r.T
    return out


def bootstrap_selected(
    x_pos: np.ndarray, y: np.ndarray, idx: np.ndarray, *, chunk: int = 2000
) -> np.ndarray:
    """Per-draw SELECTED signed r: argmax |r| over the position axis, per draw.

    Selection is re-run INSIDE every bootstrap draw (selection-symmetric).
    Returns ``(B,)`` signed r at the per-draw argmax-|r| position.
    """
    r = bootstrap_pearson(x_pos, y, idx, chunk=chunk)  # (B, P)
    absr = np.where(np.isfinite(r), np.abs(r), -np.inf)
    sel = np.argmax(absr, axis=1)
    return r[np.arange(r.shape[0]), sel]


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided percentile CI over finite draws."""
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    return (
        float(np.percentile(d, 100 * alpha / 2)),
        float(np.percentile(d, 100 * (1 - alpha / 2))),
    )


# ── null ladder (the #778 round-3 honest ladder shapes) ───────────────────────


def isotropic_null_directions(
    rng: np.random.Generator, n_draws: int, n_layers: int, d: int
) -> np.ndarray:
    """(n_draws, n_layers, d) unit-norm isotropic directions (float32)."""
    z = rng.standard_normal(size=(n_draws, n_layers, d)).astype(np.float32)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return z


def covariance_null_directions(
    rng: np.random.Generator, n_draws: int, pool: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Covariance-matched null directions from an activation pool.

    ``pool (m, n_layers, d)`` (this run's base-model capture pool). Per layer,
    directions are sampled in the m-dimensional pool span:
    ``dir = z @ centered_pool / sqrt(m - 1)`` (z ~ N(0, I_m)), giving samples
    whose covariance matches the pool's per-layer covariance, then unit-
    normalized. Returns ``(dirs (n_draws, n_layers, d) float32,
    top_evec_cos (n_layers,))`` where the second output is the circularity
    diagnostic |cos| of the pool's top principal direction per layer against
    which callers gate (reported, never silently consumed).
    """
    pool = np.asarray(pool, dtype=np.float64)
    m, n_layers, d = pool.shape
    assert m >= 3, f"covariance null needs >= 3 pool rows, got {m}"
    centered = pool - pool.mean(axis=0, keepdims=True)  # (m, L, d)
    z = rng.standard_normal(size=(n_draws, m))  # shared across layers per draw
    # dirs[b, l, :] = sum_i z[b, i] * centered[i, l, :] / sqrt(m-1)
    dirs = np.einsum("bm,mld->bld", z, centered) / np.sqrt(m - 1)
    dirs = dirs.astype(np.float32)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs /= np.maximum(norms, 1e-12)
    # top principal direction per layer (for the caller's circularity gate)
    top = np.empty((n_layers, d), dtype=np.float64)
    for layer in range(n_layers):
        _, _, vt = np.linalg.svd(centered[:, layer, :], full_matrices=False)
        top[layer] = vt[0]
    return dirs, top


def null_selected_r(
    dirs: np.ndarray,
    shifts: np.ndarray,
    y: np.ndarray,
    *,
    chunk: int = 100,
) -> np.ndarray:
    """Null battery: per draw, replace r_B by a null direction and re-select.

    ``dirs (B, L, d)`` null directions, ``shifts (n, L, d)`` per-fine-tune
    shift summaries, ``y (n,)`` trait scores. Per draw the monitor scalar is
    ``E[b, f, l] = dirs[b, l] . shifts[f, l]`` (one einsum per chunk), the
    Spearman r is computed per layer, and the same argmax-|r| selection runs
    over the layer axis. Returns ``(B,)`` selected signed r per null draw.
    """
    shifts = np.asarray(shifts, dtype=np.float32)
    assert shifts.ndim == 3, shifts.shape  # (n_finetunes, n_layers, d)
    yr = rank_transform(np.asarray(y, float)[None, :], axis=-1)[0]
    out = np.empty(dirs.shape[0], dtype=np.float64)
    for lo in range(0, dirs.shape[0], chunk):
        hi = min(lo + chunk, dirs.shape[0])
        e = np.einsum("bld,fld->bfl", dirs[lo:hi].astype(np.float32), shifts)  # (b, n, L)
        e64 = np.asarray(e, dtype=np.float64).transpose(0, 2, 1)  # (b, L, n)
        er = rank_transform(e64, axis=-1)
        r = pearson_rows(er, np.broadcast_to(yr, er.shape))  # (b, L)
        absr = np.where(np.isfinite(r), np.abs(r), -np.inf)
        sel = np.argmax(absr, axis=1)
        out[lo:hi] = r[np.arange(r.shape[0]), sel]
    return out


def score_shuffle_selected_r(
    rng: np.random.Generator, x_pos: np.ndarray, y: np.ndarray, n_draws: int, *, chunk: int = 2000
) -> np.ndarray:
    """Score-shuffle null with the SAME per-draw selection over positions."""
    n = y.shape[0]
    perms = np.stack([rng.permutation(n) for _ in range(n_draws)])  # (B, n)
    xr = rank_transform(np.asarray(x_pos, float), axis=-1)  # (P, n)
    yr = rank_transform(np.asarray(y, float)[None, :], axis=-1)[0]
    out = np.empty(n_draws, dtype=np.float64)
    for lo in range(0, n_draws, chunk):
        hi = min(lo + chunk, n_draws)
        yg = yr[perms[lo:hi]]  # (b, n)
        r = pearson_rows(xr[:, None, :], yg[None, :, :])  # (P, b)
        absr = np.where(np.isfinite(r), np.abs(r), -np.inf)
        sel = np.argmax(absr, axis=0)
        out[lo:hi] = r[sel, np.arange(r.shape[1])]
    return out


# ── folds / AUC / ordering ────────────────────────────────────────────────────


def lofo_jackknife(x: np.ndarray, y: np.ndarray, groups: list[str]) -> dict[str, float]:
    """Leave-one-FAMILY-out Spearman r: drop each family's rows, recompute."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    garr = np.asarray(groups)
    out: dict[str, float] = {}
    for fam in sorted(set(groups)):
        keep = garr != fam
        if keep.sum() < 4:
            out[fam] = float("nan")
            continue
        out[fam] = float(spearman_by_position(x[keep][None, :], y[keep])[0])
    return out


def detection_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUC of ``scores`` for binary ``labels``."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def severity_ordering(values: dict[str, dict[str, float]]) -> dict[str, bool]:
    """Per family: does the scalar rank normal < misaligned_1 < misaligned_2?"""
    out: dict[str, bool] = {}
    for fam, by_version in values.items():
        try:
            out[fam] = bool(
                by_version["normal"] < by_version["misaligned_1"] < by_version["misaligned_2"]
            )
        except KeyError:
            out[fam] = False
    return out


# ── arm scalar computation ────────────────────────────────────────────────────


def arm_scalars_for_model(
    *,
    rb: np.ndarray,
    v_ctx_shift: np.ndarray | None,
    v_pfx_shift_states: tuple[np.ndarray, np.ndarray] | None,
    v_ctx_states: tuple[np.ndarray, np.ndarray] | None,
    v_ans_shift: np.ndarray | None,
    map_ctx: dict | None,
    map_pfx: dict | None,
) -> dict[str, np.ndarray]:
    """Per-layer monitor scalars for ONE fine-tune against ONE trait direction.

    Args:
        rb: (28, 3584) trait direction.
        v_ctx_shift: (28, 3584) mean last-prompt-token shift (final - base).
        v_pfx_shift_states: (v_pfx_final, v_pfx_base) raw prefix-end states
            (each (28, 3584)) — mapped arm c_pfx needs the STATES (difference
            of mapped states), never the raw shift.
        v_ctx_states: (v_ctx_final, v_ctx_base) raw context states for c_ctx.
        v_ans_shift: (28, 3584) mean response-avg shift (final - base).
        map_ctx / map_pfx: loaded affine maps (or None to skip mapped arms).

    Returns {arm: (28,) per-layer scalar} for every computable arm.
    """
    from .loaders import apply_map_shift

    out: dict[str, np.ndarray] = {}
    ln = C.N_LAYERS
    if v_ctx_shift is not None:
        assert v_ctx_shift.shape == (ln, C.HIDDEN_DIM), v_ctx_shift.shape
        out["a_rb_ctx"] = np.einsum("ld,ld->l", rb, v_ctx_shift)
    if v_ans_shift is not None:
        assert v_ans_shift.shape == (ln, C.HIDDEN_DIM), v_ans_shift.shape
        out["b_rb_ans"] = np.einsum("ld,ld->l", rb, v_ans_shift)
    if map_ctx is not None and v_ctx_states is not None:
        vf, vb = v_ctx_states
        mapped = np.stack(
            [apply_map_shift(map_ctx, vf[layer], vb[layer], layer) for layer in range(ln)]
        )
        out["c_map_ctx"] = np.einsum("ld,ld->l", rb, mapped)
        if v_ans_shift is not None:
            num = np.einsum("ld,ld->l", mapped, v_ans_shift)
            den = np.linalg.norm(mapped, axis=1) * np.linalg.norm(v_ans_shift, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["d_transport"] = np.where(den > 0, num / den, np.nan)
    if map_pfx is not None and v_pfx_shift_states is not None:
        vf, vb = v_pfx_shift_states
        mapped = np.stack(
            [apply_map_shift(map_pfx, vf[layer], vb[layer], layer) for layer in range(ln)]
        )
        out["c_map_pfx"] = np.einsum("ld,ld->l", rb, mapped)
    return out
