"""Issue #537 combiner-track scoring (design doc §5 "Combiner track", 2026-06-11).

Scores multi-predictor COMBINERS over the per-cell scalars the shipped
single-metric leaderboard (``scripts/i537_score_metric.py``) already
computes, against the same G ground truth, the same leave-two-contexts-out
(LTCO) CV folds, and the same quarantine masking. Zero GPU: every feature is
derived from artifacts already on disk (clouds, first-token caches,
marker_base_slots, P0 headroom rates).

Combiner forms (registered spec, design doc §5 at commit 0cc4053fc):

1. **Regularized linear stacker** (ridge; alpha chosen INSIDE each fold via
   inner leave-one-context-out) over the shipped baseline scalars plus the
   NEW source-side prior logP(behavior | c_train) feature (row-indexed base
   rates -- the write-magnitude proxy). PV projection (``pv_dp``) and the A5
   sequence-level output-KL rows are registered-not-implemented (GPU passes)
   and are SKIPPED, not proxied; the shipped first-token KL fwd/rev rows
   stand in as the only shipped output divergences.
2. **Theory-shaped write x gate form**: G_hat[i,j] = (a . x_i) * (b . y_j)
   with row features x_i (write proxies: source-side prior, centered row
   norm) and column features y_j (gate proxies: bystander prior, centered
   column norm, cosine-to-neutral column effect), fit by alternating least
   squares INSIDE folds. Free per-context factors are unfittable for the two
   held-out contexts under LTCO, so the factors are feature-parameterized --
   the rank-1 bilinear form named by the design doc (the referenced
   ``docs/notes/rank1_leakage_model.pdf`` is not in the tree; fallback per
   the registered spec).
3. **Per-behavior z-normalized pooled variants** of (1)-(2) (z within fold,
   train-cell statistics only).
4. **Combination ladder**: centroid cosine -> +norm ratio -> +whitened
   projection -> +bystander prior, each rung scored.

Protocol invariants:

- Folds + masks are the harness's: same quarantine manifest, never the
  final-test split (this script has NO ``--final-test``). Two mask variants
  are scored per row: ``quarantine_only`` (EXACTLY the shipped leaderboard
  protocol -- the marker leaderboard's n=193 includes the 13 implant-failed
  cells, verified against ``baseline_scores.json``) and ``strict``
  (additionally drops G_meta ``implant_failed`` + ``saturated`` cells, the
  registered §5 exclusion). Single predictors are RE-SCORED under each mask
  so ``delta_r2_vs_best_single`` is always mask-consistent.
- Sanity gates: the harness-path centroid_cosine + rbf_mmd2 marker rows must
  reproduce ``baseline_scores.json`` (atol 1e-6) or the run aborts, and the
  marker ladder rung-1 cosine row must land on the stored leaderboard
  numbers (raw spearman exact, oof R² within 0.01 of the OLS path).
- Antisymmetric-component R²: out-of-fold prediction matrix P decomposed as
  A_P = (P - P^T)/2 vs A_G = (G - G^T)/2 over cells with both orientations
  usable (#502/#524 decomposition applied to OOF predictions).
- Checkpoint per behavior: the output JSON is rewritten after each behavior
  block (no accumulate-and-write-at-end).

Output: ``eval_results/issue_537/analysis/combiner_scores.json``.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import i537_score_metric as harness  # noqa: E402  (shipped P3 harness; folds/masks of record)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_score_combiners")

EVAL = harness.EVAL
BEHAVIORS = ["marker", "fact", "refusal", "sycophancy", "em"]
ALPHA_GRID = (1e-6, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
ALS_RIDGE = 1e-3  # fixed small ridge on the standardized bilinear sides (stability)
MIN_TRAIN_CELLS = 8  # harness ltco_cv_predictions guard, mirrored

# Cache cloud slices: each metric_matrix call reloads 16 clouds (~1.1 s each);
# the combiner run calls it ~10x per behavior. Read-only use (callers never
# mutate in place -- verified: _drop_nan_rows and the centering ops allocate).
harness._load_cloud = functools.lru_cache(maxsize=64)(harness._load_cloud)

# Stacker feature set (design doc §5 form 1). pv_dp + A5 seq-level output KL
# are registered-not-implemented (GPU) -> deliberately absent.
STACKER_FEATURES = [
    "base_prior_bystander",
    "source_prior",  # NEW: row-indexed base rates (write-magnitude proxy)
    "gauss_kl_act",
    "kl_first_token_fwd",
    "kl_first_token_rev",
    "content_free",
    "rbf_mmd2",
    "bures_w2",
    "euclidean",
    "centroid_cosine",
]
LADDER = [
    ("ladder_1_cosine", ["centroid_cosine"]),
    ("ladder_2_plus_norm_ratio", ["centroid_cosine", "norm_ratio"]),
    ("ladder_3_plus_whitened_proj", ["centroid_cosine", "norm_ratio", "rank1_proj_whitened"]),
    (
        "ladder_4_plus_bystander_prior",
        ["centroid_cosine", "norm_ratio", "rank1_proj_whitened", "base_prior_bystander"],
    ),
]
ALL_FEATURE_IDS = sorted({*STACKER_FEATURES, "norm_ratio", "rank1_proj_whitened"})


# ── Feature construction ─────────────────────────────────────────────────────


def source_prior_matrix(behavior: str, cids: list[str]) -> np.ndarray:
    """Row-effect matrix of the source-side prior logP(behavior | c_train).

    Reuses the harness base-rate loader (marker: mean base logP(marker) at the
    slot from marker_base_slots; judge rows: P0 headroom rates), indexed by the
    TRAIN context i instead of the eval context j. Distance polarity to match
    the harness convention: -base[c_i] (a higher source-side prior predicts
    MORE transfer -> LESS distant).
    """
    base = harness._base_rates_for(behavior, cids)
    n = len(cids)
    d = np.full((n, n), np.nan)
    for i, ci in enumerate(cids):
        for j in range(n):
            if i != j:
                d[i, j] = -base[ci]
    return d


def feature_matrices(behavior: str, cids: list[str]) -> dict[str, np.ndarray]:
    """All combiner features as (16, 16) matrices in harness distance polarity."""
    feats: dict[str, np.ndarray] = {}
    for fid in ALL_FEATURE_IDS:
        if fid == "source_prior":
            feats[fid] = source_prior_matrix(behavior, cids)
        else:
            feats[fid] = harness.metric_matrix(fid, cids, behavior=behavior)
        assert feats[fid].shape == (len(cids), len(cids)), (fid, feats[fid].shape)
    return feats


def load_or_compute_features(
    behavior: str, cids: list[str], cache_dir: Path
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Disk-cached feature matrices + write x gate tables (chunked invocations).

    The combiner run is chunked into <=10-min foreground invocations; feature
    computation (4 PCA-SVD metrics + RBF-MMD over 120 pairs + full-vocab
    first-token loads) dominates wall time, so it is cached per behavior. The
    cache is keyed to the git HEAD commit -- a commit mismatch recomputes.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{behavior}.npz"
    commit = harness._git_commit()
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if str(z["git_commit"]) == commit:
            feats = {fid: z[f"feat__{fid}"] for fid in ALL_FEATURE_IDS}
            logger.info("[cache] loaded features for %s from %s", behavior, p)
            return feats, (z["wg_x"], z["wg_y"])
        logger.warning("[cache] %s stale (commit mismatch) -- recomputing", p)
    feats = feature_matrices(behavior, cids)
    wg = writegate_context_features(behavior, cids)
    np.savez(
        p,
        git_commit=np.array(commit),
        wg_x=wg[0],
        wg_y=wg[1],
        **{f"feat__{fid}": m for fid, m in feats.items()},
    )
    logger.info("[cache] saved features for %s to %s", behavior, p)
    return feats, wg


def _std_sides(t: np.ndarray) -> np.ndarray:
    """Standardize non-constant columns of a per-context side-feature table."""
    t = t.copy()
    for k in range(1, t.shape[1]):
        sd = t[:, k].std()
        t[:, k] = (t[:, k] - t[:, k].mean()) / (sd if sd > 1e-12 else 1.0)
    return t


def writegate_context_features(behavior: str, cids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-context (row, col) feature tables for the write x gate form.

    Rows (write proxies): [1, source-side prior, centered mean-shift norm].
    Cols (gate proxies): [1, bystander prior, centered mean-shift norm,
    cosine-to-neutral column effect]. Priors enter in RAW polarity here (signs
    are absorbed by the fitted weights); norms use the grand-mean-centered
    context means at the harness primary anchor/layer. Returned tables are
    already standardized (non-constant columns).
    """
    base = harness._base_rates_for(behavior, cids)
    clouds = {
        c: harness._drop_nan_rows(
            harness._load_cloud(c, harness.PRIMARY_ANCHOR, harness.PRIMARY_LAYER)
        )
        for c in cids
    }
    grand = np.mean([clouds[c].mean(axis=0) for c in cids], axis=0)
    norms = {c: float(np.linalg.norm(clouds[c].mean(axis=0) - grand)) for c in cids}
    cos_neutral = harness.metric_matrix("cos_to_neutral", cids, behavior=behavior)
    # column effect: identical down each column off-diagonal
    cos_col = np.nanmean(cos_neutral, axis=0)
    x_rows = np.array([[1.0, base[c], norms[c]] for c in cids])
    y_cols = np.array([[1.0, base[c], norms[c], cos_col[k]] for k, c in enumerate(cids)])
    assert np.all(np.isfinite(x_rows)) and np.all(np.isfinite(y_cols))
    return _std_sides(x_rows), _std_sides(y_cols)


# ── Masks ────────────────────────────────────────────────────────────────────


def usable_masks(behavior: str, cids: list[str]) -> dict[str, np.ndarray]:
    """quarantine_only (leaderboard protocol) + strict (G_meta flags) masks."""
    n = len(cids)
    qmask = harness.quarantine_mask(
        behavior, cids, cids, final_test=False, invocation_note="combiner-track"
    )
    z = np.load(EVAL / "G_tensor/G_tensor.npz", allow_pickle=True)
    behaviors = list(z["behaviors"])
    bi = behaviors.index(behavior)
    at = list(z["train_cids"][bi])
    ae = list(z["eval_cids"][bi])
    flagged = np.zeros((n, n), dtype=bool)
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            cell = (bi, at.index(ci), ae.index(cj), 0)
            flagged[i, j] = bool(z["implant_failed"][cell]) or bool(z["saturated"][cell])
    offdiag = ~np.eye(n, dtype=bool)
    return {
        "quarantine_only": qmask & offdiag,
        "strict": qmask & offdiag & ~flagged,
    }


# ── Fold machinery (LTCO, mirroring harness.ltco_cv_predictions) ─────────────


def _cells(idx: list[int], mask: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i in idx for j in idx if i != j and mask[i, j]]


def _design(feats: dict[str, np.ndarray], fids: list[str], cells) -> np.ndarray:
    x = np.array([[feats[f][i, j] for f in fids] for (i, j) in cells])
    assert np.all(np.isfinite(x)), "non-finite feature value in design matrix"
    return x


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    """Ridge on standardized features (intercept via centering, unpenalized)."""
    mu_x, sd_x = x.mean(axis=0), x.std(axis=0)
    sd_x = np.where(sd_x < 1e-12, 1.0, sd_x)
    xs = (x - mu_x) / sd_x
    mu_y = y.mean()
    w = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ (y - mu_y))
    return w, {"mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y}


def _ridge_predict(x: np.ndarray, w: np.ndarray, st: dict) -> np.ndarray:
    return ((x - st["mu_x"]) / st["sd_x"]) @ w + st["mu_y"]


def _fit_stacker(train_cells, feats, fids, g, keep_idx):
    """Inner leave-one-context-out alpha selection, then refit on all train cells."""
    x_tr = _design(feats, fids, train_cells)
    y_tr = np.array([g[i, j] for (i, j) in train_cells])
    if len(fids) == 1 or len(keep_idx) < 4:
        best_alpha = ALPHA_GRID[0]
    else:
        sse = dict.fromkeys(ALPHA_GRID, 0.0)
        cnt = dict.fromkeys(ALPHA_GRID, 0)
        for c in keep_idx:
            in_cells = [(i, j) for (i, j) in train_cells if i != c and j != c]
            val_cells = [(i, j) for (i, j) in train_cells if i == c or j == c]
            if len(in_cells) < MIN_TRAIN_CELLS or not val_cells:
                continue
            x_in = _design(feats, fids, in_cells)
            y_in = np.array([g[i, j] for (i, j) in in_cells])
            x_val = _design(feats, fids, val_cells)
            y_val = np.array([g[i, j] for (i, j) in val_cells])
            for a in ALPHA_GRID:
                w, st = _ridge_fit(x_in, y_in, a)
                sse[a] += float(((y_val - _ridge_predict(x_val, w, st)) ** 2).sum())
                cnt[a] += len(val_cells)
        usable = [a for a in ALPHA_GRID if cnt[a] > 0]
        best_alpha = min(usable, key=lambda a: sse[a] / cnt[a]) if usable else ALPHA_GRID[0]
    w, st = _ridge_fit(x_tr, y_tr, best_alpha)
    return w, st, best_alpha


def oof_stacker(
    feats: dict[str, np.ndarray],
    fids: list[str],
    g: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Pooled out-of-fold ridge predictions over LTCO folds -> (P matrix, n_folds)."""
    n = g.shape[0]
    pred = np.full((n, n), np.nan)
    n_folds = 0
    for a, b in itertools.combinations(range(n), 2):
        keep = [i for i in range(n) if i not in (a, b)]
        train_cells = _cells(keep, mask)
        held = [(i, j) for (i, j) in ((a, b), (b, a)) if mask[i, j] and np.isfinite(g[i, j])]
        if len(train_cells) < MIN_TRAIN_CELLS or not held:
            continue
        w, st, _alpha = _fit_stacker(train_cells, feats, fids, g, keep)
        x_h = _design(feats, fids, held)
        for (i, j), p in zip(held, _ridge_predict(x_h, w, st), strict=True):
            pred[i, j] = p
        n_folds += 1
    return pred, n_folds


def _als_fit(x_rows, y_cols, train_cells, g, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """ALS for G_hat[i,j] = (x_i . a) * (y_j . b) on standardized side features."""
    y_t = np.array([g[i, j] for (i, j) in train_cells])
    rows = np.array([x_rows[i] for (i, _j) in train_cells])
    cols = np.array([y_cols[j] for (_i, j) in train_cells])
    # init: b s.t. y_j.b approximates column means of fold-train G
    col_mean: dict[int, list[float]] = {}
    for _i, j in train_cells:
        col_mean.setdefault(j, []).append(g[_i, j])
    cj = sorted(col_mean)
    yc = np.array([y_cols[j] for j in cj])
    gv = np.array([np.mean(col_mean[j]) for j in cj])
    b = np.linalg.solve(yc.T @ yc + ALS_RIDGE * np.eye(yc.shape[1]), yc.T @ gv)
    a = None
    prev = np.inf
    for _ in range(iters):
        s = cols @ b  # (n_cells,)
        xa = rows * s[:, None]
        a = np.linalg.solve(xa.T @ xa + ALS_RIDGE * np.eye(xa.shape[1]), xa.T @ y_t)
        t = rows @ a
        yb = cols * t[:, None]
        b = np.linalg.solve(yb.T @ yb + ALS_RIDGE * np.eye(yb.shape[1]), yb.T @ y_t)
        resid = (rows @ a) * (cols @ b) - y_t
        sse = float(resid @ resid)
        if abs(prev - sse) < 1e-12 * max(prev, 1.0):
            break
        prev = sse
    return a, b


def oof_writegate(
    x_rows: np.ndarray, y_cols: np.ndarray, g: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, int]:
    """Pooled out-of-fold write x gate ALS predictions over LTCO folds.

    ``x_rows`` / ``y_cols`` arrive standardized from writegate_context_features.
    """
    n = g.shape[0]
    pred = np.full((n, n), np.nan)
    n_folds = 0
    for a, b in itertools.combinations(range(n), 2):
        keep = [i for i in range(n) if i not in (a, b)]
        train_cells = _cells(keep, mask)
        held = [(i, j) for (i, j) in ((a, b), (b, a)) if mask[i, j] and np.isfinite(g[i, j])]
        if len(train_cells) < MIN_TRAIN_CELLS or not held:
            continue
        wa, wb = _als_fit(x_rows, y_cols, train_cells, g)
        for i, j in held:
            pred[i, j] = float((x_rows[i] @ wa) * (y_cols[j] @ wb))
        n_folds += 1
    return pred, n_folds


# ── Scoring ──────────────────────────────────────────────────────────────────


def score_oof(pred: np.ndarray, g: np.ndarray, mask: np.ndarray) -> dict:
    """R² + rank corr of pooled OOF predictions, + antisymmetric-component R²."""
    m = np.isfinite(pred) & np.isfinite(g) & mask
    y_true, y_pred = g[m], pred[m]
    assert y_true.size >= 10, f"too few scored cells ({y_true.size})"
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    out = {
        "oof_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rank_corr": float(spearmanr(y_pred, y_true).statistic),
        "n_cells": int(y_true.size),
    }
    out.update(_antisym_r2(pred, g, m))
    return out


def _antisym_r2(pred: np.ndarray, g: np.ndarray, m: np.ndarray) -> dict:
    """Antisymmetric-component R² over cells with BOTH orientations predicted."""
    both = m & m.T
    if both.sum() < 10:
        return {"antisym_r2": float("nan"), "antisym_n_cells": int(both.sum())}
    a_g = (0.5 * (g - g.T))[both]
    a_p = (0.5 * (pred - pred.T))[both]
    ss_res = float(((a_g - a_p) ** 2).sum())
    ss_tot = float(((a_g - a_g.mean()) ** 2).sum())
    return {
        "antisym_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "antisym_n_cells": int(both.sum()),
    }


# ── Pooled z-normalized variants ─────────────────────────────────────────────


def _fold_z_stats(per_behavior: dict, keep: list[int]) -> dict:
    """Per-behavior fold-train cells + z-norm stats (fold-train cells only)."""
    fold_stats = {}
    for beh, (_feats, g, mask) in per_behavior.items():
        tc = _cells(keep, mask)
        if len(tc) < MIN_TRAIN_CELLS:
            continue
        yv = np.array([g[i, j] for (i, j) in tc])
        sd = yv.std()
        fold_stats[beh] = (yv.mean(), sd if sd > 1e-12 else 1.0, tc)
    return fold_stats


def _pooled_xy(per_behavior, fold_stats, fids, cells_by_beh):
    """Pooled z-normed (X, y) over the given per-behavior cell lists."""
    xs, ys = [], []
    for beh, cells in cells_by_beh.items():
        if not cells:
            continue
        feats, g, _mask = per_behavior[beh]
        mu, sd, _tc = fold_stats[beh]
        xs.append(_design(feats, fids, cells))
        ys.append((np.array([g[i, j] for (i, j) in cells]) - mu) / sd)
    if not xs:
        return None, None
    return np.vstack(xs), np.concatenate(ys)


def _pooled_stacker_fit(per_behavior, fold_stats, fids, keep):
    """ONE ridge on the pooled z-normed fold-train cells; inner-LOO alpha."""
    train_by_beh = {beh: tc for beh, (_mu, _sd, tc) in fold_stats.items()}
    x_tr, y_tr = _pooled_xy(per_behavior, fold_stats, fids, train_by_beh)
    if len(fids) == 1:  # single feature: OLS-equivalent, no alpha search (mirrors _fit_stacker)
        return _ridge_fit(x_tr, y_tr, ALPHA_GRID[0])
    sse = dict.fromkeys(ALPHA_GRID, 0.0)
    cnt = dict.fromkeys(ALPHA_GRID, 0)
    for c in keep:
        in_by = {b: [(i, j) for (i, j) in tc if c not in (i, j)] for b, tc in train_by_beh.items()}
        va_by = {b: [(i, j) for (i, j) in tc if c in (i, j)] for b, tc in train_by_beh.items()}
        xi, yi = _pooled_xy(per_behavior, fold_stats, fids, in_by)
        xv, yv = _pooled_xy(per_behavior, fold_stats, fids, va_by)
        if xi is None or xv is None or len(yi) < MIN_TRAIN_CELLS:
            continue
        for al in ALPHA_GRID:
            w, st = _ridge_fit(xi, yi, al)
            sse[al] += float(((yv - _ridge_predict(xv, w, st)) ** 2).sum())
            cnt[al] += len(yv)
    usable = [al for al in ALPHA_GRID if cnt[al] > 0]
    best_alpha = min(usable, key=lambda al: sse[al] / cnt[al]) if usable else ALPHA_GRID[0]
    return _ridge_fit(x_tr, y_tr, best_alpha)


def _pooled_score(preds: dict, gz_full: dict) -> dict:
    """Pool z-scale OOF cells across behaviors; overall + antisym R²."""
    yt, yp, at_, ap_ = [], [], [], []
    for beh in preds:
        pm = np.isfinite(preds[beh]) & np.isfinite(gz_full[beh])
        yt.append(gz_full[beh][pm])
        yp.append(preds[beh][pm])
        both = pm & pm.T
        at_.append((0.5 * (gz_full[beh] - gz_full[beh].T))[both])
        ap_.append((0.5 * (preds[beh] - preds[beh].T))[both])
    y_true, y_pred = np.concatenate(yt), np.concatenate(yp)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    ag, ap = np.concatenate(at_), np.concatenate(ap_)
    ss_res_a = float(((ag - ap) ** 2).sum())
    ss_tot_a = float(((ag - ag.mean()) ** 2).sum())
    return {
        "oof_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rank_corr": float(spearmanr(y_pred, y_true).statistic),
        "n_cells": int(y_true.size),
        "antisym_r2": 1.0 - ss_res_a / ss_tot_a if ss_tot_a > 0 else float("nan"),
        "antisym_n_cells": int(ag.size),
    }


def pooled_zscore_cells(
    per_behavior: dict[str, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]],
    fids: list[str] | None,
    form: str,
    wg_feats: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    """Pooled z-normalized combiner: per fold, z-norm G per behavior on the
    fold-train cells, fit on the pooled cells (one ridge for the stacker;
    per-behavior ALS for write x gate), predict each behavior's held-out
    cells in z-space; score pooled.

    Fold index k = unordered context-position pair; positions 0-14 are the
    shared row-independent contexts across behaviors, position 15 is each
    behavior's own behavior-instruction cell.
    """
    n = 16
    preds = {b: np.full((n, n), np.nan) for b in per_behavior}
    gz_full = {b: np.full((n, n), np.nan) for b in per_behavior}
    n_folds = 0
    for a, b_idx in itertools.combinations(range(n), 2):
        keep = [i for i in range(n) if i not in (a, b_idx)]
        fold_stats = _fold_z_stats(per_behavior, keep)
        if not fold_stats:
            continue
        if form == "stacker":
            w, st = _pooled_stacker_fit(per_behavior, fold_stats, fids, keep)
        for beh, (feats, g, mask) in per_behavior.items():
            if beh not in fold_stats:
                continue
            mu, sd, tc = fold_stats[beh]
            held = [
                (i, j) for (i, j) in ((a, b_idx), (b_idx, a)) if mask[i, j] and np.isfinite(g[i, j])
            ]
            if not held:
                continue
            if form == "stacker":
                ph = _ridge_predict(_design(feats, fids, held), w, st)
            else:  # write x gate: per-behavior ALS on z-normed fold-train G
                xs_b, ys_b = wg_feats[beh]
                wa, wb = _als_fit(xs_b, ys_b, tc, (g - mu) / sd)
                ph = [float((xs_b[i] @ wa) * (ys_b[j] @ wb)) for (i, j) in held]
            for (i, j), p in zip(held, ph, strict=True):
                preds[beh][i, j] = p
                gz_full[beh][i, j] = (g[i, j] - mu) / sd
        n_folds += 1
    out = _pooled_score(preds, gz_full)
    out["folds"] = n_folds
    return out


# ── Sanity gate ──────────────────────────────────────────────────────────────


def sanity_reproduce_leaderboard() -> None:
    """Harness-path marker rows must reproduce baseline_scores.json exactly."""
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    stored = json.loads((EVAL / "baselines/baseline_scores.json").read_text())["scores"]
    cids = train_cids_for("marker")
    g = harness._load_g("marker", cids, cids)
    qm = harness.quarantine_mask(
        "marker", cids, cids, final_test=False, invocation_note="combiner-sanity"
    )
    g = np.where(qm, g, np.nan)
    for mid in ("centroid_cosine", "rbf_mmd2"):
        d = harness.metric_matrix(mid, cids, behavior="marker")
        res = harness.score_metric_vs_g(d, g)
        ref = stored[f"marker:{mid}"]
        for k in ("spearman", "oof_r2"):
            assert abs(res[k] - ref[k]) < 1e-6, (
                f"sanity FAIL: {mid}.{k} recomputed {res[k]:.6f} != stored {ref[k]:.6f} "
                "-- folds/masks diverge from the shipped leaderboard"
            )
        assert res["n_cells"] == ref["n_cells"], (mid, res["n_cells"], ref["n_cells"])
        logger.info(
            "[sanity] %s reproduces leaderboard (rho=%.3f oof_R²=%.3f n=%d)",
            mid,
            res["spearman"],
            res["oof_r2"],
            res["n_cells"],
        )


def sanity_ladder_rung1(rows: list[dict]) -> None:
    """Marker ladder rung-1 (cosine) must land on the stored leaderboard row."""
    stored = json.loads((EVAL / "baselines/baseline_scores.json").read_text())["scores"]
    ref = stored["marker:centroid_cosine"]
    row = next(
        r
        for r in rows
        if r["combiner"] == "ladder_1_cosine"
        and r["behavior"] == "marker"
        and r["mask"] == "quarantine_only"
    )
    assert abs(row["raw_spearman"] - ref["spearman"]) < 1e-6, (row, ref)
    assert abs(row["oof_r2"] - ref["oof_r2"]) < 0.01, (
        f"ladder rung-1 cosine oof_R² {row['oof_r2']:.4f} diverges from stored "
        f"{ref['oof_r2']:.4f} beyond the ridge-vs-OLS tolerance -- folds/masks differ"
    )
    logger.info(
        "[sanity] ladder rung-1 cosine lands on leaderboard (oof_R²=%.3f vs %.3f, raw rho=%.3f)",
        row["oof_r2"],
        ref["oof_r2"],
        row["raw_spearman"],
    )


# ── Row assembly + output ────────────────────────────────────────────────────


def make_row(combiner, beh, mask_name, scored, n_folds, best, notes=""):
    """One output row; ΔR² fields are mask-consistent vs the in-run best single."""
    d = {
        "combiner": combiner,
        "behavior": beh,
        "mask": mask_name,
        "oof_r2": scored["oof_r2"],
        "rank_corr": scored["rank_corr"],
        "antisym_r2": scored["antisym_r2"],
        "n_cells": scored["n_cells"],
        "antisym_n_cells": scored["antisym_n_cells"],
        "folds": n_folds,
        "notes": notes,
    }
    if best is not None:
        d["best_single"] = {"id": best["id"], "oof_r2": best["oof_r2"]}
        d["delta_r2_vs_best_single"] = scored["oof_r2"] - best["oof_r2"]
        d["antisym_delta_r2"] = scored["antisym_r2"] - best["antisym_r2"]
    if "raw_spearman" in scored:
        d["raw_spearman"] = scored["raw_spearman"]
    return d


def _jsonsafe(obj):
    """Map non-finite floats to None (strict-JSON downstream parsers)."""
    if isinstance(obj, dict):
        return {k: _jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonsafe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def score_behavior_block(
    beh: str,
    mask_name: str,
    feats: dict[str, np.ndarray],
    g: np.ndarray,
    mask: np.ndarray,
    wg: tuple[np.ndarray, np.ndarray],
) -> tuple[list[dict], dict]:
    """All per-behavior rows for one mask: singles, ladder, stacker, write x gate."""
    gm = np.where(mask, g, np.nan)
    rows: list[dict] = []
    # 1. single-predictor reference rows (mask-consistent best-single source)
    singles: dict[str, dict] = {}
    for fid in ALL_FEATURE_IDS:
        pred, n_folds = oof_stacker(feats, [fid], gm, mask)
        sc = score_oof(pred, gm, mask)
        m = mask & np.isfinite(feats[fid]) & np.isfinite(gm)
        sc["raw_spearman"] = float(spearmanr(feats[fid][m], gm[m]).statistic)
        singles[fid] = sc
        rows.append(
            make_row(
                f"single:{fid}",
                beh,
                mask_name,
                sc,
                n_folds,
                None,
                notes="single-predictor reference row (per-fold OLS-equivalent ridge)",
            )
        )
    best_id = max(singles, key=lambda f: singles[f]["oof_r2"])
    best = {"id": best_id, **singles[best_id]}
    logger.info("[%s/%s] best single: %s oof_R²=%.3f", beh, mask_name, best_id, best["oof_r2"])
    # 2. ladder rungs
    for name, fids in LADDER:
        pred, n_folds = oof_stacker(feats, fids, gm, mask)
        sc = score_oof(pred, gm, mask)
        if len(fids) == 1:
            m = mask & np.isfinite(feats[fids[0]]) & np.isfinite(gm)
            sc["raw_spearman"] = float(spearmanr(feats[fids[0]][m], gm[m]).statistic)
        rows.append(make_row(name, beh, mask_name, sc, n_folds, best))
        logger.info(
            "[%s/%s] %s: oof_R²=%.3f ΔR²=%.3f",
            beh,
            mask_name,
            name,
            sc["oof_r2"],
            sc["oof_r2"] - best["oof_r2"],
        )
    # 3. full ridge stacker
    pred, n_folds = oof_stacker(feats, STACKER_FEATURES, gm, mask)
    sc = score_oof(pred, gm, mask)
    rows.append(
        make_row(
            "stacker_ridge_full",
            beh,
            mask_name,
            sc,
            n_folds,
            best,
            notes=f"features={STACKER_FEATURES}",
        )
    )
    logger.info(
        "[%s/%s] stacker_ridge_full: oof_R²=%.3f ΔR²=%.3f antisym_ΔR²=%.3f",
        beh,
        mask_name,
        sc["oof_r2"],
        sc["oof_r2"] - best["oof_r2"],
        sc["antisym_r2"] - best["antisym_r2"],
    )
    # 4. write x gate
    pred, n_folds = oof_writegate(wg[0], wg[1], gm, mask)
    sc = score_oof(pred, gm, mask)
    rows.append(
        make_row(
            "write_gate_rank1",
            beh,
            mask_name,
            sc,
            n_folds,
            best,
            notes="bilinear (a.x_i)(b.y_j); rows=[1,source_prior,norm], "
            f"cols=[1,bystander_prior,norm,cos_to_neutral]; ALS, fixed ridge {ALS_RIDGE}",
        )
    )
    logger.info(
        "[%s/%s] write_gate_rank1: oof_R²=%.3f ΔR²=%.3f",
        beh,
        mask_name,
        sc["oof_r2"],
        sc["oof_r2"] - best["oof_r2"],
    )
    return rows, singles


def score_pooled_block(
    mask_name: str,
    behaviors: list[str],
    cache: dict,
) -> list[dict]:
    """Pooled z-normalized rows (form 3) for one mask variant."""
    per_behavior, wg_feats = {}, {}
    for beh in behaviors:
        feats, g, masks, wg = cache[beh]
        per_behavior[beh] = (feats, np.where(masks[mask_name], g, np.nan), masks[mask_name])
        wg_feats[beh] = wg
    partial = "" if len(behaviors) == len(BEHAVIORS) else f" PARTIAL POOL over {behaviors}"
    rows: list[dict] = []
    pooled_singles = {}
    for fid in ALL_FEATURE_IDS:
        sc = pooled_zscore_cells(per_behavior, [fid], "stacker")
        pooled_singles[fid] = sc
        rows.append(
            make_row(
                f"single:{fid}",
                "pooled",
                mask_name,
                sc,
                sc.pop("folds"),
                None,
                notes="pooled z-normed single reference" + partial,
            )
        )
    pbest_id = max(pooled_singles, key=lambda f: pooled_singles[f]["oof_r2"])
    pbest = {"id": pbest_id, **pooled_singles[pbest_id]}
    sc = pooled_zscore_cells(per_behavior, STACKER_FEATURES, "stacker")
    rows.append(
        make_row(
            "stacker_ridge_full_pooled_z",
            "pooled",
            mask_name,
            sc,
            sc.pop("folds"),
            pbest,
            notes="per-behavior z-norm inside folds" + partial,
        )
    )
    logger.info(
        "[pooled/%s] stacker: oof_R²=%.3f ΔR²=%.3f",
        mask_name,
        sc["oof_r2"],
        sc["oof_r2"] - pbest["oof_r2"],
    )
    sc = pooled_zscore_cells(per_behavior, None, "writegate", wg_feats=wg_feats)
    rows.append(
        make_row(
            "write_gate_rank1_pooled_z",
            "pooled",
            mask_name,
            sc,
            sc.pop("folds"),
            pbest,
            notes="per-behavior z-norm inside folds; per-behavior ALS" + partial,
        )
    )
    logger.info(
        "[pooled/%s] write_gate: oof_R²=%.3f ΔR²=%.3f",
        mask_name,
        sc["oof_r2"],
        sc["oof_r2"] - pbest["oof_r2"],
    )
    return rows


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--behaviors",
        nargs="+",
        default=BEHAVIORS,
        choices=BEHAVIORS,
        help="behavior rows to score (smoke: --behaviors marker)",
    )
    ap.add_argument(
        "--out",
        default=str(EVAL / "analysis/combiner_scores.json"),
        help="output JSON path (rows MERGE into an existing file by "
        "(combiner, behavior, mask) key -- chunked invocations are safe)",
    )
    ap.add_argument(
        "--sanity-only",
        action="store_true",
        help="run only the leaderboard-reproduction gate, then exit 0",
    )
    ap.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip the leaderboard-reproduction gate (chunked invocations after "
        "the first; the gate must have PASSed earlier in the same session)",
    )
    ap.add_argument(
        "--pooled-only",
        action="store_true",
        help="score only the pooled z-normed rows over ALL behaviors "
        "(per-behavior features come from --feature-cache-dir)",
    )
    ap.add_argument(
        "--feature-cache-dir",
        default="/tmp/i537_combiner_feature_cache",
        help="disk cache for per-behavior feature matrices (derived, not committed)",
    )
    args = ap.parse_args()

    if not args.skip_sanity:
        sanity_reproduce_leaderboard()
    if args.sanity_only:
        return 0
    return _run(args)


def _run(args) -> int:
    """Score the requested per-behavior blocks (+ pooled when applicable)."""
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict] = []
    if out_path.exists():
        existing_rows = json.loads(out_path.read_text()).get("rows", [])
    payload: dict = {
        "schema_version": 1,
        "spec": "design doc §5 Combiner track (commit 0cc4053fc); folds/masks = "
        "scripts/i537_score_metric.py LTCO protocol",
        "masks": {
            "quarantine_only": "the shipped leaderboard protocol (quarantine manifest only; "
            "marker n=193 includes 13 implant-failed cells, matching baseline_scores.json)",
            "strict": "quarantine + G_meta implant_failed/saturated flags (registered §5 "
            "exclusion; singles re-scored under this mask for ΔR² consistency)",
        },
        "skipped_features": {
            "pv_dp": "registered-not-implemented (needs P3 ΔP GPU pass) -- skipped per spec",
            "kl_out_seq_fwd/rev": "A5 sequence-level output KL registered-not-implemented "
            "(GPU); shipped first-token KL fwd/rev used as the output divergences instead",
        },
        "rank_corr_semantics": "spearman(pooled OOF prediction, G truth); raw_spearman on "
        "single rows is the leaderboard's feature-vs-G convention (distance polarity)",
        "rows": [],
    }
    rows: list[dict] = []  # rows produced by THIS invocation; merged at flush
    cache: dict[str, tuple] = {}
    cache_dir = Path(args.feature_cache_dir)

    def _flush():
        def _key(r):
            return (r["combiner"], r["behavior"], r["mask"])

        merged = {_key(r): r for r in existing_rows}
        for r in rows:
            merged[_key(r)] = r
        payload["rows"] = list(merged.values())
        payload.update(
            {
                "git_commit": harness._git_commit(),
                "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "env": {"numpy": np.__version__, "scipy": scipy.__version__},
            }
        )
        out_path.write_text(json.dumps(_jsonsafe(payload), indent=1))

    def _prep(beh: str):
        if beh not in cache:
            cids = train_cids_for(beh)
            feats, wg = load_or_compute_features(beh, cids, cache_dir)
            g = harness._load_g(beh, cids, cids)
            masks = usable_masks(beh, cids)
            cache[beh] = (feats, g, masks, wg)
        return cache[beh]

    if not args.pooled_only:
        for beh in args.behaviors:
            feats, g, masks, wg = _prep(beh)
            for mask_name, mask in masks.items():
                block_rows, _singles = score_behavior_block(beh, mask_name, feats, g, mask, wg)
                rows.extend(block_rows)
            _flush()  # checkpoint per behavior

        if "marker" in args.behaviors:
            sanity_ladder_rung1(rows)

    # Pooled rows: only meaningful over the full behavior set -- run when this
    # invocation covers all behaviors (single-command path) or via --pooled-only
    # (final chunked invocation; features from the disk cache).
    if args.pooled_only or set(args.behaviors) == set(BEHAVIORS):
        for beh in BEHAVIORS:
            _prep(beh)
        for mask_name in ("quarantine_only", "strict"):
            rows.extend(score_pooled_block(mask_name, BEHAVIORS, cache))
            _flush()

    _flush()
    logger.info(
        "[combiners] wrote %s (%d rows this invocation, %d total)",
        out_path,
        len(rows),
        len(payload["rows"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
