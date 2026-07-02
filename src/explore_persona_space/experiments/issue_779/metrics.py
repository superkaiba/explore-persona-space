"""Issue #779 metrics: within-condition Pearson, bootstrap CI, AUROC.

The primary DV is within-condition Pearson r (matched to Persona Vectors, arXiv
2507.21509 App. "Correlation analysis"): the Pearson correlation between a
monitor projection and the trait score computed SEPARATELY within each
experimental condition (excluding conditions with trait-score std < 1), then
averaged across conditions within a mode (system-prompting / many-shot).
Bootstrap 95% CI resamples conditions within-mode.
"""

from __future__ import annotations

import numpy as np


def within_condition_pearson(
    cond_x: list[np.ndarray], cond_y: list[np.ndarray], *, min_y_std: float = 1.0, min_n: int = 3
) -> dict:
    """Mean within-condition Pearson r over a list of per-condition (x, y) arrays.

    PV rule: exclude a condition whose trait-score (y) std < ``min_y_std`` or
    which has < ``min_n`` points or degenerate x. Returns
    {"r": mean_r, "n_conditions": kept, "per_condition_r": [...]}. ``r`` is NaN
    when no condition qualifies.
    """
    per = []
    for x, y in zip(cond_x, cond_y, strict=True):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(y) < min_n or float(np.std(y)) < min_y_std or float(np.std(x)) == 0.0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(r):
            per.append(r)
    return {
        "r": float(np.mean(per)) if per else float("nan"),
        "n_conditions": len(per),
        "per_condition_r": per,
    }


def overall_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r across ALL points (PV's "overall correlation")."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 3 or float(np.std(y)) == 0.0 or float(np.std(x)) == 0.0:
        return float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else float("nan")


def bootstrap_within_condition_ci(
    cond_x: list[np.ndarray],
    cond_y: list[np.ndarray],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    min_y_std: float = 1.0,
    min_n: int = 3,
    ci: float = 0.95,
) -> dict:
    """Bootstrap 95% CI for the within-condition mean r (resample CONDITIONS).

    Resamples the CONDITIONS (not points) with replacement — the correct unit for
    a within-condition mean, matching PV's condition-averaged statistic. Returns
    {"point": r, "lo": .., "hi": .., "n_conditions": .., "n_boot_valid": ..}.
    """
    rng = np.random.default_rng(seed)
    base = within_condition_pearson(cond_x, cond_y, min_y_std=min_y_std, min_n=min_n)
    n_cond = len(cond_x)
    if n_cond == 0 or base["n_conditions"] == 0:
        return {
            "point": base["r"],
            "lo": float("nan"),
            "hi": float("nan"),
            "n_conditions": base["n_conditions"],
            "n_boot_valid": 0,
        }
    boot_rs = []
    idx_all = np.arange(n_cond)
    for _ in range(n_boot):
        samp = rng.choice(idx_all, size=n_cond, replace=True)
        bx = [cond_x[i] for i in samp]
        by = [cond_y[i] for i in samp]
        r = within_condition_pearson(bx, by, min_y_std=min_y_std, min_n=min_n)["r"]
        if np.isfinite(r):
            boot_rs.append(r)
    if not boot_rs:
        return {
            "point": base["r"],
            "lo": float("nan"),
            "hi": float("nan"),
            "n_conditions": base["n_conditions"],
            "n_boot_valid": 0,
        }
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_rs, alpha))
    hi = float(np.quantile(boot_rs, 1.0 - alpha))
    return {
        "point": base["r"],
        "lo": lo,
        "hi": hi,
        "n_conditions": base["n_conditions"],
        "n_boot_valid": len(boot_rs),
    }


def bootstrap_delta_ci(
    cond_x_a: list[np.ndarray],
    cond_x_b: list[np.ndarray],
    cond_y: list[np.ndarray],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    min_y_std: float = 1.0,
    min_n: int = 3,
    ci: float = 0.95,
) -> dict:
    """Bootstrap CI for the within-condition r DIFFERENCE (method A - method B).

    Resamples conditions ONCE per bootstrap replicate and computes the paired
    difference r_A - r_B on the SAME resampled conditions (so the CI reflects the
    paired comparison the success criterion uses: "R1 beats pv_raw by >= +0.05,
    CI excludes 0"). Returns {"delta", "lo", "hi", "excludes_zero"}.
    """
    rng = np.random.default_rng(seed)
    ra = within_condition_pearson(cond_x_a, cond_y, min_y_std=min_y_std, min_n=min_n)["r"]
    rb = within_condition_pearson(cond_x_b, cond_y, min_y_std=min_y_std, min_n=min_n)["r"]
    n_cond = len(cond_y)
    deltas = []
    idx_all = np.arange(n_cond)
    for _ in range(n_boot):
        samp = rng.choice(idx_all, size=n_cond, replace=True)
        r_a = within_condition_pearson(
            [cond_x_a[i] for i in samp],
            [cond_y[i] for i in samp],
            min_y_std=min_y_std,
            min_n=min_n,
        )["r"]
        r_b = within_condition_pearson(
            [cond_x_b[i] for i in samp],
            [cond_y[i] for i in samp],
            min_y_std=min_y_std,
            min_n=min_n,
        )["r"]
        if np.isfinite(r_a) and np.isfinite(r_b):
            deltas.append(r_a - r_b)
    delta = (ra - rb) if (np.isfinite(ra) and np.isfinite(rb)) else float("nan")
    if not deltas:
        return {"delta": delta, "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(deltas, alpha))
    hi = float(np.quantile(deltas, 1.0 - alpha))
    return {
        "delta": delta,
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of a continuous monitor score against binary labels (rank-based).

    labels: 1 = positive (trait-expressing). Uses the Mann-Whitney U form
    (rank-sum), no sklearn dependency. Returns NaN if a class is empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg_rank_by_val = sums / counts
    ranks = avg_rank_by_val[inv]
    r_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def top_k_precision(scores: np.ndarray, labels: np.ndarray, frac: float = 0.10) -> float:
    """Precision among the top-``frac`` highest-scoring items (label==1 = positive)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if len(scores) == 0:
        return float("nan")
    k = max(1, round(frac * len(scores)))
    top_idx = np.argsort(-scores, kind="mergesort")[:k]
    return float(labels[top_idx].mean())
