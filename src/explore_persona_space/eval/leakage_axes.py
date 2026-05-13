"""Cross-phase leakage-axis utilities for issue #368.

This module holds the statistics primitives that both ``i368_phase1_analysis``
and ``i368_phase2_analysis`` reuse:

  * paired-bootstrap Δρ (cluster bootstrap) — T5 / T6
  * cluster-bootstrap CI on a single Spearman ρ
  * permutation-null off-diagonal mean (H3a / H3b — R5 split)
  * source-shuffle permutation null (T13) — Phase 2 only but reusable
  * Benjamini-Hochberg FDR (α=0.10) — R8 scope = single-axis Spearman p-values
  * partial-Spearman ρ — used by both phases
  * within-source nanmean partial ρ with bootstrap CI — R3 + R9 (Phase 2)

The module is import-safe: no GPU touched at import. All functions accept
numpy arrays and return native floats / dicts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
from scipy import stats

DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_PERMUTATION_N = 1000
DEFAULT_CI_LEVEL = 95


# ── Single-axis statistics ──────────────────────────────────────────────────


def spearman_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (rho, p_value) handling zero-variance with NaN.

    scipy.stats.spearmanr returns (nan, nan) on zero-variance input; we keep
    that contract.
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 3:
        return (float("nan"), float("nan"))
    r = stats.spearmanr(x, y, nan_policy="omit")
    rho = float(r.statistic) if hasattr(r, "statistic") else float(r[0])
    p = float(r.pvalue) if hasattr(r, "pvalue") else float(r[1])
    return (rho, p)


def partial_spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """Partial Spearman ρ(x, y | z) via rank-residual regression.

    Spearman-rank x, y, z; regress out z linearly from both x and y; compute
    Pearson r on the residuals (this is the standard partial Spearman
    construction).
    """
    if len(x) != len(y) or len(x) != len(z):
        raise ValueError("length mismatch in partial_spearman_rho")
    if len(x) < 4:
        return float("nan")
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    def _residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Regress a on b (with intercept); return residuals.
        b_centered = b - b.mean()
        denom = float((b_centered**2).sum())
        if denom == 0.0:
            return a - a.mean()
        slope = float(((a - a.mean()) * b_centered).sum()) / denom
        intercept = float(a.mean() - slope * b.mean())
        return a - (slope * b + intercept)

    rx_resid = _residual(rx, rz)
    ry_resid = _residual(ry, rz)
    if rx_resid.std() == 0 or ry_resid.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx_resid, ry_resid)[0, 1])


def conditional_spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, int]:
    """ρ(x, y) on the subset ``mask=True``. Returns (rho, p, n_subset).

    Used for T14 conditional-on-nonzero-marker_rate diagnostic.
    """
    mask = np.asarray(mask, dtype=bool)
    xs = np.asarray(x)[mask]
    ys = np.asarray(y)[mask]
    rho, p = spearman_with_p(xs, ys)
    return (rho, p, int(mask.sum()))


# ── Bootstrap CIs ───────────────────────────────────────────────────────────


def _percentile_ci(samples: np.ndarray, level: int = DEFAULT_CI_LEVEL) -> tuple[float, float]:
    lo = (100 - level) / 2.0
    hi = 100 - lo
    return (float(np.nanpercentile(samples, lo)), float(np.nanpercentile(samples, hi)))


def cluster_bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    level: int = DEFAULT_CI_LEVEL,
    seed: int = 42,
) -> dict:
    """Cluster-bootstrap 95% CI on Spearman ρ.

    ``clusters`` assigns each row to a cluster id. Resample clusters with
    replacement; recompute ρ on the resulting (possibly duplicated) rows.
    Plan T6: primary cluster = test_id (32 clusters on Phase 1);
    secondary = train_family (4 clusters).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idxs = np.concatenate([np.where(clusters == c)[0] for c in drawn])
        rho, _ = spearman_with_p(x[idxs], y[idxs])
        if not np.isnan(rho):
            samples.append(rho)
    if not samples:
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_valid": 0,
        }
    samples_arr = np.array(samples)
    lo, hi = _percentile_ci(samples_arr, level=level)
    point, _ = spearman_with_p(x, y)
    return {
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "n_valid": len(samples),
        "n_resamples": n_resamples,
        "level": level,
        "cluster_count": n_clusters,
    }


def cluster_bootstrap_delta_spearman_ci(
    x_new: np.ndarray,
    x_base: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    level: int = DEFAULT_CI_LEVEL,
    seed: int = 42,
) -> dict:
    """Paired-bootstrap 95% CI on Δρ = ρ(x_new, y) − ρ(x_base, y).

    Same cluster-resample scheme as ``cluster_bootstrap_spearman_ci``. T5
    primary statistic for H1 verdict.
    """
    rng = np.random.default_rng(seed)
    x_new = np.asarray(x_new)
    x_base = np.asarray(x_base)
    y = np.asarray(y)
    clusters = np.asarray(clusters)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idxs = np.concatenate([np.where(clusters == c)[0] for c in drawn])
        rho_new, _ = spearman_with_p(x_new[idxs], y[idxs])
        rho_base, _ = spearman_with_p(x_base[idxs], y[idxs])
        if not (np.isnan(rho_new) or np.isnan(rho_base)):
            samples.append(rho_new - rho_base)
    if not samples:
        return {
            "point_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_valid": 0,
            "excludes_zero": False,
        }
    samples_arr = np.array(samples)
    lo, hi = _percentile_ci(samples_arr, level=level)
    point_new, _ = spearman_with_p(x_new, y)
    point_base, _ = spearman_with_p(x_base, y)
    point_delta = float(point_new - point_base)
    return {
        "point_delta": point_delta,
        "rho_new": float(point_new),
        "rho_base": float(point_base),
        "ci_low": lo,
        "ci_high": hi,
        "n_valid": len(samples),
        "n_resamples": n_resamples,
        "level": level,
        "cluster_count": n_clusters,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# ── Within-source partial Spearman (R3 + R9 — Phase 2 T9) ───────────────────


def within_source_nanmean_partial_rho(
    axis_values: np.ndarray,
    leakage: np.ndarray,
    sources: Sequence[str],
    *,
    epsilon_variance: float = 1e-12,
) -> dict:
    """Per-source ρ + nanmean over contributing (non-degenerate) sources.

    R3 spec:
      - Per-source ρ via ``spearmanr`` on each source's rows.
      - If leakage variance < ``epsilon_variance``, record NaN and mark
        the source ``degenerate``.
      - Aggregate via ``np.nanmean`` over contributing sources (degenerate
        excluded).

    Returns a dict containing per-source rho, per-source n, leakage variance
    per source, list of degenerate (excluded) sources, list of contributing
    sources, and the nanmean.
    """
    axis_values = np.asarray(axis_values)
    leakage = np.asarray(leakage)
    sources_arr = np.asarray(list(sources))

    per_source_rho: dict[str, float] = {}
    per_source_n: dict[str, int] = {}
    leakage_var_per_source: dict[str, float] = {}
    degenerate: list[str] = []
    low_nonzero: list[str] = []
    contributing: list[str] = []

    for s in sorted(set(sources_arr.tolist())):
        mask = sources_arr == s
        x_s = axis_values[mask]
        y_s = leakage[mask]
        n_s = int(mask.sum())
        per_source_n[s] = n_s
        var_s = float(y_s.var()) if n_s > 1 else 0.0
        leakage_var_per_source[s] = var_s
        nonzero_count = int((y_s != 0).sum())
        if var_s < epsilon_variance:
            per_source_rho[s] = float("nan")
            degenerate.append(s)
            continue
        # M3 (documented rule, plan §R3 + round-3 Codex Methodology minor):
        # Exclude sources with fewer than 3 nonzero leakage values from the
        # within-source nanmean. Rationale: the variance criterion alone
        # (epsilon_variance=1e-12) admits sources where almost all targets
        # have leakage=0 plus a single outlier — e.g. villain in #142
        # (9/10 zeros + 1 nonzero=0.015 → var ≈ 2.25e-5, exceeds epsilon
        # but gives a near-degenerate 1-point Spearman). Spearman ρ on
        # <3 distinct values is dominated by the tie-break rule and is
        # not a meaningful within-source signal. Recorded under
        # `excluded_low_nonzero_count` in the output dict so the analyzer
        # can see which sources were dropped and why; the variance-only
        # exclusion is reported separately under `degenerate_sources_excluded`.
        if nonzero_count < 3:
            low_nonzero.append(s)
            per_source_rho[s] = float("nan")
            continue
        rho, _ = spearman_with_p(x_s, y_s)
        per_source_rho[s] = rho
        contributing.append(s)

    valid = [r for r in per_source_rho.values() if not np.isnan(r)]
    nanmean = float(np.nanmean(valid)) if valid else float("nan")

    return {
        "per_source_rho": per_source_rho,
        "per_source_n": per_source_n,
        "leakage_variance_per_source": leakage_var_per_source,
        "degenerate_sources_excluded": sorted(degenerate),
        "excluded_low_nonzero_count": sorted(low_nonzero),
        "contributing_sources": sorted(contributing),
        "nanmean_partial_rho": nanmean,
    }


def within_source_partial_rho_bootstrap_ci(
    axis_values: np.ndarray,
    leakage: np.ndarray,
    sources: Sequence[str],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    level: int = DEFAULT_CI_LEVEL,
    seed: int = 42,
    epsilon_variance: float = 1e-12,
) -> dict:
    """R9: cluster-bootstrap CI on the within-source nanmean.

    Resample within each contributing source's rows with replacement
    (keep the source label fixed, resample the 10 rows within each
    source). For each resample, recompute the per-source ρ + nanmean.
    """
    rng = np.random.default_rng(seed)
    axis_values = np.asarray(axis_values)
    leakage = np.asarray(leakage)
    sources_arr = np.asarray(list(sources))

    # Identify contributing sources from the point estimate first.
    point = within_source_nanmean_partial_rho(
        axis_values, leakage, sources, epsilon_variance=epsilon_variance
    )

    samples: list[float] = []
    for _ in range(n_resamples):
        rhos: list[float] = []
        for s in point["contributing_sources"]:
            idx_s = np.where(sources_arr == s)[0]
            if len(idx_s) < 3:
                continue
            drawn = rng.choice(idx_s, size=len(idx_s), replace=True)
            x_s = axis_values[drawn]
            y_s = leakage[drawn]
            if y_s.var() < epsilon_variance:
                continue
            rho, _ = spearman_with_p(x_s, y_s)
            if not np.isnan(rho):
                rhos.append(rho)
        if rhos:
            samples.append(float(np.mean(rhos)))
    if not samples:
        ci = {"ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0}
    else:
        samples_arr = np.array(samples)
        lo, hi = _percentile_ci(samples_arr, level=level)
        ci = {"ci_low": lo, "ci_high": hi, "n_valid": len(samples)}

    return {
        **point,
        "bootstrap_ci_95": [ci["ci_low"], ci["ci_high"]],
        "bootstrap_n_resamples": n_resamples,
        "bootstrap_n_valid": ci["n_valid"],
        "ci_excludes_zero": bool(
            (not np.isnan(ci["ci_low"]))
            and (not np.isnan(ci["ci_high"]))
            and (ci["ci_low"] > 0 or ci["ci_high"] < 0)
        ),
    }


# ── Permutation nulls ───────────────────────────────────────────────────────


def recipe_agreement_matrix(
    score_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """K×K Spearman matrix of per-row similarity-score vectors.

    Input: ``{axis_name: per_row_score_vector}`` with identical length.
    Returns: (matrix, ordered_axis_names) where matrix[i, j] is the
    Spearman ρ of axis i's score vector against axis j's score vector.
    """
    axes = list(score_vectors.keys())
    k = len(axes)
    mat = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if i == j:
                mat[i, j] = 1.0
            else:
                rho, _ = spearman_with_p(score_vectors[axes[i]], score_vectors[axes[j]])
                mat[i, j] = rho
    return mat, axes


def off_diagonal_stats(matrix: np.ndarray) -> dict:
    """Mean / min / count of off-diagonal entries (ignoring NaN)."""
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        raise ValueError("matrix must be square")
    off = []
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(matrix[i, j]):
                off.append(float(matrix[i, j]))
    if not off:
        return {"mean": float("nan"), "min": float("nan"), "n": 0}
    return {"mean": float(np.mean(off)), "min": float(np.min(off)), "n": len(off)}


def marker_shuffle_permutation_null(
    score_vectors: dict[str, np.ndarray],
    marker_rate: np.ndarray,
    *,
    n_permutations: int = DEFAULT_PERMUTATION_N,
    seed: int = 42,
) -> dict:
    """H3b — shuffle marker_rate, recompute per-axis ρ, build K×K matrix.

    NB (round-3 Codex Methodology SR): the K×K recipe-agreement matrix is
    built from cosine-score vectors that are INDEPENDENT of marker_rate, so
    shuffling marker_rate does NOT change the matrix → the permutation
    null is degenerate (point mass at the observed value).

    We still implement the procedure faithfully (it's the H3b verdict
    statistic as specified) but flag ``null_is_degenerate=True`` in the
    returned dict so the analyzer knows H3b PASS = "null test confirms
    algebraic structure," not "leakage-aligned signal."
    """
    rng = np.random.default_rng(seed)
    # The matrix doesn't depend on marker_rate; computing it once is sufficient.
    mat, _axes = recipe_agreement_matrix(score_vectors)
    observed = off_diagonal_stats(mat)
    # Simulate the procedure for transparency; record the null distribution.
    null_means: list[float] = []
    for _ in range(n_permutations):
        _ = rng.permutation(marker_rate)
        # By construction the matrix is unchanged, but we keep the loop so
        # the recorded n_permutations matches what the verdict claims.
        null_means.append(observed["mean"])
    pct95 = float(np.percentile(null_means, 95))
    return {
        "observed_off_diagonal_mean": observed["mean"],
        "observed_off_diagonal_min": observed["min"],
        "null_95th_percentile": pct95,
        "exceeds_null": bool(observed["mean"] > pct95),
        "n_permutations": n_permutations,
        "null_is_degenerate": True,
        "null_degeneracy_note": (
            "K×K matrix built from cosine score vectors independent of "
            "marker_rate; shuffling marker_rate does not change the matrix. "
            "H3b PASS confirms algebraic recipe agreement, not leakage "
            "alignment. See round-3 Codex Methodology SR."
        ),
    }


def source_shuffle_permutation_null(
    axis_values: np.ndarray,
    leakage: np.ndarray,
    sources: Sequence[str],
    *,
    n_permutations: int = DEFAULT_PERMUTATION_N,
    seed: int = 42,
) -> dict:
    """T13 — shuffle the source-vector-to-source-name assignment.

    For each permutation, permute the source labels across rows (so
    "source = villain" attaches to a different set of rows each draw) and
    recompute marginal Spearman ρ(axis, leakage). The observed ρ must exceed
    the 95th percentile of the null to be considered "source-specific
    signal" rather than "any-vector correlated with leakage."

    Plan §4.2.3 T13 calls this an "informative" alternative null distinct
    from the degenerate marker-shuffle for H3b.
    """
    rng = np.random.default_rng(seed)
    axis_values = np.asarray(axis_values, dtype=float).copy()
    leakage = np.asarray(leakage, dtype=float)
    sources_arr = np.asarray(list(sources))
    unique_sources = np.unique(sources_arr)

    # Per source, the axis values are the same across all targets within that
    # source (chenstyle vector per source × leakage rate per (source, target)).
    # Build a source→axis_value mapping (mean over within-source values, since
    # within-source axis values per (source, target_panel) differ but the
    # per-source persona vector is identical). Then permute the source-label
    # mapping.
    null_rhos: list[float] = []

    # Snapshot original per-row (source, axis_value) to avoid mutating input.
    # The permutation reassigns each source name to another source's row-set.
    for _ in range(n_permutations):
        perm = rng.permutation(unique_sources)
        relabel = dict(zip(unique_sources, perm, strict=True))
        # Build a new axis-value vector by looking up the permuted source's
        # original axis values in row-order. We approximate per-row
        # reassignment by, for each row, taking the original axis value of
        # one randomly-sampled row from the permuted source.
        new_axis = np.empty_like(axis_values)
        for s in unique_sources:
            target_rows = np.where(sources_arr == s)[0]
            donor_source = relabel[s]
            donor_rows = np.where(sources_arr == donor_source)[0]
            # Sample with replacement; vectorize.
            drawn = rng.choice(donor_rows, size=len(target_rows), replace=True)
            new_axis[target_rows] = axis_values[drawn]
        rho, _ = spearman_with_p(new_axis, leakage)
        if not np.isnan(rho):
            null_rhos.append(rho)

    observed_rho, _ = spearman_with_p(axis_values, leakage)
    if not null_rhos:
        return {
            "observed_rho": float(observed_rho),
            "null_95th_percentile": float("nan"),
            "exceeds_null": False,
            "n_permutations": n_permutations,
            "n_valid": 0,
        }
    pct95 = float(np.percentile(null_rhos, 95))
    return {
        "observed_rho": float(observed_rho),
        "null_95th_percentile": pct95,
        "exceeds_null": bool(abs(observed_rho) > pct95),
        "null_mean": float(np.mean(null_rhos)),
        "null_std": float(np.std(null_rhos)),
        "n_permutations": n_permutations,
        "n_valid": len(null_rhos),
    }


# ── Benjamini-Hochberg FDR (R8 scope = 9 single-axis Spearman p-values) ─────


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.10) -> dict[str, dict]:
    """BH-FDR correction over a family of p-values.

    Returns a {name: {p_raw, p_adjusted, significant_at_alpha}} dict.

    Plan R8: applied ONLY to the 9 single-axis Spearman ρ p-values (the
    canonical 8 new axes + the 1 pos-only descriptive). NOT applied to
    ΔR², partial ρ, conditional ρ, within-source partial ρ (those are
    descriptive decomposition statistics, not multiple-test artifacts).
    """
    names = list(p_values.keys())
    ps = np.array([p_values[n] for n in names], dtype=float)
    n = len(ps)
    order = np.argsort(ps)
    ranked = ps[order]
    # BH adjusted p = min over j>=i of (n / (j+1)) * p[j], then clamped to 1.
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for j in range(n - 1, -1, -1):
        adj = ranked[j] * n / (j + 1)
        if adj < running_min:
            running_min = adj
        adjusted[j] = min(running_min, 1.0)
    # Un-sort back to original order.
    out_adj = np.empty(n, dtype=float)
    out_adj[order] = adjusted
    return {
        names[i]: {
            "p_raw": float(ps[i]),
            "p_adjusted_bh": float(out_adj[i]),
            "significant_at_alpha": bool(out_adj[i] < alpha),
        }
        for i in range(n)
    }


# ── Pretty-print + IO helpers ───────────────────────────────────────────────


def to_jsonable(obj):
    """Recursively convert numpy types / arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump_json(obj, path) -> None:
    """JSON dump with the numpy-tolerant converter + sorted keys + indent."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True)


# ── Reproducibility metadata ───────────────────────────────────────────────


def build_run_metadata(extra: dict | None = None) -> dict:
    """Standard metadata block: git commit, timestamps, env versions.

    Plan §"Reproducibility metadata": every result JSON should include these.
    """
    import datetime
    import platform
    import subprocess
    import sys as _sys

    def _safe_run(cmd: list[str]) -> str:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            return out.stdout.strip()
        except Exception:
            return ""

    md = {
        "git_commit": _safe_run(["git", "rev-parse", "HEAD"]),
        "git_branch": _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_safe_run(["git", "status", "--porcelain"])),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": _sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": stats.__name__ + "@scipy",  # placeholder
    }
    try:
        import scipy

        md["scipy_version"] = scipy.__version__
    except Exception:
        md["scipy_version"] = ""
    if extra:
        md.update(extra)
    return md


__all__ = [
    "AXIS_SPECS_RECIPE_AGREEMENT",
    "DEFAULT_BOOTSTRAP_N",
    "DEFAULT_PERMUTATION_N",
    "benjamini_hochberg",
    "build_run_metadata",
    "cluster_bootstrap_delta_spearman_ci",
    "cluster_bootstrap_spearman_ci",
    "conditional_spearman_rho",
    "dump_json",
    "marker_shuffle_permutation_null",
    "off_diagonal_stats",
    "partial_spearman_rho",
    "recipe_agreement_matrix",
    "source_shuffle_permutation_null",
    "spearman_with_p",
    "to_jsonable",
    "within_source_nanmean_partial_rho",
    "within_source_partial_rho_bootstrap_ci",
]

# The 8 axes that go into the H3a / H3b recipe-agreement matrix (excludes
# pcentroid_chenstyle_pos_only_L20, which is descriptive-only per T10).
AXIS_SPECS_RECIPE_AGREEMENT: list[str] = [
    "pvec_chenstyle_L20",
    "pvec_chenstyle_L15",
    "pvec_chenstyle_L25",
    "pvec_chenstyle_lasttoken",
    "pvec_chenstyle_orthog",
    "pvec_chenstyle_L20_projdiff",
    "pcentroid_methodA_L20",
    "pcentroid_methodB_L20",
]

# Phase 1 / Phase 2 R4 disclosure — projdiff identical to chenstyle_L20
# within-source by construction; analyzers / tests should reference this
# canonical pair for sanity-checking algebraic identity.
R4_DEGENERATE_PAIR: tuple[str, str] = ("pvec_chenstyle_L20", "pvec_chenstyle_L20_projdiff")
