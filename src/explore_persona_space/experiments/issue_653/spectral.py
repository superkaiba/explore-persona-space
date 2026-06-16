# ruff: noqa: RUF002, RUF003
# Intentional Unicode (σ, λ, Σ, ρ, Δ, ≥, ≤) in scientific docstrings + logs.
"""Spectral DVs for #653 — pinned on the EIGENVALUE (variance) spectrum λ = σ².

This module is the single source of truth for the round-1-REVISE spectral
definitions (plan §3.0). EVERY spectral quantity is computed on the eigenvalue
spectrum ``λ_i := σ_i²``, where ``σ_i`` are the singular values of the
row-centered shift / Δx cloud (Arm B) or of the fitted Jacobian J (Arm A).

The two registered DVs, named explicitly so they are never confused with the
raw-σ storage key ``s_top1_frac = σ₁/Σσ`` in eval_results/issue_521/svd/*.json
(which is NOT a registered DV — it under-reads concentration by summing
un-squared σ):

* ``top_share_lambda := σ₁² / Σ_i σ_i²`` — leading variance share.
* ``pr_lambda := (Σ_i σ_i²)² / Σ_i σ_i⁴`` — participation ratio on λ = σ².

Cross-check (§3, §11): the verified #521 EM exemplar
(eval_results/issue_521/svd/on_policy_em_seed{42,137,256}.json) reads
top_share_lambda 0.81/0.86/0.89 and pr_lambda 1.49/1.34/1.25 under these
formulae — the H1-clean exemplar by construction. ``EM_EXEMPLAR`` records the
verified numbers and ``assert_exemplar_calibration`` asserts the thresholds keep
it H1. The data-shape-only API (numpy in, dict out) has no live-model
dependency, so it is fully CPU-smoke-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import (
    COS_ALIGNED_FLOOR,
    CROSS_SEED_ROTATION_FLOOR,
    MIN_SPECTRUM_ROWS,
    PR_LAMBDA_H3,
    PR_LAMBDA_LOWRANK,
    RANK_K_H3,
    TOP_SHARE_LOWRANK,
)

# Verified #521 EM exemplar (the H1-clean reference; §3/§11). seed 42/137/256.
EM_EXEMPLAR: dict[str, tuple[float, float, float]] = {
    "top_share_lambda": (0.81, 0.86, 0.89),
    "pr_lambda": (1.49, 1.34, 1.25),
}


def spectral_dvs(singular_values: np.ndarray) -> dict[str, float]:
    """Compute the registered spectral DVs from a vector of singular values σ.

    Args:
        singular_values: 1-D array of σ_i ≥ 0 (the SVD of a shift / Δx cloud or
            of a Jacobian). Zeros are allowed (padded spectra); a length-1
            spectrum gives top_share=1.0, pr=1.0.

    Returns:
        ``{top_share_lambda, pr_lambda, rank_k_at_90, n_modes,
        eff_rank_entropy}`` — all on the eigenvalue spectrum λ = σ².
    """
    sigma = np.asarray(singular_values, dtype=np.float64).ravel()
    assert sigma.ndim == 1, sigma.shape
    if (sigma < -1e-9).any():
        raise ValueError("singular values must be non-negative")
    sigma = np.clip(sigma, 0.0, None)
    lam = sigma**2  # the eigenvalue (variance) spectrum
    total = float(lam.sum())
    if total <= 0.0:
        raise ValueError("degenerate spectrum (Σσ² == 0) — cannot compute DVs")

    top_share = float(lam.max() / total)
    pr = float((lam.sum() ** 2) / (lam**2).sum())  # (Σλ)² / Σλ²

    # rank-K@90%: smallest K with cumulative variance share ≥ 0.9.
    order = np.argsort(lam)[::-1]
    cum = np.cumsum(lam[order]) / total
    rank_k = int(np.searchsorted(cum, 0.9) + 1)

    # Entropy-based effective rank (descriptive companion, NOT a registered DV).
    p = lam / total
    nz = p[p > 0]
    eff_rank_entropy = float(np.exp(-(nz * np.log(nz)).sum()))

    return {
        "top_share_lambda": top_share,
        "pr_lambda": pr,
        "rank_k_at_90": rank_k,
        "n_modes": int(lam.size),
        "eff_rank_entropy": eff_rank_entropy,
    }


def svd_of_cloud(cloud: np.ndarray, *, center_rows: bool = True) -> np.ndarray:
    """Singular values of a (n_rows, d_model) shift / Δx cloud.

    Per §3.3 the cloud is the on-policy response-mean shift rows pooled across
    the evaluated context panel × the prompt set (≥ MIN_SPECTRUM_ROWS rows).
    Row-centering removes a shared mean-shift component so the spectrum measures
    the *spread* of shifts, matching the #521 SVD convention.
    """
    X = np.asarray(cloud, dtype=np.float64)
    assert X.ndim == 2, X.shape
    if center_rows:
        X = X - X.mean(axis=0, keepdims=True)
    # economy SVD; singular values only.
    return np.linalg.svd(X, compute_uv=False)


def top_direction(cloud: np.ndarray, *, center_rows: bool = True) -> np.ndarray:
    """The leading right-singular vector (top variance direction) of the cloud.

    Returns a unit vector in R^d_model — the direction compared to ``r_B`` for
    the alignment read.
    """
    X = np.asarray(cloud, dtype=np.float64)
    assert X.ndim == 2, X.shape
    if center_rows:
        X = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def norm_matched_random_cos_ci(
    target: np.ndarray,
    *,
    n_directions: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Norm-matched random-direction cosine CI (the #503 cosine null).

    A 7B residual has |cos| ≈ 0.06 to a random direction by chance; without
    this CI any 0.1 cosine reads as "aligned". Returns the empirical
    (1-alpha) CI on |cos(target, random_unit)|.
    """
    target = np.asarray(target, dtype=np.float64).ravel()
    d = target.shape[0]
    rng = np.random.default_rng(seed)
    V = rng.standard_normal(size=(n_directions, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    t = target / (np.linalg.norm(target) + 1e-12)
    cosines = np.abs(V @ t)
    return {
        "n_directions": n_directions,
        "ci_low": float(np.quantile(cosines, alpha / 2)),
        "ci_high": float(np.quantile(cosines, 1 - alpha / 2)),
        "mean": float(cosines.mean()),
    }


def cluster_bootstrap_dv(
    cloud: np.ndarray,
    cluster_ids: np.ndarray,
    dv_name: str,
    *,
    n_boot: int = 10_000,
    seed: int = 653,
    center_rows: bool = True,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Cluster-bootstrap CI on a single spectral DV of the cloud.

    Resamples CLUSTERS (e.g. context-personas) with replacement, recomputes the
    SVD on the resampled rows, reads the named DV. Per §6 the resampling unit is
    (context-persona, question) for Arm B Δx; pass that cluster id per row.

    Returns ``{point, ci_low, ci_high, n_boot}``.
    """
    X = np.asarray(cloud, dtype=np.float64)
    ids = np.asarray(cluster_ids)
    assert X.ndim == 2 and ids.shape[0] == X.shape[0], (X.shape, ids.shape)
    point = spectral_dvs(svd_of_cloud(X, center_rows=center_rows))[dv_name]

    unique = np.unique(ids)
    rows_by_cluster = {c: np.where(ids == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in picked])
        try:
            val = spectral_dvs(svd_of_cloud(X[idx], center_rows=center_rows))[dv_name]
        except ValueError:
            continue  # degenerate resample
        boot.append(val)
    boot_arr = np.asarray(boot)
    return {
        "point": float(point),
        "ci_low": float(np.quantile(boot_arr, alpha / 2)) if boot_arr.size else float("nan"),
        "ci_high": float(np.quantile(boot_arr, 1 - alpha / 2)) if boot_arr.size else float("nan"),
        "n_boot": int(boot_arr.size),
    }


# ── Per-cell H1/H2/H3 verdict (§3.2 thresholds, σ² spectrum) ─────────────────


@dataclass
class CellVerdict:
    """Per-(behavior × context) verdict with the deciding-quantity flags."""

    cell_group: str
    rung: str
    top_share_lambda: float
    pr_lambda: float
    rank_k_at_90: int
    n_rows: int
    cos_top_to_rb: float | None
    random_ci_high: float | None
    cross_seed_top_cos: float | None
    label: str  # "H1" | "H2" | "H3" | "underdetermined"
    is_low_rank: bool
    is_aligned: bool | None
    ambiguous: bool  # deciding quantity's bootstrap CI crosses a threshold
    notes: list[str] = field(default_factory=list)


def classify_cell(
    *,
    cell_group: str,
    rung: str,
    spec: dict[str, float],
    n_rows: int,
    cos_top_to_rb: float | None = None,
    random_ci_high: float | None = None,
    cross_seed_top_cos: float | None = None,
    deciding_ci: tuple[float, float] | None = None,
) -> CellVerdict:
    """Apply the §3.2 thresholds on the σ² spectrum to one cell-rung.

    Args:
        spec: output of :func:`spectral_dvs`.
        n_rows: rows in the SVD cloud (< MIN_SPECTRUM_ROWS ⇒ underdetermined).
        cos_top_to_rb: |cos(top dir, r_B)| (None for Arm A geometry-only).
        random_ci_high: upper bound of the #503 norm-matched random-cos CI.
        cross_seed_top_cos: leading-direction cosine across seeds (rotation
            stability; None at a single headline seed).
        deciding_ci: (lo, hi) bootstrap CI on the deciding quantity; if it
            crosses the threshold the verdict is flagged ambiguous (§3.4).
    """
    notes: list[str] = []
    top_share = spec["top_share_lambda"]
    pr = spec["pr_lambda"]
    rank_k = int(spec["rank_k_at_90"])

    if n_rows < MIN_SPECTRUM_ROWS:
        return CellVerdict(
            cell_group=cell_group,
            rung=rung,
            top_share_lambda=top_share,
            pr_lambda=pr,
            rank_k_at_90=rank_k,
            n_rows=n_rows,
            cos_top_to_rb=cos_top_to_rb,
            random_ci_high=random_ci_high,
            cross_seed_top_cos=cross_seed_top_cos,
            label="underdetermined",
            is_low_rank=False,
            is_aligned=None,
            ambiguous=True,
            notes=[f"spectrum-underdetermined: {n_rows} rows < {MIN_SPECTRUM_ROWS} (§3.3)"],
        )

    is_low_rank = (pr <= PR_LAMBDA_LOWRANK) or (top_share >= TOP_SHARE_LOWRANK)
    is_h3 = (pr >= PR_LAMBDA_H3) or (rank_k >= RANK_K_H3)

    is_aligned: bool | None = None
    if cos_top_to_rb is not None:
        aligned_by_floor = abs(cos_top_to_rb) >= COS_ALIGNED_FLOOR
        aligned_by_ci = random_ci_high is None or abs(cos_top_to_rb) > random_ci_high
        is_aligned = aligned_by_floor and aligned_by_ci
        if aligned_by_floor and random_ci_high is not None and abs(cos_top_to_rb) <= random_ci_high:
            notes.append(
                "|cos| ≥ 0.5 floor but does NOT exceed the random-CI upper bound — "
                "alignment label rests on the bare 0.5 cut (§3-bis)"
            )

    # Label precedence: H3 (diffuse) wins over low-rank labels.
    if is_h3:
        label = "H3"
    elif is_low_rank and is_aligned:
        label = "H1"
    elif is_low_rank and is_aligned is False:
        # low-rank but not aligned — rotated if the rotation is reproducible.
        if cross_seed_top_cos is not None and cross_seed_top_cos >= CROSS_SEED_ROTATION_FLOOR:
            label = "H2"
        else:
            label = "H2"
            notes.append(
                "rotation NOT confirmed across seeds (cross-seed cos missing or "
                f"< {CROSS_SEED_ROTATION_FLOOR}) — 1-seed rotation is cross-seed-unverified (§3-bis)"
            )
    elif is_low_rank and is_aligned is None:
        # Arm A geometry-only: low-rank with no r_B alignment read.
        label = "H1/H2(low-rank, alignment-not-read)"
    else:
        # neither clearly low-rank nor clearly H3 — boundary.
        label = "H2/H3(boundary)"
        notes.append("between the low-rank and H3 boundaries — boundary cell")

    ambiguous = False
    if deciding_ci is not None:
        lo, hi = deciding_ci
        # The most load-bearing threshold for the label.
        for thr in (TOP_SHARE_LOWRANK, PR_LAMBDA_LOWRANK, PR_LAMBDA_H3):
            if lo <= thr <= hi:
                ambiguous = True
                notes.append(f"deciding-quantity bootstrap CI [{lo:.3f}, {hi:.3f}] crosses {thr}")
                break

    return CellVerdict(
        cell_group=cell_group,
        rung=rung,
        top_share_lambda=top_share,
        pr_lambda=pr,
        rank_k_at_90=rank_k,
        n_rows=n_rows,
        cos_top_to_rb=cos_top_to_rb,
        random_ci_high=random_ci_high,
        cross_seed_top_cos=cross_seed_top_cos,
        label=label,
        is_low_rank=is_low_rank,
        is_aligned=is_aligned,
        ambiguous=ambiguous,
        notes=notes,
    )


def assert_exemplar_calibration() -> None:
    """Assert the §3.2 thresholds keep the verified #521 EM exemplar H1-clean.

    A guard against silently re-categorizing the exemplar to H3 (the v2 defect):
    top_share 0.81-0.89 must pass the ≥0.7 cut and PR_λ 1.25-1.49 must pass the
    ≤2 cut and sit below the H3 ≥5 boundary, on EVERY seed.
    """
    for i, seed in enumerate((42, 137, 256)):
        ts = EM_EXEMPLAR["top_share_lambda"][i]
        pr = EM_EXEMPLAR["pr_lambda"][i]
        assert ts >= TOP_SHARE_LOWRANK, (
            f"EM exemplar seed {seed}: top_share {ts} < {TOP_SHARE_LOWRANK}"
        )
        assert pr <= PR_LAMBDA_LOWRANK, f"EM exemplar seed {seed}: PR_λ {pr} > {PR_LAMBDA_LOWRANK}"
        assert pr < PR_LAMBDA_H3, (
            f"EM exemplar seed {seed}: PR_λ {pr} >= H3 boundary {PR_LAMBDA_H3}"
        )
