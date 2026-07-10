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

from collections.abc import Callable, Sequence
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


def spectral_dvs_from_lambda(lam: np.ndarray) -> dict[str, float]:
    """Registered spectral DVs from an EIGENVALUE (variance) spectrum λ = σ².

    Shared reduction for :func:`spectral_dvs` (σ input) and the Gram-space
    batched bootstrap (λ input directly from ``eigvalsh``). Zeros allowed;
    raises on a degenerate (all-zero) spectrum.
    """
    lam = np.asarray(lam, dtype=np.float64).ravel()
    lam = np.clip(lam, 0.0, None)
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
    return spectral_dvs_from_lambda(sigma**2)


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
    import warnings

    warnings.warn(
        "cluster_bootstrap_dv is the SERIAL per-draw reference; production "
        "batteries use the Gram-space batched twin batched_cluster_bootstrap / "
        "batched_dvs_over_indices (vectorize-many-cell-fits.md supersede "
        "contract, #1112). The serial body is retained for exactness gates + "
        "prior-issue reproducibility.",
        FutureWarning,
        stacklevel=2,
    )
    import os

    if os.environ.get("EPM_FORBID_SERIAL_FITS") == "1":
        raise RuntimeError(
            "cluster_bootstrap_dv (serial per-draw SVD loop) blocked under "
            "EPM_FORBID_SERIAL_FITS=1 — use batched_cluster_bootstrap."
        )
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


# ── Gram-space BATCHED cluster bootstrap (#1112 source-module fix) ────────────
#
# The #1112 battery is ~1,344 clouds × 1,000 draws ≈ 1.3e6 spectra — a serial
# per-draw SVD loop is the vectorize-many-cell-fits.md failure signature. The
# batched path precomputes each cloud's row-Gram G = X Xᵀ ONCE (one
# d-contraction), then evaluates every bootstrap draw as an eigvalsh of the
# double-centered sub-Gram G[idx][:, idx] (≤ m×m), batched over draws in
# chunks via torch.linalg.eigvalsh. Row-centering commutes: the eigenvalues of
# C S C (C = I − 11ᵀ/m, S the resampled sub-Gram) are exactly the σ² of the
# row-centered resampled cloud, so spectral_dvs_from_lambda reproduces the
# serial spectral_dvs(svd_of_cloud(X[idx])) numbers to float tolerance
# (pinned by tests/test_issue1112_spectral_batched.py).


BOOTSTRAPPABLE_DVS = ("top_share_lambda", "pr_lambda", "rank_k_at_90")


def bootstrap_index_matrix(
    cluster_ids: Sequence,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """(n_boot, n_rows) row-index matrix for cluster bootstrap draws.

    Resamples CLUSTERS with replacement (the ``cluster_bootstrap_dv``
    convention) and expands each pick to its member rows. Requires EQUAL
    cluster sizes so every draw has a fixed row count (the batched eigvalsh
    needs a fixed shape) — the #1112 clouds have exactly one row per
    (context, question) cluster, which trivially satisfies this. Raises on
    unequal cluster sizes (fall back to the serial reference there).

    PAIRED cross-cell resampling (#1112 plan §6): build this matrix ONCE per
    cloud-pair grouping from the SHARED (context, question) cluster ids and
    apply the SAME matrix to both cells' clouds via
    :func:`batched_dvs_over_indices`.
    """
    ids = np.asarray(cluster_ids)
    unique, first_idx = np.unique(ids, return_index=True)
    # Deterministic cluster order = first-appearance order (stable across
    # cells that share the same (context, question) row ordering).
    unique = unique[np.argsort(first_idx)]
    rows_by_cluster = [np.where(ids == c)[0] for c in unique]
    sizes = {len(r) for r in rows_by_cluster}
    if len(sizes) != 1:
        raise ValueError(
            f"bootstrap_index_matrix requires equal cluster sizes, got sizes {sorted(sizes)} "
            "— use the serial cluster_bootstrap_dv reference for unequal clusters"
        )
    members = np.stack(rows_by_cluster)  # (n_clusters, cluster_size)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique), size=(n_boot, len(unique)))  # cluster picks
    idx = members[picks]  # (n_boot, n_clusters, cluster_size)
    return idx.reshape(n_boot, -1)


def batched_dvs_over_indices(
    cloud: np.ndarray,
    idx: np.ndarray,
    *,
    dv_names: Sequence[str] = BOOTSTRAPPABLE_DVS,
    center_rows: bool = True,
    chunk: int = 250,
) -> dict[str, np.ndarray]:
    """Per-draw spectral DVs for every row-index draw in ``idx`` (batched).

    Args:
        cloud: (n_rows, d) Δx cloud (float; promoted to float64).
        idx: (n_boot, m) row-index matrix (e.g. from
            :func:`bootstrap_index_matrix`).
        dv_names: subset of ``spectral_dvs`` keys to return per draw.
        center_rows: row-center each resampled cloud (the #653 convention).
        chunk: draws per batched eigvalsh call ((chunk, m, m) float64 stack).

    Returns:
        ``{dv_name: (n_boot,) float64 array}`` — the per-draw DV matrix the
        selection-symmetric-nulls rule requires persisting.
    """
    import torch

    X = np.asarray(cloud, dtype=np.float64)
    assert X.ndim == 2, X.shape
    idx = np.asarray(idx)
    assert idx.ndim == 2, idx.shape
    n_boot, m = idx.shape
    d = X.shape[1]
    n_keep = min(m, d)  # svd_of_cloud returns min(m, d) singular values

    G = torch.from_numpy(X @ X.T)  # (n_rows, n_rows) float64, ONE d-contraction
    idx_t = torch.from_numpy(np.ascontiguousarray(idx, dtype=np.int64))

    out = {name: np.empty(n_boot, dtype=np.float64) for name in dv_names}
    for start in range(0, n_boot, chunk):
        sel = idx_t[start : start + chunk]  # (c, m)
        c = sel.shape[0]
        # Sub-Grams via advanced indexing: S[b] = G[sel[b]][:, sel[b]].
        S = G[sel.unsqueeze(2), sel.unsqueeze(1)]  # (c, m, m)
        assert S.shape == (c, m, m), S.shape
        if center_rows:
            row_mean = S.mean(dim=2, keepdim=True)
            col_mean = S.mean(dim=1, keepdim=True)
            grand = S.mean(dim=(1, 2), keepdim=True)
            S = S - row_mean - col_mean + grand
        lam = torch.linalg.eigvalsh(S)  # (c, m) ascending
        lam = lam.clamp_min(0.0)
        # Keep the n_keep LARGEST eigenvalues (parity with min(m, d) SVD modes).
        lam_np = lam.numpy()[:, ::-1][:, :n_keep]
        for b in range(c):
            try:
                dvs = spectral_dvs_from_lambda(lam_np[b])
            except ValueError:
                # Degenerate resample (all-identical rows) — mirror the serial
                # cluster_bootstrap_dv skip semantics: NaN, dropped at quantile.
                for name in dv_names:
                    out[name][start + b] = np.nan
                continue
            for name in dv_names:
                out[name][start + b] = dvs[name]
    return out


def batched_cluster_bootstrap(
    cloud: np.ndarray,
    cluster_ids: Sequence,
    *,
    dv_names: Sequence[str] = BOOTSTRAPPABLE_DVS,
    n_boot: int = 1000,
    seed: int = 653,
    center_rows: bool = True,
    alpha: float = 0.05,
    chunk: int = 250,
    idx: np.ndarray | None = None,
) -> dict:
    """Gram-space batched cluster-bootstrap CIs on the spectral DVs.

    The batched twin of :func:`cluster_bootstrap_dv` (identical resampling
    unit + point estimate; the per-draw spectra come from batched eigvalsh of
    double-centered sub-Grams instead of a per-draw SVD).

    Args:
        idx: optional pre-built (n_boot, n_rows) index matrix (pass the SAME
            matrix to both cells of a pair for PAIRED difference CIs); default
            builds one via :func:`bootstrap_index_matrix`.

    Returns:
        ``{"point": {dv: float}, "ci": {dv: [lo, hi]}, "draws": {dv: (n_boot,)
        array}, "n_boot": int, "resampling": "paired-capable-cluster"}``.
    """
    X = np.asarray(cloud, dtype=np.float64)
    point = spectral_dvs(svd_of_cloud(X, center_rows=center_rows))
    if idx is None:
        idx = bootstrap_index_matrix(cluster_ids, n_boot=n_boot, seed=seed)
    draws = batched_dvs_over_indices(
        X, idx, dv_names=dv_names, center_rows=center_rows, chunk=chunk
    )
    ci = {
        name: [
            float(np.nanquantile(vals, alpha / 2)),
            float(np.nanquantile(vals, 1 - alpha / 2)),
        ]
        for name, vals in draws.items()
    }
    n_valid = int(np.isfinite(next(iter(draws.values()))).sum()) if draws else 0
    return {
        "point": {name: float(point[name]) for name in dv_names},
        "ci": ci,
        "draws": draws,
        "n_boot": int(idx.shape[0]),
        "n_valid": n_valid,
        "resampling": "paired-capable-cluster",
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
    ambiguous: bool  # the DECIDING DV's bootstrap CI crosses that DV's threshold
    # §3.4.CI: which DV's threshold the label rests on — the ambiguity CI is
    # bootstrapped on THIS DV, against THIS DV's thresholds (never hardcoded
    # top-share; round-4 BLOCKER deciding-ci-hardcoded-top-share).
    deciding_dv: str | None = None
    # cos_top_to_rb is alignment-driven: the cosine + random-CI exceedance check IS
    # the decision criterion, so a cluster bootstrap is not meaningful. Recorded as
    # explicit unavailability (fail-loud-style), never a silent top-share fallback.
    deciding_ci_unavailable: bool = False
    deciding_ci_reason: str | None = None
    notes: list[str] = field(default_factory=list)


# §3.4.CI: the threshold(s) each candidate deciding DV is checked against. The
# ambiguity flag fires iff the deciding DV's bootstrap CI brackets ITS OWN
# threshold — a top-share CI ∈ [0,1] can never reach the PR thresholds 2.0/5.0,
# which is exactly why round-4's hardcoded top-share bootstrap made every
# PR-decided H3 cell trivially unambiguous. Source: plan §3.4.CI.
DV_THRESHOLDS: dict[str, tuple[float, ...]] = {
    "top_share_lambda": (TOP_SHARE_LOWRANK,),  # 0.7
    "pr_lambda": (PR_LAMBDA_LOWRANK, PR_LAMBDA_H3),  # 2.0, 5.0
    "rank_k_at_90": (float(RANK_K_H3),),  # 10
    "cos_top_to_rb": (COS_ALIGNED_FLOOR,),  # 0.5 (+ random_ci_high, handled at call site)
}


def deciding_dv_for_label(
    *,
    top_share: float,
    pr: float,
    rank_k: float,
    cos_top_to_rb: float | None,
    is_low_rank: bool,
    is_h3: bool,
    is_aligned: bool | None,
) -> str | None:
    """Name the single DV whose threshold the cell's H-label rests on (§3.4.CI).

    The label is set by a disjunction (``is_low_rank = pr ≤ 2 OR top_share ≥ 0.7``;
    ``is_h3 = pr ≥ 5 OR rank_k ≥ 10``), so the deciding DV is NOT always top-share.
    Selection mirrors the label precedence in :func:`classify_cell` (H3 wins
    first): pick whichever criterion actually crossed, breaking ties by tightest
    fractional margin so the CI gate is tightest. Returns ``None`` for boundary /
    underdetermined cells (no single deciding DV → already ambiguous).
    """
    if is_h3:
        cand: list[tuple[str, float]] = []
        if pr >= PR_LAMBDA_H3:
            cand.append(("pr_lambda", abs(pr - PR_LAMBDA_H3) / PR_LAMBDA_H3))
        if rank_k >= RANK_K_H3:
            cand.append(("rank_k_at_90", abs(rank_k - RANK_K_H3) / RANK_K_H3))
        return min(cand, key=lambda t: t[1])[0]  # tightest-margin H3 criterion
    # H1 / H2 both require low-rank; the deciding low-rank criterion:
    if is_low_rank:
        cand = []
        if top_share >= TOP_SHARE_LOWRANK:
            cand.append(
                ("top_share_lambda", abs(top_share - TOP_SHARE_LOWRANK) / TOP_SHARE_LOWRANK)
            )
        if pr <= PR_LAMBDA_LOWRANK:
            cand.append(("pr_lambda", abs(pr - PR_LAMBDA_LOWRANK) / PR_LAMBDA_LOWRANK))
        low_rank_dv = min(cand, key=lambda t: t[1])[0]
        # If the H1 vs H2 split (alignment) is live, the alignment cos decides.
        if is_aligned is not None and cos_top_to_rb is not None:
            return "cos_top_to_rb"  # H1↔H2 turns on |cos| vs the 0.5 floor + random CI
        return low_rank_dv
    # boundary cells (H2/H3 boundary, underdetermined): no single deciding DV
    return None


def classify_cell(  # noqa: C901 — the §3.2 label precedence + §3.4.CI deciding-DV ambiguity branches ARE the spec; flattening would obscure it.
    *,
    cell_group: str,
    rung: str,
    spec: dict[str, float],
    n_rows: int,
    cos_top_to_rb: float | None = None,
    random_ci_high: float | None = None,
    cross_seed_top_cos: float | None = None,
    deciding_ci: tuple[float, float] | None = None,
    bootstrap_fn: Callable[[str], tuple[float, float]] | None = None,
) -> CellVerdict:
    """Apply the §3.2 thresholds on the σ² spectrum to one cell-rung.

    Args:
        spec: output of :func:`spectral_dvs`.
        n_rows: rows in the SVD cloud (< MIN_SPECTRUM_ROWS ⇒ underdetermined).
        cos_top_to_rb: |cos(top dir, r_B)| (None for Arm A geometry-only).
        random_ci_high: upper bound of the #503 norm-matched random-cos CI.
        cross_seed_top_cos: leading-direction cosine across seeds (rotation
            stability; None at a single headline seed).
        deciding_ci: (lo, hi) bootstrap CI ON THE DECIDING DV (§3.4.CI). Mutually
            exclusive with ``bootstrap_fn``; pass this when the caller already
            knows the deciding DV (e.g. the off-pod refresh re-reads a stored
            ``deciding_dv``) and bootstrapped it. NEVER a top-share CI for a
            PR/rank-decided label.
        bootstrap_fn: optional ``(dv_name) -> (ci_lo, ci_hi)`` callback. When
            given, :func:`classify_cell` selects the deciding DV first
            (:func:`deciding_dv_for_label`) and invokes the callback for THAT DV
            only — the §3.4.CI contract (the bootstrap follows the label, never a
            hardcoded top-share). Ignored for ``cos_top_to_rb``-decided labels,
            where the bootstrap is not meaningful (the cosine + random-CI
            exceedance check is the decision criterion).
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
            deciding_dv=None,
            deciding_ci_unavailable=True,
            deciding_ci_reason="spectrum-underdetermined; no DV decides the label",
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
                f"< {CROSS_SEED_ROTATION_FLOOR}) — 1-seed rotation cross-seed-unverified (§3-bis)"
            )
    elif is_low_rank and is_aligned is None:
        # Arm A geometry-only: low-rank with no r_B alignment read.
        label = "H1/H2(low-rank, alignment-not-read)"
    else:
        # neither clearly low-rank nor clearly H3 — boundary.
        label = "H2/H3(boundary)"
        notes.append("between the low-rank and H3 boundaries — boundary cell")

    # ── §3.4.CI ambiguity flag: bootstrap THE DECIDING DV, vs ITS thresholds ──
    deciding_dv = deciding_dv_for_label(
        top_share=top_share,
        pr=pr,
        rank_k=float(rank_k),
        cos_top_to_rb=cos_top_to_rb,
        is_low_rank=is_low_rank,
        is_h3=is_h3,
        is_aligned=is_aligned,
    )
    ambiguous = False
    deciding_ci_unavailable = False
    deciding_ci_reason: str | None = None

    if deciding_dv is None:
        # boundary / underdetermined: no single deciding DV → already ambiguous.
        ambiguous = True
        deciding_ci_unavailable = True
        deciding_ci_reason = "boundary/underdetermined label; no single deciding DV (§3.4.CI)"
        notes.append("no single deciding DV (boundary label) — flagged ambiguous (§3.4.CI)")
    elif deciding_dv == "cos_top_to_rb":
        # Alignment-driven: the |cos| floor + #503 random-CI exceedance check IS the
        # decision criterion; a cluster bootstrap on cos is not meaningful. Explicit
        # unavailability (fail-loud-style), NOT a silent top-share fallback (§3.4.CI).
        deciding_ci_unavailable = True
        deciding_ci_reason = "alignment-driven; norm-matched random CI is the ambiguity flag"
        # The alignment ambiguity is already captured by the random-CI note above
        # (|cos| ≥ floor but ≤ random_ci_high → label rests on the bare 0.5 cut).
        if (
            cos_top_to_rb is not None
            and random_ci_high is not None
            and abs(cos_top_to_rb) <= random_ci_high
        ):
            ambiguous = True
    else:
        # top_share_lambda / pr_lambda / rank_k_at_90 — bootstrap THIS DV (via the
        # caller's bootstrap_fn) OR use a pre-bootstrapped deciding_ci, and check
        # it against THIS DV's own thresholds (DV_THRESHOLDS), never top-share's.
        ci = deciding_ci
        if ci is None and bootstrap_fn is not None:
            ci = bootstrap_fn(deciding_dv)
        if ci is not None:
            lo, hi = ci
            for thr in DV_THRESHOLDS[deciding_dv]:
                if lo <= thr <= hi:
                    ambiguous = True
                    notes.append(
                        f"deciding DV {deciding_dv} bootstrap CI [{lo:.3f}, {hi:.3f}] "
                        f"crosses its threshold {thr} (§3.4.CI)"
                    )
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
        deciding_dv=deciding_dv,
        deciding_ci_unavailable=deciding_ci_unavailable,
        deciding_ci_reason=deciding_ci_reason,
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
