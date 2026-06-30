# ruff: noqa: RUF002, RUF003
"""Issue #742 — decoding-ceiling, linear-information-loss, and sample-complexity
statistical primitives for the #658 base-model behavior representations (n=50).

This module is the CANONICAL implementation surface for the issue-742 analysis
(plan v7 §4 "New code" + §10 Code row + §13 TDD). The six orchestration scripts
``scripts/issue742_*.py`` import every estimator from here; the per-symbol API is
the contract the ``tests/issue_742/`` suite binds to (see ``conftest.EXPECTED_API``).

Everything here is a closed-form / sampling-based statistical estimator over the
frozen #658 tensors — reliability decompositions, the closed-form LEACE eraser,
distance correlation + its refit-per-permutation null, cluster bootstrap, and the
genre-aware artifact loaders. 0 GPU; no model weights. Determinism: every random
estimator takes an explicit ``numpy.random.Generator`` (seeds in the 742X family,
plan §10 reproducibility card).

The single load-bearing methodological commitments encoded here (plan §4/§6/§11):

* **Report everything relative to √(r_yy); NEVER disattenuate** — the ceiling is a
  reliability estimate, the bracket is ``[ρ_lin, √(r_yy)]``, and ``ρ_lin / √(r_yy)``
  is never computed as a headline.
* **MF3 (single full-sample PCA→LEACE→dCor frame, refit per permutation)** — the
  ``dcor_permutation_test`` re-fits the WHOLE pipeline inside every label
  permutation so the observed statistic and every null draw are produced by the
  literally identical procedure (no cross-fitted / cached coordinate frame leaks).
  The ``pca_fit_fn`` / ``leace_fit_fn`` hooks default to the real fits and exist so
  a counting stub can prove the refit-per-permutation contract.
* **binomial reliability uses the cell-actual m, NEVER a blanket m=2000** (plan
  §11 row 1; the heterogeneous ``per_probe`` shape across behaviors, §12 row 2).
* **ρ_lin reads ``analyzer_body_data.json`` ``/<genre>/a33/<beh>/lin_rho``, NEVER
  ``assumption_verdicts.json``** (the A3.2 MLP ``best_rho`` is a different quantity,
  §12 row 4).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Genre alias map                                                              #
# --------------------------------------------------------------------------- #
# The two genres carry DISTINCT v0 tensors (MF1). The eval-results JSON keys the
# UltraChat genre as "g1" while the user-facing / loader genre name is
# "ultrachat"; "betley" is keyed as-is.
_GENRE_TO_A33_KEY = {"betley": "betley", "ultrachat": "g1", "g1": "g1"}

# The four read-out behaviors that carry a33 read-out contrasts (plan §1).
READOUT_BEHAVIORS = ("broad_em", "harmful_compliance", "sycophancy", "refusal")


# --------------------------------------------------------------------------- #
# 1. Reliability decompositions (plan §4 Stage-0 step 1 + §11 row 1)            #
# --------------------------------------------------------------------------- #
def _spearman_brown(r_half: float) -> float:
    """Spearman-Brown prophecy step for a split-half correlation.

    Maps a half-test reliability ``r_half`` to the full-test reliability
    ``r_yy = 2 r_half / (1 + r_half)``. Clamped to [0, 1] (a negative or >1
    half-correlation from small-n sampling noise yields a clamped reliability).
    """
    denom = 1.0 + r_half
    if abs(denom) < 1e-12:
        return 0.0
    r_yy = 2.0 * r_half / denom
    return float(np.clip(r_yy, 0.0, 1.0))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning 0.0 for a degenerate (zero-variance) input."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def reliability_split_half_over_rollouts(
    rollout_labels: np.ndarray,
    *,
    n_split_seeds: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Split-half-over-ROLLOUTS reliability + Spearman-Brown (≥2-rollout regime).

    ``rollout_labels``: int array ``(n_contexts, n_probes, n_rollouts)`` of {0,1}.
    For each random split seed, split the n_rollouts dimension into two halves per
    probe, recompute the per-context rate from each half (averaging over probes ×
    half-rollouts), correlate the two half-rate vectors ACROSS contexts to get
    ``r_half``, then apply Spearman-Brown. Average the resulting r_yy over the
    ``n_split_seeds`` random rollout-splits.

    Returns the mean Spearman-Brown-corrected reliability in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    labels = np.asarray(rollout_labels)
    n_contexts, _n_probes, n_rollouts = labels.shape
    if n_rollouts < 2:
        raise ValueError(
            f"split-half-over-rollouts is undefined for n_rollouts={n_rollouts} (<2); "
            "use reliability_split_half_over_probes for the 1-rollout regime"
        )
    half = n_rollouts // 2
    r_yys: list[float] = []
    for _ in range(n_split_seeds):
        perm = rng.permutation(n_rollouts)
        idx_a, idx_b = perm[:half], perm[half : 2 * half]
        # per-context rate from each rollout-half (mean over probes AND half-rollouts)
        rate_a = labels[:, :, idx_a].reshape(n_contexts, -1).mean(axis=1)
        rate_b = labels[:, :, idx_b].reshape(n_contexts, -1).mean(axis=1)
        r_half = _corr(rate_a, rate_b)
        r_yys.append(_spearman_brown(r_half))
    return float(np.mean(r_yys))


def reliability_split_half_over_probes(
    probe_rates: np.ndarray,
    *,
    n_split_seeds: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Split-half-over-PROBES reliability + Spearman-Brown (1-rollout regime).

    ``probe_rates``: float array ``(n_contexts, n_probes)`` — the per-(context,
    probe) rate (for a 1-rollout-per-probe behavior this is the single {0,1}
    label). For each random split seed, split the PROBE set into two halves,
    compute ``E0`` per context from each probe-half (mean over that half's
    probes), correlate the two half-E0 vectors ACROSS contexts to get ``r_half``,
    then apply Spearman-Brown. Average over ``n_split_seeds`` random probe-splits.

    Returns the mean Spearman-Brown-corrected reliability in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    rates = np.asarray(probe_rates, dtype=float)
    _n_contexts, n_probes = rates.shape
    if n_probes < 2:
        raise ValueError(f"split-half-over-probes needs >=2 probes, got {n_probes}")
    half = n_probes // 2
    r_yys: list[float] = []
    for _ in range(n_split_seeds):
        perm = rng.permutation(n_probes)
        idx_a, idx_b = perm[:half], perm[half : 2 * half]
        e0_a = rates[:, idx_a].mean(axis=1)
        e0_b = rates[:, idx_b].mean(axis=1)
        r_half = _corr(e0_a, e0_b)
        r_yys.append(_spearman_brown(r_half))
    return float(np.mean(r_yys))


def reliability_binomial_variance(
    rates: np.ndarray,
    m_cell: np.ndarray | int,
) -> float:
    """Binomial-variance reliability decomposition with the CELL-ACTUAL m.

    ``rates``: per-context rate array ``(n_contexts,)`` (E0 per context).
    ``m_cell``: the per-context measurement count (rollouts judged) — either a
    scalar applied to every context or a ``(n_contexts,)`` array. This MUST be the
    cell-actual ``n_judged`` (≈2000 sycophancy / ≈400 broad_em / ≈115
    harmful_compliance / ≈214 refusal), NEVER a blanket m=2000 — under-subtracting
    the binomial noise (which scales as ``p(1-p)/m``) inflates the ceiling toward 1.

    Returns the VARIANCE-RATIO reliability ``r_yy = SP / Var_C(E0)`` where
    ``SP = Var_C(E0) − mean_C[p̂(1−p̂)/m_cell]`` (signal power = total
    context-variance minus mean within-context binomial sampling variance),
    clamped to [0, 1]. This is the SAME space the split-half + Spearman-Brown
    estimators return (a variance ratio), so the two are directly comparable for
    the §4-step-1 agreement cross-check. The bracket's noise CEILING is the
    correlation ``√(r_yy)`` — the orchestration applies ``√`` to this return when
    forming ``[ρ_lin, √(r_yy)]`` (a correlation comparable to ρ_lin); the
    primitive deliberately returns the un-sqrt'd reliability so it lives in the
    same space as the split-half reads it is cross-checked against.
    """
    r = np.asarray(rates, dtype=float)
    n = r.shape[0]
    if np.isscalar(m_cell):
        m = np.full(n, float(m_cell))
    else:
        m = np.asarray(m_cell, dtype=float)
        if m.shape != r.shape:
            raise ValueError(f"m_cell shape {m.shape} != rates shape {r.shape}")
    var_total = float(np.var(r))  # population variance across contexts
    if var_total < 1e-12:
        return 0.0
    within = r * (1.0 - r) / np.clip(m, 1.0, None)  # binomial sampling var of the rate
    sp = var_total - float(np.mean(within))
    ratio = sp / var_total
    if ratio <= 0.0:
        return 0.0
    return float(np.clip(ratio, 0.0, 1.0))


@dataclass
class ReliabilityEstimates:
    """Both reliability-estimator reads for one (behavior, genre) cell.

    Surfaces BOTH values separately (never an average) plus a ``disagree`` flag
    that fires when |split_half − binomial| > 0.10 (plan §7 1-rollout-disagreement
    row). When the two disagree the analyzer trusts the split-half read; this
    dataclass never collapses them to a mean.
    """

    behavior: str
    genre: str
    split_half: float
    binomial: float
    estimator_kind: str  # "split_half_over_rollouts" | "split_half_over_probes"
    disagree_threshold: float = 0.10

    @property
    def disagree(self) -> bool:
        return abs(float(self.split_half) - float(self.binomial)) > self.disagree_threshold


def load_reliability_estimates(
    *,
    behavior: str,
    genre: str,
    probe_rates: np.ndarray | None = None,
    rollout_labels: np.ndarray | None = None,
    n_rollouts_per_probe: int = 1,
    rng: np.random.Generator | None = None,
    n_split_seeds: int = 200,
) -> ReliabilityEstimates:
    """Compute BOTH reliability estimators for a (behavior, genre) cell.

    The estimator pair FORKS on ``n_rollouts_per_probe`` (plan §11 row 1):
    ≥2 rollouts/probe → split-half-over-rollouts; 1 rollout/probe → split-half-
    over-probes. The binomial read uses the cell-actual m (= total measurements per
    context). Returns a :class:`ReliabilityEstimates` exposing both values + a
    ``disagree`` flag — both are surfaced, never averaged.

    Caller passes ``probe_rates`` (n_contexts, n_probes) for the over-probes path,
    or ``rollout_labels`` (n_contexts, n_probes, n_rollouts) for the over-rollouts
    path. The per-context rate (for the binomial term) is derived from whichever is
    supplied.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_rollouts_per_probe >= 2:
        if rollout_labels is None:
            raise ValueError("rollout_labels required for the >=2-rollout regime")
        labels = np.asarray(rollout_labels)
        n_contexts = labels.shape[0]
        m_cell = labels.shape[1] * labels.shape[2]
        per_context_rate = labels.reshape(n_contexts, -1).mean(axis=1)
        split = reliability_split_half_over_rollouts(labels, n_split_seeds=n_split_seeds, rng=rng)
        kind = "split_half_over_rollouts"
    else:
        if probe_rates is None:
            raise ValueError("probe_rates required for the 1-rollout regime")
        rates2d = np.asarray(probe_rates, dtype=float)
        m_cell = rates2d.shape[1]  # one rollout per probe -> m = n_probes
        per_context_rate = rates2d.mean(axis=1)
        split = reliability_split_half_over_probes(rates2d, n_split_seeds=n_split_seeds, rng=rng)
        kind = "split_half_over_probes"
    binom = reliability_binomial_variance(per_context_rate, m_cell)
    return ReliabilityEstimates(
        behavior=behavior,
        genre=genre,
        split_half=split,
        binomial=binom,
        estimator_kind=kind,
    )


def bayes_error_ceiling(rates: np.ndarray) -> float:
    """Model-free binary Bayes-error ceiling ``β = E_C[min(E0, 1−E0)]`` (Ishida 2022).

    The mean of the labels expressing class-assignment uncertainty — a free
    (no-fit, no-hyperparameter) ceiling read bounding achievable error from the
    label uncertainty alone (plan §4 Stage-0 step 7).
    """
    r = np.asarray(rates, dtype=float)
    return float(np.mean(np.minimum(r, 1.0 - r)))


# --------------------------------------------------------------------------- #
# 2. PCA basis + closed-form LEACE eraser (plan §4 Stage-1 steps 1-2)           #
# --------------------------------------------------------------------------- #
@dataclass
class PCABasis:
    """A fitted PCA basis: mean + the top-``d_eff`` principal-component loadings.

    ``transform(X)`` projects ``(n, d)`` onto ``(n, d_eff)`` in the fitted basis.
    Single full-sample fit (Option A) — no per-fold rotation.
    """

    mean: np.ndarray  # (d,)
    components: np.ndarray  # (d_eff, d) — rows are the top principal directions
    explained_variance_ratio: np.ndarray  # (d_eff,)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=float) - self.mean[None, :]
        return Xc @ self.components.T


def fit_pca_basis(X: np.ndarray, d_eff: int) -> PCABasis:
    """Fit a single full-sample PCA basis reducing ``(n, d)`` to ``d_eff`` dims.

    Centered SVD of the full sample (Option A — the single commensurable frame, MF3).
    ``d_eff`` is clamped to ``min(d_eff, n, d)``. Returns a :class:`PCABasis`.
    """
    Xm = np.asarray(X, dtype=float)
    n, d = Xm.shape
    k = int(min(d_eff, n, d))
    mean = Xm.mean(axis=0)
    Xc = Xm - mean[None, :]
    # economy SVD: Xc = U S Vt; principal directions are the rows of Vt
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]  # (k, d)
    total_var = float((S**2).sum())
    evr = (S[:k] ** 2) / total_var if total_var > 0 else np.zeros(k)
    return PCABasis(mean=mean, components=components, explained_variance_ratio=evr)


@dataclass
class LeaceEraser:
    """A fitted closed-form LEACE eraser for a (1-D continuous) concept E0.

    Erases the linearly-decodable component of a continuous concept from ``v0``
    via the whitened oblique projection of Belrose et al. 2023 (closed form,
    minimal change). ``transform(v0)`` returns the residual (mean re-added so the
    embedding is only minimally changed). ``P`` is the (d, d) projection applied to
    the centered data.

    Construction (one full-sample fit on all rows): with centered ``X̃`` and the
    centered scalar concept ``z̃``, the linearly-decodable direction in X-space is
    the whitened cross-covariance ``W = Σ_xx^{-1/2} σ_xz`` (a unit direction after
    normalization in the whitened basis). The oblique eraser projects out exactly
    that direction in the whitened basis, then un-whitens::

        P = I − Σ^{1/2} (ŵ ŵᵀ) Σ^{-1/2}

    where ``ŵ`` is the unit whitened cross-cov direction. Applying ``P`` to the
    centered data zeroes ``cov(z, P X̃)`` along every coordinate (the closed-form
    guarantee) while leaving directions orthogonal to the concept ~unchanged.
    """

    mean_x: np.ndarray  # (d,)
    P: np.ndarray  # (d, d) projection applied to centered X

    def transform(self, v0: np.ndarray) -> np.ndarray:
        X = np.asarray(v0, dtype=float)
        Xc = X - self.mean_x[None, :]
        return (Xc @ self.P.T) + self.mean_x[None, :]


def _symmetric_inv_sqrt(M: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Return (M^{1/2}, M^{-1/2}) for a symmetric PSD matrix via eigendecomposition.

    Eigenvalues are floored at ``eps`` before the inverse-square-root so a rank-
    deficient covariance (d ≥ n) does not blow up — the floored directions carry no
    concept covariance to erase, so flooring them is the minimal-change-safe choice.
    """
    Msym = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(Msym)
    evals_clipped = np.clip(evals, eps, None)
    sqrt = (evecs * np.sqrt(evals_clipped)) @ evecs.T
    inv_sqrt = (evecs * (1.0 / np.sqrt(evals_clipped))) @ evecs.T
    return sqrt, inv_sqrt


def fit_leace(v0: np.ndarray, E0: np.ndarray) -> LeaceEraser:
    """Fit a single full-sample closed-form LEACE eraser for the concept E0.

    ``v0``: ``(n, d)`` features. ``E0``: ``(n,)`` continuous concept. Returns a
    :class:`LeaceEraser` whose ``.transform`` yields the residual with
    ``cov(E0, residual) ≈ 0`` along every coordinate (the closed-form guarantee,
    Belrose 2023) and orthogonal directions minimally changed.
    """
    X = np.asarray(v0, dtype=float)
    z = np.asarray(E0, dtype=float)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    zc = z - z.mean()

    sigma_xx = (Xc.T @ Xc) / n  # (d, d)
    sigma_xz = (Xc.T @ zc) / n  # (d,)

    sqrt, inv_sqrt = _symmetric_inv_sqrt(sigma_xx)
    # whitened cross-covariance direction
    w_white = inv_sqrt @ sigma_xz  # (d,)
    norm = float(np.linalg.norm(w_white))
    if norm < 1e-12:
        # no linearly-decodable concept signal -> identity eraser
        return LeaceEraser(mean_x=X.mean(axis=0), P=np.eye(d))
    w_hat = w_white / norm
    # oblique projection in the original basis: P = I - Σ^{1/2} ŵ ŵᵀ Σ^{-1/2}
    P = np.eye(d) - sqrt @ np.outer(w_hat, w_hat) @ inv_sqrt
    return LeaceEraser(mean_x=X.mean(axis=0), P=P)


def leace_residual(v0: np.ndarray, E0: np.ndarray) -> np.ndarray:
    """Full-sample-erased embedding: ``fit_leace(v0, E0).transform(v0)``."""
    return fit_leace(v0, E0).transform(v0)


# --------------------------------------------------------------------------- #
# 3. Distance correlation + permutation null (plan §4 Stage-1 steps 0/3)        #
# --------------------------------------------------------------------------- #
def _pairwise_euclidean(A: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix for ``(n, d)`` (or ``(n,)``) input."""
    Am = np.asarray(A, dtype=float)
    if Am.ndim == 1:
        Am = Am[:, None]
    sq = np.sum(Am**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Am @ Am.T)
    d2 = np.clip(d2, 0.0, None)
    return np.sqrt(d2)


def _double_center(D: np.ndarray) -> np.ndarray:
    """U-centering / double-centering of a distance matrix (Székely 2007)."""
    row = D.mean(axis=1, keepdims=True)
    col = D.mean(axis=0, keepdims=True)
    grand = D.mean()
    return D - row - col + grand


def distance_correlation(X: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation ``dCor(X, y)`` in [0, 1] (Székely, Rizzo, Bakirov 2007).

    ``dCor = 0`` iff X and y are independent; captures nonlinear dependence a
    Pearson/Spearman residual would miss. ``X``: ``(n, d)``; ``y``: ``(n,)`` or
    ``(n, p)``.
    """
    A = _double_center(_pairwise_euclidean(X))
    B = _double_center(_pairwise_euclidean(y))
    dcov2 = float((A * B).mean())
    dvarx2 = float((A * A).mean())
    dvary2 = float((B * B).mean())
    denom = np.sqrt(dvarx2 * dvary2)
    if denom < 1e-12:
        return 0.0
    dcor2 = dcov2 / denom
    return float(np.sqrt(max(0.0, dcor2)))


@dataclass
class DcorPermutationResult:
    """Result of a refit-per-permutation dCor test (plan §4 Stage-1 step 3, MF3)."""

    dcor: float
    null: np.ndarray
    p_value: float
    d_eff: int
    n_perm: int


def _pipeline_dcor(
    v0: np.ndarray,
    E0: np.ndarray,
    d_eff: int,
    pca_fit_fn: Callable[..., PCABasis],
    leace_fit_fn: Callable[..., LeaceEraser],
) -> float:
    """One pass of the full PCA→LEACE→dCor pipeline (a single commensurable frame).

    Fits a fresh PCA basis on ``v0``, reduces, fits a fresh LEACE eraser against
    ``E0`` on the reduced points, erases, and returns dCor(residual, E0). Both fits
    route through the injected hooks so a counting wrapper proves the
    refit-per-permutation contract.
    """
    basis = pca_fit_fn(v0, d_eff)
    reduced = basis.transform(v0)
    eraser = leace_fit_fn(reduced, E0)
    residual = eraser.transform(reduced)
    return distance_correlation(residual, E0)


def dcor_permutation_test(
    v0: np.ndarray,
    E0: np.ndarray,
    *,
    d_eff: int = 10,
    n_perm: int = 1000,
    rng: np.random.Generator | None = None,
    pca_fit_fn: Callable[..., PCABasis] | None = None,
    leace_fit_fn: Callable[..., LeaceEraser] | None = None,
) -> DcorPermutationResult:
    """dCor permutation test that REFITS the full PCA→LEACE→dCor pipeline per draw.

    MF3 contract (plan §13 item 3 + §10 unit test 3): the observed statistic AND
    every null draw are produced by the literally identical procedure — for each of
    ``n_perm`` permutations the ``E0`` labels are permuted, then PCA + LEACE are
    RE-FIT (with the permuted E0) and dCor recomputed on the freshly-erased points.
    So no cross-fitted / cached coordinate frame can leak in.

    ``pca_fit_fn`` / ``leace_fit_fn`` default to the real fits (:func:`fit_pca_basis`
    / :func:`fit_leace`); they are test-injection hooks so a counting stub can prove
    EACH is called exactly ``n_perm + 1`` times (once observed, once per perm).

    Returns a :class:`DcorPermutationResult` with the observed ``dcor``, the ``null``
    array, and the right-tail ``p_value``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if pca_fit_fn is None:
        pca_fit_fn = fit_pca_basis
    if leace_fit_fn is None:
        leace_fit_fn = fit_leace
    v = np.asarray(v0, dtype=float)
    z = np.asarray(E0, dtype=float)

    # observed statistic: one full pipeline fit (call #1 of n_perm+1 for each hook)
    observed = _pipeline_dcor(v, z, d_eff, pca_fit_fn, leace_fit_fn)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        z_perm = z[rng.permutation(len(z))]
        # refit the WHOLE pipeline on the permuted labels (call #(i+2) for each hook)
        null[i] = _pipeline_dcor(v, z_perm, d_eff, pca_fit_fn, leace_fit_fn)

    # right-tail p-value with the +1 finite-sample correction
    p_value = float((1.0 + np.sum(null >= observed)) / (1.0 + n_perm))
    return DcorPermutationResult(
        dcor=observed, null=null, p_value=p_value, d_eff=int(d_eff), n_perm=int(n_perm)
    )


def dcor_at_subsample(
    v0_layer: np.ndarray,
    E0: np.ndarray,
    *,
    n_prime: int,
    d_eff: int,
    rng: np.random.Generator,
) -> float:
    """dCor of the single-frame LEACE residual vs E0 on a size-``n_prime`` subsample.

    Plan §4 Stage-2 step 2: the Stage-2 learning curve needs ``dCor(n′)`` alongside
    ``ρ_lin(n′)`` and ``√(r_yy)(n′)``. Draws a without-replacement subsample of
    ``n_prime`` contexts, fits the single full-sample PCA→LEACE pipeline on JUST that
    subsample (the same commensurable-frame procedure as the full Stage-1 test, MF3),
    and returns the dCor of the erased residual against E0. ``d_eff`` is clamped to
    ``min(d_eff, n_prime − 1)`` so the PCA fit is well-posed at small ``n′``.
    """
    v = np.asarray(v0_layer, dtype=float)
    z = np.asarray(E0, dtype=float)
    n = v.shape[0]
    k = min(int(n_prime), n)
    idx = rng.choice(n, size=k, replace=False)
    vs, zs = v[idx], z[idx]
    d = int(min(d_eff, max(1, k - 1)))
    basis = fit_pca_basis(vs, d)
    reduced = basis.transform(vs)
    residual = fit_leace(reduced, zs).transform(reduced)
    return distance_correlation(residual, zs)


def _planted_nonlinear_dataset(
    *, d_eff: int, n: int, effect: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (v0, E0) with a planted nonlinear residual at ~``effect`` partial-corr.

    ``E0`` is a convex blend of a NONLINEAR function of v0 (squared-norm of the
    leading dims, standardized — its linear component is ~0) and pure noise; the
    blend weight ``effect`` sets the partial-correlation effect size at the floor
    the power check probes (plan §11 row 3 / §12 row 6).
    """
    v0 = rng.normal(0.0, 1.0, size=(n, d_eff))
    r2 = (v0[:, : max(1, d_eff // 2)] ** 2).sum(axis=1)
    r2_std = (r2 - r2.mean()) / (r2.std() + 1e-12)
    noise = rng.normal(0.0, 1.0, size=n)
    signal = 1.0 / (1.0 + np.exp(-r2_std))
    signal = (signal - signal.mean()) / (signal.std() + 1e-12)
    E0 = effect * signal + np.sqrt(max(0.0, 1.0 - effect**2)) * noise
    return v0, E0


def dcor_power_check(
    *,
    d_eff: int = 10,
    n: int = 50,
    n_perm: int = 1000,
    effect: float = 0.10,
    n_trials: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Realized power of the dCor permutation test at the stated effect floor.

    Repeats the whole experiment ``n_trials`` times: each trial draws a fresh
    synthetic ``(v0, E0)`` with a planted nonlinear residual at ``effect`` partial
    correlation, runs the dCor permutation test, and counts a detection
    (p < 0.05). Returns the detection fraction = realized power. Codifies the
    runtime power check the plan promised (plan §4 Stage-1 step 0 + §13 item 3c).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    detections = 0
    for _ in range(n_trials):
        v0, E0 = _planted_nonlinear_dataset(d_eff=d_eff, n=n, effect=effect, rng=rng)
        res = dcor_permutation_test(v0, E0, d_eff=d_eff, n_perm=n_perm, rng=rng)
        if res.p_value < 0.05:
            detections += 1
    return float(detections / n_trials)


@dataclass
class PowerSelection:
    """Result of the plan §4 Stage-1 step-0 adaptive ``d_eff`` power selection.

    ``chosen_d_eff``: the largest candidate ``d_eff`` whose realized power clears
    ``target_power`` (subject to the PCA cumulative-variance floor when a
    ``variance_retained_fn`` is supplied); falls back to the candidate with the
    HIGHEST realized power when none clears the bar.
    ``realized_power``: the realized power AT ``chosen_d_eff``.
    ``variance_limited``: True iff NO candidate cleared ``target_power`` — the
    plan's "report any null as indistinguishable-from-null given variance, never
    as no-signal" branch (§4 Stage-1 step 0a). A variance-limited verdict is the
    EXPECTED outcome of a genuine nonlinear-residual test at n=50 (Reddi 2015:
    nonparametric independence-test power drops polynomially with dimension and is
    intrinsically low at small n) — it is an honest readout, never a code failure.
    ``per_d_eff_power``: the realized power probed at each candidate (auditable).
    """

    chosen_d_eff: int
    realized_power: float
    variance_limited: bool
    target_power: float
    per_d_eff_power: dict[int, float] = field(default_factory=dict)


def select_d_eff_for_power(
    *,
    candidates: tuple[int, ...] = (10, 15, 20),
    target_power: float = 0.8,
    n: int = 50,
    n_perm: int = 1000,
    effect: float = 0.10,
    n_trials: int = 200,
    rng: np.random.Generator | None = None,
    variance_retained_fn: Callable[[int], float] | None = None,
    variance_floor: float = 0.80,
) -> PowerSelection:
    """Plan §4 Stage-1 step-0 adaptive ``d_eff`` selection from a runtime power check.

    Implements the plan's verbatim contract (§4 Stage-1 step 0, §11 row 3): run the
    synthetic dCor power check at each candidate ``d_eff`` (skipping any candidate
    whose retained PCA variance would drop below ``variance_floor``, when a
    ``variance_retained_fn`` is supplied), and pick the LARGEST candidate whose
    realized power clears ``target_power``. If NONE clears it, return the
    highest-power candidate with ``variance_limited=True`` — the plan's mandated
    "report the null as indistinguishable-from-null given variance" branch, NEVER a
    silent no-signal claim.

    This is the function the §13 item-3c power check codifies: a variance-limited
    verdict at the ~0.10 partial-correlation floor is the HONEST, expected outcome
    at n=50 (dCor-after-LEACE has intrinsically low power for a nonlinear residual
    at this sample size), and the chosen ``d_eff`` + realized power + the
    variance-limited flag are what land in ``stage1_leace_dcor.json``.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    per: dict[int, float] = {}
    eligible: list[int] = []
    for d in candidates:
        if variance_retained_fn is not None and variance_retained_fn(d) < variance_floor:
            continue  # candidate violates the PCA cumulative-variance floor
        eligible.append(d)
        per[d] = dcor_power_check(
            d_eff=d, n=n, n_perm=n_perm, effect=effect, n_trials=n_trials, rng=rng
        )
    if not eligible:  # variance floor excluded everything -> probe the smallest candidate
        d = min(candidates)
        eligible.append(d)
        per[d] = dcor_power_check(
            d_eff=d, n=n, n_perm=n_perm, effect=effect, n_trials=n_trials, rng=rng
        )
    clearing = [d for d in eligible if per[d] >= target_power]
    if clearing:
        chosen = max(clearing)  # largest d_eff that still clears the power floor
        return PowerSelection(
            chosen_d_eff=int(chosen),
            realized_power=float(per[chosen]),
            variance_limited=False,
            target_power=float(target_power),
            per_d_eff_power=per,
        )
    # none clears the bar -> variance-limited; surface the highest-power candidate
    best = max(eligible, key=lambda d: per[d])
    return PowerSelection(
        chosen_d_eff=int(best),
        realized_power=float(per[best]),
        variance_limited=True,
        target_power=float(target_power),
        per_d_eff_power=per,
    )


# --------------------------------------------------------------------------- #
# 4. Cluster bootstrap over contexts (plan §4 Stage-0 step 4 + §11 row 6)       #
# --------------------------------------------------------------------------- #
def cluster_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    data: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile cluster-bootstrap CI over the rows (clusters) of ``data``.

    Resamples the ROWS of ``data`` (each row = one context/cluster) WITH replacement
    ``n_boot`` times, applies ``stat_fn`` to each resample, and returns the
    ``(alpha/2, 1−alpha/2)`` percentile interval. Bootstrapping over contexts is the
    honest CV-variance estimate — NEVER std-across-folds (Bengio-Grandvalet 2004 +
    Varoquaux 2018).
    """
    if rng is None:
        rng = np.random.default_rng(742)
    arr = np.asarray(data)
    n = arr.shape[0]
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(arr[idx])
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(finite, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


# --------------------------------------------------------------------------- #
# 5. ρ_lin source-path loader (plan §4 Stage-0 step 5 + §12 row 4)              #
# --------------------------------------------------------------------------- #
def load_rho_lin(behavior: str, genre: str, *, eval_dir: str | Path) -> float:
    """Read #658's ridge ``ρ_lin`` from ``analyzer_body_data.json``.

    Reads ``/<genre-key>/a33/<behavior>/lin_rho`` where the genre key maps
    ``"betley"→"betley"`` and ``"ultrachat"/"g1"→"g1"``. This is the a33 RIDGE
    ``lin_rho`` — NEVER the A3.2 MLP ``best_rho`` in ``assumption_verdicts.json``
    (a different quantity that would silently substitute, §12 row 4).

    Raises (KeyError / FileNotFoundError) when the a33/lin_rho key is absent — a
    clear error, never a silent substitution (fail-loud, CLAUDE.md Critical Rules).
    """
    eval_path = Path(eval_dir)
    abd = json.loads((eval_path / "analyzer_body_data.json").read_text())
    a33_key = _GENRE_TO_A33_KEY.get(genre)
    if a33_key is None:
        raise KeyError(f"unknown genre {genre!r}; expected one of {sorted(_GENRE_TO_A33_KEY)}")
    try:
        return float(abd[a33_key]["a33"][behavior]["lin_rho"])
    except (KeyError, TypeError) as e:
        raise KeyError(
            f"a33 lin_rho for ({behavior!r}, {genre!r}) not present at "
            f"/{a33_key}/a33/{behavior}/lin_rho in {eval_path / 'analyzer_body_data.json'} "
            "— refusing to silently substitute assumption_verdicts.json best_rho (the A3.2 "
            "MLP quantity)"
        ) from e


def load_a33_layer(behavior: str, genre: str, *, eval_dir: str | Path) -> int:
    """Read #658's A3.3 best layer from ``analyzer_body_data.json`` ``/<genre>/a33/<beh>/layer``.

    The A3.3 read-out layers are PER-BEHAVIOR and live ONLY here — NEVER in
    ``locked_recipe.json`` (which has no ``per_behavior`` key, so a Stage-1 layer
    read from it silently falls back to a wrong default, §12 trap). Verified this
    session: Betley sycophancy → 27, refusal → 8, broad_em → 0, harmful_compliance
    → 8; UltraChat (g1) refusal → 6, sycophancy → 26, etc. Raises (no silent
    fallback) when the key is absent.
    """
    eval_path = Path(eval_dir)
    abd = json.loads((eval_path / "analyzer_body_data.json").read_text())
    a33_key = _GENRE_TO_A33_KEY.get(genre)
    if a33_key is None:
        raise KeyError(f"unknown genre {genre!r}; expected one of {sorted(_GENRE_TO_A33_KEY)}")
    try:
        return int(abd[a33_key]["a33"][behavior]["layer"])
    except (KeyError, TypeError) as e:
        raise KeyError(
            f"a33 layer for ({behavior!r}, {genre!r}) not present at "
            f"/{a33_key}/a33/{behavior}/layer in {eval_path / 'analyzer_body_data.json'} "
            "— refusing to fall back to locked_recipe.json (no per_behavior key)"
        ) from e


# --------------------------------------------------------------------------- #
# 5b. LOCO nested-CV ridge re-fit (plan §4 Stage-0 step 0c — join-integrity)    #
# --------------------------------------------------------------------------- #
RIDGE_LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)  # #658's a33 nested-CV grid


@dataclass
class RidgeRefitResult:
    """Result of the Stage-0 step-0c LOCO nested-CV ridge re-fit (join-integrity).

    ``refit_rho``: the held-out LOCO Spearman ρ of the re-fit ridge
    ``v0(C)[layer] → E0(C, behavior)``. ``persisted_rho``: #658's persisted a33
    ``lin_rho`` for the cell. ``delta``: ``|refit_rho − persisted_rho|``.
    ``join_ok``: True iff ``delta <= tol`` — when False the inputs are mis-joined
    (a wrong genre tensor / a Betley↔UltraChat swap) and the bracket interpretation
    for that genre must BLOCK (a join-integrity REVISE, plan §4 step 0c).
    """

    behavior: str
    genre: str
    layer: int
    refit_rho: float
    persisted_rho: float
    delta: float
    join_ok: bool
    tol: float
    lambdas: tuple[float, ...]


def loco_ridge_refit_rho(
    v0_layer: np.ndarray,
    E0: np.ndarray,
    *,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
) -> float:
    """Held-out LOCO Spearman ρ of a nested-CV ridge ``v0_layer → E0``.

    For each held-out context, pick λ minimizing inner leave-one-out PRESS MSE on the
    training contexts (no λ leakage into the held-out read), fit the ridge on the
    standardized training design, predict the held-out point, then Spearman-correlate
    the N held-out predictions against the measured ``E0``. Reuses #658's exact
    closed-form dual-ridge LOCO (``issue658_fit_predictors._ridge_predict_loco``) so
    the re-fit reproduces #658's a33 ``lin_rho`` within numerical tolerance (the
    join-integrity contract, plan §4 Stage-0 step 0c).

    ``v0_layer``: ``(n_contexts, d)`` single-layer features. ``E0``: ``(n_contexts,)``.
    Returns the held-out Spearman ρ in [-1, 1] (0.0 on a degenerate prediction).
    """
    import importlib

    from scipy.stats import spearmanr

    X = np.asarray(v0_layer, dtype=float)
    y = np.asarray(E0, dtype=float).reshape(-1, 1)  # (n, 1) single-output target
    # reuse #658's exact closed-form dual-ridge LOCO (bit-equivalent to the
    # primal refit; standardizes per fold, nested-CV λ via PRESS)
    fp = importlib.import_module("issue658_fit_predictors")
    preds = fp._ridge_predict_loco(X, y, list(lambdas))  # (n, 1)
    pred_vec = np.asarray(preds).reshape(-1)
    if np.std(pred_vec) < 1e-12 or np.std(y.reshape(-1)) < 1e-12:
        return 0.0
    rho = spearmanr(pred_vec, y.reshape(-1)).correlation
    return 0.0 if (rho is None or np.isnan(rho)) else float(rho)


def ridge_join_integrity(
    v0_layer: np.ndarray,
    E0: np.ndarray,
    *,
    behavior: str,
    genre: str,
    layer: int,
    persisted_rho: float,
    tol: float = 0.05,
    lambdas: tuple[float, ...] = RIDGE_LAMBDAS,
) -> RidgeRefitResult:
    """Re-fit the LOCO ridge and RECORD its delta vs #658's persisted ``lin_rho``.

    v8 [REPLAN] — plan §4 Stage-0 step 0c + §11 row 17 + §12 rows 13/14: this is a
    RECORDED DIAGNOSTIC, NOT a gate. It re-fits a LOCO-CV ridge and reports the delta
    between the re-fit held-out ρ and #658's persisted a33 ``lin_rho`` — which is a
    diff-of-means ``r_B`` PROJECTION ρ (``a33_cells.json`` ``rb_recipe: "diffmeans"``),
    a DIFFERENT estimator from a regularized cross-validated ridge, so the two
    legitimately diverge (the sycophancy ridge≈0.65 vs projection 0.1268 gap IS the §7
    chat-re-analysis finding). The orchestration REPORTS this delta in
    ``stage0_brackets.json`` and NEVER raises on it — the v7 ``|refit − projection| ≤
    tol → raise`` gate was unsatisfiable by construction and is removed.

    The actual join-integrity gate (catching a Betley↔UltraChat tensor mis-join) is the
    DETERMINISTIC per-genre ``probe_pool_hash`` assert in :func:`load_inputs` (0a): a
    swap loads the wrong genre's tensor whose hash will not match the expected per-genre
    value, so the assert fires directly on the swap.

    Returns a :class:`RidgeRefitResult` carrying ``join_ok = (delta <= tol)`` as a
    REPORTED flag only — no caller raises on it.
    """
    refit = loco_ridge_refit_rho(v0_layer, E0, lambdas=lambdas)
    delta = abs(float(refit) - float(persisted_rho))
    return RidgeRefitResult(
        behavior=behavior,
        genre=genre,
        layer=int(layer),
        refit_rho=float(refit),
        persisted_rho=float(persisted_rho),
        delta=float(delta),
        join_ok=bool(delta <= tol),
        tol=float(tol),
        lambdas=tuple(lambdas),
    )


# --------------------------------------------------------------------------- #
# 5c. CV-matched reliability CI (plan §4 Stage-0 step 3 — fold-matched to LOCO) #
# --------------------------------------------------------------------------- #
def cv_matched_reliability_ci(
    rates: np.ndarray,
    m_cell: np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Fold-matched ``√(r_yy)`` CI: one estimate per LOCO fold, held-out ctx excluded.

    Plan §4 Stage-0 step 3 (Storrs 2020): the ceiling MUST be CV-matched to the LOCO
    folds that produced ``ρ_lin`` — for each held-out context the reliability is
    re-estimated on the OTHER n−1 contexts (the held-out context excluded), exactly
    as that context is excluded from its LOCO ridge fit. NEVER a pooled bootstrap
    against a CV'd ρ. Returns ``(mean, lo, hi)`` where ``mean`` is the
    fold-averaged ``√(r_yy)`` and ``(lo, hi)`` is the ``(alpha/2, 1−alpha/2)``
    percentile interval ACROSS the n fold estimates (the fold-to-fold spread is the
    honest CV variance, Bengio-Grandvalet 2004).

    ``rates``: ``(n_contexts,)`` per-context E0. ``m_cell``: ``(n_contexts,)`` cell-
    actual n_judged (the binomial m, NEVER a blanket 2000).
    """
    r = np.asarray(rates, dtype=float)
    m = np.asarray(m_cell, dtype=float)
    n = r.shape[0]
    if m.shape != r.shape:
        raise ValueError(f"m_cell shape {m.shape} != rates shape {r.shape}")
    fold_vals = np.empty(n, dtype=float)
    for i in range(n):
        keep = np.arange(n) != i  # exclude held-out context i (LOCO-matched)
        fold_vals[i] = float(np.sqrt(reliability_binomial_variance(r[keep], m[keep])))
    mean = float(np.mean(fold_vals))
    lo = float(np.percentile(fold_vals, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(fold_vals, 100.0 * (1.0 - alpha / 2.0)))
    return mean, lo, hi


# --------------------------------------------------------------------------- #
# 6. Stage-0 raw-completion source resolution (plan §4 Stage-0 step 2a, MF2)    #
# --------------------------------------------------------------------------- #
class RawCompletionShortfallError(RuntimeError):
    """Raised before the judge batch when a cache cell has < J completions.

    Names the offending ``(context_id, behavior, count)`` triple so the shortfall
    is diagnosable without paging the raw-completion files into context.
    """


# The per-genre HF raw-completion source paths (plan §4 Stage-0 step 2, MF2).
_GENRE_RAW_COMPLETION_PREFIX = {
    "betley": "issue658_theory_assumptions/raw_completions/e0_gen",
    "ultrachat": (
        "issue658_theory_assumptions/raw_completions_genre-generalization-ultrachat/e0_gen"
    ),
}
# The canonical #658 context ids are the HF raw-completion filename context tokens:
# filename = f"{prefix}/{context_id}__{behavior}.json". The 50 ids span several
# families (f1_house_*, f1_phub_*, f2_wc_*, f3_icl_*, f4_reph_*, f5_fmt_*, f6_*,
# f8_*) — they are NOT a 6-persona house list. The default below is the f1_house
# sextet present on HF (verified this session: data_scientist, librarian,
# medical_doctor, programmer, software_engineer, surgeon), used ONLY when no
# explicit ``context_ids`` is supplied; production passes the genre's full 50-id
# list from ``load_inputs`` so the snapshot covers every read-out cell.
_DEFAULT_F1_HOUSE_CONTEXT_IDS = (
    "f1_house_data_scientist",
    "f1_house_librarian",
    "f1_house_medical_doctor",
    "f1_house_programmer",
    "f1_house_software_engineer",
    "f1_house_surgeon",
)


@dataclass
class RawCompletionRecord:
    """Provenance record for one snapshotted raw-completion cell."""

    context_id: str
    behavior: str
    n_completions: int
    snapshot_path: str
    sha256: str


def _count_completions(obj: dict) -> int:
    """Total completions in a #658 raw-completion file, across both on-disk schemas.

    The REAL #658 ``e0_gen`` file nests completions under ``cells`` —
    ``{context_id, dv, n_samples, cells: [{probe, completions: [...]}, ...]}`` — so the
    cell total is ``sum(len(cell["completions"]))`` over the probe cells. The flat test
    fixture shape ``{context_id, behavior, completions: [...]}`` is also supported (one
    flat list). Returns the total completion count under whichever schema is present.
    """
    if isinstance(obj.get("cells"), list):
        return sum(len(c.get("completions", [])) for c in obj["cells"] if isinstance(c, dict))
    return len(obj.get("completions", []))


def snapshot_raw_completions(
    genre: str,
    *,
    dest_dir: str | Path,
    hf_download_fn: Callable[..., str],
    rerun_probe_set_size: int = 20,
    context_ids: tuple[str, ...] | list[str] | None = None,
    behaviors: tuple[str, ...] = READOUT_BEHAVIORS,
) -> list[RawCompletionRecord]:
    """Snapshot HF-resolved raw completions into an issue-owned dir with sha256s.

    For each (context_id, behavior) cell of ``genre``, resolves the per-genre HF path
    ``{prefix}/{context_id}__{behavior}.json`` via ``hf_download_fn`` (a per-file
    ``hf_hub_download``-shaped callable taking ``repo_id`` / ``filename`` /
    ``repo_type``), copies it under ``dest_dir`` with a recorded sha256, and FAILS LOUD
    before any judge batch on a short / missing cell (plan §4 Stage-0 step 2a). Returns
    the per-cell provenance manifest.

    ``context_ids`` defaults to the f1_house sextet (back-compat); production passes the
    genre's full 50-id list from :func:`load_inputs` so every read-out cell is covered.

    Raises :class:`RawCompletionShortfallError` on a cell with fewer than
    ``rerun_probe_set_size`` completions, naming the offending triple; a missing
    cell (``hf_download_fn`` raises) propagates loud — never a silent skip.
    """
    if genre not in _GENRE_RAW_COMPLETION_PREFIX:
        raise KeyError(
            f"unknown genre {genre!r}; expected one of {sorted(_GENRE_RAW_COMPLETION_PREFIX)}"
        )
    prefix = _GENRE_RAW_COMPLETION_PREFIX[genre]
    ctx_ids = tuple(context_ids) if context_ids is not None else _DEFAULT_F1_HOUSE_CONTEXT_IDS
    dest = Path(dest_dir) / genre
    dest.mkdir(parents=True, exist_ok=True)

    manifest: list[RawCompletionRecord] = []
    for ctx in ctx_ids:
        for behavior in behaviors:
            filename = f"{prefix}/{ctx}__{behavior}.json"
            # hf_download_fn raises on a missing cell -> propagates loud (no skip)
            src = hf_download_fn(
                repo_id="superkaiba1/explore-persona-space-data",
                filename=filename,
                repo_type="dataset",
            )
            obj = json.loads(Path(src).read_text())
            context_id = obj.get("context_id", ctx)
            cell_behavior = obj.get("behavior") or obj.get("column_id") or behavior
            n_completions = _count_completions(obj)
            if n_completions < rerun_probe_set_size:
                raise RawCompletionShortfallError(
                    f"raw-completion shortfall for context_id={context_id!r} "
                    f"behavior={cell_behavior!r}: {n_completions} completions "
                    f"< required {rerun_probe_set_size} (genre={genre!r}, file={filename!r})"
                )
            snap_path = dest / f"{ctx}__{behavior}.json"
            shutil.copyfile(src, snap_path)
            sha = hashlib.sha256(snap_path.read_bytes()).hexdigest()
            manifest.append(
                RawCompletionRecord(
                    context_id=context_id,
                    behavior=cell_behavior,
                    n_completions=n_completions,
                    snapshot_path=str(snap_path),
                    sha256=sha,
                )
            )
    return manifest


# --------------------------------------------------------------------------- #
# 6b. Per-behavior judge rate (plan §4 Stage-0 step 2 — the CORRECT construct)  #
# --------------------------------------------------------------------------- #
def sample_completions_for_judge(
    obj: dict,
    *,
    j_completions: int,
    seed: int,
) -> dict:
    """Deterministically sample exactly ``j_completions`` completions across a cell.

    Plan §11 row 8 / §9: the judge rerun re-judges a FIXED sample of ``J``
    completions per (context, behavior) cell, NOT all of them — sending every
    completion balloons the batch far past the registered ~16k calls. This pools
    every (probe × rollout) completion in the #658 ``cells`` schema, samples exactly
    ``min(J, total)`` of them WITHOUT replacement under a seeded RNG (the 742X
    family, reproducible), and returns a NEW gen-shaped dict with one synthetic
    ``cells`` entry carrying the sampled completions — so the downstream per-behavior
    judge (:func:`per_behavior_judge_rate`) reconstructs the rate over EXACTLY ``J``
    completions. Completion TEXT is held in the returned dict, never logged.

    Returns a ``{context_id, column_id, dv, cells: [{probe, completions}]}`` dict.
    """
    rng = np.random.default_rng(seed)
    cells = obj.get("cells")
    flat: list[dict] = []
    if isinstance(cells, list):
        for cell in cells:
            probe = cell.get("probe", "")
            for c in cell.get("completions", []):
                if isinstance(c, dict):
                    flat.append({"probe": probe, **c})
                else:
                    flat.append({"probe": probe, "text": str(c), "logp_norm": 0.0})
    else:  # flat fixture shape {completions: [...]}
        for c in obj.get("completions", []):
            if isinstance(c, dict):
                flat.append({"probe": c.get("probe", ""), **c})
            else:
                flat.append({"probe": "", "text": str(c), "logp_norm": 0.0})
    total = len(flat)
    k = min(int(j_completions), total)
    if total == 0:
        sampled: list[dict] = []
    else:
        idx = rng.choice(total, size=k, replace=False)
        sampled = [flat[i] for i in sorted(idx)]
    # rebuild a gen-shaped dict; each sampled completion becomes its own 1-completion
    # cell so judge_column flattens them back to exactly k completions.
    new_cells = [
        {
            "probe": s.get("probe", ""),
            "completions": [{"text": s["text"], "logp_norm": s.get("logp_norm", 0.0)}],
        }
        for s in sampled
    ]
    return {
        "context_id": obj.get("context_id", ""),
        "column_id": obj.get("column_id", obj.get("behavior", "")),
        "dv": "judged_rate",
        "n_sampled_for_judge": k,
        "cells": new_cells,
    }


def per_behavior_judge_rate(
    gen: dict,
    *,
    behavior: str,
    judge_model: str,
    judge_fn: Callable[..., dict] | None = None,
) -> dict:
    """Judge a cell's completions with the CORRECT PER-BEHAVIOR rubric (plan §4 step 2).

    BLOCKER-FIX (judge-rerun-wrong-judge-construct): #658 judged each read-out
    behavior with its OWN binary rubric — ``E0_COLUMNS[behavior].judge_prompt`` +
    ``_verdict_truthy(verdict, col.e0_verdict_key, behavior)`` — NOT the default
    alignment judge (``mean_aligned``). This reuses #658's own
    ``issue658_judge_e0.judge_column``, which (a) reads ``comp["text"]`` from the
    cells schema (BLOCKER-FIX judge-rerun-completion-key-crash), (b) applies the
    per-behavior rubric (broad_em→misaligned, harmful_compliance→complied,
    sycophancy→agreed, refusal→refused), and (c) reconstructs the judge-positive
    rate. So ``Var_judge`` is measured on the SAME construct #658's E0 rate used.

    ``judge_fn`` defaults to ``issue658_judge_e0.judge_column`` (the real per-behavior
    judge); it is a test-injection hook (signature ``(col_id, gen, model) -> dict``)
    so a counting stub can prove the J-sampling + per-behavior construct without an
    API call. Returns the ``judge_column`` dict ``{rate, n_judged, n_positive, ...}``.
    """
    import importlib

    if behavior not in READOUT_BEHAVIORS:
        raise KeyError(f"per_behavior_judge_rate only for read-out behaviors, got {behavior!r}")
    if judge_fn is None:
        judge_e0 = importlib.import_module("issue658_judge_e0")
        judge_fn = judge_e0.judge_column
    return judge_fn(behavior, gen, judge_model)


# --------------------------------------------------------------------------- #
# 7. Held-out post-LEACE linear-leakage diagnostic (plan §13 NEW, Phase-2 REVISE)#
# --------------------------------------------------------------------------- #
@dataclass
class HeldOutLeakageResult:
    """Held-out post-LEACE linear-leakage diagnostic result.

    ``rho``: held-out ridge-probe correlation of E0 on the LEACE residual.
    ``null_ci``: the held-out permutation-null CI for that rho.
    ``post_leace_linear_pass``: True iff |rho| sits within the null CI (no residual
    linear leakage) — the Stage-1 verdict must NOT read 'nonlinear-yes' when this is
    False (the alternatives critic Must-fix: leftover linear leakage masquerading as
    a nonlinear residual).
    """

    rho: float
    null_ci: tuple[float, float]
    post_leace_linear_pass: bool


def _linear_probe_rho(features: np.ndarray, target: np.ndarray) -> float:
    """Best-fit linear-probe correlation of ``target`` on ``features`` (with bias)."""
    F = np.asarray(features, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    y = np.asarray(target, dtype=float)
    design = np.column_stack([np.ones(len(y)), F])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ beta
    if np.std(y_hat) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(y_hat, y)[0, 1])


def held_out_linear_leakage(
    v0_train: np.ndarray,
    E0_train: np.ndarray,
    v0_held: np.ndarray,
    E0_held: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    n_perm: int = 1000,
) -> HeldOutLeakageResult:
    """Diagnose residual LINEAR leakage in the held-out LEACE residual.

    LEACE's closed-form guarantee is on the TRAIN fold only (Belrose 2023). On a
    held-out split the residual can still carry linear E0-correlation from sampling
    variance, which at d_eff=10 / n=50 would masquerade as a nonlinear residual in
    the dCor test. This fits LEACE on (v0_train, E0_train), applies the train eraser
    to v0_held, probes the held-out residual for residual linear correlation with
    E0_held, and compares against a held-out permutation null.

    ``post_leace_linear_pass`` is True iff the held-out rho sits within the null CI.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    eraser = fit_leace(v0_train, E0_train)
    resid_held = eraser.transform(v0_held)
    rho = _linear_probe_rho(resid_held, E0_held)

    z = np.asarray(E0_held, dtype=float)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = _linear_probe_rho(resid_held, z[rng.permutation(len(z))])
    lo = float(np.percentile(null, 2.5))
    hi = float(np.percentile(null, 97.5))
    passed = bool(lo <= rho <= hi)
    return HeldOutLeakageResult(rho=rho, null_ci=(lo, hi), post_leace_linear_pass=passed)


def classify_stage1_verdict(*, dcor_pass: bool, linear_pass: bool) -> str:
    """Stage-1 verdict enum from (dcor_pass, linear_pass) (plan §13 Phase-2 REVISE).

    * dCor pass + linear pass → ``"nonlinear-yes"`` (genuine nonlinear residual).
    * dCor pass + linear FAIL → ``"linear-erasure-leakage-unresolved"`` (the
      apparent residual is unresolved leftover linear leakage, not nonlinearity).
    * dCor null (either linear verdict) → ``"ceiling-limited"`` (no residual the
      test can resolve at this n).
    """
    if not dcor_pass:
        return "ceiling-limited"
    return "nonlinear-yes" if linear_pass else "linear-erasure-leakage-unresolved"


# --------------------------------------------------------------------------- #
# 8. Genre-aware input loader (plan §4 Stage-0 step 0a + §10 inputs)            #
# --------------------------------------------------------------------------- #
# Per-genre v0 tensor source paths + expected probe_pool_hash (MF1).
GENRE_V0_PATHS = {
    "betley": "data/issue_658/store/v0_summaries.pt",
    "ultrachat": (
        "data/issue_658/g1_dl/issue658_theory_assumptions/"
        "store_genre-generalization-ultrachat/v0_summaries.pt"
    ),
}
GENRE_E0_PATHS = {
    "betley": "eval_results/issue_658/E0_expression.json",
    "ultrachat": "eval_results/issue_658/E0_expression_g1.json",
}
GENRE_EXPECTED_PROBE_POOL_HASH = {
    "betley": "ad687becec266286549aaaa1af3b35e246d593e012e233564e58ff75fb015dd7",
    "ultrachat": "f277f8c3e2550b2ce3e4545a8ad6473498d070e7343eb7c9398a6aac31525455",
}


@dataclass
class GenreInputs:
    """Structured per-genre inputs for the issue-742 analysis.

    ``v0_dict``: the raw dict torch.load returns (``summaries`` + ``context_ids`` +
    ``probe_pool_hash`` + ...). ``E0_per_behavior``: ``e0[context][behavior]`` JSON
    dict. ``lin_rho_per_behavior``: the reproduced/persisted a33 ridge ρ_lin per
    read-out behavior (from ``analyzer_body_data.json``, NOT assumption_verdicts).
    """

    genre: str
    v0_dict: dict[str, Any]
    E0_per_behavior: dict[str, Any]
    lin_rho_per_behavior: dict[str, float]
    context_ids: list[str] = field(default_factory=list)


def stack_v0(v0_dict: dict[str, Any], recipe: str = "last") -> np.ndarray:
    """Stack a genre's per-context v0 summaries into ``(n_contexts, n_layers, d)``.

    ``v0_dict['summaries'][recipe]`` is a ``dict[context_id → Tensor(28, 3584)]``
    (NOT a stacked tensor); stack over ``context_ids`` in canonical order. Returns a
    float numpy array. Never index ``summaries[recipe].shape`` (it is a dict).
    """
    import torch

    summaries = v0_dict["summaries"][recipe]
    context_ids = list(v0_dict["context_ids"])
    stacked = torch.stack([summaries[c] for c in context_ids])  # (n, n_layers, d)
    return stacked.to(torch.float32).cpu().numpy()


def load_inputs(genre: str, *, repo_root: str | Path | None = None) -> GenreInputs:
    """Load the per-genre v0 / E0 / ρ_lin inputs for one genre.

    Reads the genre-specific v0 tensor (asserting its ``probe_pool_hash`` matches the
    expected per-genre value, MF1), the genre's E0 JSON, and the a33 ridge ρ_lin per
    read-out behavior from ``analyzer_body_data.json`` (NEVER assumption_verdicts.json,
    the §12-row-4 trap). ``repo_root`` defaults to the project repo root.
    """
    import torch

    if repo_root is None:
        from explore_persona_space.task_workflow import repo_root as _rr

        root = _rr()
    else:
        root = Path(repo_root)
    if genre not in GENRE_V0_PATHS:
        raise KeyError(f"unknown genre {genre!r}; expected one of {sorted(GENRE_V0_PATHS)}")

    v0_dict = torch.load(root / GENRE_V0_PATHS[genre], weights_only=False)
    expected_hash = GENRE_EXPECTED_PROBE_POOL_HASH[genre]
    actual_hash = v0_dict.get("probe_pool_hash")
    if actual_hash != expected_hash:
        raise ValueError(
            f"probe_pool_hash mismatch for genre {genre!r}: expected {expected_hash} "
            f"got {actual_hash} — a Betley↔UltraChat tensor swap (MF1)"
        )

    e0 = json.loads((root / GENRE_E0_PATHS[genre]).read_text())
    eval_dir = root / "eval_results" / "issue_658"
    lin_rho = {beh: load_rho_lin(beh, genre, eval_dir=eval_dir) for beh in READOUT_BEHAVIORS}

    return GenreInputs(
        genre=genre,
        v0_dict=v0_dict,
        E0_per_behavior=e0.get("e0", {}),
        lin_rho_per_behavior=lin_rho,
        context_ids=list(v0_dict["context_ids"]),
    )


def snapshot_inputs(genre: str, *, repo_root: str | Path | None = None) -> dict[str, Any]:
    """Snapshot a genre's v0 tensor into ``data/issue_742/inputs/`` with sha256.

    Copies the genre's ``v0_summaries.pt`` to
    ``data/issue_742/inputs/v0_summaries_<genre>.pt`` and records the per-genre
    sha256 + expected/actual probe_pool_hash in
    ``data/issue_742/inputs/provenance.json`` (plan §4 step 0a, content-identity pin
    so a mid-run cache reap cannot strand the run). Returns the provenance record.
    """
    if repo_root is None:
        from explore_persona_space.task_workflow import repo_root as _rr

        root = _rr()
    else:
        root = Path(repo_root)
    if genre not in GENRE_V0_PATHS:
        raise KeyError(f"unknown genre {genre!r}; expected one of {sorted(GENRE_V0_PATHS)}")

    src = root / GENRE_V0_PATHS[genre]
    inputs_dir = root / "data" / "issue_742" / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    dest = inputs_dir / f"v0_summaries_{genre}.pt"
    shutil.copyfile(src, dest)
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()

    prov_path = inputs_dir / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    rec = {
        "genre": genre,
        "source_path": str(src),
        "snapshot_path": str(dest),
        "sha256": sha,
        "expected_probe_pool_hash": GENRE_EXPECTED_PROBE_POOL_HASH[genre],
    }
    prov[genre] = rec
    prov_path.write_text(json.dumps(prov, indent=2))
    return rec
