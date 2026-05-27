"""Multi-seed aggregator: per-factor matched-pair Δ, Page's L, Kendall-τ (task #397).

Plan v4 §5.9 + §7 are authoritative. This module covers:

- ``per_factor_matched_pair_delta_multiseed`` — per-seed widest-of-three
  (per-pair / source-cluster / source-FE) bootstrap THEN across-seed bootstrap
  on the 3 seed-level point estimates.
- ``pages_l_test`` — Page's L one-tailed over K=3 ordered conditions × n=108
  blocks (with normal approximation; n_blocks ≫ 12). Operationalizes H2.
- ``h1_sign_and_ordering`` — per-factor sign agreement with #383 + Kendall-τ
  between the v397 per-factor Δ vector and the #383 per-factor Δ vector.
  Operationalizes H1 (reframed in v4 to sign/ordering invariance).

Phase 1 (TDD): stubs raise ``NotImplementedError`` on call. Phase 2 wires
the real implementations after user approves the proposed tests via
``epm:approve-tests v1``.
"""

from __future__ import annotations

from typing import Any

# Plan v4 §1 + §2 reference vector — #383's per-factor selectivity Δ on the
# E0+E2-restricted canonical estimator. Used as the comparison vector for
# Kendall-τ in H1.
H383_FACTOR_DELTAS: dict[str, float] = {
    "A": +33.6,  # long-system → selectively positive
    "B": +27.8,  # long-answer → selectively positive
    "C": -26.9,  # neutral framing vs persona → selectively negative
    "D": +11.2,  # Claude data → selectively positive (borderline)
}

# Sign predictions from #383 directionality (plan v4 §1 H1).
H383_FACTOR_SIGNS: dict[str, int] = {
    "A": +1,
    "B": +1,
    "C": -1,
    "D": +1,  # borderline; no requirement
}


def per_factor_matched_pair_delta_multiseed(
    records_by_seed: dict[int, list[dict[str, Any]]],
    factor: str,
    metric: str,
    e_subset: tuple[int, ...] | None = None,
    n_bootstrap: int = 1000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Per-factor matched-pair Δ pooled across seeds with nested CI construction.

    Plan v4 §5.9:
      - per-seed view: each seed's per-factor matched-pair Δ + within-seed
        widest-of-three CI (per-pair / source-cluster / source-FE);
      - across-seed view: 3-seed mean Δ + across-seed bootstrap CI on the mean
        treating each seed as one cluster.

    Returns a dict with keys ``per_seed`` (mapping seed → {point, ci_lo, ci_hi,
    n_pairs}), ``across_seed`` ({point, ci_lo, ci_hi}), ``n_pairs_per_seed``,
    and ``total_runs``.

    Phase 1 (TDD) stub.
    """
    raise NotImplementedError("per_factor_matched_pair_delta_multiseed is a Phase 1 (TDD) stub.")


def pages_l_test(
    blocks: list[list[float]],
    alternative: str = "increasing",
) -> dict[str, Any]:
    """Page's L trend test over K ordered conditions × n blocks.

    Plan v4 §5.9 + §7 (H2):
      - ``blocks`` is a list of length-K inner lists; each inner list is one
        block's K values in ordered-condition order.
      - ``alternative`` is ``"increasing"`` (the H2 primary direction:
        E0 < E1 < E2 on selectivity Δ) or ``"decreasing"`` (kill-criterion
        check). The two-sided form is computed by symmetry.

    Page's L = Σ_k k · R_k where R_k is the sum across blocks of the within-
    block rank of the k-th condition. Under H0 of no trend, L is
    asymptotically normal with mean n·K·(K+1)²/4 and variance
    n·K²·(K-1)·(K+1)²/144. We return the L statistic, the expected L under
    H0, the variance, the z-score, the one-tailed p-value in the requested
    direction, n_blocks, and the alternative.

    The test surface uses synthetic monotonic/random/reversed blocks to
    verify p << 0.05 (monotonic↑ with ``alternative="increasing"``),
    p ≈ 0.5 (random/flat with either alternative), and p ≫ 0.05
    (reversed with ``alternative="increasing"``).

    Phase 1 (TDD) stub.
    """
    raise NotImplementedError("pages_l_test is a Phase 1 (TDD) stub.")


def kendall_tau(
    v1: list[float],
    v2: list[float],
) -> float:
    """Kendall-τ rank correlation between two equal-length numeric vectors.

    For two 4-element vectors there are C(4, 2) = 6 unordered pairs. τ = 1 if
    every pair has matching relative order, τ = -1 if every pair is inverted.
    Plan v4 §1 + §13 use τ ≥ +0.67 as the H1 ordering-invariance pass
    threshold (at most 1 of 6 inversions across A/B/C/D).

    Phase 1 (TDD) stub.
    """
    raise NotImplementedError("kendall_tau is a Phase 1 (TDD) stub.")


def h1_sign_and_ordering(
    factor_deltas_v397: dict[str, float],
    factor_cis_v397: dict[str, tuple[float, float]] | None = None,
    factor_deltas_v383: dict[str, float] | None = None,
    tau_threshold: float = 0.67,
) -> dict[str, Any]:
    """Combined H1 sign-and-ordering test (plan v4 §1 reframe).

    Pass = (sign matches #383 for A, B, C; D no CI requirement)
           AND Kendall-τ between v397 and v383 per-factor Δ vectors ≥ ``tau_threshold``.

    Returns a dict with keys:
      - ``per_factor_sign_match``: ``{factor: bool}`` for A, B, C, D.
      - ``kendall_tau``: τ between v397 and v383 per-factor Δ vectors.
      - ``h1_pass``: overall PASS / FAIL boolean.

    ``factor_cis_v397`` is reserved for the strict CI-strictly-above/below-zero
    check on A, B, C; if None, the function returns the sign of the point
    estimate only (used by the synthetic test surface; Phase 2 wiring threads
    the real CIs through).

    Phase 1 (TDD) stub.
    """
    raise NotImplementedError("h1_sign_and_ordering is a Phase 1 (TDD) stub.")
