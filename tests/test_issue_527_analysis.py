"""Unit tests for issue #527's pure-numpy analyzer (Major recommendation).

Pins the deterministic DV1 cosine + GD1/GD2/GD3 SVD primitives against
hand-computable synthetic inputs so future refactors of ``analyze_cell``
can't silently change the headline semantics.

Run with: ``uv run pytest tests/test_issue_527_analysis.py -x``
"""

# math symbols in docstrings

from __future__ import annotations

import numpy as np
import pytest

from explore_persona_space.experiments.issue_527.analysis import (
    _cosine,
    _svd_spectrum,
    analyze_cell,
)


def test_cosine_orthogonal_zero():
    assert _cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_parallel_one():
    assert _cosine(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.0)


def test_cosine_antiparallel_minus_one():
    assert _cosine(np.array([1.0, 1.0]), np.array([-1.0, -1.0])) == pytest.approx(-1.0)


def test_cosine_zero_vector_short_circuits():
    """A zero-norm vector returns 0.0 (avoid NaN)."""
    assert _cosine(np.zeros(3), np.ones(3)) == 0.0


def test_svd_spectrum_rank1_full_share():
    """A pure rank-1 matrix has top1_sv_share == 1, effective_rank == 1."""
    # All rows are a scalar multiple of (1,1,1,1) → rank 1.
    M = np.outer(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0, 1.0]))
    top1, eff_rank, s = _svd_spectrum(M)
    assert top1 == pytest.approx(1.0)
    assert eff_rank == pytest.approx(1.0)
    # All other singular values are 0 (rank 1).
    assert s[1:].sum() == pytest.approx(0.0, abs=1e-10)


def test_svd_spectrum_rank2_balanced():
    """Two orthogonal equal-magnitude directions: top1_share ≈ 0.5, eff_rank ≈ 2."""
    # M has two rows: (1, 0, 0) and (0, 1, 0) — orthonormal, equal magnitude.
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    top1, eff_rank, _s = _svd_spectrum(M)
    assert top1 == pytest.approx(0.5)
    assert eff_rank == pytest.approx(2.0)


def test_svd_spectrum_rank2_skewed():
    """Two orthogonal directions with magnitudes (3, 1): top1_share = 9/10, eff_rank ≈ 1.22."""
    M = np.array([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    top1, eff_rank, _s = _svd_spectrum(M)
    # s = [3, 1]; s² = [9, 1]; top1_share = 9/10 = 0.9
    assert top1 == pytest.approx(0.9)
    # Participation ratio = (Σs²)² / Σs⁴ = 100 / 82 ≈ 1.2195
    assert eff_rank == pytest.approx(100.0 / 82.0, rel=1e-6)


def test_svd_spectrum_zero_matrix():
    """A zero matrix returns (0, 0, [0…])."""
    M = np.zeros((3, 4))
    top1, eff_rank, _s = _svd_spectrum(M)
    assert top1 == 0.0
    assert eff_rank == 0.0


def test_analyze_cell_perfect_additivity_dv1_one():
    """When shift_joint == shift_a + shift_b exactly, DV1 cosines are all 1."""
    rng = np.random.default_rng(0)
    h = 4
    contexts = ["c0", "c1", "c2"]
    M_a = {c: rng.standard_normal(h) for c in contexts}
    M_b = {c: rng.standard_normal(h) for c in contexts}
    M_joint = {c: M_a[c] + M_b[c] for c in contexts}
    # Marker log-prob: additivity in log-prob space too.
    dlp_a = {c: 1.0 for c in contexts}
    dlp_b = {c: 2.0 for c in contexts}
    dlp_j = {c: 3.0 for c in contexts}  # joint = A + B

    cell = analyze_cell(
        pair_id="c0__c1",
        seed=42,
        pair_a="c0",
        pair_b="c1",
        contexts=contexts,
        shift_a=M_a,
        shift_b=M_b,
        shift_joint=M_joint,
        delta_logp_a=dlp_a,
        delta_logp_b=dlp_b,
        delta_logp_joint=dlp_j,
    )
    assert cell.dv1_median == pytest.approx(1.0)
    assert cell.dv1_coverage_at_threshold == pytest.approx(1.0)
    # DV3 (magnitude residual) is dj - (da + db) = 0 everywhere.
    assert cell.dv3_residual_median == pytest.approx(0.0)
    # DV2 normalized residual is zero (perfect additivity).
    assert cell.dv2_residual_norm_median == pytest.approx(0.0)
    # H2 must PASS (|dv3_residual_median| < 1.0 nat).
    assert cell.h2_pass is True


def test_analyze_cell_orthogonal_residual():
    """If shift_joint is orthogonal to (shift_a + shift_b), DV1 cosines are 0."""
    contexts = ["c0", "c1", "c2"]
    # Sum direction = e0; joint direction = e1 (orthogonal).
    M_a = {c: np.array([0.5, 0.0, 0.0, 0.0]) for c in contexts}
    M_b = {c: np.array([0.5, 0.0, 0.0, 0.0]) for c in contexts}
    M_joint = {c: np.array([0.0, 1.0, 0.0, 0.0]) for c in contexts}
    dlp = {c: 0.0 for c in contexts}

    cell = analyze_cell(
        pair_id="c0__c1",
        seed=42,
        pair_a="c0",
        pair_b="c1",
        contexts=contexts,
        shift_a=M_a,
        shift_b=M_b,
        shift_joint=M_joint,
        delta_logp_a=dlp,
        delta_logp_b=dlp,
        delta_logp_joint=dlp,
    )
    assert cell.dv1_median == pytest.approx(0.0, abs=1e-10)
    # Coverage at threshold 0.85 should be 0 (no contexts have cos >= 0.85).
    assert cell.dv1_coverage_at_threshold == pytest.approx(0.0)


def test_analyze_cell_gd1_rank1_fails_gate():
    """A perfectly rank-1 shift_joint matrix triggers GD1 FAIL (top1_share >> 0.75)."""
    n, h = 5, 8
    contexts = [f"c{i}" for i in range(n)]
    direction = np.zeros(h)
    direction[0] = 1.0
    # All rows are scalar multiples of `direction`.
    M_a = {c: np.zeros(h) for c in contexts}
    M_b = {c: np.zeros(h) for c in contexts}
    M_joint = {c: (i + 1) * direction for i, c in enumerate(contexts)}
    dlp = {c: 0.0 for c in contexts}

    cell = analyze_cell(
        pair_id="c0__c1",
        seed=42,
        pair_a="c0",
        pair_b="c1",
        contexts=contexts,
        shift_a=M_a,
        shift_b=M_b,
        shift_joint=M_joint,
        delta_logp_a=dlp,
        delta_logp_b=dlp,
        delta_logp_joint=dlp,
    )
    # The joint matrix is exactly rank 1.
    assert cell.gd1_top1_sv_share == pytest.approx(1.0)
    assert cell.gd1_effective_rank == pytest.approx(1.0)
    assert cell.gd1_pass is False  # top1_share > 0.75 OR eff_rank < 2.0


def test_analyze_cell_gd2_high_cosine_fails_gate():
    """When per-context shift_a and shift_b are highly correlated, GD2 fails."""
    n, h = 4, 4
    contexts = [f"c{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    # shift_a and shift_b are parallel per context (cosine ≈ 1).
    base_vecs = [rng.standard_normal(h) for _ in range(n)]
    M_a = {c: base_vecs[i] for i, c in enumerate(contexts)}
    M_b = {c: 2.0 * base_vecs[i] for i, c in enumerate(contexts)}  # parallel
    # Joint shifts are also along the same direction.
    M_joint = {c: 3.0 * base_vecs[i] for i, c in enumerate(contexts)}
    dlp = {c: 0.0 for c in contexts}

    cell = analyze_cell(
        pair_id="c0__c1",
        seed=42,
        pair_a="c0",
        pair_b="c1",
        contexts=contexts,
        shift_a=M_a,
        shift_b=M_b,
        shift_joint=M_joint,
        delta_logp_a=dlp,
        delta_logp_b=dlp,
        delta_logp_joint=dlp,
    )
    # Singleton cosines per context are ~1.0.
    assert cell.gd2_singleton_cosine_median == pytest.approx(1.0)
    assert cell.gd2_pass is False  # median > 0.6 fails the gate


def test_analyze_cell_dv4_pass_requires_all_four_emissions_above_gate():
    """DV4 PASS requires source_a, source_b, joint_a, joint_b ALL >= 0.5."""
    contexts = ["a_persona", "b_persona", "c0"]
    M_a = {c: np.array([1.0, 0.0]) for c in contexts}
    M_b = {c: np.array([0.0, 1.0]) for c in contexts}
    M_joint = {c: np.array([1.0, 1.0]) for c in contexts}
    dlp = {c: 0.5 for c in contexts}

    # All four above gate.
    cell_pass = analyze_cell(
        pair_id="a_persona__b_persona",
        seed=42,
        pair_a="a_persona",
        pair_b="b_persona",
        contexts=contexts,
        shift_a=M_a,
        shift_b=M_b,
        shift_joint=M_joint,
        delta_logp_a=dlp,
        delta_logp_b=dlp,
        delta_logp_joint=dlp,
        source_emission_a={"a_persona": 0.9, "joint_a_persona": 0.85},
        source_emission_b={"b_persona": 0.8, "joint_b_persona": 0.75},
    )
    assert cell_pass.dv4_pass is True

    # One source below gate → FAIL.
    cell_fail = analyze_cell(
        pair_id="a_persona__b_persona",
        seed=42,
        pair_a="a_persona",
        pair_b="b_persona",
        contexts=contexts,
        shift_a=M_a,
        shift_b=M_b,
        shift_joint=M_joint,
        delta_logp_a=dlp,
        delta_logp_b=dlp,
        delta_logp_joint=dlp,
        source_emission_a={"a_persona": 0.3, "joint_a_persona": 0.85},
        source_emission_b={"b_persona": 0.8, "joint_b_persona": 0.75},
    )
    assert cell_fail.dv4_pass is False
