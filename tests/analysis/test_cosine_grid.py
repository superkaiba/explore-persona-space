"""Tests for explore_persona_space.analysis.cosine_grid helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.cosine_grid import (
    cosine_matrix,
    mc_r_distance,
    mean_center_cosine_matrix,
    noise_floor_cross_half,
    off_diag_upper,
)


def test_cosine_matrix_shape_and_diagonal():
    """cosine_matrix on (N, D) yields (N, N) with diagonal == 1.0 (unit-vector self-cos)."""
    torch.manual_seed(42)
    cents = torch.randn(5, 8)
    M = cosine_matrix(cents)
    assert M.shape == (5, 5)
    np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-5)


def test_cosine_matrix_known_values():
    """Two parallel + one orthogonal vectors give known cosines."""
    cents = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],  # parallel to row 0 -> cos = 1
            [0.0, 1.0],  # orthogonal to row 0 -> cos = 0
        ]
    )
    M = cosine_matrix(cents)
    assert M[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert M[0, 2] == pytest.approx(0.0, abs=1e-5)
    assert M[1, 2] == pytest.approx(0.0, abs=1e-5)


def test_cosine_matrix_rejects_non_2d():
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        cosine_matrix(torch.zeros(3, 4, 5))


def test_mean_center_cosine_matrix_differs_from_raw():
    """Mean-centering should change the cosine matrix when centroids are not zero-mean."""
    torch.manual_seed(7)
    cents = torch.randn(6, 16) + 5.0  # heavy global mean
    raw = cosine_matrix(cents)
    mc = mean_center_cosine_matrix(cents)
    # The two should not be identical
    assert not np.allclose(raw, mc, atol=1e-3)
    # Diagonal still 1
    np.testing.assert_allclose(np.diag(mc), 1.0, atol=1e-5)


def test_off_diag_upper_count():
    """Upper-triangle off-diag of (N, N) has N*(N-1)/2 entries."""
    M = np.arange(16, dtype=float).reshape(4, 4)
    upper = off_diag_upper(M)
    assert upper.size == 4 * 3 // 2  # 6 entries
    # Should NOT include diagonal entries (0, 5, 10, 15)
    for diag_val in (0.0, 5.0, 10.0, 15.0):
        assert diag_val not in upper.tolist() or M[0, 1] == diag_val  # protect against coincidence


def test_off_diag_upper_rejects_non_square():
    with pytest.raises(ValueError, match=r"square"):
        off_diag_upper(np.zeros((3, 4)))


def test_mc_r_distance_self_is_zero():
    """A matrix vs itself has Pearson r = 1, distance = 0."""
    torch.manual_seed(11)
    cents = torch.randn(10, 16)
    M = mean_center_cosine_matrix(cents)
    d = mc_r_distance(M, M)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_mc_r_distance_different_matrices():
    """Different matrices give a positive distance."""
    torch.manual_seed(13)
    cents_a = torch.randn(10, 16)
    cents_b = torch.randn(10, 16)
    M_a = mean_center_cosine_matrix(cents_a)
    M_b = mean_center_cosine_matrix(cents_b)
    d = mc_r_distance(M_a, M_b)
    assert d > 0.0
    # Bounded above by 2.0
    assert d <= 2.0


def test_mc_r_distance_shape_mismatch():
    with pytest.raises(ValueError, match=r"same-shape"):
        mc_r_distance(np.zeros((3, 3)), np.zeros((4, 4)))


def test_noise_floor_cross_half_smoke():
    """Smoke: noise floor on (N, n_q, n_layers, D) returns a dict with all expected keys."""
    torch.manual_seed(17)
    # 5 personas, 20 questions, 2 layers, 8-dim
    per_q = torch.randn(5, 20, 2, 8)
    out = noise_floor_cross_half(per_q, layer_idx=0)
    for k in (
        "per_persona_min",
        "per_persona_p5",
        "per_persona_mean",
        "matrix_mc_pearson_r",
    ):
        assert k in out
        assert isinstance(out[k], float)


def test_noise_floor_cross_half_consistent_persona_yields_high_cos():
    """If each persona's vectors are tightly clustered, cross-half cosine should be high."""
    torch.manual_seed(23)
    n_personas, n_q, D = 4, 40, 16
    # Construct: each persona has a fixed direction + small noise
    directions = torch.randn(n_personas, D)
    per_q = directions[:, None, None, :].expand(n_personas, n_q, 1, D).clone()
    per_q = per_q + 0.01 * torch.randn(n_personas, n_q, 1, D)
    out = noise_floor_cross_half(per_q, layer_idx=0)
    # With tiny noise, cross-half mean cos should be very close to 1
    assert out["per_persona_mean"] > 0.99


def test_noise_floor_cross_half_rejects_wrong_dims():
    with pytest.raises(ValueError, match=r"N_personas"):
        noise_floor_cross_half(torch.zeros(5, 16), layer_idx=0)
