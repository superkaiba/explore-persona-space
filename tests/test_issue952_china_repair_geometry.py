"""Pin row-vector read/write orientation and missing/degenerate contrast behavior."""

import numpy as np
import pytest

from scripts import issue952_china_repair_geometry as geom


def test_factorial_contrasts_keep_both_country_controls_and_constituents():
    values = np.zeros((3, 2, 4, 2, 5))
    for content, base in enumerate((13, 10, 2, 7)):
        values[:, :, content, 0] = base
        values[:, :, content, 1] = base + 2
    contrasts = geom.factorial_contrasts(values)
    for key, expected in {
        "subject": 8,
        "china_cue": 3,
        "control_cue": 5,
        "cue_difference_in_differences": -2,
        "framing": 2,
        "neutral": 11,
        "subject:direct": 8,
        "framing:matched_non_china_country": 2,
    }.items():
        assert contrasts[key].shape == (3, 2, 5)
        np.testing.assert_array_equal(contrasts[key], expected)


def test_missing_factorial_arm_is_not_silently_dropped():
    values = np.zeros((2, 2, 4, 2))
    values[0, 0, 3, 0] = np.nan
    result = geom.factorial_contrasts(values)
    assert np.isnan(result["control_cue"][0, 0])
    assert np.isnan(result["cue_difference_in_differences"][0, 0])
    assert np.isfinite(result["subject"]).all()


def test_row_vector_modes_pair_read_with_write():
    operator = np.array([[0.0, 3.0], [1.0, 0.0]])
    sd = np.array([2.0, 4.0])
    modes = geom.operator_modes(operator * sd[:, None], sd)
    np.testing.assert_allclose(modes["operator"], operator)
    for index, singular in enumerate(modes["singular"]):
        np.testing.assert_allclose(
            modes["read"][:, index] @ operator,
            singular * modes["write"][:, index],
        )
    # This nonsymmetric map's leading read and write axes are different.
    assert abs(modes["read"][:, 0] @ modes["write"][:, 0]) < 1e-12


def test_orthogonal_parts_and_zero_direction():
    values = np.array([[3.0, 4.0], [0.0, 0.0]])
    result = geom.orthogonal_parts(values, np.array([[1.0], [0.0]]))
    np.testing.assert_array_equal(result["retained"] + result["low"], values)
    assert result["low_share"][0] == pytest.approx(16 / 25)
    assert np.isnan(result["low_share"][1])
    np.testing.assert_array_equal(result["defined"], [True, False])


def test_answer_split_uses_write_basis_not_read_basis():
    modes = geom.operator_modes(np.array([[0.0, 3.0], [1.0, 0.0]]), np.ones(2))
    observed = np.array([[0.0, 5.0]])
    assert geom.orthogonal_parts(observed, modes["write"][:, :1])["low_share"][0] == 0
    assert geom.orthogonal_parts(observed, modes["read"][:, :1])["low_share"][0] == 1


def test_zero_cosine_and_r2_are_not_reported_as_zero():
    zero = np.zeros((2, 3))
    assert np.isnan(geom.cosine_rows(zero, zero)).all()
    assert geom.delta_r2(zero, zero)["r2"] is None
    assert geom.delta_r2(np.ones((2, 3)), np.ones((2, 3)))["r2"] == 1


def test_mass_cutoff_and_zero_operator():
    assert geom.mass_rank(np.array([3.0, 1.0]), 0.9) == 1
    assert geom.mass_rank(np.array([3.0, 1.0]), 0.99) == 2
    assert geom.mass_rank(np.zeros(2), 0.99) == 0


def test_nonorthogonal_basis_is_rejected():
    with pytest.raises(ValueError, match="orthonormal"):
        geom.orthogonal_parts(np.ones((1, 2)), np.ones((2, 1)))
