"""Independent direct-SVD and known-rank checks for character-shift geometry."""

import numpy as np
import pytest

from scripts.issue2054_k5_matched_rank import analyze_difference, mean_shift_spectrum


@pytest.mark.parametrize("n,d", [(75, 20), (45, 80)])
def test_gram_projection_matches_direct_svd(n, d):
    """Exercise both sides of n=d and every train/test centering calculation."""
    rng = np.random.default_rng(81)
    delta = rng.normal(size=(n, d)) + rng.normal(size=d)
    folds = np.arange(n) % 5
    report, arrays = analyze_difference(delta, folds)
    for mode in ("raw", "centered"):
        x = delta if mode == "raw" else delta - delta.mean(0)
        s = np.linalg.svd(x, compute_uv=False)
        np.testing.assert_allclose(arrays[f"{mode}_energy"][: len(s)], s * s, rtol=1e-10, atol=1e-9)
        for f in range(5):
            train, test = delta[folds != f], delta[folds == f]
            if mode == "centered":
                mean = train.mean(0)
                train, test = train - mean, test - mean
            _, s, vt = np.linalg.svd(train, full_matrices=False)
            tol = report["folds"][f]["modes"][mode]["gram_eigenvalue_tolerance"]
            basis = vt[s * s > tol]
            expected = np.r_[0.0, np.cumsum(np.square(test @ basis.T).sum(0))]
            np.testing.assert_allclose(
                arrays[f"fold{f}_{mode}_projection_energy"], expected, rtol=1e-9, atol=1e-9
            )
    changed = delta.copy()
    changed[folds == 0] += 7
    _, altered = analyze_difference(changed, folds)
    for mode in ("raw", "centered"):
        np.testing.assert_allclose(
            arrays[f"fold0_{mode}_train_energy"], altered[f"fold0_{mode}_train_energy"], atol=1e-10
        )
    np.testing.assert_array_equal(
        arrays["training_mean_shift"][0], altered["training_mean_shift"][0]
    )


def test_rank_one_does_not_mean_constant():
    """A common direction with changing amplitude is rank one but not a constant vector."""
    rng = np.random.default_rng(6)
    delta = np.linspace(0.2, 3, 100)[:, None] * rng.normal(size=(1, 25))
    report, arrays = analyze_difference(delta, np.arange(100) % 5)
    assert report["spectra"]["raw"]["r95"] == 1
    assert report["spectra"]["centered"]["r95"] == 1
    assert report["query_constancy"]["constant_vector_fraction"] < 0.9
    assert np.isclose(report["query_constancy"]["variable_amplitude_mean_direction_fraction"], 1)
    np.testing.assert_allclose(arrays["mean_direction_cosine"], 1)


def test_exact_constant_shift_has_no_query_varying_remainder():
    """The literal constant-vector hypothesis has raw rank one and centered rank zero."""
    rng = np.random.default_rng(19)
    delta = np.tile(rng.normal(size=35), (75, 1))
    report, arrays = analyze_difference(delta, np.arange(75) % 5)
    assert report["spectra"]["raw"]["r95"] == 1
    assert report["spectra"]["centered"]["r95"] == 0
    assert report["spectra"]["centered"]["top1"] is None
    assert report["heldout"]["centered"]["status"] == "zero_residual_energy"
    assert report["query_constancy"]["constant_vector_fraction"] == 1
    np.testing.assert_array_equal(arrays["centered_energy"], 0)
    np.testing.assert_array_equal(arrays["constant_error"], 0)
    np.testing.assert_allclose(arrays["mean_direction_coefficient"], 1)


def test_four_mean_contrasts_have_at_most_three_directions():
    """Check the centroid/pairwise equivalence and a known collinear mean configuration."""
    rng = np.random.default_rng(9)
    report, arrays = mean_shift_spectrum(rng.normal(size=(4, 30)))
    assert report["r95"] <= 3
    np.testing.assert_allclose(arrays["pair_energy"][:3], 4 * arrays["centroid_energy"][:3])
    means = np.arange(4)[:, None] * rng.normal(size=(1, 30)) + rng.normal(size=30)
    report, _ = mean_shift_spectrum(means)
    assert report["r95"] == 1
