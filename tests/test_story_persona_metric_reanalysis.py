"""Independent dense oracles for the no-centering metric calculation."""

import numpy as np
import pytest

from scripts.story_persona_metric_reanalysis import cosine_from_gram, metric_gram, select_ridge


@pytest.mark.parametrize("offset,ridge", [(0.0, 0.001), (12.0, 0.3), (0.0, 10.0)])
def test_dual_metric_matches_direct_whitening(offset, ridge):
    """Compare dual-space metric to actual dense Cholesky whitening, with offsets."""
    rng = np.random.default_rng(2673)
    x = rng.normal(size=(21, 35)) + offset
    centroids = rng.normal(size=(6, 35)) + offset
    moment = x.T @ x / len(x) + ridge * np.eye(x.shape[1])
    white = np.linalg.solve(np.linalg.cholesky(moment), centroids.T).T
    actual, residual = metric_gram(x, centroids, ridge)
    np.testing.assert_allclose(actual, white @ white.T, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        cosine_from_gram(actual), cosine_from_gram(white @ white.T), atol=2e-10
    )
    assert residual < 1e-8


def test_ridge_is_smallest_admissible_and_uses_second_moment():
    """A common offset must affect the uncentered estimate and its ridge choice."""
    x = np.ones((20, 35)) * 7
    ridge, info = select_ridge(x)
    np.testing.assert_allclose(info["largest_second_moment_eigenvalue"], 35 * 49)
    assert info["condition_number"] <= 1e4
    previous = [
        c
        for lam, c in zip(info["ridge_grid"], info["condition_by_grid"], strict=True)
        if lam < ridge
    ]
    assert all(c > 1e4 for c in previous)


def test_isotropic_metric_reduces_to_raw_cosine():
    """Whitening an isotropic second moment preserves angles and self-cosine."""
    x = np.eye(5) * 3
    centroids = np.random.default_rng(1).normal(size=(4, 5))
    gram, _ = metric_gram(x, centroids, 0.1)
    np.testing.assert_allclose(
        cosine_from_gram(gram), cosine_from_gram(centroids @ centroids.T), atol=1e-12
    )
    np.testing.assert_allclose(np.diag(cosine_from_gram(gram)), 1, atol=1e-12)
