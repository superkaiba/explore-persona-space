"""Algebraic checks for the dispatched cached-analysis path."""

import numpy as np

from scripts.issue1739_million_cached import extreme_ids, inverse_direction, rank_errors


def test_rank_sweep_matches_direct_inverse_and_zero_baseline():
    rng = np.random.default_rng(1739)
    w = rng.normal(size=(5, 5))
    u, s, vt = np.linalg.svd(w)
    z = rng.normal(size=(11, 5))
    yc = z @ w + rng.normal(size=(11, 5)) * 0.1
    errors = rank_errors(z, yc, u, s, vt)
    for k in range(6):
        predicted = (yc @ vt[:k].T / s[:k]) @ u[:, :k].T
        np.testing.assert_allclose(errors[k], np.sum((z - predicted) ** 2), atol=1e-10)
    v = rng.normal(size=5)
    direction = inverse_direction(u, s, vt, v, 5)
    np.testing.assert_allclose(direction @ w, v, atol=1e-10)


def test_covariance_recovery_and_coordinate_score_parity():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(20, 4)) * np.arange(1, 5) + 3
    mu = x.mean(0)
    sd = x.std(0, ddof=1) + 1e-9
    z = (x - mu) / sd
    cov = sd[:, None] * (z.T @ z) * sd[None, :] / 19
    np.testing.assert_allclose(cov, np.cov(x, rowvar=False))
    v = rng.normal(size=4)
    np.testing.assert_allclose(z @ (sd * v), (x - mu) @ v)
    np.testing.assert_allclose(z @ v, (x - mu) @ (v / sd))


def test_inverse_ignores_zero_singular_components_and_ties_are_id_ordered():
    u = np.eye(3)
    s = np.array([2.0, 1.0, 0.0])
    vt = np.eye(3)
    errors = rank_errors(np.ones((2, 3)), np.ones((2, 3)), u, s, vt)
    assert len(errors) == 3 and np.isfinite(errors).all()
    lo, hi = extreme_ids(np.ones(3), np.array(["c", "a", "b"]), 2)
    assert lo.tolist() == hi.tolist() == [1, 2]
