"""Coordinate restoration and undefined-correlation reporting for the small map."""

import numpy as np

from scripts.issue1739_small_map_fixed import deltas, predict_answers, projection


def test_raw_answer_direction_after_native_coordinate_map():
    """A nonorthogonal transform exposes wrong inverse or transpose conventions."""
    rng = np.random.default_rng(1739)
    transform = rng.normal(size=(4, 4)) + 4 * np.eye(4)
    payload = dict(
        S=transform,
        invS=np.linalg.inv(transform),
        mu=rng.normal(size=4),
        x_mu=rng.normal(size=4),
        x_sd=np.exp(rng.normal(size=4)),
        y_mu=rng.normal(size=4),
        w=rng.normal(size=(4, 4)),
    )
    contexts, direction = rng.normal(size=(7, 4)), rng.normal(size=4)
    explicit = predict_answers(payload, contexts) @ direction
    np.testing.assert_allclose(
        projection(payload, contexts, direction), explicit, rtol=1e-12, atol=1e-12
    )


def test_undefined_comparison_remains_null():
    """Empty paired draws or undefined point estimates must never become zero."""
    boot = np.array([[0.1, 0.2], [np.nan, np.nan], [0.1, 0.3], [0.1, 0.2]])
    result = deltas([0.15, None, 0.2, 0.15], boot)
    undefined = result["mapped_minus_real_answer"]
    assert undefined == dict(delta=None, ci95=None, valid_bootstrap_draws=0)
    assert result["mapped_minus_context_native"]["ci95"] is not None
