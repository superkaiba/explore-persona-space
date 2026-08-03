"""Tests for analysis/mapping_baselines.py (identity+bias baseline + kNN retrieval)."""

import numpy as np
import pytest

from explore_persona_space.analysis.mapping_baselines import (
    identity_bias_predict,
    knn_retrieval,
)


def test_identity_bias_exact_recovery():
    rng = np.random.default_rng(0)
    x_tr = rng.standard_normal((40, 8))
    b = rng.standard_normal(8)
    y_tr = x_tr + b
    x_ev = rng.standard_normal((10, 8))
    pred = identity_bias_predict(x_tr, y_tr, x_ev)
    np.testing.assert_allclose(pred, x_ev + b, atol=1e-10)


def test_identity_bias_shape_mismatch_raises():
    with pytest.raises(ValueError):
        identity_bias_predict(np.zeros((4, 3)), np.zeros((4, 5)), np.zeros((2, 3)))


def test_knn_perfect_predictions():
    rng = np.random.default_rng(1)
    true = rng.standard_normal((30, 6))
    out = knn_retrieval(true.copy(), true, ks=(1, 5), metric="euclidean")
    assert out["acc_at_k"][1] == 1.0
    assert out["median_rank"] == 1.0
    assert out["chance_at_k"][5] == pytest.approx(5 / 30)


def test_knn_constant_predictor_scores_exactly_chance():
    rng = np.random.default_rng(2)
    true = rng.standard_normal((50, 6))
    pred = np.tile(true.mean(0), (50, 1))
    out = knn_retrieval(pred, true, ks=(1, 5), metric="euclidean")
    # fixed ordering ⇒ each pool row's rank is unique ⇒ acc@k == chance == k/n
    assert out["acc_at_k"][5] == pytest.approx(out["chance_at_k"][5])
    assert out["median_rank"] == pytest.approx(25.5)


def test_knn_cosine_metric_runs():
    rng = np.random.default_rng(3)
    true = rng.standard_normal((20, 6))
    out = knn_retrieval(true + 0.01 * rng.standard_normal((20, 6)), true, metric="cosine")
    assert out["acc_at_k"][1] > 0.9
