"""Tests for analysis/mapping_baselines.py (identity+bias baseline + kNN retrieval)."""

import numpy as np
import pytest

from explore_persona_space.analysis.mapping_baselines import (
    identity_bias_predict,
    identity_bias_predict_blocked,
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


def test_identity_bias_blocked_matches_exact_helper():
    """Plan #1901 mlp-scaling-densify equivalence bar: max |Δpred| ≤ 1e-6 vs the
    exact helper on the same train subset, with block small enough to force
    multiple accumulation blocks (7 over 173 rows) and fp32 store inputs."""
    rng = np.random.default_rng(4)
    x_all = rng.standard_normal((500, 16)).astype(np.float32)
    y_all = rng.standard_normal((500, 16)).astype(np.float32)
    tr_idx = np.sort(rng.choice(500, size=173, replace=False))
    x_ev = rng.standard_normal((50, 16)).astype(np.float32)
    exact = identity_bias_predict(x_all[tr_idx], y_all[tr_idx], x_ev)
    blocked = identity_bias_predict_blocked(x_all, y_all, tr_idx, x_ev, block=7)
    assert np.max(np.abs(blocked - exact)) <= 1e-6


def test_identity_bias_blocked_return_bias():
    rng = np.random.default_rng(5)
    x_all = rng.standard_normal((60, 5))
    b_true = rng.standard_normal(5)
    y_all = x_all + b_true
    tr_idx = np.arange(60)
    x_ev = rng.standard_normal((9, 5))
    pred, b = identity_bias_predict_blocked(x_all, y_all, tr_idx, x_ev, block=16, return_bias=True)
    np.testing.assert_allclose(b, b_true, atol=1e-10)
    np.testing.assert_allclose(pred, x_ev + b, atol=0)  # pred is exactly x_ev + b


def test_identity_bias_blocked_invalid_inputs_raise():
    x = np.zeros((10, 3))
    y = np.zeros((10, 3))
    with pytest.raises(ValueError):  # empty index set
        identity_bias_predict_blocked(x, y, np.array([], dtype=np.int64), np.zeros((2, 3)))
    with pytest.raises(ValueError):  # train dim mismatch
        identity_bias_predict_blocked(x, np.zeros((10, 4)), np.arange(10), np.zeros((2, 3)))
    with pytest.raises(ValueError):  # eval dim mismatch
        identity_bias_predict_blocked(x, y, np.arange(10), np.zeros((2, 4)))
    with pytest.raises(ValueError):  # non-positive block
        identity_bias_predict_blocked(x, y, np.arange(10), np.zeros((2, 3)), block=0)


def test_identity_bias_blocked_row_count_and_index_bounds_validated():
    """#1901 round-2 review (Codex Minor): row-count equality between the full
    stores, negative-index rejection, and index upper-bound validation."""
    x = np.zeros((10, 3))
    with pytest.raises(ValueError, match="row counts"):  # x/y row-count mismatch
        identity_bias_predict_blocked(x, np.zeros((9, 3)), np.arange(5), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="non-negative"):  # negative indexing rejected
        identity_bias_predict_blocked(x, np.zeros((10, 3)), np.array([-1, 2]), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="out of bounds"):  # index >= n_rows
        identity_bias_predict_blocked(x, np.zeros((10, 3)), np.array([0, 10]), np.zeros((2, 3)))


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
