"""Independent numerical software fixtures, never fits to experiment outcomes."""

import warnings

import numpy as np
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from scripts import context_risk_followup_probe_core as core


def test_integer_frequency_fit_equals_literal_binary_expansion():
    """Compare the production weighted fit with an independently expanded fit."""
    rng = np.random.default_rng(621)
    train = rng.normal(size=(12, 7))
    test = rng.normal(size=(4, 7))
    n = np.array([1, 2, 4, 3, 5, 1, 4, 2, 5, 3, 1, 4])
    k = np.array([0, 1, 4, 1, 2, 1, 0, 2, 3, 1, 0, 4])
    repeated = np.repeat(train, n, axis=0)
    labels = np.concatenate(
        [np.r_[np.ones(p), np.zeros(t - p)] for p, t in zip(k, n, strict=True)]
    )
    for c_value in (1e-6, 0.01, 100.0):
        actual = core.fit_logistic(train, test, k, n, c_value)
        expected = LogisticRegression(C=c_value, solver="lbfgs", max_iter=5000, tol=1e-8)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            expected.fit(repeated, labels)
        assert np.allclose(actual["coef"], expected.coef_[0], atol=1e-7, rtol=1e-6)
        assert np.allclose(
            actual["logits"], expected.decision_function(test), atol=1e-7, rtol=1e-6
        )
        assert np.isclose(actual["intercept"], expected.intercept_[0], atol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qr_preserves_row_span_norm_and_out_of_span_predictions(dtype):
    """Use weighted centering, dependent rows/columns and unseen test directions."""
    rng = np.random.default_rng(331)
    train = rng.normal(size=(18, 47)).astype(dtype)
    test = rng.normal(size=(6, 47)).astype(dtype)
    train[-1] = train[0]
    train[:, 8] = train[:, 2]
    test[:, 8] = test[:, 2]
    n = np.arange(18) % 4 + 1
    k = np.arange(18) % (n + 1)
    scaled, evaluation, _ = core.weighted_standardize(train, test, n)
    reduced, heldout, basis = core.l2_basis(scaled, evaluation)
    assert reduced.dtype == np.float64 and basis.dtype == np.float64
    assert np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
    assert np.allclose(reduced @ basis.T, scaled, atol=1e-12)
    full = core.fit_logistic(scaled, evaluation, k, n, 1.0)
    small = core.fit_logistic(reduced, heldout, k, n, 1.0, basis=basis)
    assert np.allclose(full["logits"], small["logits"], atol=2e-5, rtol=2e-5)
    assert np.isclose(np.linalg.norm(full["coef"]), np.linalg.norm(small["coef"]), atol=2e-5)
    assert np.allclose(evaluation @ small["coef"] + small["intercept"], small["logits"])


@pytest.mark.parametrize("logits", [np.array([40.0, 50.0]), np.array([-1000.0, -900.0])])
def test_ranking_metrics_do_not_collapse_saturated_probabilities(logits):
    """Finite ordered logits remain perfectly ranked even if expit values tie."""
    result = core.metric_report(logits, np.array([0, 4]), np.array([4, 4]))
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert np.isfinite(result["log_loss"])
    assert sum(row["n"] for row in result["calibration"]) == 8
    for row in result["calibration"]:
        if row["n"] == 0:
            assert row["observed"] is None and row["predicted"] is None


def test_bootstrap_matches_explicit_unequal_task_resampling():
    """Compare the vectorized implementation to a transparent per-task loop."""
    first = np.array([-1.0, 0.2, 1.7, -0.4, -0.8, 2.0])
    second = np.array([-1.8, 0.1, 0.6, -0.6, -0.2, 1.0])
    n = np.array([1, 4, 2, 3, 1, 5])
    k = np.array([0, 1, 2, 1, 0, 3])
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    actual = core.paired_task_interval(first, second, k, n, groups, replicates=257, seed=414)
    loss_a = k * np.logaddexp(0, -first) + (n - k) * np.logaddexp(0, first)
    loss_b = k * np.logaddexp(0, -second) + (n - k) * np.logaddexp(0, second)
    draws = []
    ids = np.unique(groups)
    rng = np.random.default_rng(414)
    for sampled in rng.integers(len(ids), size=(257, len(ids))):
        numerator = 0.0
        denominator = 0
        for index in sampled:
            select = groups == ids[index]
            numerator += (loss_a[select] - loss_b[select]).sum()
            denominator += n[select].sum()
        draws.append(numerator / denominator)
    low, high = np.quantile(draws, [0.025, 0.975])
    assert np.isclose(actual["ci_low"], low)
    assert np.isclose(actual["ci_high"], high)
    assert np.isclose(actual["improvement"], (loss_a - loss_b).sum() / n.sum())
    equal = core.paired_task_interval(first, first, k, n, groups)
    assert equal["ci_low"] == equal["ci_high"] == equal["improvement"] == 0
    assert equal["positive_interval"] is False


@pytest.mark.parametrize("all_positive", [False, True])
def test_degenerate_fold_and_metric_outputs_are_explicit(all_positive):
    """Both single-class cases use the declared prior and omit discrimination."""
    n = np.array([1, 4, 2])
    k = n.copy() if all_positive else np.zeros(3, dtype=int)
    result = core.fit_logistic(np.eye(3), np.ones((2, 3)), k, n, 0.1)
    assert result["coef"] is None and result["iterations"] == 0
    assert np.allclose(expit(result["logits"]), (k.sum() + 0.5) / (n.sum() + 1))
    report = core.metric_report(np.zeros(3), k, n)
    assert report["auroc"] is None and report["average_precision"] is None


@pytest.mark.parametrize("bad_c", [0.0, -1.0, np.nan, np.inf])
def test_nonfinite_or_nonpositive_c_cannot_bypass_validation(bad_c):
    """The prevalence branch must not accept an invalid declared fit setup."""
    with pytest.raises(ValueError):
        core.fit_logistic(np.eye(2), np.ones((1, 2)), np.zeros(2), np.ones(2), bad_c)


@pytest.mark.parametrize(
    "bad_weights",
    [np.array([0, 1]), np.array([-1, 2]), np.array([np.nan, 1]), np.array([1.5, 2])],
)
def test_scaling_rejects_invalid_frequency_counts(bad_weights):
    """Scaling weights represent positive integer trajectory frequencies."""
    with pytest.raises(ValueError):
        core.weighted_standardize(np.eye(2), np.ones((1, 2)), bad_weights)


def test_scaling_equals_literal_frequency_expansion_without_test_leakage():
    """Large held-out values cannot alter training mean/variance."""
    train = np.array([[1.0, 8.0], [2.0, 8.0], [4.0, 8.0]])
    test = np.array([[1e9, -1e9]])
    n = np.array([1, 2, 5])
    scaled, evaluation, scaler = core.weighted_standardize(train, test, n)
    repeated = np.repeat(train, n, axis=0)
    assert np.allclose(scaler.mean_, repeated.mean(axis=0))
    assert np.allclose(scaler.var_, repeated.var(axis=0))
    assert np.isfinite(scaled).all() and np.isfinite(evaluation).all()
    assert scaler.scale_[1] == 1.0
