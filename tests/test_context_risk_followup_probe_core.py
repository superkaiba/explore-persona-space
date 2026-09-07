"""Software fixtures for binomial weighting, exact linear fits and task uncertainty."""

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from scripts import context_risk_followup_probe_core as core


def test_frequency_weighted_metrics_equal_expanded_binary_observations():
    """Mixed contexts retain all successes and failures rather than fractional labels."""
    k = np.array([0, 1, 3, 4, 2])
    n = np.array([4, 4, 4, 4, 4])
    logits = np.array([-2.0, -0.4, 0.7, 2.0, -0.1])
    labels = np.concatenate([np.r_[np.ones(p), np.zeros(t - p)] for p, t in zip(k, n, strict=True)])
    prediction = np.repeat(expit(logits), n)
    metrics = core.metric_report(logits, k, n)
    assert np.isclose(metrics["log_loss"], log_loss(labels, prediction))
    assert np.isclose(metrics["brier"], brier_score_loss(labels, prediction))
    assert np.isclose(metrics["auroc"], roc_auc_score(labels, prediction))
    assert np.isclose(metrics["average_precision"], average_precision_score(labels, prediction))
    assert np.isfinite(
        core.loss_terms(np.array([1000.0, -1000.0]), np.array([0, 4]), np.array([4, 4]))
    ).all()


def test_exact_l2_reduction_matches_full_logistic_fit():
    """Exercise both real lbfgs bodies, including a rank-deficient input matrix."""
    rng = np.random.default_rng(817)
    train = rng.normal(size=(24, 64))
    test = rng.normal(size=(9, 64))
    train[:, 7] = train[:, 2]
    test[:, 7] = test[:, 2]
    k = rng.integers(0, 5, size=24)
    n = np.full(24, 4)
    train, test, _ = core.weighted_standardize(train, test, n)
    compressed, evaluation, basis = core.l2_basis(train, test)
    for c_value in (0.0001, 0.1, 10.0):
        full = core.fit_logistic(train, test, k, n, c_value)
        reduced = core.fit_logistic(compressed, evaluation, k, n, c_value, basis=basis)
        assert np.allclose(full["logits"], reduced["logits"], atol=2e-5, rtol=2e-5)
        assert np.allclose(test @ reduced["coef"] + reduced["intercept"], reduced["logits"])


def test_task_bootstrap_does_not_invent_precision_from_repeated_seeds():
    """Multiplying every context count leaves the task-bootstrap interval unchanged."""
    k = np.array([0, 1, 2, 4, 1, 3])
    n = np.full(6, 4)
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    raw = np.array([-0.8, -0.8, -0.4, -0.4, 0.0, 0.0])
    mapped = np.array([-1.7, -0.2, -0.5, 1.8, 0.4, 0.6])
    first = core.paired_task_interval(raw, mapped, k, n, groups)
    repeated = core.paired_task_interval(raw, mapped, k * 100, n * 100, groups)
    for key in ("improvement", "ci_low", "ci_high"):
        assert np.isclose(first[key], repeated[key])
    assert first["n_tasks"] == 3


def test_single_class_training_is_explicit_prevalence_baseline():
    """The degenerate-fold path creates no purported learned coefficient."""
    result = core.fit_logistic(np.eye(3), np.ones((2, 3)), np.zeros(3), np.full(3, 4), 0.1)
    assert result["coef"] is None
    assert result["status"] == "single_class_training_prevalence"
    assert np.allclose(expit(result["logits"]), 0.5 / 13)
