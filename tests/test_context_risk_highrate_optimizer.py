"""Numerical ceiling correction preserves strict convergence and cache identity."""

import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from scripts import context_risk_followup_analyze as consumer
from scripts import context_risk_highrate_optimizer as optimizer
from scripts.context_risk_followup_probe_core import fit_logistic


def inputs():
    x = np.array([[0.0], [1.0], [2.0]])
    return x, x, np.array([0, 1, 2]), np.array([2, 2, 2]), 1.0


def test_converged_fit_is_bitwise_identical():
    old = fit_logistic(*inputs(), basis=np.eye(1))
    new = optimizer.fit_logistic_with_limit(*inputs(), basis=np.eye(1), max_iter=20000)
    for key in ("logits", "coef", "intercept", "iterations", "status"):
        np.testing.assert_array_equal(old[key], new[key])


@pytest.mark.parametrize("limit", [0, -1, True, 20000.0, float("inf"), None])
def test_invalid_limits_rejected(limit):
    with pytest.raises(ValueError, match="positive integer"):
        optimizer.fit_logistic_with_limit(*inputs(), max_iter=limit)


def test_convergence_warning_remains_an_error(monkeypatch):
    def fail(*args, **kwargs):
        warnings.warn("test-only nonconvergence", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(optimizer.LogisticRegression, "fit", fail)
    with pytest.raises(ConvergenceWarning, match="test-only"):
        optimizer.fit_logistic_with_limit(*inputs(), max_iter=20000)


def test_cache_requires_matching_declared_limit(tmp_path):
    path = tmp_path / "fit.json"
    consumer.cached_fit(path, "same", *inputs(), np.eye(1), max_iter=20000)
    saved = json.loads(path.read_text())
    assert saved["max_iter"] == 20000
    with pytest.raises(ValueError, match="iteration limit differs"):
        consumer.cached_fit(path, "same", *inputs(), np.eye(1), max_iter=5000)
    for bad in (None, 5000, True):
        changed = dict(saved)
        if bad is None:
            changed.pop("max_iter")
        else:
            changed["max_iter"] = bad
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="iteration limit differs"):
            consumer.cached_fit(path, "same", *inputs(), np.eye(1), max_iter=20000)
    for bad in (20001, True):
        path.write_text(json.dumps({**saved, "iterations": bad}))
        with pytest.raises(ValueError, match="fitting diagnostics"):
            consumer.cached_fit(path, "same", *inputs(), np.eye(1), max_iter=20000)


def test_legacy_core_dispatch_and_legacy_cache(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Legacy 5000 fit must use unchanged core")

    monkeypatch.setattr(consumer, "fit_logistic_with_limit", unexpected)
    path = tmp_path / "legacy.json"
    first = consumer.cached_fit(path, "same", *inputs(), np.eye(1))
    saved = json.loads(path.read_text())
    saved.pop("max_iter")
    path.write_text(json.dumps(saved))
    second = consumer.cached_fit(path, "same", *inputs(), np.eye(1))
    np.testing.assert_array_equal(first["logits"], second["logits"])


def test_amended_path_uses_declared_helper(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Amended fit must not use fixed-5000 core")

    monkeypatch.setattr(consumer, "fit_logistic", unexpected)
    result = consumer.cached_fit(
        tmp_path / "new.json", "new", *inputs(), np.eye(1), max_iter=20000
    )
    assert result["status"] == "fitted"
