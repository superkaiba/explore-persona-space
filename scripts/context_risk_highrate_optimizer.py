"""Analysis-only finite optimizer ceiling; preserve the frozen capture/core closure.

The fit body matches context_risk_followup_probe_core.fit_logistic with only
an explicit max_iter argument. The legacy 5000 path still uses that frozen core.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from scripts.context_risk_followup_probe_core import validate_counts


def validate_iteration_limit(max_iter: int) -> None:
    """Reject implicit, fractional, boolean or unbounded optimizer ceilings."""
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError("Optimizer iteration limit must be a positive integer")


def fit_logistic_with_limit(
    train: np.ndarray,
    test: np.ndarray,
    positive: np.ndarray,
    trials: np.ndarray,
    c_value: float,
    *,
    max_iter: int,
    basis: np.ndarray | None = None,
) -> dict:
    """Fit the exact frequency-weighted binary objective with an unpenalized intercept."""
    validate_iteration_limit(max_iter)
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    validate_counts(positive, trials)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Invalid feature matrix shapes")
    if len(train) != len(trials) or not np.isfinite(c_value) or c_value <= 0:
        raise ValueError("Invalid training rows or regularization")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Nonfinite probe features")
    started = time.monotonic()
    if positive.sum() == 0 or positive.sum() == trials.sum():
        p = float((positive.sum() + 0.5) / (trials.sum() + 1.0))
        logit = float(np.log(p) - np.log1p(-p))
        return {
            "logits": np.full(len(test), logit),
            "coef": None,
            "intercept": logit,
            "status": "single_class_training_prevalence",
            "iterations": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
    weights = np.column_stack([trials - positive, positive]).ravel()
    expanded = np.repeat(train, 2, axis=0)
    labels = np.tile([0, 1], len(train))
    keep = weights > 0
    head = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=max_iter, tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        head.fit(expanded[keep], labels[keep], sample_weight=weights[keep])
    logits = np.asarray(head.decision_function(test), dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite fitted logits")
    coef = head.coef_[0] if basis is None else basis @ head.coef_[0]
    return {
        "logits": logits,
        "coef": coef,
        "intercept": float(head.intercept_[0]),
        "status": "fitted",
        "iterations": int(head.n_iter_[0]),
        "elapsed_seconds": time.monotonic() - started,
    }

