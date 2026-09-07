"""Frequency-weighted linear risk probes with an exact L2 subspace reduction."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def validate_counts(positive: np.ndarray, trials: np.ndarray) -> None:
    """Reject missing, fractional, impossible or empty binomial observations."""
    if positive.shape != trials.shape or positive.ndim != 1 or len(positive) == 0:
        raise ValueError("Invalid binomial count shapes")
    if (
        not np.isfinite(positive).all()
        or not np.isfinite(trials).all()
        or np.any(positive != np.floor(positive))
        or np.any(trials != np.floor(trials))
        or np.any(trials <= 0)
        or np.any(positive < 0)
        or np.any(positive > trials)
    ):
        raise ValueError("Counts must be finite integers with 0 <= positive <= trials")


def loss_terms(logits: np.ndarray, positive: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """Return context loss sums, retaining every binary trajectory's frequency."""
    validate_counts(positive, trials)
    if logits.shape != positive.shape or not np.isfinite(logits).all():
        raise ValueError("Invalid forecast logits")
    return positive * np.logaddexp(0.0, -logits) + (trials - positive) * np.logaddexp(0.0, logits)


def weighted_standardize(train: np.ndarray, test: np.ndarray, trials: np.ndarray):
    """Fit scaling only on training inputs with their actual frequency counts."""
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    validate_counts(np.zeros_like(trials), trials)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Invalid feature matrix shapes")
    if len(train) != len(trials):
        raise ValueError("Frequency counts do not match training rows")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Nonfinite input features")
    scaler = StandardScaler().fit(train, sample_weight=trials)
    return scaler.transform(train), scaler.transform(test), scaler


def l2_basis(train: np.ndarray, test: np.ndarray):
    """Reduce d to min(n,d) without changing any fitted L2 linear prediction.

    A finite-L2 optimum has no coefficient component orthogonal to all training
    rows. An orthonormal basis for their span therefore preserves both the
    logistic objective and coefficient norm. QR may include numerical null
    directions; their finite-L2 optimum is still zero. No outcome enters QR.
    """
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Invalid feature matrix shapes")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Nonfinite subspace features")
    basis, _ = np.linalg.qr(train.T, mode="reduced")
    compressed = train @ basis
    if not np.allclose(compressed @ basis.T, train, rtol=1e-9, atol=1e-9):
        raise ValueError("L2 subspace reduction failed to reconstruct training inputs")
    return compressed, test @ basis, basis


def fit_logistic(
    train: np.ndarray,
    test: np.ndarray,
    positive: np.ndarray,
    trials: np.ndarray,
    c_value: float,
    *,
    basis: np.ndarray | None = None,
) -> dict:
    """Fit the exact frequency-weighted binary objective with an unpenalized intercept."""
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
    head = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=5000, tol=1e-8)
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


def metric_report(logits: np.ndarray, positive: np.ndarray, trials: np.ndarray) -> dict:
    """Report proper losses, weighted discrimination and fixed-bin calibration."""
    terms = loss_terms(logits, positive, trials)
    probability = expit(logits)
    n = int(trials.sum())
    score = {
        "log_loss": float(terms.sum() / n),
        "brier": float(
            (positive * (1 - probability) ** 2 + (trials - positive) * probability**2).sum() / n
        ),
        "observed_prevalence": float(positive.sum() / n),
        "predicted_prevalence": float((trials * probability).sum() / n),
        "n_trajectories": n,
        "n_positive": int(positive.sum()),
        "n_contexts": len(trials),
        "auroc": None,
        "average_precision": None,
    }
    if 0 < positive.sum() < n:
        labels = np.tile([0, 1], len(positive))
        weights = np.column_stack([trials - positive, positive]).ravel()
        # Logits preserve ranking when expit saturates to exactly zero or one.
        values = np.repeat(logits, 2)
        keep = weights > 0
        score["auroc"] = float(
            roc_auc_score(labels[keep], values[keep], sample_weight=weights[keep])
        )
        score["average_precision"] = float(
            average_precision_score(labels[keep], values[keep], sample_weight=weights[keep])
        )
    calibration = []
    bins = np.minimum((probability * 5).astype(int), 4)
    for index in range(5):
        selected = bins == index
        count = int(trials[selected].sum())
        calibration.append(
            {
                "lower": index / 5,
                "upper": (index + 1) / 5,
                "n": count,
                "observed": None if count == 0 else float(positive[selected].sum() / count),
                "predicted": None
                if count == 0
                else float((trials[selected] * probability[selected]).sum() / count),
            }
        )
    score["calibration"] = calibration
    return score


def paired_task_interval(
    first_logits: np.ndarray,
    second_logits: np.ndarray,
    positive: np.ndarray,
    trials: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int = 5000,
    seed: int = 38296,
) -> dict:
    """Bootstrap base tasks for first-minus-second log-loss improvement."""
    differences = loss_terms(first_logits, positive, trials) - loss_terms(
        second_logits, positive, trials
    )
    group_ids, assignment = np.unique(groups, return_inverse=True)
    if len(group_ids) < 2:
        raise ValueError("Task uncertainty requires at least two task groups")
    numerator = np.bincount(assignment, weights=differences)
    denominator = np.bincount(assignment, weights=trials)
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(group_ids), size=(replicates, len(group_ids)))
    draws = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "improvement": float(differences.sum() / trials.sum()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_tasks": len(group_ids),
        "replicates": replicates,
        "seed": seed,
        "positive_interval": bool(low > 0),
        "uncertainty_scope": "marginal fixed-prediction paired task bootstrap; excludes fitting uncertainty",
    }


def save_json(path: Path, value: dict) -> None:
    """Persist one bounded fit/analysis unit atomically, rejecting invalid JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
