"""Numerical parity and held-out affine recovery for #825 turn calibration."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from explore_persona_space.analysis.turn_transfer_calibration import (
    adapted_predictions,
    calibrate,
    fit_batched_gcv,
)


@pytest.mark.parametrize("shape", [(23, 8), (19, 29)])
def test_batched_predictions_match_legacy_source_fit(monkeypatch, shape):
    """Execute the production helper against the real legacy-GCV parent bodies."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    parent = importlib.import_module("issue825_crossmodel_map_transfer")
    monkeypatch.setattr(parent, "GCV_DOF_CAP", None)
    monkeypatch.setattr(parent, "LAMBDA_SELECTION", "gcv")
    monkeypatch.setattr(parent, "LEGACY_UNGUARDED_GCV", True)
    monkeypatch.setattr(parent, "SELECTOR_LOG", None)
    monkeypatch.setattr(parent, "_fit_device", lambda: torch.device("cpu"))
    rng = np.random.default_rng(825)
    x = rng.normal(size=shape).astype(np.float32)
    x[:, -1] = 2.0  # Exercise the parent's constant-feature normalization.
    y = x @ rng.normal(size=(shape[1], 5)) + 0.5 * rng.normal(size=(shape[0], 5))
    folds = [np.flatnonzero(np.arange(len(x)) % 3 != fold) for fold in range(3)]
    x_eval = rng.normal(size=(7, shape[1]))
    fitted = fit_batched_gcv(x, y, folds)
    for fold, idx in enumerate(folds):
        expected = parent._ridge_predict_cached(
            parent._prep_fold(x[idx], x_eval), y[idx]
        )
        np.testing.assert_allclose(
            fitted.predict(fold, x_eval), expected, rtol=2e-7, atol=2e-7
        )
        np.testing.assert_allclose(fitted.xmu[fold], x[idx].astype(np.float64).mean(0))
        np.testing.assert_allclose(
            fitted.xsd[fold], x[idx].astype(np.float64).std(0, ddof=1) + 1e-9
        )
    np.testing.assert_array_equal(fitted.n_train, [len(idx) for idx in folds])


def test_zero_centered_padding_preserves_prediction_and_lambda():
    """Extra zero rows must not change real-row centering, GCV or the fitted map."""
    rng = np.random.default_rng(41)
    x, y = rng.normal(size=(24, 9)), rng.normal(size=(24, 7))
    folds = [np.arange(17), np.arange(3, 24)]
    ordinary = fit_batched_gcv(x, y, folds)
    padded = fit_batched_gcv(x, y, folds, pad_to=31)
    np.testing.assert_array_equal(ordinary.lambdas, padded.lambdas)
    np.testing.assert_allclose(
        ordinary.gcv_scores, padded.gcv_scores, atol=1e-12, rtol=1e-9
    )
    for fold in range(len(folds)):
        np.testing.assert_allclose(
            ordinary.predict(fold, x), padded.predict(fold, x), atol=1e-9, rtol=1e-9
        )


def test_affine_calibration_recovers_unseen_answers_and_bias_only_residual():
    """A gain and offset learned on calibration rows recover separate test rows."""
    rng = np.random.default_rng(19)
    prediction, unseen = rng.normal(size=(37, 8)), rng.normal(size=(11, 8))
    bias, gain = rng.normal(size=8), 1.75
    coefficients = calibrate(prediction, gain * prediction + bias)
    calibrated = adapted_predictions(unseen, coefficients)
    assert coefficients["gain"] == pytest.approx(gain, abs=1e-14)
    np.testing.assert_allclose(
        calibrated["bias_scale"], gain * unseen + bias, atol=1e-13
    )
    np.testing.assert_allclose(
        calibrated["bias"], unseen + (gain - 1) * prediction.mean(0) + bias, atol=1e-13
    )


def test_excluded_rows_cannot_affect_source_fit():
    """Held-out context and answer values have no influence on source coefficients."""
    rng = np.random.default_rng(61)
    x, y = rng.normal(size=(25, 7)), rng.normal(size=(25, 7))
    indices = [np.arange(17)]
    original = fit_batched_gcv(x, y, indices)
    x[17:] += 1000
    y[17:] -= 2000
    changed = fit_batched_gcv(x, y, indices)
    for name in ["beta", "xmu", "xsd", "ymu", "lambdas", "gcv_scores"]:
        np.testing.assert_array_equal(getattr(original, name), getattr(changed, name))


@pytest.mark.parametrize(
    ("prediction", "target", "match"),
    [
        (np.ones((4, 2)), np.ones((4, 2)), "constant"),
        (np.ones((1, 2)), np.ones((1, 2)), "invalid"),
        (np.zeros((4, 2)), np.zeros((4, 3)), "invalid"),
        (np.full((4, 2), np.nan), np.ones((4, 2)), "nonfinite"),
    ],
)
def test_calibration_rejects_undefined_inputs(prediction, target, match):
    """Undefined calibration must fail rather than produce substituted coefficients."""
    with pytest.raises(ValueError, match=match):
        calibrate(prediction, target)


def test_source_fit_rejects_bad_folds():
    """Duplicate/out-of-range indices and insufficient padding cannot silently fit."""
    x = np.arange(30, dtype=np.float64).reshape(10, 3)
    for idx in [np.array([0, 0, 1]), np.array([0, 1, 10]), np.array([0, 1])]:
        with pytest.raises(ValueError):
            fit_batched_gcv(x, x, [idx])
    with pytest.raises(ValueError, match="pad_to"):
        fit_batched_gcv(x, x, [np.arange(5)], pad_to=4)
