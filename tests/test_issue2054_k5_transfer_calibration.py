"""Calibration oracle, held-out behavior, and source-fold leakage checks."""

import numpy as np
import pytest
import torch

from scripts import issue2054_k5_assistant_transfer as subset
from scripts import issue2054_k5_loso_calibration as calibration


def test_scalar_and_bias_match_joint_least_squares():
    """Compare the centered formula with a joint intercept-plus-scalar solve."""
    rng = np.random.default_rng(37)
    p, y = rng.normal(size=(31, 4)), rng.normal(size=(31, 4))
    design = np.column_stack((p.ravel(), np.tile(np.eye(4), (len(p), 1))))
    oracle = np.linalg.lstsq(design, y.ravel(), rcond=None)[0]
    c = calibration.calibrate(p, y)
    np.testing.assert_allclose(c["gain"], oracle[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        c["target_mean"] - c["gain"] * c["prediction_mean"], oracle[1:], atol=1e-12
    )
    np.testing.assert_allclose(c["bias"], (y - p).mean(0), atol=1e-12)


def test_affine_transfer_recovers_independent_test_targets():
    """Fit only training samples, then recover an unseen affine target mapping."""
    rng = np.random.default_rng(19)
    train, test = rng.normal(size=(60, 7)), rng.normal(size=(20, 7))
    gain, bias = 1.7, rng.normal(size=7)
    c = calibration.calibrate(train, gain * train + bias)
    actual = calibration.adapted_predictions(test, c)
    np.testing.assert_allclose(actual["bias_scale"], gain * test + bias, atol=1e-12)
    assert np.mean((actual["bias"] - (gain * test + bias)) ** 2) > 0.1


def test_unidentifiable_scaling_fails():
    """Constant predictions cannot silently produce a fallback scale."""
    with pytest.raises(ValueError, match="cannot identify"):
        calibration.calibrate(np.ones((5, 2)), np.arange(10).reshape(5, 2))


def test_source_audit_rejects_conversation_leakage():
    """A shared test conversation mislabeled as source training must fail."""
    panel = {
        c: {"ids": ["a", "b", "c"], "membership": np.array([0, 1, 2])}
        for c in ["source__model", "target__model"]
    }
    audit = subset.audit_sources(panel, ["source__model"], "target__model", 0)
    assert audit["n_train"] == 2 and audit["test_overlap_source"] == 0
    with pytest.raises(ValueError, match="target must be excluded"):
        subset.audit_sources(panel, ["target__model"], "target__model", 0)
    panel["source__model"]["membership"][0] = 1
    with pytest.raises(RuntimeError, match="leaked"):
        subset.audit_sources(panel, ["source__model"], "target__model", 0)


def test_resume_fingerprint_binds_effective_source_settings(monkeypatch):
    """A presentation reorder must not silently reuse a different source map."""
    original = subset.fingerprint()
    changed = list(calibration.SETTINGS)
    changed[0], changed[1] = changed[1], changed[0]
    monkeypatch.setattr(calibration, "SETTINGS", changed)
    assert subset.fingerprint() != original


def test_plain_source_is_excluded_from_targets_and_has_distinct_fingerprint():
    """Changing the assistant framing changes both the source and resume identity."""
    for model in calibration.MODELS:
        regimes = subset.source_sets(model, "plain_only")
        assert list(regimes) == ["assistant_plain_only"]
        sources = regimes["assistant_plain_only"]
        assert sources == [f"conversation_paired_stories_assistant__on_policy__bare_text__{model}"]
        targets = [
            f"{prefix}__{model}"
            for _, prefix in calibration.SETTINGS
            if f"{prefix}__{model}" not in sources
        ]
        assert len(targets) == 5
        assert f"conversation_paired_stories_assistant__on_policy__chat__{model}" in targets
        assert all("bare_text" not in target for target in targets)
    assert subset.fingerprint("plain_only") != subset.fingerprint("chat_grid")
    with pytest.raises(ValueError, match="unknown source mode"):
        subset.source_sets(calibration.MODELS[0], "unknown")


@pytest.mark.parametrize("n_sources", [1, 2])
def test_subset_moments_match_materialized_ridge(n_sources):
    """The subset-bank dispatch matches the established materialized solver."""
    rng = np.random.default_rng(71)
    panel, bank = {}, {}
    for ci in range(n_sources):
        x = rng.normal(size=(60, 8)) + ci
        y = rng.normal(size=(60, 8)) + 0.7 * x
        membership = np.arange(60) % 5
        cell = f"c{ci}"
        panel[cell] = {"x": x, "y": y, "membership": membership}
        bank[cell] = []
        for f in range(5):
            xt, yt = [torch.tensor(a[membership == f]) for a in [x, y]]
            bank[cell].append(
                {
                    "n": len(xt),
                    "sum_x": xt.sum(0),
                    "sum_y": yt.sum(0),
                    "yss": float((yt**2).sum()),
                    "c_xx": xt.T @ xt,
                    "c_xy": xt.T @ yt,
                }
            )
    moments = calibration.loso.combine(bank, 5, drop_speaker=None, drop_fold=0)
    actual = calibration.loso.PooledMomentRidge(**moments)
    x, y = [
        np.concatenate([p[key][p["membership"] != 0] for p in panel.values()]) for key in ["x", "y"]
    ]
    test = rng.normal(size=(10, 8))
    reference, info = calibration.loso.SharedEighRidge(x, test, device="cpu").fit_predict(y)
    np.testing.assert_allclose(actual.predict_np(test), reference, rtol=1e-10, atol=1e-10)
    assert actual.info()["best_lambda"] == info["best_lambda"]
