"""Readout controls stay independent of test outcomes; noise follows equal-K means."""

import numpy as np
import pytest

from explore_persona_space.analysis.workspace_diagnostics import (
    evaluate_direction_readouts,
    rollout_noise_report,
)


def test_direction_control_selection_is_unchanged_by_test_labels():
    rng = np.random.default_rng(71)
    train, test = rng.normal(size=(24, 6)), rng.normal(size=(12, 6))
    basis = rng.normal(size=(6, 4))
    basis /= np.linalg.norm(basis, axis=0)
    config = {
        "seed": 71,
        "diagnostics": {
            "controls_pool_multiplier": 3,
            "maximum_absolute_log_variance_mismatch": 0.2,
        },
        "statistics": {"bootstrap": {"draws": 16, "confidence": 0.95}},
    }

    def run(target):
        return evaluate_direction_readouts(
            train,
            target,
            {"ridge": test, "mlp": test * 0.9},
            {"J": basis, "R": basis},
            [1, 2, 3, 4],
            [f"train{i}" for i in range(24)],
            [f"test{i}" for i in range(12)],
            config,
        )

    first, arrays, samples = run(test)
    second, changed, _ = run(test + 100)
    for key in first["matching"]:
        for field in first["matching"][key]:
            np.testing.assert_equal(first["matching"][key][field], second["matching"][key][field])
    for key in ("direction__random", "direction__pca"):
        np.testing.assert_array_equal(arrays[key], changed[key])
    np.testing.assert_array_equal(samples["ridge/R_minus_J"], 0)
    np.testing.assert_allclose(first["metrics"]["ridge"]["J"]["r2"], 1)
    assert "mlp_minus_ridge/J" in first["paired_context_bootstrap"]["intervals"]
    assert not np.allclose(
        first["metrics"]["ridge"]["J"]["r2"], second["metrics"]["ridge"]["J"]["r2"]
    )
    with pytest.raises(ValueError, match="no nonzero PCA"):
        evaluate_direction_readouts(
            np.full_like(train, 0.1),
            test,
            {"ridge": test},
            {"J": basis, "R": basis},
            [1, 2, 3, 4],
            [f"train{i}" for i in range(24)],
            [f"test{i}" for i in range(12)],
            config,
        )


def test_rollout_noise_preserves_component_covariance_and_flags_noisy_targets():
    rng = np.random.default_rng(12)
    full = rng.normal(size=(10, 1, 3)) * 0.1 + rng.normal(size=(10, 5, 3))
    j = full * 2
    r = full * 3
    targets = {"full": full, "J": j, "restJ": full - j, "R": r, "restR": full - r}
    report, arrays = rollout_noise_report(targets, list(range(10)), [42, 43, 44, 45, 46])
    oracle = full.var(axis=1, ddof=1).sum(1) / 5
    np.testing.assert_allclose(arrays["noise__full"], oracle)
    np.testing.assert_allclose(arrays["noise_covariance__J"], -2 * oracle)
    assert report["higher_k_trigger"]
    deterministic = {name: np.repeat(value[:, :1], 5, axis=1) for name, value in targets.items()}
    quiet, _ = rollout_noise_report(deterministic, list(range(10)), [42, 43, 44, 45, 46])
    assert not quiet["higher_k_trigger"]
    with pytest.raises(ValueError, match="reconstruction"):
        rollout_noise_report({**targets, "J": j + 1}, list(range(10)), [42, 43, 44, 45, 46])
    constant = np.full((7, 5, 2), 0.3)
    quiet, _ = rollout_noise_report(
        {
            "full": constant,
            "J": 2 * constant,
            "restJ": constant - 2 * constant,
            "R": 3 * constant,
            "restR": constant - 3 * constant,
        },
        list(range(7)),
        [42, 43, 44, 45, 46],
    )
    assert not quiet["higher_k_trigger"]
    assert all(value["mean_target_noise_trace"] == 0 for value in quiet["components"].values())
    assert all(
        value["observed_between_context_variance_trace_unbiased"] == 0
        for value in quiet["components"].values()
    )
