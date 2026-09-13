"""Supplementary plots must use the same rows, weighting and frozen directions."""

import numpy as np
import pytest

from explore_persona_space.analysis.workspace_diagnostics import (
    evaluate_direction_readouts,
    rollout_noise_report,
)
from explore_persona_space.analysis.workspace_supplement import (
    agreement_statistics,
    paired_noise,
    paired_readouts,
    row_indices,
)


def test_near_zero_components_remain_excluded_from_cosine_only():
    full = np.array([[2.0, 0], [4, 0], [6, 0]])
    j = np.array([[0.0, 0], [2, 0], [3, 0]])
    r = full / 2
    report = agreement_statistics(
        {"full": full, "J": j, "restJ": full - j, "R": r, "restR": full - r}, list("abc"), 1e-6
    )
    assert report["excluded_near_zero_contexts"] == 1
    assert np.isnan(report["component_cosine"][0])
    assert report["mean_component_cosine"] == 1
    assert report["squared_component_difference"][0] == 1
    assert report["reconstruction"]["J"]["max_abs_reconstruction_error"] == 0


def test_noise_rescore_matches_direct_rollout_reduction_on_selected_contexts():
    rng = np.random.default_rng(7)
    full = rng.normal(size=(7, 5, 3)) + np.arange(7)[:, None, None]
    rollouts = {
        "full": full,
        "J": full * 0.3,
        "restJ": full * 0.7,
        "R": full * 0.6,
        "restR": full * 0.4,
    }
    ids, selected, seeds = list("abcdefg"), list("bdf"), list(range(42, 47))
    report, arrays = rollout_noise_report(rollouts, ids, seeds)
    index = row_indices(ids, selected)
    targets = {name: value[index].mean(1) for name, value in rollouts.items()}
    actual = paired_noise(report, arrays, targets, selected)
    expected, _ = rollout_noise_report(
        {name: value[index] for name, value in rollouts.items()}, selected, seeds
    )
    for name in rollouts:
        for key in (
            "noise_fraction",
            "mean_target_noise_trace",
            "observed_between_context_variance_trace_unbiased",
        ):
            np.testing.assert_allclose(
                actual["components"][name][key], expected["components"][name][key], rtol=1e-12
            )


def test_readout_rescore_matches_direct_same_direction_evaluation():
    rng = np.random.default_rng(18)
    training, target = rng.normal(size=(30, 4)), rng.normal(size=(8, 4))
    predictors = {
        "ridge": target + rng.normal(size=target.shape) * 0.3,
        "mlp": target + rng.normal(size=target.shape) * 0.2,
    }
    directions = {name: rng.normal(size=(4, 3)) for name in ("J", "R")}
    directions = {name: value / np.linalg.norm(value, axis=0) for name, value in directions.items()}
    ids, selected = list("abcdefgh"), list("bcfg")
    config = {
        "seed": 42,
        "diagnostics": {
            "controls_pool_multiplier": 8,
            "maximum_absolute_log_variance_mismatch": 0.2,
        },
        "statistics": {"bootstrap": {"draws": 32, "confidence": 0.95}},
    }
    original, arrays, _ = evaluate_direction_readouts(
        training, target, predictors, directions, [10, 11, 12], list(range(30)), ids, config
    )
    actual, samples = paired_readouts(original, arrays, selected, config)
    index = row_indices(ids, selected)
    expected, _, direct_samples = evaluate_direction_readouts(
        training,
        target[index],
        {name: value[index] for name, value in predictors.items()},
        directions,
        [10, 11, 12],
        list(range(30)),
        selected,
        config,
    )
    assert (
        actual["paired_context_bootstrap"]["counts_sha256"]
        == expected["paired_context_bootstrap"]["counts_sha256"]
    )
    for key, values in samples.items():
        np.testing.assert_allclose(
            values, direct_samples[key], rtol=1e-12, atol=1e-12, equal_nan=True
        )
    for predictor in predictors:
        np.testing.assert_allclose(
            actual["paired_lens_r2_difference"][predictor],
            expected["paired_lens_r2_difference"][predictor],
        )


def test_missing_primary_scoring_context_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="lacks a primary"):
        row_indices(list("abc"), list("abd"))
