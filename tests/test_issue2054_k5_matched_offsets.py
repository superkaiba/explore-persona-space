"""Independent numerical checks for the direct paired constant-shift test."""

import numpy as np

from explore_persona_space.analysis.mapping_baselines import knn_retrieval
from scripts.issue2054_k5_matched_offsets import evaluate_pair


def test_constant_translation_and_retrieval_oracle():
    """A true fixed translation explains all displacement, including held-out rows."""
    rng = np.random.default_rng(77)
    source = rng.normal(size=(100, 12))
    target = source + rng.normal(size=12) * 3
    folds = np.arange(100) % 5
    records, arrays = evaluate_pair(source, target, folds)
    for record in records:
        test = folds == record["fold"]
        assert np.isclose(record["constant_fraction"], 1)
        assert np.isclose(record["metrics"]["bias"]["r2"], 1)
        for name in ("identity", "bias"):
            pred = source[test] + (arrays["bias"][record["fold"]] if name == "bias" else 0)
            oracle = knn_retrieval(pred, target[test], metric="euclidean")
            assert record["metrics"][name]["euclidean_top1"] == oracle["acc_at_k"][1]
            assert record["metrics"][name]["median_rank"] == oracle["median_rank"]


def test_no_target_test_leakage_and_reversed_pair():
    """Changing test targets cannot affect their learned shift; reversal is symmetric."""
    rng = np.random.default_rng(21)
    source = rng.normal(size=(100, 12))
    target = source * 0.4 + rng.normal(size=(100, 12)) + 2
    folds = np.arange(100) % 5
    records, arrays = evaluate_pair(source, target, folds)
    altered = target.copy()
    altered[folds == 0] += 100
    _, changed = evaluate_pair(source, altered, folds)
    np.testing.assert_array_equal(arrays["bias"][0], changed["bias"][0])
    reversed_records, reversed_arrays = evaluate_pair(target, source, folds)
    np.testing.assert_allclose(arrays["bias"], -reversed_arrays["bias"])
    for record, reverse in zip(records, reversed_records, strict=True):
        assert np.isclose(record["constant_fraction"], reverse["constant_fraction"])
        test = folds == record["fold"]
        bias = (target[~test] - source[~test]).mean(0)
        pred = source[test] + bias
        oracle = knn_retrieval(pred, target[test], metric="euclidean")
        assert record["metrics"]["bias"]["euclidean_top1"] == oracle["acc_at_k"][1]
