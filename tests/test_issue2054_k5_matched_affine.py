"""Independent least-squares, retrieval, and leakage tests for paired scalar maps."""

import numpy as np

from explore_persona_space.analysis.mapping_baselines import knn_retrieval
from scripts.issue2054_k5_matched_affine import evaluate_both


def test_affine_truth_and_reverse_scale():
    """An exact translation/scaling must be recovered in both directions."""
    rng = np.random.default_rng(31)
    x = rng.normal(size=(100, 8))
    y = 0.4 * x + rng.normal(size=8)
    folds = np.arange(100) % 5
    for direction, (records, arrays) in enumerate(evaluate_both(x, y, folds)):
        np.testing.assert_allclose(arrays["scale"], 0.4 if direction == 0 else 2.5)
        for record in records:
            assert np.isclose(record["metrics"]["bias_scale"]["r2"], 1)
            assert record["metrics"]["bias_scale"]["euclidean_top1"] == 1


def test_independent_lstsq_oracle_retrieval_and_no_leakage():
    """Compare against a full intercept design and canonical retrieval on noisy data."""
    rng = np.random.default_rng(99)
    x = rng.normal(size=(100, 8))
    y = 0.6 * x + rng.normal(size=(100, 8)) + 2
    folds = np.arange(100) % 5
    original = evaluate_both(x, y, folds)
    for direction, (records, arrays) in enumerate(original):
        source, target = (x, y) if direction == 0 else (y, x)
        for fold in range(5):
            train, test = folds != fold, folds == fold
            # Joint regression with one scalar coefficient and a separate intercept per coordinate.
            design = np.column_stack([source[train].ravel(), np.tile(np.eye(8), (train.sum(), 1))])
            fitted = np.linalg.lstsq(design, target[train].ravel(), rcond=None)[0]
            np.testing.assert_allclose(arrays["scale"][fold], fitted[0], atol=1e-12)
            np.testing.assert_allclose(arrays["affine_bias"][fold], fitted[1:], atol=1e-12)
            pred = fitted[0] * source[test] + fitted[1:]
            ref = knn_retrieval(pred, target[test], metric="euclidean")
            metric = records[fold]["metrics"]["bias_scale"]
            assert metric["euclidean_top1"] == ref["acc_at_k"][1]
            assert metric["median_rank"] == ref["median_rank"]
    changed = y.copy()
    changed[folds == 0] += 1000
    altered = evaluate_both(x, changed, folds)
    for old, new in zip(original, altered, strict=True):
        np.testing.assert_array_equal(old[1]["scale"][0], new[1]["scale"][0])
        np.testing.assert_array_equal(old[1]["affine_bias"][0], new[1]["affine_bias"][0])
