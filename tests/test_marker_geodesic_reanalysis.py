"""Check the numerical and paired-resampling contracts of the marker reanalysis."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "marker_geodesic", SCRIPTS / "analyze_marker_geodesic.py"
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_gram_coordinates_preserve_local_metric_geodesics():
    """A rotated, lower-width input must preserve local metrics and graph distances."""
    rng = np.random.default_rng(20261002)
    original = rng.normal(size=(35, 60))
    original[:, :3] *= 8
    with threadpool_limits(limits=2):
        reconstructed = ANALYSIS.coordinates_from_gram(original @ original.T)
        left, _ = ANALYSIS.fit_geometry(original, 15, 3, 1e-4)
        right, _ = ANALYSIS.fit_geometry(reconstructed, 15, 3, 1e-4)
    for name in left:
        np.testing.assert_allclose(left[name], right[name], rtol=1e-8, atol=1e-8)


def test_batched_rank_correlation_matches_scipy_with_ties_and_missing_self():
    """Duplicates from bootstrap sampling and excluded self-pairs retain tie ranks."""
    x = np.array([[1, 1, 4, np.nan, 3], [np.nan, 2, 2, 4, 3]], dtype=float)
    y = np.array([[0, 0, 1, np.nan, 0.5], [np.nan, 3, 1, 1, 4]], dtype=float)
    result = ANALYSIS.row_rank_correlation(x, y)
    for i in range(2):
        valid = np.isfinite(x[i])
        np.testing.assert_allclose(result[i], spearmanr(x[i, valid], y[i, valid]).statistic)


def test_paired_bootstrap_identical_metrics_have_zero_difference():
    """Joint resampling must produce exactly zero change when metrics are identical."""
    rng = np.random.default_rng(18)
    y = rng.uniform(size=(5, 30))
    x = rng.normal(size=(5, 30))
    mask = np.ones_like(y, dtype=bool)
    mask[np.arange(5), np.arange(5)] = False
    keys = ["centered_cosine", "raw_cosine", "whitened_cosine", "graph_euclidean", "geodesic"]
    result = ANALYSIS.paired_bootstrap({key: x for key in keys}, y, mask, {"draws": 500, "seed": 2})
    for intervals in result["geodesic_minus"].values():
        np.testing.assert_array_equal(intervals["macro_delta_ci95"], [0, 0])
        np.testing.assert_array_equal(intervals["source_delta_ci95"], np.zeros((5, 2)))
