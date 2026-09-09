"""Paired-bootstrap, rank-adjustment, and held-out calibration regression tests."""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue2569_refusal_magnitude_comparison as magnitude


def test_count_reconstruction_preserves_mathematical_ties():
    """0.3 minus 0.2 and 0.1 minus zero represent the same one-refusal gap."""
    signed, gap = magnitude.exact_count_gaps(np.array([0.3, 0.1, 0]), np.array([0.2, 0, 0.1]))
    np.testing.assert_array_equal(signed, [1, 1, -1])
    np.testing.assert_array_equal(gap, [0.1, 0.1, 0.1])


def test_identical_scores_have_zero_paired_difference(monkeypatch):
    """Shared draws preserve perfect score equivalence in every bootstrap replicate."""
    monkeypatch.setattr(magnitude.base, "N_BOOT", 80)
    x = np.arange(12, dtype=float)
    result, draws = magnitude.compare_scores(
        np.stack([x, x, x + 1], axis=1),
        x[::-1],
        np.repeat(["a", "b", "c"], 4),
        np.repeat(["v", "u"], 6),
    )
    assert result["mapped_minus_context"]["delta_rho"] == 0
    assert result["mapped_minus_context"]["ci95"] == [0, 0]
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])


def test_paired_bootstrap_matches_serial_reference(monkeypatch):
    """Every batched score uses the identical family resample and SciPy-equivalent ranks."""
    monkeypatch.setattr(magnitude.base, "N_BOOT", 80)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 3))
    y = np.array([0, 0, 1, 1, 0, 0.5, 0.5, 1, 0, 0.2, 0.8, 1])
    groups = np.repeat(["a", "b", "c", "d"], 3)
    result, draws = magnitude.compare_scores(x, y, groups, np.repeat(["v", "u"], 6))
    idx = magnitude.base.cluster_indices(groups)
    expected = np.array(
        [[spearmanr(x[r[r >= 0], j], y[r[r >= 0]]).statistic for j in range(3)] for r in idx]
    )
    np.testing.assert_allclose(draws, expected, atol=1e-14)
    np.testing.assert_allclose(
        result["mapped_minus_context"]["ci95"],
        np.quantile(expected[:, 1] - expected[:, 0], [0.025, 0.975]),
    )


def test_rank_adjustment_matches_indicator_residualization():
    """Partial rank correlation equals explicit least-squares category residuals."""
    x = np.array([1.0, 2, 3, 4, 7, 6, 5, 8])
    y = np.array([0.0, 1, 0, 1, 0.4, 0.6, 0.2, 0.8])
    categories = np.repeat(["a", "b"], 4)
    xr = magnitude.rankdata(x)
    yr = magnitude.rankdata(y)
    design = np.stack([categories == "a", categories == "b"], axis=1).astype(float)
    rx = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    ry = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    np.testing.assert_allclose(
        magnitude.category_adjusted_rho(x, y, categories), np.corrcoef(rx, ry)[0, 1]
    )


def test_adjusted_bootstrap_recomputes_ranks(monkeypatch):
    """Duplicated cluster rows are ranked anew, not fixed full-bank residuals."""
    monkeypatch.setattr(magnitude.base, "N_BOOT", 80)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 3))
    y = rng.normal(size=12)
    groups = np.repeat(["a", "b", "c", "d"], 3)
    categories = np.tile(["u", "v"], 6)
    _, draws = magnitude.compare_scores(x, y, groups, categories, adjusted=True)
    idx = magnitude.base.cluster_indices(groups)
    for b, r in enumerate(idx[:10]):
        valid = r[r >= 0]
        for j in range(3):
            expected = magnitude.category_adjusted_rho(x[valid, j], y[valid], categories[valid])
            np.testing.assert_allclose(draws[b, j], expected, atol=1e-14)


def test_calibration_train_test_disjoint_and_exact_linear():
    """Batched scalar OLS recovers an exact law without evaluated-family leakage."""
    x = np.arange(1, 13, dtype=float)
    predictors = np.stack([x, 2 * x, 3 * x], axis=1)
    y = 0.03 * x + 0.1
    groups = np.repeat(["a", "b", "c"], 4)
    pred, baseline, folds = magnitude.calibrate_lofo(predictors, y, groups)
    np.testing.assert_allclose(pred, np.repeat(y[:, None], 3, axis=1), atol=1e-12)
    changed = y.copy()
    changed[:4] += 50
    updated, _, _ = magnitude.calibrate_lofo(predictors, changed, groups)
    np.testing.assert_allclose(updated[:4], pred[:4], atol=1e-12)
    for fold in folds:
        assert not set(fold["test_indices"]) & set(fold["train_indices"])
    assert magnitude.predictive_metrics(pred, y, baseline)["context_norm"]["r2"] == 1


def test_constant_outcome_is_explicitly_undefined():
    """Floor saturation never becomes a zero correlation claim."""
    x = np.stack([np.arange(8), np.arange(8) + 1, np.arange(8) + 2], axis=1)
    result, draws = magnitude.compare_scores(
        x, np.zeros(8), np.repeat(["a", "b"], 4), np.repeat(["a", "b"], 4)
    )
    assert draws is None and result["mapped_minus_context"]["delta_rho"] is None
    assert all(r["rho"] is None for r in result["scores"].values())
