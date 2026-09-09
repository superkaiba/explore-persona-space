"""Exact small-matrix oracles for necessity-stratified rank analysis."""

import numpy as np
import pytest
import torch

from scripts import issue2546_necessity_rank as analysis


def test_row_curves_and_bootstrap_equal_explicit_projection():
    """Test the production batched kernel, including rank zero and nonmonotone curves."""
    torch.manual_seed(13)
    x, y = torch.randn(70, 8).double(), torch.randn(70, 8).double()
    model = analysis.parent.fit(analysis.parent.moments(x[:50], y[:50]), 2.0)
    prediction = analysis.parent.predict(model, x[50:])
    zero, increments = analysis.curve_rows(prediction, y[50:], model["ymu"], model["vectors"])
    counts = analysis.bootstrap_counts(np.array(["a"] * 8 + ["b"] * 12), 30, 10)
    ranks = analysis.ranks_for_counts(zero, increments, counts)
    errors = []
    for r in range(9):
        basis = model["vectors"][:, :r]
        low = (prediction - model["ymu"]) @ basis @ basis.T + model["ymu"]
        errors.append((y[50:] - low).square().sum(1).numpy())
    direct = np.array(errors).T
    np.testing.assert_allclose(analysis.aggregate_curve(zero, increments), direct.sum(0))
    for b in range(30):
        expected = analysis.parent.select_rank(counts[b] @ direct, 0.1)
        assert ranks[b] == expected
    np.testing.assert_array_equal(counts[:, :8].sum(1), [8] * 30)
    np.testing.assert_array_equal(counts[:, 8:].sum(1), [12] * 30)
    np.testing.assert_array_equal(counts, analysis.bootstrap_counts(["a"] * 8 + ["b"] * 12, 30, 10))


def test_spectral_measures_and_subset_centering():
    """Eigenvalue normalization matches the prior chat control, including translation invariance."""
    eigen = torch.diag(torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64))
    result = analysis.measures(eigen)
    assert result["participation_ratio"] == pytest.approx(25 / 17)
    assert result["stable_rank"] == pytest.approx(1.25)
    assert result["effective_rank_entropy"] == pytest.approx(
        np.exp(-(0.8 * np.log(0.8) + 0.2 * np.log(0.2)))
    )
    torch.manual_seed(2)
    x = torch.randn(50, 6).double()
    y = x @ torch.randn(6, 6).double() + 2
    model = analysis.parent.fit(analysis.parent.moments(x, y), 5)
    stats = analysis.diversity(model, x[:15], y[:15])
    shifted = analysis.diversity(model, x[:15] + 8, y[:15] + 20)
    for key in ("input", "answer", "fitted_output"):
        assert stats[key]["participation_ratio"] == pytest.approx(
            shifted[key]["participation_ratio"]
        )
    preds = analysis.parent.predict(model, x[:15])
    pc = preds - preds.mean(0)
    direct = analysis.measures(pc.T @ pc)
    assert stats["fitted_output"]["effective_rank_entropy"] == pytest.approx(
        direct["effective_rank_entropy"]
    )


def test_denominator_uses_training_rows_only():
    """Subset scoring must not recenter the reference on the held-out outcomes."""
    y = torch.tensor(
        [[1.0, 1.0], [3.0, 3.0], [10.0, 10.0], [12.0, 12.0], [5.0, 5.0], [20.0, 20.0]],
        dtype=torch.float64,
    )
    corpus = np.array(["a", "a", "b", "b", "a", "b"])
    train = np.array([True] * 4 + [False] * 2)
    global_sst, within = analysis.row_sst(y, train, ~train, corpus)
    np.testing.assert_allclose(within, [18, 162])
    np.testing.assert_allclose(global_sst, ((y[~train] - y[train].mean(0)) ** 2).sum(1))


def test_empty_and_non_psd_fail_loud():
    """No empty subset or malformed spectrum is silently replaced by zero."""
    with pytest.raises(ValueError):
        analysis.bootstrap_counts([], 3, 1)
    with pytest.raises(ValueError):
        analysis.measures(torch.diag(torch.tensor([1.0, -1.0])))
