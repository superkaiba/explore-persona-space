"""Scientific correctness checks for conditional rank comparisons and joins."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from issue1482_answer_property_analysis import (
    concordance_bootstrap,
    nested_bins,
    quantile_bins,
)
from issue1482_answer_property_matryoshka import join_index


def test_concordance_matches_auc_with_ties_and_paired_outcomes():
    y = np.array([0, 1, 1, 2, 3, 3, 4, 1, 5, 2], float)
    g = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0], bool)
    result = concordance_bootstrap(
        np.column_stack([y, 4 * y, -y]), g, np.zeros(len(y)), n_boot=40
    )
    np.testing.assert_allclose(
        result["point"],
        [roc_auc_score(g, y), roc_auc_score(g, y), roc_auc_score(g, -y)],
    )
    np.testing.assert_array_equal(result["draws"][:, 0], result["draws"][:, 1])
    np.testing.assert_allclose(result["draws"][:, 0] + result["draws"][:, 2], 1)


def test_stratified_concordance_exact_brute_pairs():
    y = np.array([0, 4, 3, 1, 5, 5, 9, 8], float)
    g = np.array([0, 1, 0, 1, 0, 1, 1, 1], bool)
    cells = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    brute = []
    for i in range(len(y)):
        for j in range(len(y)):
            if g[i] and not g[j] and cells[i] == cells[j]:
                brute.append((y[i] > y[j]) + 0.5 * (y[i] == y[j]))
    result = concordance_bootstrap(y, g, cells, n_boot=0)
    assert result["point"][0] == np.mean(brute)
    assert result["n_pairs"] == len(brute)
    assert result["n_positive_matched"] == 3
    assert result["ci95"] is None


def test_quantile_bins_preserve_ties_and_nested_parent():
    values = np.repeat(np.arange(8), 10)
    first = quantile_bins(values, 4)
    second = nested_bins(first, np.arange(80), 5)
    np.testing.assert_array_equal(second // 5, first)
    for value in np.unique(values):
        assert len(np.unique(first[values == value])) == 1


def test_join_index_exact_and_rejects_bad_population():
    np.testing.assert_array_equal(
        join_index(np.array([2, 1, 4]), np.array([4, 2, 1])), [1, 2, 0]
    )
    with pytest.raises(ValueError, match="duplicate"):
        join_index(np.array([1, 1]), np.array([1, 2]))
    with pytest.raises(ValueError, match="population"):
        join_index(np.array([1, 3]), np.array([1, 2]))


def test_no_supported_pairs_fails():
    with pytest.raises(ValueError, match="overlapping"):
        concordance_bootstrap(np.arange(4), np.ones(4, bool), np.zeros(4))
