"""Candidate-pool invariants and parity with the existing retrieval instrument."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue1901_figure2_retrieval_pool as M


def test_exact_pool_keeps_queries_and_deduplicates_distractors():
    rng = np.random.default_rng(5)
    unique = rng.normal(size=(28, 16)).astype(np.float32)
    source = np.concatenate([unique[:14], unique[[0, 2]], unique[[1]], unique[14:]])
    view = M.exact_pool_view(source, 16, 20)
    assert len(view.pool_rows) == 20
    np.testing.assert_array_equal(view.pred_rows, np.arange(14))
    np.testing.assert_array_equal(view.true_idx, np.arange(14))
    assert view.diagnostics["source_n_pool"] == 23
    assert view.diagnostics["realized_n_excess_duplicate_classes"] == 0
    with pytest.raises(ValueError):
        M.exact_pool_view(source, 16, 13)
    with pytest.raises(ValueError):
        M.exact_pool_view(source, 16, 29)


def test_precomputed_score_matches_parent_on_large_and_original_pools():
    rng = np.random.default_rng(8)
    source = rng.normal(size=(36, 16)).astype(np.float32)
    source[15] = source[0]
    pred = source[:16] + rng.normal(size=(16, 16)) * 2.5
    pool = source + rng.normal(size=source.shape) * 0.03
    scale = np.linspace(0.6, 1.7, 16)

    def whiten(x):
        return np.asarray(x, dtype=np.float64) * scale

    full = M.FINAL._precompute_metric_arrays(pred, pool, whiten(pred), whiten(pool))
    csls_changes_ranks = False
    for size in (15, 28):
        view = M.exact_pool_view(source, 16, size)
        actual = M.score_geometry(full, view, seed=72)
        expected = M.FINAL.score_cell(pred, pool, view, whiten, seed=72)
        for metric, cell in actual.items():
            ranks = np.asarray(cell["per_query_ranks"])
            assert cell["acc_at_k"] == expected[metric]["strict"]["acc_at_k"]
            assert cell["acc1_ci95"] == expected[metric]["strict"]["acc1_ci95"]
            assert cell["mrr"] == expected[metric]["strict"]["mrr"]
            assert np.mean(ranks <= 1) == cell["acc_at_k"]["1"]
            assert np.any(ranks > 1), "parity fixture must exercise retrieval failures"
        csls_changes_ranks |= (
            actual["whiten_csls"]["per_query_ranks"] != actual["whiten_cosine"]["per_query_ranks"]
        )
    assert csls_changes_ranks, "parity fixture must detect omitted CSLS correction"
