"""Independent numerical checks for training-K regression and cluster inference."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import issue1901_training_k10_fit as fit


@pytest.mark.parametrize("rank_deficient", [False, True])
def test_primal_gcv_matches_direct_hat_matrix_with_intercept(rank_deficient):
    rng = np.random.default_rng(483)
    x = rng.normal(size=(37, 7)) + np.arange(7)
    if rank_deficient:
        x[:, 5] = 2 * x[:, 1] - x[:, 2]
        x[:, 6] = 4
    y = rng.normal(size=(37, 3)) + 12
    lambdas = np.array([0.01, 1.0, 100.0])
    fac = fit.factorize(x)
    got = fit.gcv_solution(fac, y, lambdas)
    xn = (x - x.mean(0)) / (x.std(0, ddof=1) + 1e-9)
    scores, rss, dfs, coefs = [], [], [], []
    for lam in lambdas:
        inverse = np.linalg.inv(xn.T @ xn + lam * np.eye(x.shape[1]))
        hat = np.ones((len(x), len(x))) / len(x) + xn @ inverse @ xn.T
        residual = y - hat @ y
        rss.append(np.square(residual).sum())
        dfs.append(np.trace(hat))
        scores.append(rss[-1] / (len(x) - dfs[-1]) ** 2)
        coefs.append(inverse @ xn.T @ (y - y.mean(0)))
    np.testing.assert_allclose(got["gcv"], scores, rtol=1e-10)
    np.testing.assert_allclose(got["rss"], rss, rtol=1e-10)
    np.testing.assert_allclose(got["df"], dfs, rtol=1e-10)
    selected = int(np.argmin(scores))
    assert got["selected_lambda"] == lambdas[selected]
    np.testing.assert_allclose(fac["basis"] @ got["coefficient_eigen"], coefs[selected], atol=1e-12)


def test_cluster_bootstrap_matches_explicit_variable_length_resamples():
    rng = np.random.default_rng(81)
    hashes = ["a", "a", "a", "b", "c", "d", "d"]
    row_counts, query_counts, _ = fit.cluster_resampling(hashes, np.array([0, 3, 4, 5]), 31, 142)
    assert np.unique(row_counts.sum(1)).size > 1
    np.testing.assert_array_equal(row_counts[:, 0], row_counts[:, 1])
    np.testing.assert_array_equal(query_counts.sum(1), np.full(31, 4))
    target = rng.normal(size=(7, 4))
    prediction = rng.normal(size=(2, 7, 4))
    point, boots, _ = fit.bootstrap_r2(target, prediction, row_counts)
    for b, counts in enumerate(row_counts):
        ix = np.repeat(np.arange(7), counts)
        y = target[ix]
        expected = 1 - np.square(prediction[:, ix] - y).sum((1, 2)) / np.square(y - y.mean(0)).sum()
        np.testing.assert_allclose(boots[:, b], expected, rtol=1e-12)
    expected = (
        1 - np.square(prediction - target).sum((1, 2)) / np.square(target - target.mean(0)).sum()
    )
    np.testing.assert_allclose(point, expected)


def test_retrieval_pool_cannot_repeat_or_omit_prompt_clusters():
    with pytest.raises(ValueError, match="representatives"):
        fit.cluster_resampling(["a", "a", "b", "c"], np.array([0, 1, 2]), 10, 42)


def test_gcv_rejects_nonfinite_training_target():
    fac = fit.factorize(np.arange(20).reshape(10, 2))
    y = np.zeros((10, 2))
    y[0, 0] = np.nan
    with pytest.raises(ValueError, match="targets"):
        fit.gcv_solution(fac, y)
