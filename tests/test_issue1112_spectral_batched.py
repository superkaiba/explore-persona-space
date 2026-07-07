"""#1112 — Gram-space batched bootstrap must bit-reproduce the serial spectra.

The batched path (batched_dvs_over_indices: double-centered sub-Gram +
eigvalsh) must reproduce spectral_dvs(svd_of_cloud(X[idx])) for arbitrary
index draws within float tolerance (the vectorize-many-cell-fits.md
equivalence-gate requirement), including the paired-index cross-cell shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from explore_persona_space.experiments.issue_653.spectral import (
    batched_cluster_bootstrap,
    batched_dvs_over_indices,
    bootstrap_index_matrix,
    spectral_dvs,
    svd_of_cloud,
)

torch = pytest.importorskip("torch")
torch.set_num_threads(2)  # tiny matrices; avoid shared-VM thread thrash


def _random_cloud(n: int, d: int, seed: int, *, low_rank: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if low_rank is None:
        return rng.standard_normal((n, d))
    A = rng.standard_normal((n, low_rank))
    B = rng.standard_normal((low_rank, d))
    return A @ B + 0.01 * rng.standard_normal((n, d))


@pytest.mark.parametrize(
    ("n", "d", "low_rank"),
    [(24, 64, None), (24, 64, 2), (30, 16, None)],  # incl. d < n (mode-count parity)
)
def test_batched_matches_serial_per_draw(n: int, d: int, low_rank: int | None) -> None:
    """Per-draw batched DVs == serial spectral_dvs(svd_of_cloud(X[idx]))."""
    X = _random_cloud(n, d, seed=7, low_rank=low_rank)
    ids = np.arange(n)  # one row per cluster (the #1112 cloud shape)
    idx = bootstrap_index_matrix(ids, n_boot=40, seed=11)
    draws = batched_dvs_over_indices(X, idx, chunk=16)

    for b in range(idx.shape[0]):
        ref = spectral_dvs(svd_of_cloud(X[idx[b]], center_rows=True))
        assert draws["rank_k_at_90"][b] == ref["rank_k_at_90"], b
        np.testing.assert_allclose(
            draws["top_share_lambda"][b], ref["top_share_lambda"], rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(draws["pr_lambda"][b], ref["pr_lambda"], rtol=1e-8, atol=1e-10)


def test_paired_indices_shared_across_cells() -> None:
    """The SAME index matrix applied to two clouds gives paired difference draws."""
    ids = [f"c{i}_q{j}" for i in range(3) for j in range(8)]
    Xa = _random_cloud(24, 32, seed=1)
    Xb = _random_cloud(24, 32, seed=2)
    idx = bootstrap_index_matrix(ids, n_boot=25, seed=5)
    da = batched_dvs_over_indices(Xa, idx, dv_names=("rank_k_at_90",), chunk=10)
    db = batched_dvs_over_indices(Xb, idx, dv_names=("rank_k_at_90",), chunk=10)
    diff = da["rank_k_at_90"] - db["rank_k_at_90"]
    assert diff.shape == (25,)
    # Paired = every draw used identical resampled (context, question) rows:
    ref_a = spectral_dvs(svd_of_cloud(Xa[idx[0]]))["rank_k_at_90"]
    ref_b = spectral_dvs(svd_of_cloud(Xb[idx[0]]))["rank_k_at_90"]
    assert diff[0] == ref_a - ref_b


def test_bootstrap_index_matrix_rejects_unequal_clusters() -> None:
    ids = ["a", "a", "b"]  # cluster a has 2 rows, b has 1
    with pytest.raises(ValueError, match="equal cluster sizes"):
        bootstrap_index_matrix(ids, n_boot=3, seed=0)


def test_batched_cluster_bootstrap_summary_shape() -> None:
    X = _random_cloud(20, 12, seed=3)
    res = batched_cluster_bootstrap(np.asarray(X), np.arange(20), n_boot=30, seed=9, chunk=8)
    assert res["n_boot"] == 30
    for dv in ("top_share_lambda", "pr_lambda", "rank_k_at_90"):
        lo, hi = res["ci"][dv]
        assert lo <= hi
        assert res["draws"][dv].shape == (30,)
        # the point estimate matches the serial full-cloud read
        ref = spectral_dvs(svd_of_cloud(X))[dv]
        np.testing.assert_allclose(res["point"][dv], ref, rtol=1e-8, atol=1e-10)


def test_serial_reference_tombstoned() -> None:
    """The serial twin warns FutureWarning and raises under EPM_FORBID_SERIAL_FITS=1."""
    import os
    from unittest import mock

    from explore_persona_space.experiments.issue_653.spectral import cluster_bootstrap_dv

    X = _random_cloud(12, 8, seed=4)
    with pytest.warns(FutureWarning, match="batched"):
        cluster_bootstrap_dv(X, np.arange(12), "pr_lambda", n_boot=3, seed=1)
    with (
        mock.patch.dict(os.environ, {"EPM_FORBID_SERIAL_FITS": "1"}),
        pytest.warns(FutureWarning),
        pytest.raises(RuntimeError, match="batched"),
    ):
        cluster_bootstrap_dv(X, np.arange(12), "pr_lambda", n_boot=3, seed=1)
