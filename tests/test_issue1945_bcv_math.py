"""Pure-math regression tests for the #1945 Gabriel (2,2) BCV rank battery.

Self-contained (synthetic matrices only — no staged-data dependency, <30 s):

- ``twoway_removed`` closure (grand/row/column means removed; batched == per-matrix)
- ``bcv_curve_batched`` == a naive truncated-SVD reconstruction oracle
  (the cumulative Gram-eigh trick vs the definition Ahat11(r) = A12 V_r S_r^-1 U_r^T A21)
- planted rank-3 structure is RECOVERED out-of-block (R^2 high at r >= 3),
  pure noise is not; the r=0 baseline column is exactly 0
- ``bcv_per_block`` averages to the pooled curve (true single-block curves)
- degenerate-input gates fire (zero held-out block; parent ``two_way`` n<2 / constant R)
- ``_rrr_curves`` == a naive SVD-of-fitted-values reduced-rank oracle
- ``_batched_null_ridge`` (identity draw) == ``ridge_fit_predict_fast`` on BOTH
  the test and train predictions (pins the eigenbasis-rotation contract)

Dims are chosen big enough to catch shape/transpose bugs (n=40, k=24).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1945_bcv_interaction as bcv  # noqa: E402
from issue1482_twoway_residual import two_way  # noqa: E402

N, K = 40, 24
R_GRID = [1, 2, 4, 6]


def _splits(n: int = N, k: int = K):
    return bcv.bcv_splits(n, k, seed=7)


def _naive_bcv(F: np.ndarray, r_grid: list[int], rows, cols) -> np.ndarray:
    """Definition-level oracle: per block, truncated SVD of A22, reconstruct A11."""
    curves = np.zeros((4, 1 + len(r_grid)))
    for bi, (ri, rj, ci, cj) in enumerate(bcv._bcv_blocks(rows, cols)):
        A11 = F[np.ix_(ri, ci)]
        A12 = F[np.ix_(ri, cj)]
        A21 = F[np.ix_(rj, ci)]
        A22 = F[np.ix_(rj, cj)]
        U, s, Vt = np.linalg.svd(A22, full_matrices=False)
        ss11 = float((A11**2).sum())
        for t, r in enumerate(r_grid):
            ahat = A12 @ Vt[:r].T @ np.diag(1.0 / s[:r]) @ U[:, :r].T @ A21
            curves[bi, 1 + t] = 1.0 - float(((A11 - ahat) ** 2).sum()) / ss11
    return curves


def test_twoway_removed_closure():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((N, K)) + rng.standard_normal(N)[:, None] * 3.0
    F = bcv.twoway_removed(M)
    assert abs(F.mean()) < 1e-12
    assert np.abs(F.mean(axis=1)).max() < 1e-12
    assert np.abs(F.mean(axis=0)).max() < 1e-12
    # matches the explicit mu + a_i + b_j residual
    mu = M.mean()
    a = M.mean(axis=1) - mu
    b = M.mean(axis=0) - mu
    np.testing.assert_allclose(F, M - mu - a[:, None] - b[None, :], atol=1e-12)
    # batched removal == per-matrix removal
    Mb = rng.standard_normal((3, N, K))
    Fb = bcv.twoway_removed(Mb)
    for i in range(3):
        np.testing.assert_allclose(Fb[i], bcv.twoway_removed(Mb[i]), atol=1e-12)


def test_bcv_matches_naive_svd_oracle():
    rng = np.random.default_rng(1)
    F = bcv.twoway_removed(rng.standard_normal((N, K)))
    rows, cols = _splits()
    oracle = _naive_bcv(F, R_GRID, rows, cols)
    pooled = bcv.bcv_curve_batched(F[None], R_GRID, rows, cols)[0]
    np.testing.assert_allclose(pooled, oracle.mean(axis=0), atol=1e-9)
    per_block = bcv.bcv_per_block(F, R_GRID, rows, cols)
    np.testing.assert_allclose(per_block, oracle, atol=1e-9)
    # per-block curves must actually DIFFER across blocks (true single blocks,
    # not 4 copies of the average — the drafting bug this test pins)
    assert np.std(per_block[:, -1]) > 1e-12
    np.testing.assert_allclose(per_block.mean(axis=0), pooled, atol=1e-12)
    # batched call == stacked single calls
    F2 = bcv.twoway_removed(rng.standard_normal((N, K)))
    both = bcv.bcv_curve_batched(np.stack([F, F2]), R_GRID, rows, cols)
    np.testing.assert_allclose(both[1], bcv.bcv_curve_batched(F2[None], R_GRID, rows, cols)[0])


def test_bcv_block_predictions_match_oracle():
    rng = np.random.default_rng(5)
    F = bcv.twoway_removed(rng.standard_normal((N, K)))
    rows, cols = _splits()
    r = 4
    pairs = bcv.bcv_block_predictions(F, r, rows, cols)
    for (a11, ahat), (ri, rj, ci, cj) in zip(pairs, bcv._bcv_blocks(rows, cols), strict=True):
        A12, A21, A22 = F[np.ix_(ri, cj)], F[np.ix_(rj, ci)], F[np.ix_(rj, cj)]
        U, s, Vt = np.linalg.svd(A22, full_matrices=False)
        oracle = A12 @ Vt[:r].T @ np.diag(1.0 / s[:r]) @ U[:, :r].T @ A21
        np.testing.assert_allclose(a11, F[np.ix_(ri, ci)], atol=0)
        np.testing.assert_allclose(ahat, oracle, atol=1e-9)


def test_bcv_planted_rank_recovery_and_noise_floor():
    rng = np.random.default_rng(2)
    U = rng.standard_normal((N, 3))
    V = rng.standard_normal((3, K))
    F_sig = bcv.twoway_removed(5.0 * (U @ V) + 0.01 * rng.standard_normal((N, K)))
    rows, cols = _splits()
    curve = bcv.bcv_curve_batched(F_sig[None], R_GRID, rows, cols)[0]
    assert curve[0] == 0.0  # r=0 baseline exactly zero by construction
    r_idx = {r: 1 + t for t, r in enumerate(R_GRID)}
    assert curve[r_idx[4]] > 0.9, f"planted rank-3 not recovered: {curve}"
    assert curve[r_idx[4]] > curve[r_idx[1]], "rank-3 structure should beat rank-1"
    # pure noise: held-out prediction cannot explain much variance
    F_noise = bcv.twoway_removed(rng.standard_normal((N, K)))
    noise_curve = bcv.bcv_curve_batched(F_noise[None], R_GRID, rows, cols)[0]
    assert noise_curve[1:].max() < 0.3, f"noise-only BCV too high: {noise_curve}"


def test_degenerate_gates_fire():
    rows, cols = _splits()
    with pytest.raises(AssertionError, match="degenerate held-out block"):
        bcv.bcv_curve_batched(np.zeros((1, N, K)), R_GRID, rows, cols)
    with pytest.raises(ValueError, match="n>=2 and k>=2"):
        two_way(np.ones((1, 5)))
    with pytest.raises(ValueError, match="zero total sum of squares"):
        two_way(np.ones((4, 4)))


def test_r_grid_truncation():
    assert bcv.r_grid_for_k(64) == [1, 2, 4, 8, 16]  # k2=32 -> r <= 16
    assert bcv.r_grid_for_k(256) == [1, 2, 4, 8, 16, 32, 64]  # full grid


def test_rrr_curves_match_naive():
    rng = np.random.default_rng(3)
    B, n_tr, n_te, k = 2, 30, 26, 8
    rg = [1, 2, 4]
    yhat_tr = rng.standard_normal((B, n_tr, k))
    yhat_te = rng.standard_normal((B, n_te, k))
    f_te = rng.standard_normal((B, n_te, k))
    got = bcv._rrr_curves(yhat_tr, yhat_te, f_te, rg)
    for b in range(B):
        _u, _s, Vt = np.linalg.svd(yhat_tr[b], full_matrices=False)
        ss = float((f_te[b] ** 2).sum())
        for t, r in enumerate(rg):
            proj = yhat_te[b] @ Vt[:r].T @ Vt[:r]
            naive = 1.0 - float(((f_te[b] - proj) ** 2).sum()) / ss
            np.testing.assert_allclose(got[b, 1 + t], naive, atol=1e-9)
        assert got[b, 0] == 0.0


def test_torch_twins_match_numpy():
    import torch

    rng = np.random.default_rng(6)
    Rk = rng.random((3, N, K)) + 0.1
    lam = rng.random(K) + 0.5
    for space in ("log", "raw", "normalized"):
        ref = bcv.space_transform(Rk, lam, space)
        got = bcv._space_transform_t(torch.from_numpy(Rk), torch.from_numpy(lam), space)
        np.testing.assert_allclose(got.numpy(), ref, atol=1e-12)
    with pytest.raises(ValueError, match="unknown space"):
        bcv._space_transform_t(torch.from_numpy(Rk), torch.from_numpy(lam), "bogus")
    M = rng.standard_normal((3, N, K))
    np.testing.assert_allclose(
        bcv._twoway_removed_t(torch.from_numpy(M)).numpy(), bcv.twoway_removed(M), atol=1e-12
    )


def test_batched_null_ridge_matches_fast_solver():
    from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict_fast

    rng = np.random.default_rng(4)
    n_tr, n_te, d, k = 40, 36, 12, 6
    X_tr = rng.standard_normal((n_tr, d))
    X_te = rng.standard_normal((n_te, d))
    Y = rng.standard_normal((n_tr + n_te, k))
    lambdas = np.logspace(-2, 4, 13)
    bt_tr, bt_te, best_lam, dof = bcv._batched_null_ridge(X_tr, X_te, Y[None], lambdas)
    ref, info = ridge_fit_predict_fast(
        X_tr, Y[:n_tr], np.concatenate([X_tr, X_te]), device="cpu", return_info=True
    )
    # the twin must agree on BOTH the train fitted values (feeds the RRR
    # truncation) and the test predictions — the eigenbasis-rotation contract
    np.testing.assert_allclose(bt_tr[0], ref[:n_tr], atol=1e-8)
    np.testing.assert_allclose(bt_te[0], ref[n_tr:], atol=1e-8)
    assert best_lam[0] == pytest.approx(info["best_lambda"])
    assert dof[0] == pytest.approx(info["dof"], rel=1e-6)
