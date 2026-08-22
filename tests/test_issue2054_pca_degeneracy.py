"""`_pca_topk` degeneracy contract (#2054 Gap B).

A shard died mid-run on::

    RuntimeError: per-cell PCA: negative eigenvalue in top-128 (w_min=-2.02905e-14)

``xc.T @ xc`` is PSD by construction, so every eigenvalue is >= 0 in exact
arithmetic; ``eigh`` on a near-singular PSD matrix returns machine-epsilon-scale
NEGATIVES for directions the data does not span. A bare ``w < 0`` test therefore
trips on roundoff rather than on a real defect. The absolute variance floor only
caught EXACTLY-constant X, so a NEAR-constant cell cleared the floor and then ran
out of spectrum.

The fix routes both shapes to a NAMED, RECORDED degeneracy (M2 skipped, M1
substituted) instead of killing the shard — while still RAISING on a genuinely
unusable input, which is the property a roundoff tolerance must not erode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2054_pool_specialize as ps

N, D, K = 400, 64, 16


def _rank_r_data(rng, n, d, r):
    """n x d data whose centered rank is exactly r (r << k, so top-k is starved)."""
    return rng.normal(size=(n, r)) @ rng.normal(size=(r, d))


def test_healthy_cell_returns_real_basis():
    rng = np.random.default_rng(0)
    out = ps._pca_topk(rng.normal(size=(N, D)), K)
    assert not isinstance(out, str), f"healthy cell degraded to {out!r}"
    mu, comps = out
    assert mu.shape == (D,)
    assert comps.shape == (D, K)
    # Components are an orthonormal basis of real directions.
    assert np.allclose(comps.T @ comps, np.eye(K), atol=1e-8)


def test_exactly_constant_x_is_named_not_raised():
    rng = np.random.default_rng(1)
    x = np.tile(rng.normal(size=(1, D)), (N, 1))
    assert ps._pca_topk(x, K) == "constant_x"


def test_rank_deficient_topk_is_named_not_raised():
    """rank(X) < k_max: the cell cannot supply k_max directions."""
    rng = np.random.default_rng(2)
    assert ps._pca_topk(_rank_r_data(rng, N, D, 4), K) == "rank_deficient_topk"


def test_near_constant_clears_variance_floor_but_is_still_named():
    """THE REGRESSION: near-constant X clears the absolute floor, then starves
    the top-k spectrum and previously raised on a -1e-14 roundoff eigenvalue."""
    rng = np.random.default_rng(3)
    x = np.tile(rng.normal(size=(1, D)), (N, 1)) + 1e-5 * _rank_r_data(rng, N, D, 4)
    # Precondition: it really does clear the constant-X floor, so this test
    # exercises the spectral path and not the floor.
    xc = x - x.mean(axis=0)
    assert float((xc**2).mean(axis=0).max()) >= ps.CONSTANT_X_VAR_FLOOR
    assert ps._pca_topk(x, K) == "rank_deficient_topk"


def test_nonfinite_input_still_raises():
    """A roundoff tolerance must NOT absorb an upstream defect."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(N, D))
    x[3, 7] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        ps._pca_topk(x, K)


def test_too_few_rows_still_raises():
    rng = np.random.default_rng(5)
    with pytest.raises(RuntimeError, match="n_train > k_max"):
        ps._pca_topk(rng.normal(size=(K, D)), K)


def test_psd_eig_tol_scales_with_leading_eigenvalue():
    """The tolerance must track the spectrum's scale, not be a fixed constant —
    otherwise it is either useless on large-scale Grams or absorbs real signal
    on small-scale ones."""
    small = ps._psd_eig_tol(np.array([1.0, 0.0]), N, D)
    large = ps._psd_eig_tol(np.array([1e6, 0.0]), N, D)
    assert large > small
    assert small > 0.0
    # The measured failure (-2.03e-14) must fall INSIDE tolerance at a realistic
    # Gram scale (sums of squares over thousands of rows), i.e. be treated as
    # roundoff rather than as a defect.
    assert ps._psd_eig_tol(np.array([1e3, 0.0]), 6400, 3584) > 2.03e-14
