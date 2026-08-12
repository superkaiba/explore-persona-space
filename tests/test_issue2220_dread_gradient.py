"""A9 gate (plan §12) for task #2220: the fitted scorer's analytic input-space
gradient must equal a finite-difference of the scorer, end-to-end through the
REAL #1739 fit stack (fit_whitening -> apply_whitening ->
ridge_fit_predict_primal_layer_batched), on ~3 (behavior x layer) cells.

CPU-only, tiny synthetic fits, no GPU, no HF download. This gate is what
certifies the `materialize_directions` fold `d_read = normalize(wh.w @
(w_ridge / sigma_z))` — the analytic gradient of the whiten->standardize->
linear-ridge scorer — before any pod run.
"""

from __future__ import annotations

import numpy as np
import pytest

import scripts.issue2220_readwrite as rw
from explore_persona_space.experiments.issue_1739 import fits


def _fit_cells(*, n_layers=3, d=8, n_u=200, n_l=60, seed=0):
    """Build tiny synthetic cells and fit the REAL stack.

    Returns per-layer (wh_mu, wh_w, mu_z, sigma_z, w_ridge, ymu) so a caller
    can both reconstruct the scorer (scorer_predict) AND fold d_read.
    """
    rng = np.random.default_rng(seed)
    # U pool for whitening; labeled set for the ridge fit (n_l > d -> well-posed,
    # though the primal solver is d x d and does not require it).
    x_u = rng.standard_normal((n_layers, n_u, d))
    x_lab = rng.standard_normal((n_layers, n_l, d))
    # a per-layer linear-ish target with noise so the ridge weight is non-degenerate
    beta = rng.standard_normal((n_layers, d))
    dv = np.einsum("lnd,ld->ln", x_lab, beta) + 0.1 * rng.standard_normal((n_layers, n_l))

    wh = fits.fit_whitening(x_u)  # mu (Ly,d), w (Ly,d,d) symmetric, gamma (Ly,)
    z = fits.apply_whitening(x_lab, wh)  # (Ly, n_l, d) fp64
    # y must be (n_slices, n_tr, d_out); one scalar target per layer's own DV
    y = dv[:, :, None]
    _preds, w_out = fits.ridge_fit_predict_primal_layer_batched(
        z, y, z, lambdas=rw.RIDGE_LAMBDAS, return_weights=True
    )
    cells = []
    for li in range(n_layers):
        mu_z = z[li].mean(axis=0)
        sigma_z = rw.recompute_sigma_z(z[li])
        ymu = float(dv[li].mean())
        cells.append(
            {
                "wh_mu": wh.mu[li],
                "wh_w": wh.w[li],
                "mu_z": mu_z,
                "sigma_z": sigma_z,
                "w_ridge": w_out[li, :, 0],
                "ymu": ymu,
                "z_train": z[li],
                "x_lab": x_lab[li],
            }
        )
    return cells


def test_scorer_reconstructs_helper_predictions():
    """scorer_predict must reproduce the ridge helper's own un-centered preds
    (in whitened-z space) so the finite-difference below is over the SAME
    scorer the fold differentiates."""
    rng = np.random.default_rng(1)
    n_layers, d, n_u, n_l = 3, 8, 200, 60
    x_u = rng.standard_normal((n_layers, n_u, d))
    x_lab = rng.standard_normal((n_layers, n_l, d))
    beta = rng.standard_normal((n_layers, d))
    dv = np.einsum("lnd,ld->ln", x_lab, beta) + 0.1 * rng.standard_normal((n_layers, n_l))
    wh = fits.fit_whitening(x_u)
    z = fits.apply_whitening(x_lab, wh)
    preds, w_out = fits.ridge_fit_predict_primal_layer_batched(
        z, dv[:, :, None], z, lambdas=rw.RIDGE_LAMBDAS, return_weights=True
    )
    for li in range(n_layers):
        mu_z = z[li].mean(axis=0)
        sigma_z = rw.recompute_sigma_z(z[li])
        ymu = float(dv[li].mean())
        # scorer_predict takes INPUT-space v; z = wh.w @ (v - wh.mu). Recover v
        # for each labeled row and score it; must match the helper's preds.
        for j in (0, n_l // 2, n_l - 1):
            got = rw.scorer_predict(
                x_lab[li, j], wh.mu[li], wh.w[li], mu_z, sigma_z, w_out[li, :, 0], ymu
            )
            assert np.isclose(got, preds[li, j, 0], rtol=1e-6, atol=1e-6), (li, j)


def test_dread_matches_central_finite_difference():
    """A9: analytic input-space gradient wh.w @ (w_ridge / sigma_z) == central
    finite difference of scorer_predict, per (layer) cell, within float tol."""
    cells = _fit_cells(seed=2)
    eps = 1e-4
    for li, c in enumerate(cells):
        d = c["wh_w"].shape[0]
        analytic = c["wh_w"] @ (c["w_ridge"] / c["sigma_z"])  # un-normalized gradient
        # finite-difference at the labeled centroid (a generic interior point)
        v0 = c["x_lab"].mean(axis=0)
        fd = np.empty(d)
        for k in range(d):
            e = np.zeros(d)
            e[k] = eps
            s_plus = rw.scorer_predict(
                v0 + e, c["wh_mu"], c["wh_w"], c["mu_z"], c["sigma_z"], c["w_ridge"], c["ymu"]
            )
            s_minus = rw.scorer_predict(
                v0 - e, c["wh_mu"], c["wh_w"], c["mu_z"], c["sigma_z"], c["w_ridge"], c["ymu"]
            )
            fd[k] = (s_plus - s_minus) / (2 * eps)
        # the scorer is exactly linear in v, so central FD is exact up to roundoff
        assert np.allclose(fd, analytic, rtol=1e-5, atol=1e-6), (li, np.abs(fd - analytic).max())
        # and the normalized fold agrees with the FD direction
        d_read = rw.fold_d_read(c["wh_w"], c["w_ridge"], c["sigma_z"])
        cos = float(d_read @ (fd / np.linalg.norm(fd)))
        assert cos > 0.999999, (li, cos)


def test_fold_d_read_is_unit_norm():
    cells = _fit_cells(seed=3)
    for c in cells:
        d_read = rw.fold_d_read(c["wh_w"], c["w_ridge"], c["sigma_z"])
        assert np.isclose(np.linalg.norm(d_read), 1.0, atol=1e-10)


def test_fold_d_read_rejects_degenerate_gradient():
    """A zero ridge weight yields a zero gradient -> fail loud, never a NaN
    direction silently written to the bank."""
    d = 8
    with pytest.raises(ValueError):
        rw.fold_d_read(np.eye(d), np.zeros(d), np.ones(d))


def test_direction_helpers_are_unit_norm():
    """raw mean-diff, shuffled-label fold, and random directions are all unit."""
    rng = np.random.default_rng(4)
    n_layers, d, n_u, n_l = 3, 8, 200, 60
    x_u = rng.standard_normal((n_layers, n_u, d))
    x_lab = rng.standard_normal((n_layers, n_l, d))
    beta = rng.standard_normal(d)
    dv = x_lab[0] @ beta + 0.1 * rng.standard_normal(n_l)
    wh = fits.fit_whitening(x_u)
    z = fits.apply_whitening(x_lab, wh)

    raw = rw.raw_mean_diff_direction(x_lab[0], dv)
    assert np.isclose(np.linalg.norm(raw), 1.0, atol=1e-10)

    shuf = rw.shuffled_fold(wh.w[0], z[0], dv, seed=42)
    assert np.isclose(np.linalg.norm(shuf), 1.0, atol=1e-10)

    rnd = rw.random_direction(d, seed=7)
    assert np.isclose(np.linalg.norm(rnd), 1.0, atol=1e-10)


def test_random_direction_is_seed_deterministic():
    a = rw.random_direction(16, seed=11)
    b = rw.random_direction(16, seed=11)
    c = rw.random_direction(16, seed=12)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
