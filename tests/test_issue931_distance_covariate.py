"""Tests for the issue #931 distance-covariate pure function.

Pins: (1) the batched (draws, G)-GEMM cluster-bootstrap WLS reproduces a
serial per-draw reference; (2) the per-pair decomposition's observed mean
equals delta R^2 = R^2_correct - R^2_swap from the run's own
group_bootstrap_r2 machinery (shared draws); (3) a known linear relation is
recovered (slope + intercept-at-zero).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue931_distance_covariate as dc
import issue931_fit_cells as fitc


def _toy(seed: int = 0, n: int = 60, groups: int = 6, dim: int = 5):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, size=n)
    group_ids = np.asarray([f"g{i}" for i in g])
    yc = rng.normal(size=(n, dim))
    ys = rng.normal(size=(n, dim))
    pred_c = yc + 0.3 * rng.normal(size=(n, dim))
    pred_s = ys + 0.6 * rng.normal(size=(n, dim))
    err_c = ((yc - pred_c) ** 2).sum(axis=1)
    err_s = ((ys - pred_s) ** 2).sum(axis=1)
    x = rng.uniform(-2, 5, size=n)
    G = len(np.unique(group_ids))
    picks = np.random.default_rng(7).integers(0, G, size=(50, G))
    M = np.zeros((50, G))
    for d in range(50):
        M[d] = np.bincount(picks[d], minlength=G)
    return err_c, err_s, x, group_ids, yc, ys, pred_c, pred_s, M


def _serial_reference(err_c, err_s, x, group_ids, yc, ys, M):
    """Per-draw python-loop WLS oracle (the equivalence-gate serial twin)."""
    _uniq, inv = np.unique(group_ids, return_inverse=True)
    out_a, out_b, out_d = [], [], []
    for m in M:
        w = m[inv]
        N = w.sum()
        mu_c = (w[:, None] * yc).sum(0) / N
        mu_s = (w[:, None] * ys).sum(0) / N
        ss_c = (w * ((yc - mu_c) ** 2).sum(1)).sum()
        ss_s = (w * ((ys - mu_s) ** 2).sum(1)).sum()
        u = N * (err_s / ss_s - err_c / ss_c)
        X = np.stack([np.ones_like(x), x], axis=1)
        WX = X * w[:, None]
        beta = np.linalg.solve(X.T @ WX, WX.T @ u)
        out_a.append(beta[0])
        out_b.append(beta[1])
        out_d.append((w * u).sum() / N)
    return np.asarray(out_a), np.asarray(out_b), np.asarray(out_d)


def test_batched_matches_serial_reference():
    err_c, err_s, x, gids, yc, ys, _, _, M = _toy()
    res = dc.distance_partialled_gap(err_c, err_s, x, gids, yc, ys, M)
    ref_a, ref_b, ref_d = _serial_reference(err_c, err_s, x, gids, yc, ys, M)
    assert np.allclose(res["intercept_draws"], ref_a, atol=1e-10)
    assert np.allclose(res["slope_draws"], ref_b, atol=1e-10)
    assert np.allclose(res["delta_draws"], ref_d, atol=1e-10)


def test_observed_mean_matches_group_bootstrap_delta():
    err_c, err_s, x, gids, yc, ys, pred_c, pred_s, M = _toy(seed=3)
    res = dc.distance_partialled_gap(err_c, err_s, x, gids, yc, ys, M)
    gb_c = fitc.group_bootstrap_r2(pred_c, yc, gids, n_boot=50, seed=0, draws_matrix=M)
    gb_s = fitc.group_bootstrap_r2(pred_s, ys, gids, n_boot=50, seed=0, draws_matrix=M)
    assert abs(res["r2_correct"] - gb_c["r2"]) < 1e-12
    assert abs(res["r2_swap"] - gb_s["r2"]) < 1e-12
    assert abs(res["delta"] - (gb_c["r2"] - gb_s["r2"])) < 1e-12
    assert np.allclose(res["delta_draws"], gb_c["draws"] - gb_s["draws"], atol=1e-10)
    # Observed mean of the per-pair decomposition == delta R^2 exactly.
    assert abs(res["u_obs"].mean() - res["delta"]) < 1e-12


def test_known_linear_relation_recovered():
    """u constructed as alpha + beta*x exactly => WLS recovers both; the
    intercept is the gap at zero distance difference."""
    rng = np.random.default_rng(11)
    n, dim = 80, 4
    gids = np.asarray([f"g{i % 8}" for i in range(n)])
    yc = rng.normal(size=(n, dim))
    ys = rng.normal(size=(n, dim))
    x = rng.uniform(0, 10, size=n)
    # Choose err_c fixed, then set err_s so that u = 0.5 + 0.2 * x exactly.
    err_c = rng.uniform(0.5, 2.0, size=n)
    mu_c, mu_s = yc.mean(0), ys.mean(0)
    ss_c = ((yc - mu_c) ** 2).sum()
    ss_s = ((ys - mu_s) ** 2).sum()
    u_target = 0.5 + 0.2 * x
    err_s = (u_target / n + err_c / ss_c) * ss_s
    M = np.ones((3, 8))  # identity-multiplicity draws
    res = dc.distance_partialled_gap(err_c, err_s, x, gids, yc, ys, M)
    assert abs(res["intercept"] - 0.5) < 1e-9
    assert abs(res["slope"] - 0.2) < 1e-9
    assert np.allclose(res["intercept_draws"], 0.5, atol=1e-9)
