# ruff: noqa: RUF002
"""Primal-vs-dual ridge parity for the issue #841 scaling-capture round.

`maps.fit_ridge_primal` is the large-n (n≫d) solver the scaling curve depends on
(the dual m×m Gram is n×n fp64 ≈80 GB at n=100k, infeasible). It MUST be a numeric
drop-in for the parent's `fit_ridge_split` (dual) — the KILL-B gate asserts this
at run time; this test pins it in CI so a future refactor can't silently break
the equivalence. Both operate on the SAME center-then-ridge affine contract, so
they must agree to fp64 precision: same selected λ, identical eval predictions,
identical R².
"""

from __future__ import annotations

import numpy as np
import pytest

from explore_persona_space.experiments.issue_841 import maps as MP


def _synth(seed, n, d, p):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    w = (rng.standard_normal((d, p)) * 0.1).astype(np.float32)
    y = (x @ w + 0.05 * rng.standard_normal((n, p))).astype(np.float32)
    xev = rng.standard_normal((40, d)).astype(np.float32)
    yev = (xev @ w).astype(np.float32)
    return x, y, xev, yev


@pytest.mark.parametrize("gram_chunk", [7, 137, 100000])  # non-divisor, mid, one-shot
def test_primal_matches_dual(gram_chunk):
    x, y, xev, yev = _synth(seed=0, n=400, d=64, p=8)  # n>d, well-posed for both
    pred_d, rmap_d = MP.fit_ridge_split(x, y, xev, sigma=1.0, device="cpu")
    pred_p, rmap_p = MP.fit_ridge_primal(x, y, xev, sigma=1.0, device="cpu", gram_chunk=gram_chunk)
    assert rmap_d.best_lam == rmap_p.best_lam, (rmap_d.best_lam, rmap_p.best_lam)
    assert np.max(np.abs(pred_d - pred_p)) < 1e-8
    r2_d = MP.identity_relative_r2(pred_d, yev)
    r2_p = MP.identity_relative_r2(pred_p, yev)
    assert abs(r2_d - r2_p) < 1e-9, (r2_d, r2_p)


def test_primal_weights_match_dual():
    x, y, xev, _ = _synth(seed=1, n=500, d=80, p=6)
    _, rmap_d = MP.fit_ridge_split(x, y, xev, sigma=1.0, device="cpu")
    _, rmap_p = MP.fit_ridge_primal(x, y, xev, sigma=1.0, device="cpu", gram_chunk=137)
    assert float((rmap_d.w - rmap_p.w).abs().max()) < 1e-6
    assert float((rmap_d.bias - rmap_p.bias).abs().max()) < 1e-9
    assert float((rmap_d.mu - rmap_p.mu).abs().max()) < 1e-9


def test_direct_hop_solver_dispatch_parity():
    """fit_direct_hop_ridge with n>dual_max (primal) == n≤dual_max (dual) on same data."""
    rng = np.random.default_rng(2)
    n, d = 300, 48
    h_src = rng.standard_normal((n, d)).astype(np.float32)
    h_tgt = (h_src + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    h_ev = rng.standard_normal((25, d)).astype(np.float32)
    dual = MP.fit_direct_hop_ridge(h_src, h_tgt, h_ev, device="cpu", n=n, dual_max=10_000)  # dual
    primal = MP.fit_direct_hop_ridge(h_src, h_tgt, h_ev, device="cpu", n=n, dual_max=10)  # primal
    assert dual.best_lam == primal.best_lam
    assert float((dual.w - primal.w).abs().max()) < 1e-6
