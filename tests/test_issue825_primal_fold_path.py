"""#1336 Unit B: primal (d-space) fold-path parity + regime-switch pins.

The primal route — engaged automatically at n_train > d unless
``fc.FORCE_GRAM`` — must be numerically identical to the legacy Gram route
at matched (lambda, fold): the plan v13 G0'(b) gate form, |dR^2| <= 1e-6.
At n_train <= d the switch must NOT engage (Gram path byte-preserved).
These tests fail pre-fix trivially (no primal route / no ``route`` field
existed) and pin the equivalence + the switch predicate post-fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import issue825_fit_cells as fc  # noqa: E402

# Parity-gate thread hygiene on the shared VM (tiny-matrix eighs thrash wide
# torch pools; #928 lesson).
torch.set_num_threads(2)

PARITY_TOL = 1e-6  # the plan G0'(b) gate tolerance


def _synth(n: int, d: int, layers: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, layers, d)).astype(np.float32)
    w = rng.normal(size=(d, d)) / np.sqrt(d)
    Y = (
        np.einsum("nld,de->nle", X.astype(np.float64), w) + 0.5 * rng.normal(size=(n, layers, d))
    ).astype(np.float32)
    conv = np.asarray([f"c{i}" for i in range(n)])
    return X, Y, conv


def test_primal_matches_forced_gram_per_lambda_fold(monkeypatch):
    """Per-(lambda, fold) R^2 agreement, forced-Gram vs primal, n=600 > d=64."""
    n, d = 600, 64
    X, Y, conv = _synth(n, d, layers=1, seed=0)
    X0, Y0 = X[:, 0, :], Y[:, 0, :]
    folds = fc._cv_folds(conv, 5, 0)
    lams = [1e-2, 1.0, 1e2, 1e4]
    for k in range(5):
        te = folds == k
        tr = ~te
        monkeypatch.setattr(fc, "FORCE_GRAM", True)
        cg = fc._prep_fold(X0[tr], X0[te])
        assert cg["route"] == "gram"
        monkeypatch.setattr(fc, "FORCE_GRAM", False)
        cp = fc._prep_fold(X0[tr], X0[te])
        assert cp["route"] == "primal"
        # The primal eigenbasis is d-dimensional (vs n_train for Gram).
        assert cp["w"].shape[0] <= d and cp["V"].shape == (int(tr.sum()), cp["w"].shape[0])
        true = Y0[te].astype(np.float64)
        for lam in lams:
            # 1-element forced grid pins the SAME lambda on both routes.
            pg = fc._ridge_predict_cached(cg, Y0[tr], lambdas=[lam])
            pp = fc._ridge_predict_cached(cp, Y0[tr], lambdas=[lam])
            r2_g = fc._pooled_r2(pg, true)
            r2_p = fc._pooled_r2(pp, true)
            assert abs(r2_g - r2_p) <= PARITY_TOL, (k, lam, r2_g, r2_p)


def test_full_sweep_parity_gram_vs_primal(monkeypatch):
    """heldout_r2_sweep parity — observed AND null draws — on the v2 recipe
    shape (23-pt grid, inner-group-cv, batched nulls) at n_train > d."""
    n, d = 600, 64
    X, Y, conv = _synth(n, d, layers=3, seed=1)
    kw = dict(
        n_folds=5,
        seed=0,
        null_draws=2,
        collect_cosines=False,
        frozen_layers=(0,),
        lambdas=np.logspace(-3, 8, 23),
    )
    monkeypatch.setattr(fc, "FORCE_GRAM", True)
    sg = fc.heldout_r2_sweep(X, Y, conv, **kw)
    monkeypatch.setattr(fc, "FORCE_GRAM", False)
    sp = fc.heldout_r2_sweep(X, Y, conv, **kw)
    np.testing.assert_allclose(sp["r2_obs"], sg["r2_obs"], rtol=0, atol=PARITY_TOL)
    np.testing.assert_allclose(sp["r2_null"], sg["r2_null"], rtol=0, atol=PARITY_TOL)
    # Identical lambda selections (same RSS curves to fp roundoff; seeded
    # synthetic data keeps the grid argmins tie-free).
    lg = np.asarray(sg["gcv_lambda"], dtype=np.float64)
    lp = np.asarray(sp["gcv_lambda"], dtype=np.float64)
    assert np.array_equal(np.isfinite(lg), np.isfinite(lp))
    fin = np.isfinite(lg)
    np.testing.assert_array_equal(lp[fin], lg[fin])


def test_regime_switch_not_engaged_at_n_le_d():
    """n_train <= d keeps the legacy Gram route (byte-preserved path)."""
    rng = np.random.default_rng(2)
    d = 64
    # Strictly below d.
    Xtr = rng.normal(size=(40, d)).astype(np.float32)
    Xev = rng.normal(size=(10, d)).astype(np.float32)
    cache = fc._prep_fold(Xtr, Xev)
    assert cache["route"] == "gram"
    assert cache["V"].shape == (40, 40)  # n x n Gram eigenbasis
    # Exactly d: the switch is strict `>`, so this stays Gram too.
    Xtr_eq = rng.normal(size=(d, d)).astype(np.float32)
    cache_eq = fc._prep_fold(Xtr_eq, Xev)
    assert cache_eq["route"] == "gram"


def test_inner_lambda_primal_engages_and_matches(monkeypatch):
    """Inner-fold caches take the primal route at n_fi > d and reproduce the
    Gram route's inner-CV RSS curve (hence identical lambda argmins)."""
    n, d = 400, 32
    rng = np.random.default_rng(3)
    Xtr = rng.normal(size=(n, d)).astype(np.float32)
    Ytr = torch.as_tensor(rng.normal(size=(n, d)).astype(np.float64), device=fc._fit_device())
    groups = np.asarray([f"g{i}" for i in range(n)])
    monkeypatch.setattr(fc, "FORCE_GRAM", True)
    ig = fc._prep_inner_lambda(Xtr, groups, 2, seed=7)
    monkeypatch.setattr(fc, "FORCE_GRAM", False)
    ip = fc._prep_inner_lambda(Xtr, groups, 2, seed=7)
    assert ig is not None and ip is not None
    for cg, cp in zip(ig, ip, strict=True):
        assert cg["V"].shape[0] == cg["V"].shape[1]  # gram: n_fi x n_fi
        assert cp["V"].shape[1] <= d  # primal: n_fi x k, k <= d
    lams = np.logspace(-2, 4, 13)
    rss_g = fc._inner_cv_rss_curve(ig, Ytr, lams=lams).cpu().numpy()
    rss_p = fc._inner_cv_rss_curve(ip, Ytr, lams=lams).cpu().numpy()
    np.testing.assert_allclose(rss_p, rss_g, rtol=1e-9)
    assert int(np.argmin(rss_p)) == int(np.argmin(rss_g))
    # Batched twin parity on a 2-draw batch (the null path's shape).
    Yb = torch.stack([Ytr, Ytr.flip(0)])
    rb_g = fc._inner_cv_rss_curve_batched(ig, Yb, lams=lams).cpu().numpy()
    rb_p = fc._inner_cv_rss_curve_batched(ip, Yb, lams=lams).cpu().numpy()
    np.testing.assert_allclose(rb_p, rb_g, rtol=1e-9)


def test_force_gram_default_off():
    """The module default must leave the regime switch armed (plan §4 div. 5)."""
    assert fc.FORCE_GRAM is False


@pytest.mark.parametrize("n,d,expected", [(600, 64, "primal"), (48, 64, "gram")])
def test_route_field_matches_regime(n, d, expected):
    rng = np.random.default_rng(4)
    cache = fc._prep_fold(
        rng.normal(size=(n, d)).astype(np.float32),
        rng.normal(size=(8, d)).astype(np.float32),
    )
    assert cache["route"] == expected
