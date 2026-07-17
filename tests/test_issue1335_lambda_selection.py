"""#1335 r8: GCV lambda-selection degeneracy on within-group-correlated cells.

Pins the att-20260715-210436 gate1/gate2 failure class at fixture scale: on the
ladder's fiction per-persona cells (near-duplicate within-scene rows, n_tr < D)
the train Gram is near-singular, train RSS collapses to ~0 by interpolation,
and the GCV criterion picks the grid-min lambda (0.01) — held-out R^2 lands at
-2..-46 while lambda=1e3-1e4 on the SAME folds reads +0.22..+0.35 (the #1310
anchor band). The fix is inner GROUP-level-CV lambda selection
(``heldout_r2_sweep(..., lambda_selection="inner-group-cv")``), applied
identically to observed + null draws (selection-symmetric).

The fixture is a REAL downsampled slice of the failing production cell
(store_r7_endpoint_instruct / Dana / layer 19; 40 of 300 scenario groups,
384 of 3584 dims, seed 1335) — synthetic group-noise fixtures did NOT
reproduce the degeneracy, the real within-scene correlation structure does:
GCV reads -15.0 with every fold at lambda=0.01; inner-group-cv reads +0.093
at lambda 1e3-3.2e3. Pre-fix (no ``lambda_selection`` param) the GCV number
is the ONLY behavior — this file fails; post-fix it passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue825_fit_cells as fit825

FIXTURE = Path(__file__).parent / "fixtures" / "issue1335_gcv_degeneracy_slice.npz"


@pytest.fixture(scope="module")
def real_slice():
    z = np.load(FIXTURE, allow_pickle=False)
    X = z["X"].astype(np.float32)[:, None, :]  # (n, 1 layer, d)
    Y = z["Y"].astype(np.float32)[:, None, :]
    return X, Y, z["groups"]


def test_gcv_degenerates_and_inner_group_cv_recovers(real_slice):
    """The bug (GCV grid-min collapse, catastrophic negative R^2) and the fix."""
    X, Y, groups = real_slice
    gcv = fit825.heldout_r2_sweep(
        X, Y, groups, n_folds=5, seed=0, null_draws=0, collect_lambdas=True
    )
    inner = fit825.heldout_r2_sweep(
        X,
        Y,
        groups,
        n_folds=5,
        seed=0,
        null_draws=0,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
    )
    # The documented defect: GCV picks the grid minimum on every fold and the
    # held-out R^2 is catastrophically negative (fixture reads ~-15.0).
    assert np.all(gcv["gcv_lambda"][0] == pytest.approx(float(fit825.LAMBDAS[0]))), gcv[
        "gcv_lambda"
    ]
    assert gcv["r2_obs"][0] < -2.0, gcv["r2_obs"]
    # The fix: inner-group-CV selects an honest lambda (>=100 on this slice)
    # and recovers POSITIVE held-out R^2 (fixture reads ~+0.093).
    assert np.all(inner["gcv_lambda"][0] >= 100.0), inner["gcv_lambda"]
    assert inner["r2_obs"][0] > 0.03, inner["r2_obs"]


def test_inner_cv_rss_curve_matches_bruteforce():
    """The reduced-form RSS(lambda) curve equals explicit per-lambda predictions."""
    rng = np.random.default_rng(7)
    n, d = 60, 24
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = rng.standard_normal((n, d)).astype(np.float32)
    groups = np.repeat([f"g{i}" for i in range(12)], 5)
    icaches = fit825._prep_inner_lambda(X, groups, fit825.N_INNER_LAMBDA_FOLDS, seed=3)
    assert icaches is not None and len(icaches) >= 2
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    curve = fit825._inner_cv_rss_curve(icaches, Yt)
    # brute force: per inner fold, per lambda, materialize the ridge prediction
    brute = torch.zeros(len(fit825.LAMBDAS), dtype=torch.float64)
    for ic in icaches:
        Yf = Yt.index_select(0, ic["fi_idx"])
        Yv = Yt.index_select(0, ic["va_idx"])
        ymu = Yf.mean(0)
        VtY = ic["V"].T @ (Yf - ymu)
        for li, lam in enumerate(fit825.LAMBDAS):
            alpha = VtY / (ic["w"] + lam).unsqueeze(1)
            pred_c = ic["P"] @ alpha
            brute[li] += ((Yv - ymu - pred_c) ** 2).sum()
    assert torch.allclose(curve, brute, rtol=1e-9, atol=1e-8), (curve - brute).abs().max()


def test_batched_inner_selection_matches_serial(real_slice):
    """Null-path (batched) inner selection == observed-path (serial) selection."""
    X, Y, groups = real_slice
    X0, Y0 = X[:, 0, :], Y[:, 0, :]
    folds = fit825._cv_folds(groups, 5, 0)
    te = folds == 0
    tr = ~te
    cache = fit825._prep_fold(X0[tr], X0[te])
    cache["inner"] = fit825._prep_inner_lambda(
        X0[tr], np.asarray(groups)[tr], fit825.N_INNER_LAMBDA_FOLDS, seed=4242
    )
    assert cache["inner"] is not None
    pred_serial, lam = fit825._ridge_predict_cached(cache, Y0[tr], return_lam=True)
    pred_batched = fit825._ridge_predict_cached_batched(cache, Y0[tr][None]).cpu().numpy()[0]
    assert lam >= 100.0  # the honest-lambda pick, not the degenerate grid-min
    assert np.allclose(pred_serial, pred_batched, rtol=1e-9, atol=1e-9)


def test_unknown_lambda_selection_raises(real_slice):
    X, Y, groups = real_slice
    with pytest.raises(ValueError, match="lambda_selection"):
        fit825.heldout_r2_sweep(
            X, Y, groups, n_folds=5, seed=0, null_draws=0, lambda_selection="loo"
        )


def test_issue1335_fit_pins_inner_group_cv():
    """The ladder driver's module constant is the fixed selector (provenance pin)."""
    import issue1335_fit as fit1335

    assert fit1335.LAMBDA_SELECTION == "inner-group-cv"
