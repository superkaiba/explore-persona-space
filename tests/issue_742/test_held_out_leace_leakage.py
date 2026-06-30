"""Test 7 (plan v7 §13 NEW per Phase-2 REVISE — the alternatives critic's Must-fix:
distinguish a genuine nonlinear residual from leftover LINEAR leakage LEACE's
train-fold-only guarantee does not erase out-of-sample).

LEACE's closed-form guarantee is on the TRAINING fold only (Belrose 2023). On a
held-out split the residual can still carry linear E0-correlation from sampling
variance — at d_eff=10 / n=50 that leftover linear signal would masquerade as a
"nonlinear residual" in the dCor test. This diagnostic fits LEACE on train,
applies to held-out, and probes the held-out LEACE-residual for residual LINEAR
leakage against a held-out permutation null. The Stage-1 verdict must NOT read
'nonlinear-yes' when the held-out residual still carries linear leakage.

Sub-tests:
  (a) false-positive control — LEACE perfect on train AND held-out (a clean
      2-dim subspace, no sampling variance) -> held-out |rho| within the null CI
      -> post_leace_linear_pass = True.
  (b) true positive — train-fold LEACE exact, but the held-out residual carries
      a PLANTED Spearman > 0.20 with E0_held -> post_leace_linear_pass = False.
  (c) verdict-classifier enum: (dcor pass, linear pass) -> 'nonlinear-yes';
      (dcor pass, linear fail) -> 'linear-erasure-leakage-unresolved';
      (dcor null) -> 'ceiling-limited'.

Seeds: 7425 (false-positive control), 7426 (true-positive leakage).
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import impl, impl_has


# --------------------------------------------------------------------------- #
# Sub-test (a) — LEACE perfect on BOTH splits -> diagnostic PASSES (no leakage) #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("held_out_linear_leakage"),
    reason="implementation pending round 2",
)
def test_perfect_erasure_held_out_pass():
    rng = np.random.default_rng(7425)
    # A clean low-dim construction with NO sampling-variance pathology: E0 is an
    # EXACT linear function of a 2-dim subspace of v0 on BOTH train and held-out,
    # so LEACE (fit on train) erases it exactly out-of-sample too -> held-out
    # residual carries no linear E0 signal -> diagnostic passes.
    n_train, n_held, d = 49, 25, 2
    w = np.array([1.0, -0.5])
    v0_train = rng.normal(size=(n_train, d))
    v0_held = rng.normal(size=(n_held, d))
    E0_train = v0_train @ w  # exact linear, no noise
    E0_held = v0_held @ w  # SAME linear law -> train eraser generalizes exactly

    res = impl.held_out_linear_leakage(v0_train, E0_train, v0_held, E0_held, rng=rng)

    assert res.post_leace_linear_pass is True, (
        f"perfect-erasure control must PASS (no held-out linear leakage); got "
        f"pass={res.post_leace_linear_pass}, held-out rho={getattr(res, 'rho', None)}"
    )
    # the held-out residual rho sits within its permutation null CI
    lo, hi = res.null_ci
    assert lo <= res.rho <= hi, (
        f"held-out residual rho={res.rho:.3f} should sit inside the null CI "
        f"[{lo:.3f}, {hi:.3f}] under perfect erasure"
    )


# --------------------------------------------------------------------------- #
# Sub-test (b) — train-fold exact, held-out carries planted linear leakage -> FAIL
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (impl_has("held_out_linear_leakage") and impl_has("leace_residual")),
    reason="implementation pending round 2",
)
def test_held_out_residual_with_planted_leakage_fails():
    rng = np.random.default_rng(7426)
    n_train, n_held, d = 50, 50, 10

    # Train fold: E0 linearly decodable -> LEACE erases it exactly on train.
    v0_train = rng.normal(size=(n_train, d))
    w_train = np.zeros(d)
    w_train[0] = 1.0
    E0_train = v0_train @ w_train + rng.normal(0, 0.02, n_train)

    # Assert the TRAIN-fold guarantee first: cov(E0_train, P @ v0_train) ~= 0.
    resid_train = impl.leace_residual(v0_train, E0_train)
    E0c = E0_train - E0_train.mean()
    cov_train = np.abs(((resid_train - resid_train.mean(0)) * E0c[:, None]).mean(0)).max()
    assert cov_train <= 1e-6, (
        f"train-fold LEACE guarantee broken (max|cov|={cov_train:.2e}); the test "
        "premise requires exact train erasure"
    )

    # Held-out fold: build it so the train-fitted eraser leaves a residual that
    # STILL correlates linearly with E0_held (a different effective direction the
    # train eraser does not cover) -> a Spearman > 0.20 leakage the dCor test
    # would otherwise mistake for a nonlinear residual.
    v0_held = rng.normal(size=(n_held, d))
    w_held = np.zeros(d)
    w_held[5] = 1.0  # a DIFFERENT axis than the train eraser removed (axis 0)
    E0_held = v0_held @ w_held + rng.normal(0, 0.02, n_held)

    res = impl.held_out_linear_leakage(v0_train, E0_train, v0_held, E0_held, rng=rng)

    assert res.post_leace_linear_pass is False, (
        "held-out residual carries planted linear leakage (a non-train direction) "
        f"-> diagnostic must FAIL; got pass={res.post_leace_linear_pass}, "
        f"rho={getattr(res, 'rho', None)}"
    )
    assert abs(res.rho) > 0.20, (
        f"the planted held-out linear leakage should yield |rho|>0.20, got {res.rho:.3f}"
    )


# --------------------------------------------------------------------------- #
# Sub-test (c) — the Stage-1 verdict enum classifier                            #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("classify_stage1_verdict"),
    reason="implementation pending round 2",
)
def test_stage1_verdict_enum():
    # dCor pass + linear pass -> genuine nonlinear residual
    assert impl.classify_stage1_verdict(dcor_pass=True, linear_pass=True) == "nonlinear-yes"
    # dCor pass + linear FAIL -> NOT nonlinear-yes (the alternatives critic Must-fix):
    # the apparent residual is unresolved leftover linear leakage, not nonlinearity
    assert (
        impl.classify_stage1_verdict(dcor_pass=True, linear_pass=False)
        == "linear-erasure-leakage-unresolved"
    )
    # dCor null -> the test cannot resolve a residual at this n (ceiling-limited)
    assert impl.classify_stage1_verdict(dcor_pass=False, linear_pass=True) == "ceiling-limited"
    assert impl.classify_stage1_verdict(dcor_pass=False, linear_pass=False) == "ceiling-limited"
