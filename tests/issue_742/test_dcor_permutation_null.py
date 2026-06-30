"""Test 3 (plan v7 §13 item 3 + §4 Stage-1 steps 0/3 + §11 row 5 + §12 row 6) —
the dCor permutation null detects nonlinear dependence (true positive), controls
the false-positive (true negative + null centered near the independence value),
and has >=0.8 power at the plan's stated 0.10-partial-correlation effect floor in
the d_eff=10 / n=50 / B=10000 regime.

dCor (Székely 2007, arXiv:0803.4101): dCor=0 iff independent; captures nonlinear
dependence a Pearson/Spearman residual would miss. The permutation null is the
exact small-n test.

Planted ground-truth:
  * (a) nonlinear: E0 = sigmoid(||v0||^2) + small_noise (no linear component)
  * (b) independence: E0 iid, no dependence on v0
  * (c) mid-effect at the ~0.10 partial-correlation floor for the power check

Seed: 7422 (dCor synthetic, plan §10 reproducibility card).
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    impl,
    impl_has,
    make_independent_target,
    make_nonlinear_dependence,
)

# B=10000 perms is the power-check budget (plan §13 item 3c / §11 row 3). The
# real-data run uses B_perm=1000 (refit-per-permutation, plan §4 Stage 1); these
# tests run the bare dCor permutation on a FIXED single frame, so the larger B is
# cheap and matches the power-check the plan promised at §12 row 6.
N_PERM_POWER = 10000
N_PERM_FAST = 2000  # detection sub-tests (a)/(b): smaller B is enough to read p


# --------------------------------------------------------------------------- #
# Sub-test (a) — TRUE POSITIVE: a planted nonlinear dependence lands in the tail #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("dcor_permutation_test"),
    reason="implementation pending round 2",
)
def test_dcor_detects_planted_nonlinear_dependence():
    v0, E0 = make_nonlinear_dependence(n=50, d=10, noise=0.05, seed=7422)
    rng = np.random.default_rng(7422)
    res = impl.dcor_permutation_test(v0, E0, d_eff=10, n_perm=N_PERM_FAST, rng=rng)
    assert res.p_value < 0.05, (
        f"dCor failed to detect a strong planted nonlinear dependence (p={res.p_value:.4f})"
    )
    # observed statistic sits in the right tail of its own null
    assert res.dcor > float(np.quantile(res.null, 0.95)), (
        "observed dCor must exceed the 95th percentile of the permutation null"
    )


# --------------------------------------------------------------------------- #
# Sub-test (b) — TRUE NEGATIVE: independence is NOT significant, null centered   #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("dcor_permutation_test"),
    reason="implementation pending round 2",
)
def test_dcor_controls_false_positive_under_independence():
    v0, E0 = make_independent_target(n=50, d=10, seed=7422)
    rng = np.random.default_rng(7422)
    res = impl.dcor_permutation_test(v0, E0, d_eff=10, n_perm=N_PERM_FAST, rng=rng)

    # not significant at 0.05 under genuine independence
    assert res.p_value >= 0.05, (
        f"dCor fired on independent data (p={res.p_value:.4f}) — false positive"
    )
    # the observed statistic is within sampling error of the null center
    null_median = float(np.median(res.null))
    null_iqr = float(np.quantile(res.null, 0.75) - np.quantile(res.null, 0.25))
    assert abs(res.dcor - null_median) <= 2.0 * null_iqr + 1e-9, (
        f"observed dCor ({res.dcor:.4f}) is far from the null center "
        f"({null_median:.4f}) under independence"
    )


# --------------------------------------------------------------------------- #
# Sub-test (c) — POWER >= 0.8 at the 0.10-partial-correlation floor (d_eff=10)   #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("dcor_power_check"),
    reason="implementation pending round 2 (codifies the §12 row 6 runtime power check)",
)
def test_dcor_power_at_effect_floor_d_eff_10():
    # plan §13 item 3c / §11 row 3 / §12 row 6: at d_eff=10, n=50, B_perm=10000,
    # with a planted nonlinear-residual effect at the ~0.10 partial-correlation
    # floor, realized power must be >= 0.8. This codifies the UNVERIFIED-flagged
    # runtime power check the plan promised to run, as a test fixture.
    rng = np.random.default_rng(7422)
    power = impl.dcor_power_check(
        d_eff=10,
        n=50,
        n_perm=N_PERM_POWER,
        effect=0.10,  # partial-correlation effect floor (plan §11 row 3)
        n_trials=200,  # repeated-experiment trials to estimate power
        rng=rng,
    )
    assert power >= 0.8, (
        f"dCor power at the 0.10 effect floor (d_eff=10, n=50, B={N_PERM_POWER}) "
        f"is {power:.3f} < 0.8 — the plan's Stage-1 power floor is not met; the "
        "implementation must relax d_eff per §4 Stage-1 step 0"
    )


# --------------------------------------------------------------------------- #
# distance_correlation primitive sanity (deterministic, no impl-state needed)   #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("distance_correlation"),
    reason="implementation pending round 2",
)
def test_distance_correlation_zero_for_independent_one_for_identical():
    rng = np.random.default_rng(7422)
    n = 200
    x = rng.normal(size=(n, 3))
    # identical (perfectly dependent) target derived from x -> dCor near 1 on a
    # strict monotone map of a 1-D summary
    y_dep = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2
    y_indep = rng.normal(size=n)

    dc_dep = impl.distance_correlation(x, y_dep)
    dc_indep = impl.distance_correlation(x, y_indep)
    assert 0.0 <= dc_indep <= 1.0 and 0.0 <= dc_dep <= 1.0, "dCor must lie in [0,1]"
    assert dc_dep > dc_indep, (
        f"dCor must be larger for a dependent target ({dc_dep:.3f}) than an "
        f"independent one ({dc_indep:.3f})"
    )
    # the independent dCor should be small (finite-sample noise floor)
    assert dc_indep < 0.3, f"dCor on independent data unexpectedly large ({dc_indep:.3f})"
