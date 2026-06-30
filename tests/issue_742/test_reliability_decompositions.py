"""Test 1 (plan v7 §13 item 1 + §4 Stage-0 step 1 + §11 row 1) — reliability
decompositions match a synthetic dataset with a KNOWN r_yy, in BOTH estimator
regimes, and the two estimators are surfaced separately (never averaged) when
they disagree.

Planted ground-truth (see conftest.make_bernoulli_dataset_with_reliability):
  r_yy = signal_var / (signal_var + binomial_noise_var) is tuned by construction;
  the recovered r_yy from each estimator must land within tolerance of it.

Estimator FORK (plan §11 row 1) — the pair is method-adaptive per behavior:
  * >=2 rollouts/probe (sycophancy, broad_em): split-half-OVER-ROLLOUTS + binomial.
  * 1 rollout/probe (harmful_compliance, refusal): split-half-OVER-PROBES + binomial.
  * binomial uses the cell-actual m, NEVER a blanket m=2000.

Seeds: 7423 (>=2-rollout regime), 7424 (1-rollout regime), 74230 (disagreement).
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import (
    impl,
    impl_has,
    make_bernoulli_dataset_with_reliability,
)

TOL = 0.05  # ±0.05 recovery tolerance (plan §13 item 1)
TOL_BINOMIAL_1ROLLOUT = 0.07  # slightly looser for the 1-rollout binomial (Statistics critic)


# --------------------------------------------------------------------------- #
# Sub-test (a) — >=2 rollouts/probe regime: split-half-over-rollouts + binomial #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (
        impl_has("reliability_split_half_over_rollouts")
        and impl_has("reliability_binomial_variance")
    ),
    reason="implementation pending round 2",
)
def test_reliability_ge2_rollouts_both_estimators_recover_and_agree():
    # 50 contexts x 20 probes x 10 rollouts/probe (m=200), planted r_yy ~= 0.6.
    rollout_labels, per_context_rate, true_r_yy = make_bernoulli_dataset_with_reliability(
        n_contexts=50, n_probes=20, n_rollouts=10, target_r_yy=0.6, seed=7423
    )
    rng = np.random.default_rng(7423)

    r_split = impl.reliability_split_half_over_rollouts(rollout_labels, n_split_seeds=200, rng=rng)
    m_cell = rollout_labels.shape[1] * rollout_labels.shape[2]  # cell-actual m, per context
    r_binom = impl.reliability_binomial_variance(per_context_rate, m_cell)

    assert abs(r_split - true_r_yy) <= TOL, (
        f"split-half-over-rollouts r_yy={r_split:.3f} not within {TOL} of planted {true_r_yy:.3f}"
    )
    assert abs(r_binom - true_r_yy) <= TOL, (
        f"binomial r_yy={r_binom:.3f} not within {TOL} of planted {true_r_yy:.3f}"
    )
    assert abs(r_split - r_binom) <= TOL, (
        f"estimators disagree: split={r_split:.3f} vs binomial={r_binom:.3f} (>{TOL})"
    )


# --------------------------------------------------------------------------- #
# Sub-test (b) — 1 rollout/probe regime: split-half-over-PROBES + binomial      #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (
        impl_has("reliability_split_half_over_probes") and impl_has("reliability_binomial_variance")
    ),
    reason="implementation pending round 2",
)
def test_reliability_1_rollout_split_over_probes_recovers():
    # 50 contexts x 200 probes x 1 rollout/probe (m=200), planted r_yy ~= 0.5.
    # split-half-over-rollouts is UNDEFINED here (no within-probe rollouts to
    # split) -> the implementation must use split-half-over-PROBES instead.
    rollout_labels, per_context_rate, true_r_yy = make_bernoulli_dataset_with_reliability(
        n_contexts=50, n_probes=200, n_rollouts=1, target_r_yy=0.5, seed=7424
    )
    rng = np.random.default_rng(7424)

    # per-probe rate array (n_contexts, n_probes): each probe has 1 rollout -> its
    # rate is the single 0/1 label; split-half-over-probes correlates two probe-halves.
    probe_rates = rollout_labels[:, :, 0].astype(float)  # (50, 200)

    r_split = impl.reliability_split_half_over_probes(probe_rates, n_split_seeds=200, rng=rng)
    m_cell = rollout_labels.shape[1] * rollout_labels.shape[2]  # = n_probes (m=1 rollout each)
    r_binom = impl.reliability_binomial_variance(per_context_rate, m_cell)

    assert abs(r_split - true_r_yy) <= TOL, (
        f"split-half-over-probes r_yy={r_split:.3f} not within {TOL} of planted {true_r_yy:.3f}"
    )
    assert abs(r_binom - true_r_yy) <= TOL_BINOMIAL_1ROLLOUT, (
        f"binomial r_yy={r_binom:.3f} not within {TOL_BINOMIAL_1ROLLOUT} of planted "
        f"{true_r_yy:.3f} (1-rollout regime, looser tol)"
    )


# --------------------------------------------------------------------------- #
# Sub-test (b') — binomial MUST use the cell-actual m, never a blanket m=2000   #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("reliability_binomial_variance"),
    reason="implementation pending round 2",
)
def test_binomial_uses_cell_actual_m_not_blanket_2000():
    # A 1-rollout-per-probe behavior with m=214 (refusal-like). Feeding a blanket
    # m=2000 would UNDER-subtract the binomial noise (noise ~ p(1-p)/m), inflating
    # the apparent ceiling toward 1.0 — the exact bug this asserts against.
    _rollout_labels, per_context_rate, true_r_yy = make_bernoulli_dataset_with_reliability(
        n_contexts=50, n_probes=214, n_rollouts=1, target_r_yy=0.5, seed=7424
    )
    m_actual = 214
    m_blanket_wrong = 2000

    r_correct = impl.reliability_binomial_variance(per_context_rate, m_actual)
    r_wrong = impl.reliability_binomial_variance(per_context_rate, m_blanket_wrong)

    # the cell-actual m recovers the planted value; the blanket-2000 read inflates it
    assert abs(r_correct - true_r_yy) <= TOL_BINOMIAL_1ROLLOUT
    assert r_wrong > r_correct, (
        "a blanket m=2000 must inflate the ceiling vs the cell-actual m=214 "
        f"(wrong={r_wrong:.3f} should exceed correct={r_correct:.3f})"
    )


# --------------------------------------------------------------------------- #
# Sub-test (c) — estimator DISAGREEMENT is detected and BOTH values surfaced    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("load_reliability_estimates"),
    reason="implementation pending round 2 (loader must surface both estimators)",
)
def test_estimator_disagreement_surfaces_both_never_averages(monkeypatch):
    # Construct a cell where the two estimators MUST disagree (probe-level
    # heterogeneity / over-dispersion the binomial model does not capture), then
    # assert the loader returns BOTH values + a disagree flag — plan §6 analyzer-
    # attend: "trust split-half-over-probes when estimators disagree; don't average".
    rng = np.random.default_rng(74230)
    n_contexts, n_probes = 50, 200

    # Heterogeneous probes: half the probes are "easy" (high rate), half "hard"
    # (low rate), with per-context signal on TOP. Binomial assumes homogeneous
    # within-context p -> it mis-estimates noise under this over-dispersion, so
    # the two estimators diverge by construction.
    theta_c = np.clip(0.5 + rng.normal(0, 0.12, n_contexts), 0.05, 0.95)
    probe_difficulty = np.where(np.arange(n_probes) < n_probes // 2, 0.30, -0.30)
    probe_rates = np.empty((n_contexts, n_probes))
    for c in range(n_contexts):
        p = np.clip(theta_c[c] + probe_difficulty, 0.01, 0.99)
        probe_rates[c] = rng.binomial(1, p)

    # The loader is the contract under test: feed it the heterogeneous cell and
    # require it to expose both estimators without collapsing them to a mean.
    est = impl.load_reliability_estimates(
        behavior="harmful_compliance",
        genre="betley",
        probe_rates=probe_rates,
        n_rollouts_per_probe=1,
        rng=rng,
    )

    # both estimator values are present + distinct attributes (never a single mean)
    assert hasattr(est, "split_half") and hasattr(est, "binomial"), (
        "loader must surface BOTH estimator values as separate attributes"
    )
    assert est.split_half is not None and est.binomial is not None
    # disagreement flag fires when |Δ| > 0.10 (plan §7 1-rollout-disagreement row)
    delta = abs(float(est.split_half) - float(est.binomial))
    assert hasattr(est, "disagree")
    assert est.disagree == (delta > 0.10), (
        f"disagree flag ({est.disagree}) must equal (|Δ|={delta:.3f} > 0.10)"
    )
    # and on THIS heterogeneous fixture the estimators DO disagree
    assert delta > 0.10, (
        f"heterogeneous-probe fixture must make the two estimators disagree (|Δ|={delta:.3f})"
    )
