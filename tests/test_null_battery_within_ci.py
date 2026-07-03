"""Regression test for the #778 within-condition bootstrap-CI bug.

``null_battery.compute_setting`` on the ``monitoring_within`` setting used to
call ``bootstrap_ci_matched_r`` (pooled row bootstrap of the OVERALL Pearson r)
without ``condition_ids`` — reporting the POOLED statistic's CI as the within
CI (audit marker v70 on task #778; fixed at source in the
faithful-extraction-honest-nulls-rerun round). This test builds a synthetic
2-condition dataset where the pooled and within-condition correlations MUST
differ (a strong between-condition mean shift drives the pooled r; the
within-condition r is weak), and asserts:

1. ``bootstrap_ci_within_r`` (the fix) differs materially from the pooled
   ``bootstrap_ci_matched_r`` CI on this dataset (the two estimators are
   genuinely distinct here — the construction is valid).
2. ``compute_setting(setting="monitoring_within")`` reports the WITHIN CI
   (fails pre-fix: it reported the pooled CI).
3. The within CI actually covers the within-condition point estimate, and the
   pooled CI does NOT (the numeric signature of the bug).
"""

from __future__ import annotations

import numpy as np

from explore_persona_space.analysis import null_battery as nb

N_PER_GROUP = 24
L = 3  # layers (small; the CI is read at one fixed layer)
D = 8  # hidden dim (small synthetic)


def _synthetic_two_condition_dataset(seed: int = 0):
    """Two condition blocks with a big between-block mean shift.

    The predictor projection onto ``direction`` is (block mean) + small noise;
    the target is (block mean) + independent small noise. Pooled r is ~1 (the
    block contrast dominates); within-block r is ~0 (independent noise). So the
    pooled CI hugs ~[0.9, 1.0] while the within CI straddles 0 — they cannot
    agree.
    """
    rng = np.random.default_rng(seed)
    direction = np.zeros((L, D), dtype=np.float64)
    direction[:, 0] = 1.0  # projection reads coordinate 0
    n = 2 * N_PER_GROUP
    acts = rng.normal(0.0, 0.05, size=(n, L, D))
    target = rng.normal(0.0, 0.05, size=n)
    condition_ids = np.array([0] * N_PER_GROUP + [1] * N_PER_GROUP)
    # Block 1 gets +10 on BOTH the projected coordinate and the target.
    acts[N_PER_GROUP:, :, 0] += 10.0
    target[N_PER_GROUP:] += 10.0
    return acts, direction, target, condition_ids


def test_within_ci_differs_from_pooled_ci():
    acts, direction, target, cid = _synthetic_two_condition_dataset()
    pooled = nb.bootstrap_ci_matched_r(acts, direction, target, 0, n_boot=200, seed=1)
    within = nb.bootstrap_ci_within_r(acts, direction, target, cid, 0, n_boot=200, seed=1)
    # Pooled r ~ 1.0 (block contrast); within r ~ 0. The CIs must be far apart.
    assert pooled[0] > 0.9, f"pooled CI unexpectedly low: {pooled}"
    assert within[1] < 0.9, f"within CI unexpectedly high (pooled-like): {within}"
    assert abs(pooled[0] - within[0]) > 0.3, (pooled, within)


def test_compute_setting_within_reports_within_ci():
    acts, direction, target, cid = _synthetic_two_condition_dataset()
    pos = np.random.default_rng(2).normal(size=(6, L, D))
    neg = np.random.default_rng(3).normal(size=(6, L, D))
    result, _draws = nb.compute_setting(
        "trait",
        "monitoring_within",
        predictor_acts=acts,
        rb_per_layer=direction,
        target=target,
        pos_acts=pos,
        neg_acts=neg,
        other_rbs={"other": direction},
        condition_ids=cid,
        n_draws=10,
        n_boot=200,
        seed=1,
    )
    sel = result.matched_selected_layer
    expected_within = nb.bootstrap_ci_within_r(
        acts, direction, target, cid, sel, n_boot=200, seed=1
    )
    pooled = nb.bootstrap_ci_matched_r(acts, direction, target, sel, n_boot=200, seed=1)
    got = tuple(result.matched_r_bootstrap_ci_95)
    assert np.allclose(got, expected_within), (
        f"within setting must report the WITHIN CI {expected_within}, got {got}"
    )
    # Pre-fix signature: the reported CI equalled the pooled CI. Assert it does not.
    assert not np.allclose(got, pooled), (
        f"within setting reported the POOLED CI {pooled} — the #778 bug regressed"
    )
    # The within point estimate must be covered by the within CI, not the pooled one.
    within_point = nb.within_condition_r_per_layer(acts, direction, target, cid)[sel]
    assert got[0] <= within_point <= got[1], (got, within_point)
    assert not (pooled[0] <= within_point <= pooled[1]), (pooled, within_point)


def test_compute_setting_overall_ci_unchanged():
    """The overall (pooled) setting keeps the original pooled CI path."""
    acts, direction, target, cid = _synthetic_two_condition_dataset()
    pos = np.random.default_rng(2).normal(size=(6, L, D))
    neg = np.random.default_rng(3).normal(size=(6, L, D))
    result, _draws = nb.compute_setting(
        "trait",
        "monitoring_overall",
        predictor_acts=acts,
        rb_per_layer=direction,
        target=target,
        pos_acts=pos,
        neg_acts=neg,
        other_rbs={"other": direction},
        condition_ids=cid,
        n_draws=10,
        n_boot=200,
        seed=1,
    )
    sel = result.matched_selected_layer
    pooled = nb.bootstrap_ci_matched_r(acts, direction, target, sel, n_boot=200, seed=1)
    assert np.allclose(tuple(result.matched_r_bootstrap_ci_95), pooled)
