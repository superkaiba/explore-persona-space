# ruff: noqa: RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + × intentional
"""Task #504 v5 — byte-identical guard + KL-not-a-DV regression tests.

Pins three contracts the v5 implementer changes introduce:

1. ``compute_byte_identical_rate`` correctly identifies records where
   ``|g - b| < abs_tol`` AND ``kl > kl_min`` (the v3 marker-logprob-path-bug
   signature).

2. ``assert_byte_identical_rate_below_threshold`` raises
   ``MarkerLogprobPathReadingFromBaseError`` when the rate exceeds the 5%
   de-minimis threshold (plan v5 §4.4 tightened from a draft 95%).

3. KL stays out of the marker-DV regression. The analyzer's pooled-regression
   input columns are strictly the 6 covariates from plan v5 §11
   (`d_source`, `d_nearest_neg_nd`, `shadow_angle`, `base_prior_marker`,
   `training_step`, `source_delta_g`) — `kl` MUST NOT appear as a column.
   Catches a future-implementer mistake of substituting KL for ΔG when the
   on-policy log-prob path looks "broken" (the v3 false-fix path that plan
   v5 fix #3 explicitly forbids).

CPU-only, sub-second.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_guard import (
    DEFAULT_BYTE_IDENTICAL_RATE_MAX,
    MarkerLogprobPathReadingFromBaseError,
    assert_byte_identical_rate_below_threshold,
    compute_byte_identical_rate,
)


def _records_for(g_logp: float, b_logp: float, n_pairs: int) -> dict:
    """Build a {persona: {q: {"logp": float}}} record dict with N pairs."""
    return {
        "probe_0": {f"q{i}": {"logp": g_logp, "argmax_marker": False} for i in range(n_pairs)},
    }


def _kl_for(rate_value: float, n_pairs: int) -> dict:
    return {"probe_0": {f"q{i}": rate_value for i in range(n_pairs)}}


def test_byte_identical_rate_zero_when_g_differs_from_b():
    """If g_logp differs from b_logp by >> 1e-6, the byte-identical rate is 0."""
    g = _records_for(-1.0, -1.0, 20)
    b = _records_for(-5.0, -5.0, 20)
    kl = _kl_for(2.0, 20)
    rate, n_bad, n_total = compute_byte_identical_rate(g, b, kl)
    assert rate == 0.0
    assert n_bad == 0
    assert n_total == 20


def test_byte_identical_rate_one_when_g_equals_b_and_kl_positive():
    """The v3 failure mode: g == b on every record + KL > 0 → rate = 1.0."""
    g = _records_for(-5.0, -5.0, 20)
    b = _records_for(-5.0, -5.0, 20)
    kl = _kl_for(24.35, 20)  # the actual v3 bystander recovery KL
    rate, n_bad, n_total = compute_byte_identical_rate(g, b, kl)
    assert rate == 1.0
    assert n_bad == 20
    assert n_total == 20


def test_byte_identical_rate_kl_below_min_does_not_count():
    """g == b but KL < kl_min is byte-identical-WITHOUT positive-KL — not the v3 bug."""
    g = _records_for(-5.0, -5.0, 20)
    b = _records_for(-5.0, -5.0, 20)
    kl = _kl_for(0.005, 20)  # below DEFAULT_KL_DIAGNOSTIC_MIN_NATS = 0.01
    rate, n_bad, _n_total = compute_byte_identical_rate(g, b, kl)
    # Without the positive-KL discriminator, this is just a tied float — not
    # the bug. (A clean reader on a probe where the trained model agrees
    # with base produces this exactly.)
    assert rate == 0.0
    assert n_bad == 0


def test_byte_identical_rate_partial_bug():
    """Plan v5 §4.4: a PARTIAL bug (60% of slots broken) trips the 5% threshold."""
    # 12 records broken (g==b), 8 clean records (g != b). N=20 → 60% byte-identical.
    g = {"p0": {}, "p1": {}}
    b = {"p0": {}, "p1": {}}
    kl = {"p0": {}, "p1": {}}
    for i in range(12):
        # broken: g == b, kl > 0
        g["p0"][f"q{i}"] = {"logp": -5.0, "argmax_marker": False}
        b["p0"][f"q{i}"] = {"logp": -5.0, "argmax_marker": False}
        kl["p0"][f"q{i}"] = 2.0
    for i in range(8):
        # clean: g != b
        g["p1"][f"q{i}"] = {"logp": -1.0, "argmax_marker": False}
        b["p1"][f"q{i}"] = {"logp": -5.0, "argmax_marker": False}
        kl["p1"][f"q{i}"] = 4.0
    rate, n_bad, n_total = compute_byte_identical_rate(g, b, kl)
    assert n_total == 20
    assert n_bad == 12
    assert rate == 0.60
    # 60% > 5% de-minimis threshold → guard MUST raise.
    with pytest.raises(MarkerLogprobPathReadingFromBaseError) as exc_info:
        assert_byte_identical_rate_below_threshold(g, b, kl, cell_label="test_cell")
    assert "byte_identical_rate=0.6000" in str(exc_info.value)
    assert "v3 recovery-bug signature" in str(exc_info.value)


def test_byte_identical_rate_just_below_threshold_passes():
    """1 of 20 byte-identical = 5.0% rate; the guard's max is 5.0%, so 5% > 5% is False → PASS."""
    # 1 record broken, 19 clean. 1/20 = 5.0% which equals max_rate; PASS.
    g = {"p0": {"q0": {"logp": -5.0}}, "p1": {}}
    b = {"p0": {"q0": {"logp": -5.0}}, "p1": {}}
    kl = {"p0": {"q0": 2.0}, "p1": {}}
    for i in range(19):
        g["p1"][f"q{i}"] = {"logp": -1.0}
        b["p1"][f"q{i}"] = {"logp": -5.0}
        kl["p1"][f"q{i}"] = 4.0
    diag = assert_byte_identical_rate_below_threshold(g, b, kl, cell_label="test_cell")
    assert diag["byte_identical_rate"] == 0.05
    assert diag["n_byte_identical"] == 1
    assert diag["n_probes"] == 20


def test_byte_identical_guard_passes_clean_reader():
    """A clean reader: rate = 0 → guard passes with diag dict."""
    g = _records_for(-1.0, -1.0, 20)
    b = _records_for(-5.0, -5.0, 20)
    kl = _kl_for(4.0, 20)
    diag = assert_byte_identical_rate_below_threshold(g, b, kl, cell_label="clean_test")
    assert diag["byte_identical_rate"] == 0.0
    assert diag["max_rate"] == DEFAULT_BYTE_IDENTICAL_RATE_MAX


def test_byte_identical_guard_no_kl_records():
    """When kl_records is None (--no-kl), the byte-identical rate still detects a uniform bug."""
    g = _records_for(-5.0, -5.0, 20)
    b = _records_for(-5.0, -5.0, 20)
    rate, n_bad, _n = compute_byte_identical_rate(g, b, None)
    assert rate == 1.0
    assert n_bad == 20


# ── Plan v5 fix #3: KL-from-base is NOT a fallback DV. ───────────────────────


def test_analyzer_regression_input_columns_exclude_kl():
    """Analyzer pooled regression must use ΔG only — KL must not be a DV column.

    The marker-leakage-measurement rule explicitly bans KL-from-base as a
    fallback DV; plan v5 §4.5 / §6.1 carries this forward. KL is allowed
    only as (a) a sanity diagnostic in Phase 0.6's pass condition and (b)
    the per-batch byte-identical guard (fix #1). Any future implementer
    substituting KL for the marker log-prob DV when the on-policy path
    "looks broken" is the v3 false-fix path this test catches.
    """
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import analyze

    # The covariate set is module-level and is the regression input.
    # plan v5 §11 enumerates the 6 covariates; `kl` must not appear.
    covariates = analyze.PREDICTORS
    assert "kl" not in covariates, (
        "PREDICTORS must not include 'kl' — KL-from-base is NOT a "
        "fallback DV (plan v5 §4.5 fix #3 + marker-leakage-measurement rule). "
        f"got: {covariates}"
    )
    expected = {
        "d_source",
        "d_nearest_neg_nd",
        "shadow_angle",
        "base_prior_marker",
        "training_step",
        "source_delta_g",
    }
    assert set(covariates) == expected, (
        f"PREDICTORS drift: expected {expected}, got {set(covariates)}"
    )
