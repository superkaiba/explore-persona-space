"""Unit tests for the #514 per-lever abort-on-collapse decision function.

Plan §4.1.3 requires a unit test on the abort decision. The decision rule:
abort iff (source_r_collapse_rate >= 0.50) AND (held_out_g_logprob_mean > -5.0).
"""

from __future__ import annotations

import math

import pytest

from explore_persona_space.experiments.full_ft_regime_514.abort_logic import (
    compute_source_r_collapse_rate,
    get_held_out_g_logprob_mean,
    should_abort_lever,
)


def _eval_json(
    *,
    n_source: int,
    n_collapsed_source: int,
    held_out_g_logprob_mean: float | None,
) -> dict:
    """Build a minimal eval JSON stub mirroring the #508 eval_one_cell shape."""
    source_persona = "villain"
    delta_g_source: dict = {source_persona: {}}
    for i in range(n_source):
        q = f"q{i}"
        delta_g_source[source_persona][q] = {
            "trained_logp": -1.0,
            "base_logp": -10.0,
            "delta_g": 9.0,
            "trained_argmax_marker": True,
            "base_argmax_marker": False,
            "r_collapsed": i < n_collapsed_source,
            "n_marker_in_R": 0,
        }
    aggregates: dict = {}
    if held_out_g_logprob_mean is not None:
        aggregates["held_out_g_logprob_mean"] = held_out_g_logprob_mean
    return {
        "delta_g_source": delta_g_source,
        "aggregates": aggregates,
    }


# ── compute_source_r_collapse_rate ────────────────────────────────────────────


def test_rcollapse_rate_all_collapsed() -> None:
    ej = _eval_json(n_source=20, n_collapsed_source=20, held_out_g_logprob_mean=-2.0)
    assert compute_source_r_collapse_rate(ej) == 1.0


def test_rcollapse_rate_none_collapsed() -> None:
    ej = _eval_json(n_source=20, n_collapsed_source=0, held_out_g_logprob_mean=-10.0)
    assert compute_source_r_collapse_rate(ej) == 0.0


def test_rcollapse_rate_half() -> None:
    ej = _eval_json(n_source=20, n_collapsed_source=10, held_out_g_logprob_mean=-2.0)
    assert compute_source_r_collapse_rate(ej) == pytest.approx(0.5)


def test_rcollapse_rate_no_source_returns_nan() -> None:
    ej = {"delta_g_source": {}, "aggregates": {"held_out_g_logprob_mean": -2.0}}
    rate = compute_source_r_collapse_rate(ej)
    assert math.isnan(rate)


# ── get_held_out_g_logprob_mean ──────────────────────────────────────────────


def test_glogp_present() -> None:
    ej = _eval_json(n_source=1, n_collapsed_source=0, held_out_g_logprob_mean=-3.7)
    assert get_held_out_g_logprob_mean(ej) == pytest.approx(-3.7)


def test_glogp_missing_returns_nan() -> None:
    ej = _eval_json(n_source=1, n_collapsed_source=0, held_out_g_logprob_mean=None)
    assert math.isnan(get_held_out_g_logprob_mean(ej))


# ── should_abort_lever (the canonical decision) ──────────────────────────────


def test_abort_when_both_conditions_hold() -> None:
    """19/20 collapsed (#508's ft_b2 shape) AND held-out at -2 nat (saturated) → ABORT."""
    ej = _eval_json(n_source=20, n_collapsed_source=19, held_out_g_logprob_mean=-2.0)
    abort, diag = should_abort_lever(ej)
    assert abort is True
    assert diag["source_r_collapse_rate"] == pytest.approx(19 / 20)
    assert diag["held_out_g_logprob_mean"] == -2.0
    assert ">= 0.5" in diag["reason"]


def test_no_abort_when_clean_ft_b1_like() -> None:
    """0/20 collapsed (#508's ft_b1 shape) AND held-out at -8 nat (sub-ceiling) → CONTINUE."""
    ej = _eval_json(n_source=20, n_collapsed_source=0, held_out_g_logprob_mean=-8.0)
    abort, diag = should_abort_lever(ej)
    assert abort is False
    assert "continuing" in diag["reason"]


def test_no_abort_when_collapsed_but_subceiling() -> None:
    """Collapsed BUT held-out still sub-ceiling — the dispatcher continues (the cliff hasn't fully fired)."""
    ej = _eval_json(n_source=20, n_collapsed_source=15, held_out_g_logprob_mean=-7.0)
    abort, diag = should_abort_lever(ej)
    assert abort is False
    assert diag["source_r_collapse_rate"] == pytest.approx(0.75)


def test_no_abort_when_saturated_but_not_collapsed() -> None:
    """Saturated held-out (>-5 nat) but only 25% r-collapse — no abort yet."""
    ej = _eval_json(n_source=20, n_collapsed_source=5, held_out_g_logprob_mean=-3.0)
    abort, diag = should_abort_lever(ej)
    assert abort is False


def test_abort_at_exact_threshold_rcollapse() -> None:
    """50% r-collapse is the inclusive lower bound (>=) per the plan."""
    ej = _eval_json(n_source=20, n_collapsed_source=10, held_out_g_logprob_mean=-2.0)
    abort, _diag = should_abort_lever(ej)
    assert abort is True


def test_no_abort_at_exact_threshold_glogp() -> None:
    """-5.0 is the inclusive upper bound (>); -5.0 exactly should NOT trigger."""
    ej = _eval_json(n_source=20, n_collapsed_source=20, held_out_g_logprob_mean=-5.0)
    abort, _diag = should_abort_lever(ej)
    assert abort is False


def test_nan_inputs_conservative_no_abort() -> None:
    """NaN propagation is conservative — incomplete eval must NOT abort the lever."""
    ej_no_source = {"delta_g_source": {}, "aggregates": {"held_out_g_logprob_mean": -1.0}}
    abort, diag = should_abort_lever(ej_no_source)
    assert abort is False
    assert "NaN-input" in diag["reason"]

    ej_no_glogp = _eval_json(n_source=20, n_collapsed_source=20, held_out_g_logprob_mean=None)
    abort, diag = should_abort_lever(ej_no_glogp)
    assert abort is False
    assert "NaN-input" in diag["reason"]


def test_custom_thresholds_passthrough() -> None:
    """Aggressive thresholds (rcoll>=0.3 AND glogp>-7) should fire on borderline."""
    ej = _eval_json(n_source=20, n_collapsed_source=7, held_out_g_logprob_mean=-6.5)
    abort, _diag = should_abort_lever(ej, rcollapse_threshold=0.3, g_logprob_max=-7.0)
    assert abort is True
