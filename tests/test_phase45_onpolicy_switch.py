"""Unit tests for Phase 4.5 on-policy switch verdict (review blocker #1 round-4).

Round-2/3 had a silent zero-denominator bug: `plain_mean > 0` gated the
switch, but R_canon is generated under the SYSTEM encoding so
`system_plain`'s trained-greedy R is ~identical to R_canon → plain_mean
~ 0 in the normal training regime. So `plain_mean == 0 AND role_mean > 0`
(exactly the high-role-drift case the gate exists to catch) silently
set `switch=False`.

These tests load the phase45 script as a module and call the pure
`_onpolicy_switch_verdict` helper directly so the regression is
unit-locked without a CLI invocation.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def phase45_mod():
    """Load `scripts/i464_phase45_onpolicy_validation.py` as a module."""
    spec = importlib.util.spec_from_file_location(
        "i464_phase45_onpolicy_validation",
        REPO_ROOT / "scripts" / "i464_phase45_onpolicy_validation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_normal_case_plain_positive_role_below_threshold(phase45_mod):
    """plain_mean=0.1, role_mean=0.13 → ratio=1.3 < 1.5 → switch=False."""
    ratio, switch = phase45_mod._onpolicy_switch_verdict(0.13, 0.1, 1.5)
    assert ratio == pytest.approx(1.3)
    assert switch is False


def test_normal_case_plain_positive_role_above_threshold(phase45_mod):
    """plain_mean=0.1, role_mean=0.20 → ratio=2.0 > 1.5 → switch=True."""
    ratio, switch = phase45_mod._onpolicy_switch_verdict(0.20, 0.1, 1.5)
    assert ratio == pytest.approx(2.0)
    assert switch is True


def test_zero_denom_role_drifts_should_switch(phase45_mod):
    """The bug round-4 fixes — plain_mean=0, role_mean>0 → ratio=inf, switch=True.

    R_canon is generated under the SYSTEM encoding so system_plain's
    trained-greedy R ~ identical to R_canon → plain_mean ~ 0 in the
    normal case. Round-2/3's `plain_mean > 0` gate would have returned
    switch=False here, silently disabling MF-B(2) safeguarding exactly
    when it matters most (role drifts but system doesn't).
    """
    ratio, switch = phase45_mod._onpolicy_switch_verdict(0.15, 0.0, 1.5)
    assert ratio == float("inf"), f"expected inf ratio for zero-denom + role drift, got {ratio}"
    assert math.isinf(ratio)
    assert switch is True, (
        "round-2/3 bug: plain_mean=0 + role_mean>0 silently set switch=False; "
        "round-4 must set switch=True"
    )


def test_both_zero_no_drift_no_switch(phase45_mod):
    """plain_mean=0, role_mean=0 → no drift detected anywhere → switch=False."""
    ratio, switch = phase45_mod._onpolicy_switch_verdict(0.0, 0.0, 1.5)
    assert ratio == 0.0
    assert switch is False


def test_role_mean_none_returns_no_switch(phase45_mod):
    """Missing role data → can't decide → switch=False, ratio=None."""
    ratio, switch = phase45_mod._onpolicy_switch_verdict(None, 0.1, 1.5)
    assert ratio is None
    assert switch is False


def test_plain_mean_none_returns_no_switch(phase45_mod):
    """Missing system_plain data → can't decide → switch=False, ratio=None."""
    ratio, switch = phase45_mod._onpolicy_switch_verdict(0.1, None, 1.5)
    assert ratio is None
    assert switch is False


def test_both_none_returns_no_switch(phase45_mod):
    """Both missing → can't decide → switch=False, ratio=None."""
    ratio, switch = phase45_mod._onpolicy_switch_verdict(None, None, 1.5)
    assert ratio is None
    assert switch is False


def test_compute_onpolicy_switch_inputs_extracts_means(phase45_mod):
    """_compute_onpolicy_switch_inputs returns (role_mean, plain_mean) from per_arm dict."""
    per_arm = {
        "role": {"n": 32, "mean": 0.20, "median": 0.18},
        "system_plain": {"n": 32, "mean": 0.05, "median": 0.05},
        "system_padded": {"n": 32, "mean": 0.07, "median": 0.06},
    }
    role_mean, plain_mean = phase45_mod._compute_onpolicy_switch_inputs(per_arm)
    assert role_mean == 0.20
    assert plain_mean == 0.05


def test_compute_onpolicy_switch_inputs_handles_missing_arm(phase45_mod):
    """Missing arm in per_arm dict returns None for that mean."""
    per_arm = {
        "role": {"n": 32, "mean": 0.20, "median": 0.18},
        # system_plain absent
    }
    role_mean, plain_mean = phase45_mod._compute_onpolicy_switch_inputs(per_arm)
    assert role_mean == 0.20
    assert plain_mean is None
