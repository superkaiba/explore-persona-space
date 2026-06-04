# ruff: noqa: RUF002, RUF003
"""Regression tests for issue #490's LSE/Bernoulli-union combiner.

Round-3 code-review CRIT-1 fix: `combiner('lse', [-1000, -1001])` previously
returned 0.0 (saturation) because `exp(-1000)` underflows to 0.0 and the
`log_complement_product >= 0` guard fired. The corrected implementation
uses a two-regime helper (`_lse_bernoulli_union`):

  - Small-probability regime (all v_i ≤ -37 nats): falls back to
    `logsumexp(v_i)` directly, accurate to O(exp(v_max)) relative error
    because higher-order inclusion-exclusion terms are negligible.
  - General regime: log1mexp inclusion-exclusion identity.

Cases tested:
  1. Extreme-negative inputs: `[-1000.0, -1001.0]` → expect ≈ -999.69 (≈
     -1000 + log(1 + 1/e) = -1000 + 0.3133...).
  2. Near-zero negatives: e.g. `[-0.1, -0.2]` → expect log1mexp path
     gives sensible answer between -0.1 and 0.0.
  3. Typical pair: `[-8.0, -12.0]` → general regime works, answer ≈ -7.98.
  4. Exact 0.0 input: any v_i == 0 should return 0.0 (saturated).
  5. Positive input: `[1.0, 0.5]` should raise ValueError (delta-input rejection).
  6. Mixed regime: one extreme-negative, one small-magnitude → general regime.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_combiner_module():
    """Load scripts/_issue490_common.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "issue490_common_under_test",
        REPO_ROOT / "scripts" / "_issue490_common.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_combiner_module()


def test_lse_extreme_negative_inputs(mod):
    """CRIT-1 root case. combiner('lse', [-1000, -1001]) must NOT return 0.0.

    Mathematical expected value:
        P_union = 1 − (1 − e^-1000)(1 − e^-1001)
                ≈ e^-1000 + e^-1001 − e^-2001
                ≈ e^-1000 (1 + e^-1) − e^-2001
        log P_union ≈ -1000 + log(1 + 1/e)
                    ≈ -1000 + 0.31326...
                    ≈ -999.6867...
    """
    result = mod.combiner("lse", [-1000.0, -1001.0])
    expected = -1000.0 + math.log(1.0 + math.exp(-1.0))
    assert math.isfinite(result), f"Expected finite, got {result}"
    assert result != 0.0, (
        f"CRIT-1 regression: combiner('lse', [-1000, -1001]) silently saturated to 0.0; "
        f"expected ≈ {expected:.4f}"
    )
    assert result == pytest.approx(expected, abs=1e-9), (
        f"Expected ≈ {expected:.6f}, got {result:.6f}"
    )


def test_lse_near_zero_negatives(mod):
    """Sanity: near-zero negatives use the general (log1mexp) regime.

    P_union = 1 − (1 − e^-0.1)(1 − e^-0.2)
            = 1 − (1 − 0.9048)(1 − 0.8187)
            = 1 − 0.0952 × 0.1813
            = 1 − 0.01726
            = 0.98274
    log P_union = log(0.98274) = -0.01741
    """
    result = mod.combiner("lse", [-0.1, -0.2])
    p1 = math.exp(-0.1)
    p2 = math.exp(-0.2)
    p_union = 1.0 - (1.0 - p1) * (1.0 - p2)
    expected = math.log(p_union)
    assert result == pytest.approx(expected, abs=1e-9), f"Expected {expected:.6f}, got {result:.6f}"
    assert -0.2 < result < 0.0, f"Result {result} not in plausible range (-0.2, 0)"


def test_lse_typical_pair(mod):
    """The kind of input the real analyzer sees (abs log P(※) ~ -4 to -30).

    P_union = 1 − (1 − e^-8)(1 − e^-12)
    log P_union ≈ logsumexp([-8, -12]) − tiny correction
    """
    result = mod.combiner("lse", [-8.0, -12.0])
    # log P_union > log(max(p_i)) = -8 (union ≥ either source)
    # log P_union < log(p_1 + p_2) = logsumexp([-8, -12]) ≈ -7.98169
    from scipy.special import logsumexp

    upper_bound = float(logsumexp([-8.0, -12.0]))
    assert result <= upper_bound + 1e-12, (
        f"Result {result} exceeds logsumexp upper bound {upper_bound}"
    )
    assert result >= -8.0, f"Result {result} below max-source lower bound -8.0"
    # The exact expected value via the same identity used in _lse_bernoulli_union.
    p1 = math.exp(-8.0)
    p2 = math.exp(-12.0)
    expected = math.log(1.0 - (1.0 - p1) * (1.0 - p2))
    assert result == pytest.approx(expected, abs=1e-9)


def test_lse_zero_input_saturates(mod):
    """A saturated source (v == 0.0, p == 1.0) → union is saturated → return 0.0."""
    assert mod.combiner("lse", [0.0, -5.0]) == 0.0
    assert mod.combiner("lse", [-5.0, 0.0]) == 0.0
    assert mod.combiner("lse", [0.0, 0.0]) == 0.0


def test_lse_positive_input_rejected(mod):
    """Delta-input rejection: any v_i > 0 raises ValueError (round-2 contract)."""
    with pytest.raises(ValueError, match="all values must be ≤ 0"):
        mod.combiner("lse", [1.0, 0.5])
    with pytest.raises(ValueError, match="all values must be ≤ 0"):
        mod.combiner("lse", [-5.0, 0.001])


def test_lse_mixed_regime(mod):
    """One extreme-negative, one small-magnitude → general regime should be used.

    The "all v_i ≤ -37" small-probability regime threshold should NOT trigger
    when ANY input is above the threshold.

    P_union = 1 − (1 − e^-0.5)(1 − e^-1000)
            ≈ 1 − (1 − 0.6065) × 1.0
            ≈ 0.6065
    log P_union ≈ log(0.6065) ≈ -0.50 (dominated by the small-magnitude source)
    """
    result = mod.combiner("lse", [-0.5, -1000.0])
    p1 = math.exp(-0.5)
    # e^-1000 is exactly 0 in float64; log(1 − 0) = 0 exactly.
    expected = math.log(p1)  # = -0.5 exactly
    assert result == pytest.approx(expected, abs=1e-9), f"Expected {expected:.6f}, got {result:.6f}"


def test_lse_single_value(mod):
    """Single source: log P_union = log(p_1) = v_1."""
    assert mod.combiner("lse", [-3.5]) == pytest.approx(-3.5, abs=1e-12)
    assert mod.combiner("lse", [-1000.0]) == pytest.approx(-1000.0, abs=1e-12)


def test_lse_empty_input(mod):
    with pytest.raises(ValueError, match="empty"):
        mod.combiner("lse", [])


def test_combiner_lse_delta_from_absolutes_handles_extreme(mod):
    """The delta-from-absolutes helper must inherit the underflow fix.

    For trained=[-4.0, -8.0] vs base=[-12.0, -16.0]:
        LSE(trained) ≈ log(e^-4 + e^-8 − e^-12) ≈ -3.982
        LSE(base)    ≈ log(e^-12 + e^-16 − e^-28) ≈ -11.982
        delta        ≈ +8.000
    """
    delta = mod.combiner_lse_delta_from_absolutes(
        trained_values=[-4.0, -8.0],
        base_values=[-12.0, -16.0],
    )
    assert math.isfinite(delta)
    assert delta == pytest.approx(8.0, abs=0.05), f"Expected ≈ 8.0, got {delta:.4f}"


def test_combiner_lse_delta_from_absolutes_underflow_regime(mod):
    """Round-3 regression: extreme-negative absolute logps must not silently
    collapse to 0.0 − 0.0 == 0.0 delta.

    For trained=[-100.0, -101.0] vs base=[-200.0, -201.0]:
        LSE(trained) ≈ -100 + log(1 + 1/e) ≈ -99.687
        LSE(base)    ≈ -200 + log(1 + 1/e) ≈ -199.687
        delta        ≈ +100.0
    """
    delta = mod.combiner_lse_delta_from_absolutes(
        trained_values=[-100.0, -101.0],
        base_values=[-200.0, -201.0],
    )
    assert math.isfinite(delta), f"Expected finite, got {delta}"
    assert delta != 0.0, (
        "CRIT-1 regression: combiner_lse_delta_from_absolutes silently "
        "collapsed extreme-negative legs to 0.0 delta"
    )
    assert delta == pytest.approx(100.0, abs=0.001), f"Expected ≈ 100.0, got {delta:.6f}"
