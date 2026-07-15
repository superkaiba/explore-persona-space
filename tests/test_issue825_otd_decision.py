"""#825 onpolicy-turn-depth: pre-registered §6 outcome map over CI sign patterns.

Pins the plan §6 "and/or" H2 routing (code-review v18 Minor, task #825
`onpolicy-turn-depth-map` round 10): a ONE-SIDED positive exclusion (one CI
excluding 0 with Δ > 0 while the other includes 0) routes to the H2 /
descriptive read, never to a "mixed" bucket; any negative exclusion (absent
both-positive) is H-neg; both CIs including 0 is H0; anchor demotion
downgrades confirms to descriptive.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue825_onpolicy_turn_depth_fit import _decision_outcome  # noqa: E402


def test_both_positive_high_r2_is_h1():
    out = _decision_outcome([0.02, 0.10], [0.01, 0.12], 0.55, False)
    assert out.startswith("H1 ")


def test_both_positive_low_r2_is_h2():
    out = _decision_outcome([0.02, 0.10], [0.01, 0.12], 0.20, False)
    assert out.startswith("H2 ")


def test_one_sided_positive_pooled_routes_to_h2_not_mixed():
    out = _decision_outcome([0.02, 0.10], [-0.05, 0.12], 0.20, False)
    assert out.startswith("H2")
    assert "one-sided" in out and "pooled" in out
    assert "mixed" not in out


def test_one_sided_positive_t1_routes_to_h2_not_mixed():
    out = _decision_outcome([-0.02, 0.10], [0.01, 0.12], 0.20, False)
    assert out.startswith("H2")
    assert "one-sided" in out and "t1" in out
    assert "mixed" not in out


def test_one_sided_positive_with_high_r2_is_descriptive_boundary():
    out = _decision_outcome([0.02, 0.10], [-0.05, 0.12], 0.55, False)
    assert out.startswith("H1/H2 boundary")
    assert "descriptively" in out


def test_any_negative_exclusion_is_hneg():
    assert _decision_outcome([-0.10, -0.02], [-0.05, 0.12], 0.20, False).startswith("H-neg")
    assert _decision_outcome([0.02, 0.10], [-0.12, -0.01], 0.20, False).startswith("H-neg")


def test_both_include_zero_is_h0():
    assert _decision_outcome([-0.02, 0.10], [-0.05, 0.12], 0.20, False).startswith("H0")


def test_anchor_demotion_downgrades_to_descriptive():
    assert "descriptively" in _decision_outcome([0.02, 0.10], [0.01, 0.12], 0.55, True)
    assert "descriptively" in _decision_outcome([0.02, 0.10], [-0.05, 0.12], 0.20, True)


def test_missing_t1_ci_treated_as_non_excluding():
    out = _decision_outcome([0.02, 0.10], None, 0.20, False)
    assert out.startswith("H2") and "one-sided" in out
