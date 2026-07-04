"""Unit pins for the issue #952 `cross-layer-decision-cells` registered statistics.

Pins the follow-up plan §2 constructions EXACTLY as registered: the pinned
bootstrap-p (shift-to-null-center, add-one counting; two-sided H1 / one-sided
positive-tail H2), Holm step-down, the three-way per-layer status enums
(NON-STRICT ±0.03 equivalence containment for H1; CI-upper < 0.02 affirmative
non-replication for H2), and the TOTAL outcome lattice — including the plan's
mechanizable check that `L20_local` can NEVER fire off indeterminate reads
(affirmative >= 2/3 in either family is the only route).
"""

import importlib.util
import itertools
import pathlib

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "issue952_stats_under_test",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "issue952_stats.py",
)
stats = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stats)

STATUSES = ("replicates", "affirmative_nonreplication", "indeterminate")


def test_pinned_p_add_one_counting_two_sided():
    """p = (1 + #{|null| >= |obs|}) / (1 + B) with null = draws - observed."""
    obs = 1.0
    # draws - obs = [-2, -1, 0, 1, 2] -> |null| >= 1 for 4 of 5 draws.
    draws = np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0])
    p = stats.pinned_bootstrap_p(obs, draws, tail="two")
    assert p == pytest.approx((1 + 4) / (1 + 5))


def test_pinned_p_one_sided_positive_tail():
    """H2 tail: p = (1 + #{null >= obs}) / (1 + B)."""
    obs = 1.0
    # null = [-2, -1, 0, 1, 2] -> null >= 1 for 2 of 5 draws.
    draws = np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0])
    p = stats.pinned_bootstrap_p(obs, draws, tail="greater")
    assert p == pytest.approx((1 + 2) / (1 + 5))


def test_pinned_p_guards():
    """Non-finite observed / empty draws -> None (dropped, never coerced)."""
    assert stats.pinned_bootstrap_p(float("nan"), np.asarray([1.0]), tail="two") is None
    assert stats.pinned_bootstrap_p(1.0, np.asarray([]), tail="two") is None
    with pytest.raises(ValueError):
        stats.pinned_bootstrap_p(1.0, np.asarray([1.0]), tail="less")


def test_holm_step_down_matches_registered_convention():
    """Holm: sort ascending; adj = max-running (k - rank) * p, capped at 1."""
    out = stats.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"] == pytest.approx(0.03)  # 3 * 0.01
    assert out["c"] == pytest.approx(0.06)  # max(0.03, 2 * 0.03)
    assert out["b"] == pytest.approx(0.06)  # max(0.06, 1 * 0.04)
    assert stats.holm_adjust({"a": 0.9, "b": 0.8})["a"] == 1.0  # cap
    assert stats.holm_adjust({"a": 0.01, "b": None})["b"] is None  # missing read


def test_h1_status_nonstrict_containment_boundary():
    """Equivalence containment is NON-STRICT: lo == -0.03 AND hi == +0.03 replicates."""
    assert stats.h1_status(0.0, -0.03, 0.03, holm_p=1.0) == "replicates"
    # Straddling a margin boundary without a Holm-significant detection.
    assert stats.h1_status(0.02, -0.05, 0.09, holm_p=0.5) == "indeterminate"
    # Affirmative route 1: contrast >= +0.03 AND Holm p < 0.05.
    assert stats.h1_status(0.05, 0.01, 0.09, holm_p=0.01) == "affirmative_nonreplication"
    # Same point estimate WITHOUT the Holm detection -> indeterminate.
    assert stats.h1_status(0.05, 0.01, 0.09, holm_p=0.20) == "indeterminate"
    # Affirmative route 2: CI entirely outside the band (either side).
    assert stats.h1_status(0.06, 0.035, 0.09, holm_p=1.0) == "affirmative_nonreplication"
    assert stats.h1_status(-0.06, -0.09, -0.035, holm_p=1.0) == "affirmative_nonreplication"
    # Missing read -> indeterminate.
    assert stats.h1_status(None, float("nan"), float("nan"), None) == "indeterminate"


def test_h2_status_enum():
    """Replicates = ΔG >= 0.02 AND Holm p < 0.05; affirmative = CI upper < 0.02."""
    assert stats.h2_status(0.03, hi=0.05, holm_p=0.01) == "replicates"
    assert stats.h2_status(0.03, hi=0.05, holm_p=0.20) == "indeterminate"
    assert stats.h2_status(0.005, hi=0.015, holm_p=0.5) == "affirmative_nonreplication"
    assert stats.h2_status(0.01, hi=0.05, holm_p=0.5) == "indeterminate"
    assert stats.h2_status(None, hi=float("nan"), holm_p=None) == "indeterminate"


def test_lattice_totality_and_forbidden_route():
    """Every 3^3 x 3^3 status combination maps to exactly one of the four
    registered outcomes, and `L20_local` requires >= 2/3 affirmative in a family
    (indeterminate reads NEVER open it) — plan §2 mechanizable check."""
    layers = ("14", "23", "26")
    verdicts = set()
    for combo1 in itertools.product(STATUSES, repeat=3):
        for combo2 in itertools.product(STATUSES, repeat=3):
            h1 = dict(zip(layers, combo1, strict=True))
            h2 = dict(zip(layers, combo2, strict=True))
            rec = stats.map_outcome_lattice(h1, h2)
            v = rec["overall_verdict"]
            verdicts.add(v)
            assert v in {
                "full_replication",
                "L20_local",
                "inconclusive_layer_scope",
                "partial_band_map",
            }
            aff1 = combo1.count("affirmative_nonreplication")
            aff2 = combo2.count("affirmative_nonreplication")
            if v == "L20_local":
                assert max(aff1, aff2) >= 2  # the ONLY route
            if max(aff1, aff2) >= 2:
                assert v == "L20_local"
            if v == "full_replication":
                assert combo1 == ("replicates",) * 3 and combo2 == ("replicates",) * 3
            if v == "inconclusive_layer_scope":
                assert combo1 == ("indeterminate",) * 3 and combo2 == ("indeterminate",) * 3
    assert verdicts == {
        "full_replication",
        "L20_local",
        "inconclusive_layer_scope",
        "partial_band_map",
    }


def test_lattice_rejects_mismatched_layer_sets():
    """The lattice needs the SAME non-empty added-layer set in both families."""
    with pytest.raises(AssertionError):
        stats.map_outcome_lattice({"14": "replicates"}, {"23": "replicates"})
    with pytest.raises(AssertionError):
        stats.map_outcome_lattice({}, {})
