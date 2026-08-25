"""Issue #2588 P3 mechanizable acceptance — verdict lattice + pair metadata.

Plan §3 (MF3): enumerate synthetic three-point orderings and assert every
state labeled Capability-tracks-ordered satisfies BOTH the endpoint predicate
AND the registered ordering predicate (Δadj_min >= 0); the lattice consumes
CALIBRATED fields only. Plan §4.3 (MF2): mixed contrast pairs (different
checkpoint AND different input-position semantics) are REJECTED.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from issue2588_trend import (
    MapRef,
    assert_calibrated_inputs,
    assert_pair_metadata,
    verdict_lattice,
)


def _lattice(a35: float, a36: float, a38: float, ci_lo: float, ci_hi: float) -> dict:
    return verdict_lattice(
        {
            "delta_endpoint_cal": a38 - a35,
            "ci_low_endpoint_cal": ci_lo,
            "ci_high_endpoint_cal": ci_hi,
            "delta_step1_cal": a36 - a35,
            "delta_step2_cal": a38 - a36,
        }
    )


def test_synthetic_orderings_enumeration():
    """Every permutation of three distinct calibrated levels, with the CI
    positioned to make the endpoint predicate true where the endpoint delta is
    positive: tracks_ordered fires IFF monotone non-decreasing ordering."""
    levels = (0.10, 0.20, 0.30)
    for a35, a36, a38 in itertools.permutations(levels):
        d = a38 - a35
        ci_lo, ci_hi = d - 0.05, d + 0.05  # excludes 0 positively iff d >= 0.1
        v = _lattice(a35, a36, a38, ci_lo, ci_hi)
        monotone = a35 <= a36 <= a38
        endpoint_pred = d > 0 and ci_lo > 0
        assert v["capability_tracks_ordered"] == (endpoint_pred and monotone), (a35, a36, a38, v)
        # tracks_ordered NEVER fires without BOTH predicates (the MF3 pin)
        if v["capability_tracks_ordered"]:
            assert v["endpoint_up"] and v["order_consistent"]


def test_midpoint_dip_is_not_capability_tracking():
    """The v2 lattice defect: 3.6 below BOTH endpoints with a positive
    endpoint CI must NOT be labeled capability-tracking."""
    v = _lattice(0.20, 0.10, 0.30, ci_lo=0.02, ci_hi=0.18)
    assert v["endpoint_up"] and not v["order_consistent"]
    assert not v["capability_tracks_ordered"]
    assert v["ordering_label"] == "order_inconsistent"


def test_endpoint_partition_is_disjoint_exhaustive():
    up = _lattice(0.1, 0.2, 0.3, 0.05, 0.35)
    inv = _lattice(0.3, 0.2, 0.1, -0.35, -0.05)
    ind = _lattice(0.1, 0.2, 0.3, -0.05, 0.35)
    assert up["endpoint_label"] == "endpoint_up"
    assert inv["endpoint_label"] == "capability_inverts"
    assert ind["endpoint_label"] == "indistinguishable"
    for v in (up, inv, ind):
        assert v["endpoint_up"] + v["capability_inverts"] <= 1


def test_lattice_rejects_uncalibrated_fields():
    with pytest.raises(ValueError, match="NON-calibrated"):
        verdict_lattice(
            {
                "delta_endpoint_cal": 0.1,
                "ci_low_endpoint_cal": 0.05,
                "ci_high_endpoint_cal": 0.2,
                "delta_step1_cal": 0.05,
                "delta_step2_raw": 0.05,  # raw field smuggled in
            }
        )
    with pytest.raises(ValueError):
        assert_calibrated_inputs({"delta_endpoint": 0.1})


def test_lattice_requires_all_fields():
    with pytest.raises(AssertionError, match="missing fields"):
        verdict_lattice(
            {
                "delta_endpoint_cal": 0.1,
                "ci_low_endpoint_cal": 0.0,
                "ci_high_endpoint_cal": 0.2,
                "delta_step1_cal": 0.1,
            }
        )


def test_pair_metadata_legal_pairs():
    # same checkpoint, different positions (H2 pairs / OLMo-P)
    assert_pair_metadata(
        MapRef("q35_27b", "b", "cot_boundary"), MapRef("q35_27b", "a", "prompt_last")
    )
    assert_pair_metadata(
        MapRef("o3_7b_t", "b", "pre_think"), MapRef("o3_7b_t", "b", "cot_boundary")
    )
    # different checkpoints, SAME prompt-side semantics (OLMo-R; column pairs)
    assert_pair_metadata(MapRef("o3_7b_i", "a", "prompt_last"), MapRef("o3_7b_t", "b", "pre_think"))
    assert_pair_metadata(
        MapRef("q35_27b", "a", "prompt_last"), MapRef("q38_27b", "a", "prompt_last")
    )


def test_pair_metadata_rejects_mixed_pair():
    """Different checkpoint AND different position semantics = the banned
    mixed object (checkpoint identity confounded with read position)."""
    with pytest.raises(ValueError, match="mixed contrast pair REJECTED"):
        assert_pair_metadata(
            MapRef("o3_7b_i", "a", "prompt_last"), MapRef("o3_7b_t", "b", "cot_boundary")
        )
    with pytest.raises(ValueError, match="mixed contrast pair REJECTED"):
        assert_pair_metadata(
            MapRef("q35_27b", "a", "prompt_last"), MapRef("q38_27b", "b", "cot_boundary")
        )
