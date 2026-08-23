"""Pins for the #823 pilot refusal-attribution helpers.

These numbers feed a registered plan threshold (the refusal-attrition budget in
plan v11), so the arithmetic is pinned rather than trusted: a silent change to
the interval or the arm-impact logic would move a gate.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "issue823_pilot_refusal_attribution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("i823_attr", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


# --- wilson_interval -------------------------------------------------------


def test_wilson_reproduces_the_pilot_interval():
    """19/300 -> [4.09%, 9.68%], the interval the plan threshold is derived from."""
    lo, hi = mod.wilson_interval(19, 300)
    assert lo == pytest.approx(0.0409, abs=5e-4)
    assert hi == pytest.approx(0.0968, abs=5e-4)
    # The point estimate must lie inside its own interval.
    assert lo < 19 / 300 < hi


def test_wilson_brackets_the_point_estimate_and_stays_in_unit_range():
    for k, n in [(0, 10), (1, 10), (5, 10), (10, 10), (1, 1000), (25, 896)]:
        lo, hi = mod.wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0, (k, n, lo, hi)
        assert lo <= k / n <= hi, (k, n, lo, hi)


def test_wilson_does_not_degenerate_at_zero_events():
    """The reason Wilson is used over the normal approximation: k=0 stays informative."""
    lo, hi = mod.wilson_interval(0, 300)
    assert lo == 0.0
    assert hi > 0.0, "a zero-event upper bound of exactly 0 would understate risk"


def test_wilson_rejects_empty_denominator():
    with pytest.raises(ValueError):
        mod.wilson_interval(0, 0)


def test_wilson_interval_narrows_with_n():
    """Same proportion, more data => strictly narrower interval."""
    small = mod.wilson_interval(19, 300)
    large = mod.wilson_interval(190, 3000)
    assert (large[1] - large[0]) < (small[1] - small[0])


# --- attribute() over a synthetic record set -------------------------------


def _records():
    """Two contexts x two personas, with one refusal and one cap hit.

    persona 0 serves arms {1,2}; persona 1 serves arm {2} only -- mirroring the
    real `persona(i,k) = i mod k` structure where arm k uses personas {0..k-1}.
    """
    return [
        {
            "context_id": 0,
            "persona_idx": 0,
            "persona_name": "P Zero",
            "arms": [1, 2],
            # Persona 0 must itself refuse, or it never enters the exclusion
            # ranking (only refusing personas are exclusion candidates) and the
            # arm-destruction case below is unreachable.
            "validity": "refusal",
            "stop_reason": "refusal",
            "cap_hit": False,
        },
        {
            "context_id": 1,
            "persona_idx": 0,
            "persona_name": "P Zero",
            "arms": [1, 2],
            "validity": "ok",
            "stop_reason": "max_tokens",
            "cap_hit": True,
        },
        {
            "context_id": 0,
            "persona_idx": 1,
            "persona_name": "P One",
            "arms": [2],
            "validity": "refusal",
            "stop_reason": "refusal",
            "cap_hit": False,
        },
        {
            "context_id": 1,
            "persona_idx": 1,
            "persona_name": "P One",
            "arms": [2],
            # The label disagreement the real pilot shows: validity says empty,
            # stop_reason says refusal. The gate must count this as a refusal.
            "validity": "empty",
            "stop_reason": "refusal",
            "cap_hit": False,
        },
    ]


def test_refusals_counted_by_stop_reason_not_validity_label():
    """The undercount defect: validity labels 2 refusals, stop_reason sees 3."""
    out = mod.attribute(_records(), n_production=100)
    assert out["pairs"]["n_refused_by_stop_reason"] == 3
    assert out["pairs"]["validity_counts"]["refusal"] == 2, (
        "fixture must preserve the disagreement this test exists to pin"
    )
    assert out["pairs"]["validity_counts"]["empty"] == 1


def test_context_drop_rate_uses_intersected_mask_semantics():
    """Both contexts lose a pair => both drop, even though most pairs are ok."""
    out = mod.attribute(_records(), n_production=100)
    assert out["contexts"]["n_contexts"] == 2
    assert out["contexts"]["n_dropped_contexts"] == 2
    assert out["contexts"]["drop_rate"] == 1.0
    assert out["projection"]["projected_dropped"] == 100


def test_arm_impact_flags_shrunk_and_destroyed_arms():
    """Excluding a persona changes k for every arm containing it."""
    out = mod.attribute(_records(), n_production=100)
    by_excl = {tuple(c["excluded_personas"]): c for c in out["roster_exclusion_counterfactuals"]}

    # Excluding nobody perturbs nothing.
    base = by_excl[()]
    assert base["arms_shrunk"] == []
    assert base["changes_manipulated_variable"] is False

    # Persona 0 is arm k=1's ONLY member => any exclusion set CONTAINING it
    # destroys that rung. (Exclusion sets are prefixes of a refusal ranking, so
    # persona 0 need not be first in any of them.)
    excl0 = next(c for k, c in by_excl.items() if 0 in k)
    assert 1 in excl0["arms_destroyed"]
    assert set(excl0["arms_shrunk"]) >= {1, 2}
    assert excl0["changes_manipulated_variable"] is True

    # Persona 1 serves only arm k=2, so excluding it alone shrinks k=2 and
    # leaves k=1 intact -- the asymmetry that makes the confound load onto the
    # high-k arms.
    excl1 = by_excl[(1,)]
    assert excl1["arms_shrunk"] == [2]
    # k=2 uses {0,1}: dropping persona 1 leaves persona 0, so the arm SHRINKS
    # (its realized k falls to 1) without being destroyed. That distinction is
    # the whole point -- a shrunk arm silently stops being the rung it is
    # labelled, which is worse than a loudly missing one.
    assert excl1["arms_destroyed"] == []
    assert 1 not in excl1["arms_shrunk"], "k=1 uses only persona 0, so it is untouched"


def test_both_counterfactual_orderings_are_emitted():
    """Quoting one ordering as 'the' counterfactual is how a wrong claim was made."""
    out = mod.attribute(_records(), n_production=100)
    orderings = {c["ordering"] for c in out["roster_exclusion_counterfactuals"]}
    assert orderings == {"by_refusal_count", "by_refusal_rate"}


def test_arm_set_is_derived_from_records_not_the_fallback_constant():
    """A ladder change must not leave the fallback constant silently stale."""
    recs = _records()
    for r in recs:
        r["arms"] = [3, 7]  # arms absent from ARM_K_VALUES_FALLBACK
    out = mod.attribute(recs, n_production=10)
    shrunk = {k for c in out["roster_exclusion_counterfactuals"] for k in c["arms_shrunk"]}
    assert shrunk <= {3, 7}, f"derived arms leaked the fallback constant: {shrunk}"
    assert shrunk, "expected some arm to be flagged shrunk under exclusions"


def test_load_records_fails_loud_on_empty_selection(tmp_path):
    """An empty record set would read downstream as a clean, zero-refusal pilot."""
    with pytest.raises(RuntimeError, match="no persona"):
        mod.load_records(tmp_path)


def test_load_records_fails_loud_on_a_staged_file_with_zero_records(tmp_path):
    (tmp_path / "persona00_seed42.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero records"):
        mod.load_records(tmp_path)


def test_cap_hit_is_reported_per_arm_persona_cell():
    """The plan cites the over-trigger CELL count, so cells must be in the artifact.

    A cited figure that lives only in an ad-hoc shell tally is not re-derivable.
    """
    out = mod.attribute(_records(), n_production=100)
    cap = out["cap_hit"]
    # One row in the fixture is stop_reason=max_tokens, on persona 0 (arms 1,2).
    assert cap["n_cap_hit_pairs"] == 1
    cells = {(c["arm_k"], c["persona_idx"]): c for c in cap["cells"]}
    # persona 0 appears in arms 1 and 2; its single cap hit lands in both cells.
    assert cells[(1, 0)]["n_cap_hit"] == 1
    assert cells[(2, 0)]["n_cap_hit"] == 1
    # persona 1 has no cap hits.
    assert cells[(2, 1)]["n_cap_hit"] == 0
    assert cells[(2, 1)]["over_trigger"] is False


def test_cap_hit_over_trigger_flag_uses_the_registered_2pct_threshold():
    out = mod.attribute(_records(), n_production=100)
    cap = out["cap_hit"]
    assert cap["trigger_fraction"] == 0.02
    for c in cap["cells"]:
        assert c["over_trigger"] == (c["cap_hit_fraction"] > 0.02), c


def test_cap_hit_keys_on_stop_reason_not_the_cap_hit_flag():
    """A row whose cap_hit flag disagrees with stop_reason must follow stop_reason.

    Same precedence rule as refusal counting -- the persisted stop_reason is the
    authority, so the two accountings cannot drift apart.
    """
    recs = _records()
    for r in recs:
        if r["stop_reason"] == "max_tokens":
            r["cap_hit"] = False  # flag lies; stop_reason is authoritative
    out = mod.attribute(recs, n_production=100)
    assert out["cap_hit"]["n_cap_hit_pairs"] == 1


def test_projection_is_consistent_with_the_interval():
    out = mod.attribute(_records(), n_production=1000)
    lo_n, hi_n = out["projection"]["projected_dropped_wilson95"]
    assert lo_n <= out["projection"]["projected_dropped"] <= hi_n
    assert math.isfinite(out["contexts"]["wilson95"]["lo"])
