"""Gate-3-pre per-family feasibility-aware floor pins (task #2329).

Three permanent pins from the gate-3-pre judge pre-gate relaxation (the
authorized Option-1 fix after the round-8 STOP-and-report):

1. ``_family_min_effective_floor``: a family whose arms hold >= 51 draws'
   capacity keeps the ``GATE3PRE_MIN_EFFECTIVE_DRAWS`` ceiling (51); an
   item-limited family gets its realized arm capacity (item count x n_draws).
2. The rule-26(a) truncation check is NEVER waived: ``_gate_verdict`` FAILs
   on truncation evidence even for a parse-fail-waived arm whose every other
   clause passes (the unwaivable half the relaxation must not touch).
3. Call-kwargs pin: ``phase_pilot_gate3pre`` passes
   ``allow_subresolution_pilot=True`` and a DERIVED per-family floor (never
   the bare ``GATE3PRE_MIN_EFFECTIVE_DRAWS`` constant) to ``judge_pilot_gate``.

Review-v10 follow-up pins (Minors 1-3 + the codex n_draws-clamp synthesis):

4. ``_family_resolution_fields``: ``parse_fail_resolution_pct`` divides by the
   ANSWERED draws (``n_draws - n_transport_lost - n_api_refusal`` — the
   denominator of ``judge_pilot``'s own ``parse_fail_rate``,
   judge_pilot.py:580-601), NOT the effective (verdict) count; the two differ
   exactly on api-refusal-bearing waves. Realized zero-refusal report values
   are reproduced verbatim from the committed gate artifact.
5. ``sub_resolution`` keys on the CONFIGURED relaxation
   (``floor_applied < floor_ceiling``), so a transport-hollowed FULL-strength
   family (which FAILs the gate) is never mislabelled as a deliberate
   relaxation.
6. ``_family_min_effective_floor`` fails loud on a multi-arm family (the
   scalar verdict floor would under-enforce larger arms) and on
   ``n_draws < 1`` (an unvalidated non-positive count would derive a 0 floor
   every arm trivially clears); the ``max(1, .)`` clamp is retained for
   judge_pilot d_eff parity and only sees validated input.

CPU-only, network-free, repo-root-path-safe (reads only committed sources).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2329_judge as J29  # noqa: E402

from explore_persona_space.eval.judge_pilot import (  # noqa: E402
    ArmPilotStats,
    _gate_verdict,
)


def _rows(n: int) -> list[tuple[str, str, str]]:
    return [(f"it{i}", f"q{i}", f"a{i}") for i in range(n)]


def test_family_floor_keeps_ceiling_for_large_families():
    """>= 51-item arms keep the 51 ceiling — byte-identical to the pre-fix floor."""
    assert J29.GATE3PRE_MIN_EFFECTIVE_DRAWS == 51
    assert J29._family_min_effective_floor({"anchor": _rows(4240)}, 1) == 51
    assert J29._family_min_effective_floor({"anchor": _rows(640)}, 1) == 51
    assert J29._family_min_effective_floor({"anchor": _rows(51)}, 1) == 51


def test_family_floor_is_realized_capacity_for_item_limited_family():
    """An item-limited family's floor is its realized arm capacity, DERIVED."""
    assert J29._family_min_effective_floor({"anchor": _rows(30)}, 1) == 30
    # n_draws scales capacity: 30 items x 2 draws = 60 >= 51 -> the ceiling binds again
    assert J29._family_min_effective_floor({"anchor": _rows(30)}, 2) == 51


def test_family_floor_multi_arm_assert_fires():
    """v10 Minor 3: multi-arm is REFUSED — the scalar floor would under-enforce.

    ``_gate_verdict`` compares EVERY arm against the one scalar
    ``min_effective_draws_per_arm``, so a min-over-arms floor (here 30) would
    let the 100-item arm pass at 30 < 51 effective draws. Fail loud instead of
    silently widening; per-arm floors need a judge_pilot library change.
    """
    with pytest.raises(AssertionError, match="SINGLE-ARM"):
        J29._family_min_effective_floor({"a": _rows(100), "b": _rows(30)}, 1)


def test_family_floor_rejects_non_positive_n_draws():
    """Codex-nit synthesis: n_draws < 1 raises BEFORE the parity clamp.

    An unvalidated non-positive count would derive a 0 floor that every arm
    trivially clears — a silent PASS on zero evidence. The ``max(1, .)`` clamp
    is retained for judge_pilot d_eff parity (judge_pilot.py:327) but only
    ever operates on validated input.
    """
    with pytest.raises(ValueError, match="n_draws=0"):
        J29._family_min_effective_floor({"anchor": _rows(30)}, 0)
    with pytest.raises(ValueError, match="n_draws=-1"):
        J29._family_min_effective_floor({"anchor": _rows(30)}, -1)


def _stats(**kw) -> ArmPilotStats:
    base = dict(
        n_items=30,
        n_items_zero_valid=0,
        frac_items_complete=1.0,
        n_draws=30,
        n_scored=30,
        n_content_dropped=0,
        n_refusal=0,
        n_truncation=0,
        n_transport_lost=0,
        n_api_refusal=0,
        n_unknown_stop_reason_drops=0,
        parse_fail_rate=0.0,
        stop_reason_tally={"end_turn": 30},
        waived=False,
    )
    base.update(kw)
    return ArmPilotStats(**base)


def test_truncation_is_never_waived_even_on_waived_arm():
    """Rule 26(a): truncation FAILs unconditionally — waiver + met floor don't help."""
    stats = {
        "anchor": _stats(
            n_truncation=1,
            waived=True,
            stop_reason_tally={"end_turn": 29, "max_tokens": 1},
        )
    }
    failures, _warnings = _gate_verdict(
        stats, max_tokens=2048, parse_fail_threshold=0.02, min_effective_draws_per_arm=30
    )
    assert any("truncation" in f for f in failures), failures
    # Control: identical stats minus the truncation evidence pass every clause.
    clean = {"anchor": _stats()}
    failures_clean, _w = _gate_verdict(
        clean, max_tokens=2048, parse_fail_threshold=0.02, min_effective_draws_per_arm=30
    )
    assert failures_clean == []


def test_gate3pre_call_passes_subresolution_flag_and_derived_floor():
    """AST pin on the phase's judge_pilot_gate call kwargs."""
    src = (SCRIPTS / "issue2329_judge.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "phase_pilot_gate3pre"
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "judge_pilot_gate"
    ]
    assert len(calls) == 1, "expected exactly one judge_pilot_gate call in the phase"
    kw = {k.arg: k.value for k in calls[0].keywords}
    flag = kw.get("allow_subresolution_pilot")
    assert isinstance(flag, ast.Constant) and flag.value is True, (
        "allow_subresolution_pilot=True must be passed literally"
    )
    floor = kw.get("min_effective_draws_per_arm")
    assert floor is not None, "min_effective_draws_per_arm must be passed"
    assert not (isinstance(floor, ast.Name) and floor.id == "GATE3PRE_MIN_EFFECTIVE_DRAWS"), (
        "the floor must be DERIVED per family (via _family_min_effective_floor), "
        "never the bare ceiling constant"
    )


def test_resolution_pct_denominator_is_answered_not_effective():
    """v10 Minor 1 / codex BLOCKER: divide by ANSWERED draws, not effective.

    With api-refusal draws (rule 28: transport-conditional censoring, outside
    the answered pool) the effective (verdict) count and the answered count
    differ, and the old effective-denominator formula OVERSTATES the
    parse-fail check's fineness. effective = 110 - 5 = 105; answered =
    110 - 5 - 33 = 72.
    """
    arms = {"anchor": _stats(n_items=110, n_draws=110, n_transport_lost=5, n_api_refusal=33)}
    fields = J29._family_resolution_fields(arms, 51)
    assert fields["effective_draws_min"] == 105  # the verdict quantity, unchanged
    assert fields["answered_draws_min"] == 72
    assert fields["parse_fail_resolution_pct"] == round(100.0 / 72, 2)  # 1.39
    # The pre-fix formula (effective denominator) reads finer — the defect:
    assert fields["parse_fail_resolution_pct"] != round(100.0 / 105, 2)  # 0.95
    # Fully-censored family: no answered draws -> None, never a coerced number.
    censored = {"anchor": _stats(n_items=30, n_draws=30, n_api_refusal=30)}
    assert J29._family_resolution_fields(censored, 30)["parse_fail_resolution_pct"] is None


def test_sub_resolution_keys_on_floor_relaxation_not_realized_draws():
    """v10 Minor 2: sub_resolution flags the CONFIGURED relaxation only.

    A transport-hollowed FULL-strength family (floor 51, effective 49) FAILs
    the gate but was never a deliberate relaxation — it must NOT read
    sub-resolution (the pre-fix realized-draws keying mislabelled it). A
    genuinely floor-lowered family (30 < 51) IS flagged even when healthy.
    """
    hollowed = {"anchor": _stats(n_items=110, n_draws=110, n_transport_lost=61)}
    fields = J29._family_resolution_fields(hollowed, 51)
    assert fields["effective_draws_min"] == 49
    assert fields["sub_resolution"] is False
    # ... and the gate itself still FAILs that family (no gating impact):
    failures, _w = _gate_verdict(
        hollowed, max_tokens=2048, parse_fail_threshold=0.02, min_effective_draws_per_arm=51
    )
    assert any("transport-hollowed" in f for f in failures), failures
    # Genuinely floor-lowered family: flagged sub-resolution regardless of health.
    lowered = J29._family_resolution_fields({"anchor": _stats()}, 30)
    assert lowered["sub_resolution"] is True
    assert lowered["floor_applied"] == 30
    assert lowered["floor_ceiling"] == 51


def test_realized_report_fields_reproduced_from_committed_artifact():
    """The realized (zero-api-refusal) report values are UNCHANGED by the fix.

    Rebuilds each family's arm stats from the committed gate artifact's own
    cells rows and asserts ``_family_resolution_fields`` reproduces the
    per-family fields verbatim — incl. query-rubric's 3.33 (= 100/30) and the
    healthy families' 0.91 (= 100/110).
    """
    rec = json.loads(
        (
            REPO_ROOT
            / "eval_results"
            / "issue_2329"
            / "judge"
            / "gates"
            / "pilot_gate3pre_report.json"
        ).read_text(encoding="utf-8")
    )
    cells_by_family: dict[str, list[dict]] = {}
    for cell in rec["cells"]:
        cells_by_family.setdefault(cell["family"], []).append(cell)
    assert set(cells_by_family) == {"coherence", "value-rubric", "query-rubric"}
    for family, old in rec["per_family"].items():
        arms = {}
        for c in cells_by_family[family]:
            # Realized run had zero api-refusals: every cell's tally is 100%
            # end_turn (asserted, so the 0 below is grounded, not defaulted).
            assert c["stop_reason_tally"] == {"end_turn": c["n_draws"]}, (family, c["arm"])
            arms[c["arm"]] = _stats(
                n_items=c["n_draws"],  # n_draws=1 => items == draws on this run
                n_draws=c["n_draws"],
                n_transport_lost=c["n_transport_lost"],
                n_api_refusal=c.get("n_api_refusal", 0),  # key absent in the r9 artifact
            )
        fields = J29._family_resolution_fields(arms, old["floor_applied"])
        for key in (
            "floor_applied",
            "floor_ceiling",
            "effective_draws_min",
            "sub_resolution",
            "parse_fail_resolution_pct",
        ):
            assert fields[key] == old[key], (family, key, fields[key], old[key])
    qr = rec["per_family"]["query-rubric"]
    assert (qr["sub_resolution"], qr["parse_fail_resolution_pct"]) == (True, 3.33)
    assert rec["per_family"]["coherence"]["parse_fail_resolution_pct"] == 0.91
