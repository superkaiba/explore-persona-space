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

CPU-only, network-free, repo-root-path-safe (reads only committed sources).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

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
    # multi-arm: the SMALLEST arm binds (the verdict floor is one scalar per gate call)
    assert J29._family_min_effective_floor({"a": _rows(100), "b": _rows(30)}, 1) == 30
    # n_draws scales capacity: 30 items x 2 draws = 60 >= 51 -> the ceiling binds again
    assert J29._family_min_effective_floor({"anchor": _rows(30)}, 2) == 51


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
