"""Tests for ``pre_split_review_gate`` + ``scripts/pre_split_review_guard.py`` (#2158).

Incident-replay fixtures are FAITHFUL to the live artifacts and assert
ORDERING (relative event indices) — NEVER a file row total: events.jsonl is
append-only and grows, so any hard-coded row count rots on the next marker.

- #1336 r4 (``tasks/awaiting_promotion/1336/events.jsonl``): the v131/v132
  arm-B shape (premature Unit-A review dispatch, 2026-08-04 — the latest
  implementation-class marker, ``epm:results`` v7, PREDATES the v131 unit=A
  implementing dispatch; a breadcrumb-only guard reads REVIEW-OK there), the
  v147 arm-A breadcrumb, its v155 "NEARLY COMPLETE" variant, and the
  implementation-marker clear point (v14).
- #2061 (``tasks/awaiting_promotion/2061/events.jsonl``): the LETTERED
  ``A/5`` / ``B/5`` breadcrumbs with ``Remaining units:`` / ``Remaining:``
  spellings (rows 52 / 65) — the fail-open shape a digits-only parser reads
  REVIEW-OK on. Kill criterion (plan v2 §8): these replays must return a
  NONZERO verdict (exit 2 or 3, never 0).

The CLI exit-code mapping (0/2/3) is tested by importing the script module
with ``list_events`` monkeypatched — including the plan's named Fail-loud pin
``test_unparseable_breadcrumb_exits_3``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from explore_persona_space.task_workflow import pre_split_review_gate

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import pre_split_review_guard  # noqa: E402


def _ev(kind: str, note: str = "", version: int = 1) -> dict:
    """One events.jsonl row in the live shape ``{by, kind, note, ts, version}``."""
    return {
        "by": "tester",
        "kind": kind,
        "note": note,
        "ts": "2026-08-17T00:00:00Z",
        "version": version,
    }


# --- Faithful #1336 fixtures (shapes measured off the live rows 2026-08-17) --

V131_NOTE = (
    "stage-dispatch stage=followup-implementing round=4 "
    "subagent=experiment-implementer-lean unit=A "
    "worktree=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/"
    "issue-1336-fullcorpora label=pooled-multidataset-onoff-policy-stage-transfer"
)

V132_NOTE = (
    "stage-dispatch stage=followup-reviewing round=4 subagent=code-reviewer unit=A "
    "worktree=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/"
    "issue-1336-fullcorpora label=pooled-multidataset-onoff-policy-stage-transfer"
)

# Grammar line mid-note (internal line 4, column 0) — the live v147 shape,
# including the NON-note-anchored "stage-dispatch:" telemetry line arm B must
# NOT key on (colon form, not at note start).
V147_NOTE = "\n".join(
    [
        "Round-4 pre-split RESUME BREADCRUMB + a contract finding on why round 4 stalled.",
        "",
        "stage-dispatch: worktree=.claude/worktrees/issue-1336-fullcorpora "
        "label=pooled-multidataset-onoff-policy-stage-transfer unit=B round=4",
        "",
        "pre-split unit 1/3 complete: f02bb56eb95a675b5a436b1538a68b5b7f253ea5; "
        "remaining: Unit B (extract+fit), Unit C (ladder+figures+full per-phase smoke+marker)",
        "",
        "=== WHY ROUND 4 STALLED ===",
    ]
)

V155_NOTE = "\n".join(
    [
        "Round-4 Unit B-i COMPLETE. Unit B-iii (last Unit B unit) dispatched.",
        "",
        "stage-dispatch: worktree=.claude/worktrees/issue-1336-fullcorpora "
        "label=pooled-multidataset-onoff-policy-stage-transfer unit=B-iii round=4",
        "",
        "pre-split unit 2/3 NEARLY COMPLETE: 9e648053b1, beed23dbae, 4cbf8d8df2, "
        "546232fb2e4525db1911256506c196789469ea2d; remaining: Unit B-iii "
        "(dispatcher phases), Unit C (ladder+figures)",
        "",
        "=== UNIT B-i — LANDED AND VERIFIED ===",
    ]
)


# --- Faithful #2061 fixtures (lettered units; measured off rows 52/65) -------

# Row 52 (epm:progress v33): candidate at note line 0, LETTERED unit A/5,
# remaining spelled "Remaining units:" (no leading semicolon).
ROW52_NOTE = "\n".join(
    [
        "pre-split unit A/5 complete: commit aaf9c75d5c (+651/-141, 5 files) — C1 "
        "`turnstore-schema-loaders-fabricated` closed. Remaining units: B (C2 kNN + "
        "C3 null + M4 dof-cap/parity + M5 group folds), C (M1 storage + C5 Hub retry "
        "+ M2 parse + M3 resume keys), D (P5 figures + fitness minors), E (C4 staging).",
        "",
        "Details of unit A follow (fixture filler standing in for the live 74-line note).",
    ]
)

# Row 65 (epm:progress v34): SINGLE-line note; unit-scoped stage-dispatch text
# EMBEDDED mid-note (offset ~617 in the live row — a QUOTE, not a dispatch).
ROW65_NOTE = (
    "pre-split unit B/5 complete (durable record): commits d2b36fc0ce (C2 k-keyed kNN "
    "+ plan-k, dof-cap via opt-in fit_h kwarg + parity gate, group folds, headers) and "
    "d095ea09cf (C3 registered permute-and-refit null engine + stats test). "
    "Remaining: C (M1 storage + C5 Hub retry + M2 parse + M3 resume keys), "
    "D (P5 figures), E (C4 staging + final marker). | stage-dispatch "
    "stage=implementing round=3 subagent=experiment-implementer "
    "worktree=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/"
    "issue-2061 unit=C | dispatched"
)


# --- Predicate tests ----------------------------------------------------------


def test_empty_events_review_ok():
    result = pre_split_review_gate([])
    assert result["verdict"] == "REVIEW-OK"
    assert result["breadcrumb_index"] is None
    assert result["unit_dispatch_index"] is None


def test_1336_v147_arm_a_fires():
    events = [
        _ev("epm:progress", "routine progress note", 100),
        _ev("epm:progress", V147_NOTE, 147),
    ]
    result = pre_split_review_gate(events)
    assert result["verdict"] == "PRE-SPLIT-INCOMPLETE"
    assert "Unit B" in result["remaining"]
    assert result["breadcrumb_index"] == 1
    # the mid-note "stage-dispatch:" telemetry line is NOT note-anchored — arm B silent
    assert result["unit_dispatch_index"] is None


def test_1336_v155_nearly_complete_variant_fires():
    # "NEARLY COMPLETE" matches: the arm keys on prefix + same-line remaining,
    # not the word "complete".
    events = [_ev("epm:progress", V155_NOTE, 155)]
    result = pre_split_review_gate(events)
    assert result["verdict"] == "PRE-SPLIT-INCOMPLETE"
    assert "Unit B-iii" in result["remaining"]


def test_arm_a_cleared_by_later_experiment_implementation():
    events = [
        _ev("epm:progress", V147_NOTE, 147),
        _ev("epm:experiment-implementation", "round-4 split complete", 14),
    ]
    result = pre_split_review_gate(events)
    assert result["verdict"] == "REVIEW-OK"
    assert result["impl_index"] == 1


def test_arm_a_cleared_by_later_results():
    events = [_ev("epm:progress", V147_NOTE, 147), _ev("epm:results", "infra round done", 8)]
    result = pre_split_review_gate(events)
    assert result["verdict"] == "REVIEW-OK"
    assert result["impl_index"] == 1


@pytest.mark.parametrize("rem", ["none", "", "-", "(none)", "NONE"])
def test_empty_remaining_review_ok(rem: str):
    # An EMPTY parsed remaining field is a COMPLETED split — parseable, never
    # unparseable, and arm A does not fire.
    note = f"pre-split unit 3/3 complete: abc123def456; remaining: {rem}"
    result = pre_split_review_gate([_ev("epm:progress", note)])
    assert result["verdict"] == "REVIEW-OK"
    assert result["breadcrumb_index"] == 0


def test_1336_v131_arm_b_incident_replay():
    # THE #1336 incident-replay test (plan kill criterion: the plan is
    # falsified if this cannot pass). Ordering per the measured trace: the
    # latest implementation-class marker (epm:results v7) PREDATES the v131
    # unit=A implementing dispatch; NO remaining:-bearing breadcrumb exists
    # yet (the first is v147, two days later) — a breadcrumb-only guard reads
    # REVIEW-OK at exactly the dispatch that burned subagent deaths 7-8.
    events = [
        _ev("epm:experiment-implementation", "round-3 implementation", 12),
        _ev("epm:results", "round-3 results", 7),
        _ev("epm:progress", V131_NOTE, 131),
        _ev("epm:progress", V132_NOTE, 132),
    ]
    result = pre_split_review_gate(events)
    assert result["verdict"] == "PRE-SPLIT-INCOMPLETE"
    assert result["unit_dispatch_index"] == 2  # v131; v132 is stage=followup-reviewing
    assert result["impl_index"] == 1
    assert result["breadcrumb_index"] is None


def test_non_unit_scoped_implementing_dispatch_review_ok():
    # Arm B requires the unit= token: an ordinary (non-split) implementing
    # dispatch never trips the gate.
    note = (
        "stage-dispatch stage=implementing round=2 subagent=experiment-implementer "
        "worktree=/x label=ordinary-single-deliverable-round"
    )
    result = pre_split_review_gate([_ev("epm:progress", note)])
    assert result["verdict"] == "REVIEW-OK"
    assert result["unit_dispatch_index"] is None


def test_unparseable_candidate_predicate_verdict():
    # A recognized candidate with NO parseable same-line remaining field NEVER
    # falls through to REVIEW-OK (the #2061 fail-open shape) — fail loud.
    note = "pre-split unit 2/3 complete: 9e648053b1, beed23dbae"
    result = pre_split_review_gate([_ev("epm:progress", note)])
    assert result["verdict"] == "BREADCRUMB-UNPARSEABLE"
    assert "pre-split unit 2/3" in result["reason"]


def test_template_quote_not_a_candidate():
    # The literal 08-step-4.md grammar TEMPLATE quoted at column 0 of an
    # internal line: "M" is not a digit, so the structural exclusion (M
    # digits-only) keeps it a non-candidate — REVIEW-OK, not exit 3.
    note = "\n".join(
        [
            "Quoting the #1810 grammar for reference:",
            "pre-split unit k/M complete: <commit SHAs>; remaining: <deliverables>",
        ]
    )
    result = pre_split_review_gate([_ev("epm:progress", note)])
    assert result["verdict"] == "REVIEW-OK"
    assert result["breadcrumb_index"] is None


def test_2061_row52_lettered_prefix_replay():
    # THE #2061 fail-open replay (plan kill criterion: the plan is falsified
    # if this reads REVIEW-OK). Lettered unit A/5 + "Remaining units:"
    # spelling, no later implementation marker: NONZERO verdict, never 0.
    events = [
        _ev("epm:experiment-implementation", "unit-A prelude round", 1),
        _ev("epm:progress", ROW52_NOTE, 33),
    ]
    result = pre_split_review_gate(events)
    assert result["verdict"] != "REVIEW-OK"
    assert result["verdict"] in {"PRE-SPLIT-INCOMPLETE", "BREADCRUMB-UNPARSEABLE"}


def test_2061_row65_embedded_dispatch_replay():
    # Row-65 shape: single-line lettered candidate with the unit-C dispatch
    # text EMBEDDED mid-note. NONZERO verdict, AND arm B must NOT key on the
    # embedded quote (note-anchored by design).
    events = [
        _ev("epm:experiment-implementation", "unit-A prelude round", 1),
        _ev("epm:progress", ROW65_NOTE, 34),
    ]
    result = pre_split_review_gate(events)
    assert result["verdict"] != "REVIEW-OK"
    assert result["verdict"] in {"PRE-SPLIT-INCOMPLETE", "BREADCRUMB-UNPARSEABLE"}
    udi = result["unit_dispatch_index"]
    assert udi is None or udi < result["breadcrumb_index"]


# --- CLI exit-code mapping (list_events monkeypatched) -------------------------


def _run_cli(monkeypatch, capsys, events: list[dict]) -> tuple[int, str]:
    monkeypatch.setattr(pre_split_review_guard, "list_events", lambda task_id: events)
    rc = pre_split_review_guard.main(["9999"])
    return rc, capsys.readouterr().out


def test_unparseable_breadcrumb_exits_3(monkeypatch, capsys):
    # Fail-loud pin (plan §3 `Fail-loud pin:` line): exit code 3 + the
    # BREADCRUMB-UNPARSEABLE lead token on an unparseable-breadcrumb fixture.
    events = [_ev("epm:progress", "pre-split unit 2/3 complete: 9e648053b1, beed23dbae")]
    rc, out = _run_cli(monkeypatch, capsys, events)
    assert rc == 3
    assert out.startswith("BREADCRUMB-UNPARSEABLE")


def test_cli_exits_2_on_firing_fixture(monkeypatch, capsys):
    rc, out = _run_cli(monkeypatch, capsys, [_ev("epm:progress", V147_NOTE, 147)])
    assert rc == 2
    assert out.startswith("PRE-SPLIT-INCOMPLETE")
    assert "remaining:" in out


def test_cli_exits_0_on_cleared_fixture(monkeypatch, capsys):
    events = [
        _ev("epm:progress", V147_NOTE, 147),
        _ev("epm:experiment-implementation", "round-4 split complete", 14),
    ]
    rc, out = _run_cli(monkeypatch, capsys, events)
    assert rc == 0
    assert out.startswith("REVIEW-OK")
