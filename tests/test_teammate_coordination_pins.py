"""Pin the CLAUDE.md teammate-coordination sub-clauses (#1583/#1586/#1598).

CLAUDE.md's "Teammate coordination" bullet (section "Orchestrator vs
subagent re-invocation") was created by #1583 and amended by #1586
(channel scope) and #1598 (report-delivery contract) within 8 days --
the same amendment-velocity profile that motivated
tests/test_adhoc_summary_disclosure_pins.py (#1458). These pins keep a
future rewrite from silently dropping a sub-clause or its incident
citations. Assertions run on whitespace-NORMALIZED file text so prose
re-wrapping never breaks a pin; citation pins use distinctive
multi-token forms because bare issue ids pre-exist elsewhere in
CLAUDE.md (a bare-id pin would be vacuous).
"""

from __future__ import annotations

import re
from pathlib import Path

CLAUDE_MD = Path(__file__).resolve().parent.parent / "CLAUDE.md"


def _normalized() -> str:
    return re.sub(r"\s+", " ", CLAUDE_MD.read_text(encoding="utf-8"))


def test_subclause_headers_pinned() -> None:
    text = _normalized()
    assert "(a) **ONE implementer per file set.**" in text
    assert "(b) **An idle notification is NOT a done/stall verdict.**" in text
    assert (
        "(c) **Deliver stand-downs and teammate-directed reports on the teammate channel**" in text
    )
    assert "(d) **The report SendMessage is the teammate's FINAL action of its turn.**" in text


def test_delivery_contract_clauses_pinned() -> None:
    text = _normalized()
    # Teammate side (#1598): the report SendMessage ends the turn.
    assert "work finished with the report undelivered is an INCOMPLETE turn" in text
    # Orchestrator side (#1598): one nudge, then the durable fallback.
    assert "nudge ONCE" in text
    assert "the Agent result is the durable fallback report channel" in text


def test_channel_scope_note_pinned() -> None:
    # The #1586 trailer must survive alongside (d).
    text = _normalized()
    assert "SendMessage reaches only SUBAGENTS spawned via the Agent tool" in text
    assert "NOT SendMessage-addressable" in text


def test_incident_citations_pinned() -> None:
    text = _normalized()
    assert "#1112: a second implementer spawned" in text
    assert "#958: an idle teammate's work was finished" in text
    assert "(#1586, 2026-07-21)" in text
    assert "idled with reports unsent; #1598)" in text


def test_standdown_release_clauses_pinned() -> None:
    """Pin sub-clause (e) (#2034): stand-down effect confirmation, the
    ownership-RELEASE record for session-to-session handoffs, and the
    orchestrator-side pre-spawn ownership probe -- three same-day incidents
    (2026-08-02, sessions 472284ce / a0400dd4 / 75f66748)."""
    text = _normalized()
    assert "(e) **A stand-down is not effective until CONFIRMED" in text
    assert "never write an aborted/stood-down claim into a durable marker unconfirmed" in text
    assert "explicit ownership-RELEASE record" in text
    assert "the ownership probe runs in the ORCHESTRATOR before spawning a runner" in text
    # Citation pins (distinctive multi-token forms, per the module docstring).
    assert "the agent had already completed and committed, session 472284ce" in text
    assert "a0400dd4: near double-dispatch" in text
    assert "75f66748: a runner spawned over a live owner" in text
