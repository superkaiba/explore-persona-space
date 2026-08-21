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

#2041 added sub-clauses (f)/(g) (fan-out same-turn durable landing +
synchronous delegated gate-waits) and the SKILL.md Step 4b "Fan-out
completion contract" paragraph; both are pinned here.

#2111 added two probed sub-cases to the sibling ownership-check bullet
(pre-stop probe + post-death detached-children probe); pinned here too
(test_ownership_probe_subcases_pinned) -- same section, same
amendment-velocity profile.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

CLAUDE_MD = Path(__file__).resolve().parent.parent / "CLAUDE.md"
ISSUE_SKILL_MD = (
    Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"
)
STEP4_MD = (
    Path(__file__).resolve().parent.parent
    / ".claude"
    / "skills"
    / "issue"
    / "steps"
    / "08-step-4.md"
)


def _normalized() -> str:
    return re.sub(r"\s+", " ", CLAUDE_MD.read_text(encoding="utf-8"))


def _normalized_skill() -> str:
    return re.sub(r"\s+", " ", issue_skill_text())


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


def test_claude_md_fanout_subclauses_pinned() -> None:
    """Pin sub-clauses (f)/(g) (#2041): fan-out work products land durably in
    the producing turn (+ the join-time consolidation default), and a brief
    delegating a gate-wait mandates synchronous in-turn waiting via the
    sanctioned bounded Monitor until-loop."""
    text = _normalized()
    assert (
        "(f) **Fan-out work products land durably IN the producing turn — "
        "spawn briefs RESTATE this (#2041).**" in text
    )
    assert "a turn ending with staged-but-uncommitted work or a /tmp-only deliverable" in text
    assert "At every fan-out JOIN the ORCHESTRATOR consolidates the returned reports" in text
    assert "(g) **A brief delegating a gate-wait mandates SYNCHRONOUS waiting (#2041).**" in text
    # The sanctioned-wait naming (Monitor + until-loop) is itself pinned.
    assert "a bounded `Monitor` until-loop (foreground `sleep` chains are hook-blocked)" in text
    # Citation pins (distinctive multi-token forms, per the module docstring).
    assert "#2041: three fold subagents idled mid-delivery with staged-uncommitted work" in text
    assert "four scout reports lived only in /tmp for ~11h" in text
    assert (
        "#2041: a delegated Step 10d subagent ended its turn blocked on a background gate" in text
    )


def test_ownership_probe_subcases_pinned() -> None:
    """Pin the #2111 ownership-check sub-cases: the pre-stop probe (live
    subagent children + a fresh stage-dispatch breadcrumb => presume live)
    and the post-death detached-children probe (probe for detached children
    a dead subagent launched BEFORE dispatching same-path work) -- two
    2026-08-05 incidents (#2054 session stop; #1491 duplicate run)."""
    text = _normalized()
    assert "**Pre-stop probe (#2111):**" in text
    assert "probe for live subagent children AND a `stage-dispatch` breadcrumb" in text
    assert "younger than ~10 min" in text
    assert "never stop on an idle-looking self-report" in text
    assert "**Post-death detached-children probe (#2111):**" in text
    assert "a death notification never enumerates detached children" in text
    # Citation pins (distinctive multi-token forms, per the module docstring).
    assert "killing a 10-min-old live implementer" in text
    assert "#1491: a 529-killed analyzer's detached run" in text


def test_fanout_completion_contract_pinned() -> None:
    """Pin the SKILL.md Step 4b "Fan-out completion contract" paragraph
    (#2041): the header, its three numbered clauses, and the join-time
    consolidation sentence."""
    text = _normalized_skill()
    assert "**Fan-out completion contract in every work-producing brief (#2041).**" in text
    # Clause (1): same-turn durable landing + the lint-gate certification duty.
    assert "(1) deliverables land durably IN the producing turn" in text
    assert "carries the Step 9a-ter § Worker-brief composition duty" in text
    # Clause (2): the report ends the turn.
    assert "(2) the report is the turn's FINAL action" in text
    # Clause (3): synchronous delegated gate-waits via the sanctioned wait shape.
    assert "(3) a delegated gate-wait is waited out SYNCHRONOUSLY inside the turn" in text
    assert "a bounded `Monitor` until-loop — foreground `sleep` chains are hook-blocked" in text
    assert "never end the turn on a background call the subagent itself armed" in text
    # Join-time consolidation default.
    assert "At every fan-out JOIN the orchestrator consolidates the returned reports" in text
    assert "offer-to-save is the banned shape" in text


def test_implementer_brief_names_wait_mechanism() -> None:
    """Pin the #2422 folded-in sibling (plan Edit 7): the implementer-brief
    checklist in 08-step-4.md restates BOTH halves of the #2041 fan-out
    completion contract — the staged-but-uncommitted prohibition AND the
    synchronous bounded `Monitor` wait mechanism — inside the
    `Brief passed to the implementer:` region. A brief stating the
    prohibition without the mechanism invites an invented
    background-watcher shape that orphans the landing (#2422 folded
    sibling: ~62 min of work left uncommitted behind an unobserved lint
    gate)."""
    text = re.sub(r"\s+", " ", STEP4_MD.read_text(encoding="utf-8"))
    start = text.find("Brief passed to the implementer:")
    assert start >= 0, "start anchor missing from 08-step-4.md"
    end = text.find("Move status to ", start)
    region = text[start:end] if end >= 0 else text[start:]
    assert "Monitor" in region, "the wait mechanism (`Monitor`) is not named in the brief checklist"
    assert "staged-but-uncommitted" in region, (
        "the staged-but-uncommitted prohibition is not named in the brief checklist"
    )
