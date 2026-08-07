"""Pin the #1855 round-boundary durable-decision duty in `/issue` SKILL.md.

Incident #1776 (session 97867df6, 2026-07-29): a Step 5 code-review round
resolved its ensemble decision (CONCERNS with one prescribed 2-line fix), the
orchestrator announced the direct apply in chat text only, then died at the
context ceiling two turns later (`Prompt is too long`) — the decision existed
nowhere durable, and the successor re-derived + re-did the fix ~40 min later.
#1855 adds the canonical Step 5c-quater duty: the moment a round's ensemble
decision is RESOLVED, land it durably — one `epm:progress` decision note + an
explicit-path commit of the round's uncommitted worktree edits — BEFORE
dispatching the next round's subagents or beginning any orchestrator-applied
inline fix. Unconditional (a session cannot introspect its own headroom).

These tests pin, against `.claude/skills/issue/SKILL.md`:

1. the `5c-quater` heading exists;
2. placement — the canonical block sits AFTER the 5c-ter heading and BEFORE
   the `**5d. Loop on FAIL` heading;
3. the literal `5c-quater` is referenced >= 3 times (the canonical block plus
   the Step 9a and Step 9a-bis REVISE-paragraph cross-references);
4. the load-bearing phrases live INSIDE the block span (not merely anywhere
   in the file — `cannot introspect` and `Prompt is too long` already occur
   elsewhere in SKILL.md, so file-global greps would pass on a gutted block):
   the no-headroom-predicate rationale, the incident's error string, the
   `epm:progress` decision-note leg, and the explicit-path-commit leg.

Prose assertions run on whitespace-NORMALIZED text (the file wraps prose
mid-phrase, so a required phrase can span lines).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

QUATER_HEADING = "**5c-quater. Round-boundary durable-decision duty (#1855).**"
TER_HEADING = "**5c-ter. Binding-concerns post-strip check"
FIVE_D_HEADING = "**5d. Loop on FAIL"


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _skill_text() -> str:
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    return issue_skill_text()


def test_quater_heading_present() -> None:
    """The canonical 5c-quater duty heading exists in SKILL.md."""
    assert QUATER_HEADING in _skill_text(), (
        "SKILL.md lacks the '**5c-quater. Round-boundary durable-decision duty (#1855).**' heading"
    )


def test_quater_placement_between_5c_ter_and_5d() -> None:
    """The 5c-quater block sits AFTER 5c-ter and BEFORE the 5d loop heading."""
    text = _skill_text()
    ter = text.find(TER_HEADING)
    quater = text.find(QUATER_HEADING)
    five_d = text.find(FIVE_D_HEADING)
    assert ter != -1, f"anchor {TER_HEADING!r} not found in SKILL.md"
    assert quater != -1, f"anchor {QUATER_HEADING!r} not found in SKILL.md"
    assert five_d != -1, f"anchor {FIVE_D_HEADING!r} not found in SKILL.md"
    assert ter < quater < five_d, (
        "5c-quater block is misplaced: expected 5c-ter heading < 5c-quater heading"
        f" < 5d heading, got indices {ter} / {quater} / {five_d}"
    )


def test_quater_referenced_at_least_three_times() -> None:
    """`5c-quater` appears >= 3 times: canonical block + 9a + 9a-bis cross-refs."""
    count = _skill_text().count("5c-quater")
    assert count >= 3, (
        f"expected >= 3 occurrences of '5c-quater' (canonical block + the Step 9a"
        f" and Step 9a-bis REVISE-paragraph cross-references), found {count}"
    )


def test_quater_block_carries_load_bearing_phrases() -> None:
    """The load-bearing phrases live INSIDE the 5c-quater block span.

    `cannot introspect` and `Prompt is too long` already exist elsewhere in
    SKILL.md (the residual-conflict dispatch + the resume section), so these
    asserts slice the block span first — a file-global grep would pass even
    if the block were gutted.
    """
    text = _skill_text()
    start = text.find(QUATER_HEADING)
    assert start != -1, f"anchor {QUATER_HEADING!r} not found in SKILL.md"
    end = text.find(FIVE_D_HEADING, start)
    assert end != -1, f"anchor {FIVE_D_HEADING!r} not found after the 5c-quater heading"
    block = _norm(text[start:end])
    # Why-unconditional rationale: headroom is non-introspectable (#1338 lesson).
    assert "cannot introspect" in block, (
        "5c-quater block lacks the no-headroom-predicate rationale ('cannot introspect')"
    )
    # The incident's terminal error string (no in-session recovery after one).
    assert "Prompt is too long" in block, (
        "5c-quater block lacks the 'Prompt is too long' no-recovery rationale"
    )
    # Duty leg 1: the one-line decision note reuses epm:progress.
    assert "epm:progress" in block, "5c-quater block lacks the `epm:progress` decision-note leg"
    # Duty leg 2: explicit-path (pathspec-limited) commit, never git add -A.
    assert "never `git add -A`" in block, (
        "5c-quater block lacks the explicit-path-commit leg (never `git add -A`)"
    )
    assert "pathspec-limited" in block, (
        "5c-quater block lacks the pathspec-limited commit-form citation"
        " (`git commit -m <msg> -- <paths>`)"
    )
