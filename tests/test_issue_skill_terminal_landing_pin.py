"""Pin the #1868 Step 10d terminal landing confirmation in `/issue` SKILL.md.

Incident #1792 (2026-07-29): `scripts/sync_repo_root.py` printed its
in-flight advisory at 13:20:47Z (exit 0 with ``state=in-flight`` means
"your push has NOT landed; re-run after the in-flight sync completes" —
sync_repo_root.py L33-35), and the Step 10d terminal teardown proceeded
12 s later with no re-run — `completed` + `epm:done` existed only locally,
a crash-window, and the terminal commits reached origin only via
concurrent sessions' pushes. #1868 adds step 4 to the Terminal-teardown
sequence: a bounded 2-attempt `sync_repo_root.py` re-run whose LANDED
arbiter is a fetched-origin blob check (the task's canonical
``events.jsonl`` on ``origin/main`` carries ``"epm:done"``), never the
helper's exit code, with a loud echo + one `epm:progress`
``terminal-landing-unconfirmed`` note on the still-unconfirmed arm.

These tests pin, against `.claude/skills/issue/SKILL.md` (span-scoped to
the Terminal-teardown section — from the ``#### Terminal teardown``
heading to ``## Resume semantics`` — several tokens exist elsewhere in
the file, so file-global greps would pass on a gutted section):

1. the step-4 heading (``Terminal landing confirmation`` + ``#1868``)
   exists inside the span;
2. the ``state=in-flight`` caller-retry rationale lives in the span;
3. the origin-blob arbiter tokens (``cat-file -e`` + ``"epm:done"``)
   live in the span;
4. the step-4 block actually re-runs ``sync_repo_root.py``;
5. the 2-attempt bound (``for ATTEMPT in 1 2``) is in the step-4 block;
6. the UNCONFIRMED arm fails loud (the ``terminal landing UNCONFIRMED``
   echo + the ``terminal-landing-unconfirmed`` marker token) — a
   silent-swallow rewording of the unconfirmed arm fails this pin;
7. the Terminal-failure branch says ``SAME four-step sequence`` and no
   ``SAME three-step sequence`` survives in the span;
8. the empty-CANON guard arm (``-z "$CANON"``) is in the step-4 block.

Prose assertions run on whitespace-NORMALIZED text (the file wraps prose
mid-phrase, so a required phrase can span lines).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

TEARDOWN_HEADING = "#### Terminal teardown (code-change path only"
RESUME_HEADING = "## Resume semantics"
STEP4_HEADING = "4. **Terminal landing confirmation (#1868; incident #1792).**"
FAILURE_BRANCH_HEADING = "**Terminal-failure branch.**"


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _skill_text() -> str:
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    return issue_skill_text()


def _teardown_span() -> str:
    """The Terminal-teardown section: its heading up to ## Resume semantics."""
    text = _skill_text()
    start = text.find(TEARDOWN_HEADING)
    assert start != -1, f"anchor {TEARDOWN_HEADING!r} not found in SKILL.md"
    end = text.find(RESUME_HEADING, start)
    assert end != -1, f"anchor {RESUME_HEADING!r} not found after the Terminal-teardown heading"
    return text[start:end]


def _step4_block() -> str:
    """The step-4 block: its heading up to the Terminal-failure branch."""
    span = _teardown_span()
    start = span.find(STEP4_HEADING)
    assert start != -1, f"anchor {STEP4_HEADING!r} not found in the Terminal-teardown span"
    end = span.find(FAILURE_BRANCH_HEADING, start)
    assert end != -1, f"anchor {FAILURE_BRANCH_HEADING!r} not found after the step-4 heading"
    return span[start:end]


def test_step4_heading_present_in_teardown_span() -> None:
    """Step 4 'Terminal landing confirmation (#1868; ...)' exists inside the span."""
    span = _teardown_span()
    assert STEP4_HEADING in span, (
        "Terminal-teardown span lacks the step-4 heading"
        " '4. **Terminal landing confirmation (#1868; incident #1792).**'"
    )
    # Redundant-by-construction spot checks so a heading reword keeps the intent.
    assert "Terminal landing confirmation" in span
    assert "#1868" in span


def test_in_flight_rationale_in_span() -> None:
    """The exit-0-includes-state=in-flight caller-retry rationale lives in the span."""
    span = _norm(_teardown_span())
    assert "state=in-flight" in span, (
        "Terminal-teardown span lacks the `state=in-flight` rationale (sync_repo_root.py"
        " exit 0 does NOT prove the push landed; the retry duty is caller-owned)"
    )
    assert "sync_repo_root.py L33-35" in span, (
        "Terminal-teardown span lacks the sync_repo_root.py L33-35 exit-contract citation"
    )


def test_origin_blob_arbiter_tokens_in_span() -> None:
    """The LANDED arbiter is the fetched-origin blob check, never exit codes."""
    span = _norm(_teardown_span())
    assert "cat-file -e" in span, (
        "Terminal-teardown span lacks the `git cat-file -e` origin-blob arbiter"
    )
    assert '"epm:done"' in span, (
        'Terminal-teardown span lacks the quoted "epm:done" arbiter grep token'
    )


def test_step4_block_reruns_sync_repo_root() -> None:
    """The step-4 block re-runs scripts/sync_repo_root.py (the bounded re-run)."""
    block = _norm(_step4_block())
    assert "sync_repo_root.py" in block, "step-4 block lacks the sync_repo_root.py re-run"


def test_step4_block_has_two_attempt_bound() -> None:
    """The re-run is bounded: exactly the 2-attempt `for ATTEMPT in 1 2` loop."""
    block = _norm(_step4_block())
    assert "for ATTEMPT in 1 2" in block, (
        "step-4 block lacks the 2-attempt bound (`for ATTEMPT in 1 2`)"
    )


def test_step4_unconfirmed_arm_fails_loud() -> None:
    """The still-unconfirmed arm keeps its loud echo + the marker-note token.

    A silent-swallow rewording of the unconfirmed arm (dropping the echo or
    the `terminal-landing-unconfirmed` epm:progress note) fails this pin.
    """
    block = _norm(_step4_block())
    assert "terminal landing UNCONFIRMED" in block, (
        "step-4 block lacks the loud '[step10d] terminal landing UNCONFIRMED' echo"
    )
    assert "terminal-landing-unconfirmed" in block, (
        "step-4 block lacks the `terminal-landing-unconfirmed` epm:progress note token"
    )
    assert "epm:progress" in block, (
        "step-4 block lacks the epm:progress marker post on the unconfirmed arm"
    )


def test_failure_branch_says_four_step_sequence() -> None:
    """The Terminal-failure branch runs the SAME four-step sequence; no stale three-step."""
    span = _norm(_teardown_span())
    assert "SAME four-step sequence" in span, (
        "Terminal-failure branch no longer says 'Run the SAME four-step sequence'"
    )
    assert "SAME three-step sequence" not in span, (
        "stale 'SAME three-step sequence' wording survives in the Terminal-teardown span"
    )


def test_step4_block_has_empty_canon_guard() -> None:
    """The empty-CANON guard arm exists: a resolution failure is not a landing failure."""
    block = _norm(_step4_block())
    assert '-z "$CANON"' in block, 'step-4 block lacks the `[ -z "$CANON" ]` empty-CANON guard arm'
    assert "resolution failure, not a landing failure" in block, (
        "step-4 block lacks the resolution-failure-is-not-a-landing-failure diagnostic"
    )
