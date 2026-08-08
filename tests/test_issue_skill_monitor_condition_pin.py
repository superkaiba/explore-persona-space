"""Pin the #2014 Monitor until-CONDITION composition rules in `/issue` SKILL.md.

Incidents (both 2026-08-01):

- #1739: a box-completion Monitor keyed its until-condition on an ABSOLUTE
  count (`live=N`) already true at arm time, so every re-arm fired an
  immediate no-op event; the Monitor timed out at its cap 9 times overnight
  and the session spent >=6 turns classifying the events as "baseline tick
  -- not a real completion".
- #1947: a judge fan-out monitor's terminal branch wedged on
  `pgrep -c ... || echo 0` inside its until-condition -- `pgrep -c` prints
  `0` AND exits non-zero, so the captured value is the two-line `"0\\n0"`,
  the gate never matches, and the watch wedges OPEN (verified by hand and
  `TaskStop`'d).

#2014 adds one block to the § Long-phase heartbeat duty region (Step 6d.2),
after the Monitor heartbeat emission (#1850) block: item 1 segments the
WAIT; the new clauses compose the CONDITION -- (a) key completion on a
count DECREASE from the count captured AT ARM TIME (probe emits exactly
one integer), (b) never `<count> || echo 0` in a condition (cross-linking
`gotchas.md` § count-keyed liveness), (c) a session-length watch is
`persistent: true` (timeout_ms defaults 300000 ms / caps 3600000 ms /
IGNORED when persistent), with `poll_pipeline.py` bg-Bash preferred for a
pipeline-polled run.

These tests pin the block's load-bearing phrases INSIDE the duty block
(between the `**Long-phase heartbeat duty` anchor and the Revival-trigger
paragraph). Prose assertions run on whitespace-NORMALIZED text (the file
wraps prose mid-phrase, so a required phrase can span lines) -- the
`tests/test_issue_skill_remote_landing_watch_pin.py` convention.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_MD = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

HEARTBEAT_HEADING = "**Long-phase heartbeat duty"
REVIVAL_HEADING = "Revival trigger for the deferred watcher-side option"


def _norm(text: str) -> str:
    """Collapse all whitespace runs to single spaces (wrap-tolerant match)."""
    return re.sub(r"\s+", " ", text)


def _heartbeat_block() -> str:
    """The § Long-phase heartbeat duty block, whitespace-normalized."""
    assert SKILL_MD.exists(), f"missing {SKILL_MD}"
    text = SKILL_MD.read_text(encoding="utf-8")
    start = text.find(HEARTBEAT_HEADING)
    assert start != -1, f"anchor {HEARTBEAT_HEADING!r} not found in SKILL.md"
    end = text.find(REVIVAL_HEADING, start)
    assert end != -1, f"anchor {REVIVAL_HEADING!r} not found after the heartbeat heading"
    return _norm(text[start:end])


def test_monitor_condition_block_present() -> None:
    """The until-condition composition block + its item-1 orientation sentence."""
    block = _heartbeat_block()
    assert "Monitor until-condition composition (#1739, #1947)." in block, (
        "duty block lacks the Monitor until-condition composition heading (#2014)"
    )
    # Orientation: item 1 governs the WAIT; these clauses compose the CONDITION.
    assert "Item 1 segments the WAIT; these compose the CONDITION." in block, (
        "until-condition block lacks the item-1 WAIT-vs-CONDITION orientation sentence"
    )


def test_clause_shapes_present() -> None:
    """Clauses (a)/(b)/(c): decrease-from-baseline, no `|| echo 0`, persistent watch."""
    block = _heartbeat_block()
    # (a) completion keyed on a count DECREASE from the arm-time baseline.
    assert "count DECREASE" in block, "clause (a) lacks the count-DECREASE completion key (#1739)"
    assert "AT ARM TIME" in block, "clause (a) lacks the captured-AT-ARM-TIME baseline"
    # The DIRECTION of the comparison is the whole point of clause (a): a
    # `-lt` -> `-gt` mutation inverts the guidance while every prose assertion
    # above stays green (code-review round 1, Minor 2).
    assert '-lt "$base"' in block, (
        'clause (a) lost the canonical decrease shape `[ "$(probe)" -lt "$base" ]` — '
        "an inverted comparison would silently invert the rule (#1739)"
    )
    assert "never an absolute `live=N`" in block, "clause (a) lacks the never-an-absolute-count ban"
    assert "no-op event" in block, "clause (a) lacks the immediate-no-op-event consequence (#1739)"
    # (a) probe hygiene: exactly one integer, else the test op errors and spins.
    assert "exactly one integer" in block, (
        "clause (a) lacks the probe-emits-exactly-one-integer hygiene rule"
    )
    # (b) never `<count> || echo 0` inside a condition; the two-line wedge.
    assert "|| echo 0" in block, "clause (b) lacks the `|| echo 0` ban (#1947)"
    assert "wedges OPEN" in block, (
        "clause (b) lacks the gate-never-matches / watch-wedges-OPEN consequence (#1947)"
    )
    assert "count-keyed liveness" in block, (
        "clause (b) lacks the gotchas.md count-keyed-liveness cross-link"
    )
    assert "patter[n]" in block, "clause (b) lacks the bracketed self-match-safe pgrep alternative"
    assert "rc-keyed probe" in block, "clause (b) lacks the rc-keyed probe alternative"
    # (c) session-length watch: persistent, not a re-armed bounded arm.
    assert "persistent: true" in block, "clause (c) lacks the persistent-watch prescription"
    assert "TaskStop" in block, "clause (c) lacks the TaskStop stop mechanism"
    assert "300000" in block, "clause (c) lacks the 300000 ms timeout_ms default"
    assert "3600000" in block, "clause (c) lacks the 3600000 ms timeout_ms cap"
    assert "IGNORED when `persistent`" in block, (
        "clause (c) lacks the timeout_ms-IGNORED-when-persistent semantics"
    )
    assert "poll_pipeline.py" in block, (
        "clause (c) lacks the poll_pipeline.py bg-Bash preference for pipeline-polled runs"
    )


def test_clause_zero_not_reintroduced() -> None:
    """Regression: round 1's refuted clause-(0) advice must NOT reappear."""
    block = _heartbeat_block()
    assert "needing ONE notification is a bg-Bash" not in block, (
        "round 1's refuted primitive-selection advice (clause 0) was reintroduced -- "
        "it prescribed the #1310-banned bg-Bash sleep-loop shape (plan v6 section 1)"
    )
