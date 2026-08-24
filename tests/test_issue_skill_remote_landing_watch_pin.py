"""Pin the #1850 remote-landing watch rules in `/issue` SKILL.md.

Incidents (both 2026-07-29):

- #1738: a background until-loop keyed on a REMOTE artifact landing (an HF
  chunk appearing) ran silently past the producing GCE instance's ~15:08Z
  poweroff — a landing keyed on a dead producer never fires, so the session
  had no assistant turn 14:32->17:53Z until the watcher respawned it.
- #1739: after one healthy `Monitor` wake the session idled ~58 min with no
  wake on a 3-lane GCP run; a healthy-but-quiet Monitor and a dead one are
  indistinguishable in-session without periodic heartbeat lines, and the
  watcher stall-alert was the only recovery.

#1850 adds two rules to the § Long-phase heartbeat duty block (Step 6d.2):

1. a producer-fence DEADLINE on every remote-landing watch — overall
   deadline = the producer's own lifetime bound (GCE `--max-run-duration`
   fence, pod TTL, Batch-API `expires_at`) + grace; deadline expiry routes
   to a PRODUCER re-check, never a blind re-arm of the same landing watch
   (generalizing the deadline-bounded `batch_judge` poll, #658/#663);
2. Monitor heartbeat emission — a long-interval `Monitor` until-loop emits
   a `[watch-heartbeat]` stdout line every 2-3 cycles so a dead/lost
   Monitor is detectable by heartbeat absence at the next wake, re-armed
   only after the kill-before-relaunch probe.

These tests pin both rules' load-bearing phrases INSIDE the duty block
(between the `**Long-phase heartbeat duty` anchor and the Revival-trigger
paragraph). Prose assertions run on whitespace-NORMALIZED text (the file
wraps prose mid-phrase, so a required phrase can span lines) — the
`tests/test_issue_skill_detached_harvest_pin.py` convention.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

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
    text = issue_skill_text()
    start = text.find(HEARTBEAT_HEADING)
    assert start != -1, f"anchor {HEARTBEAT_HEADING!r} not found in SKILL.md"
    end = text.find(REVIVAL_HEADING, start)
    assert end != -1, f"anchor {REVIVAL_HEADING!r} not found after the heartbeat heading"
    return _norm(text[start:end])


def test_remote_landing_deadline_rule_present() -> None:
    """(a) The producer-fence deadline rule lives inside the duty block."""
    block = _heartbeat_block()
    assert "producer-fence deadline" in block, (
        "heartbeat duty block lacks the remote-landing producer-fence deadline rule (#1850)"
    )
    # Deadline derivation: the producer's own lifetime bound + grace.
    assert "producer's own lifetime bound" in block, (
        "deadline rule lacks the producer-lifetime-bound derivation"
    )
    assert "--max-run-duration" in block, "deadline rule does not name the GCE fence"
    assert "expires_at" in block, "deadline rule does not name the Batch-API expires_at bound"
    assert "grace" in block, "deadline rule lacks the grace window"
    # Additive to (not replacing) the per-segment wait-segmentation cap.
    assert "on top of the per-segment" in block, (
        "deadline rule does not state it is additive to the item-1 per-segment cap"
    )
    # Why: a landing keyed on a dead producer never fires (#1738).
    assert "dead producer NEVER fires" in block, (
        "deadline rule lacks the dead-producer-never-fires rationale (#1738)"
    )
    # Precedent: generalizes the deadline-bounded batch_judge poll.
    assert "batch_judge" in block, (
        "deadline rule does not cite the deadline-bounded batch_judge poll precedent (#658/#663)"
    )


def test_deadline_exit_rechecks_producer_never_blind_rearms() -> None:
    """(b) Deadline expiry routes to a producer re-check, never a blind re-arm."""
    block = _heartbeat_block()
    assert "RE-CHECKS THE PRODUCER" in block, (
        "deadline rule lacks the deadline-exit -> producer re-check instruction"
    )
    assert "never blind re-arms" in block, "deadline rule lacks the never-blind-re-arm clause"
    # The re-check surfaces: instance/pod status + crash-persist prefixes.
    assert "list-ephemeral" in block, "producer re-check does not name pod.py list-ephemeral"
    assert "issue<N>_partial/" in block, (
        "producer re-check does not name the crash-persist prefixes"
    )


def test_verify_covers_producer_clause() -> None:
    """(c) Item 2(i)'s per-resume verify covers the PRODUCER, not just the landing."""
    block = _heartbeat_block()
    assert "verify covers the PRODUCER" in block, (
        "duty block lacks the 'per-resume verify covers the PRODUCER' clause (#1850)"
    )


def test_monitor_heartbeat_emission_rule_present() -> None:
    """(d) Monitor heartbeat emission + dead-Monitor detection + re-arm probe."""
    block = _heartbeat_block()
    assert "Monitor heartbeat emission" in block, (
        "heartbeat duty block lacks the Monitor heartbeat-emission rule (#1850)"
    )
    # The heartbeat line shape + the notification mechanism.
    assert "[watch-heartbeat]" in block, "rule lacks the [watch-heartbeat] line shape"
    assert "each stdout line is a notification" in block, (
        "rule lacks the stdout-line-is-a-notification mechanism"
    )
    # Cadence: every 2-3 cycles of a long-interval loop.
    assert "2-3 cycles" in block, "rule lacks the every-2-3-cycles cadence"
    # Dead-Monitor detection by heartbeat absence at a later wake.
    assert "means the Monitor died" in block, (
        "rule lacks the heartbeat-gap dead-Monitor detection (#1739)"
    )
    # Re-arm only after the kill-before-relaunch probe; never assume alive.
    assert "kill-before-relaunch" in block, (
        "rule lacks the kill-before-relaunch re-arm probe reference"
    )
    assert "never assume it is still watching" in block, (
        "rule lacks the never-assume-still-watching clause"
    )
    # Scope fence: stdout only — the [long-phase-heartbeat] marker convention
    # (shared watcher/tick_triage machinery) is untouched.
    assert "NEVER a task marker" in block, (
        "rule lacks the stdout-only / never-a-task-marker scope fence"
    )
    assert "[long-phase-heartbeat]" in block, (
        "rule does not distinguish the [long-phase-heartbeat] marker convention"
    )
