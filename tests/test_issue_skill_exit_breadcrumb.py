"""Prose-side pin for the #1053 Step 0 exit-breadcrumb contract (MF-3).

``.claude/skills/issue/SKILL.md`` prescribes EXACTLY TWO ``task.py
post-marker <N> epm:progress --by issue-session-guard`` command blocks — the
Step 0 single-orchestrator collision exit and the stale-wake YIELD. Their
``--note`` payloads are load-bearing: each must lstrip-start with the
``deliberate-stop `` prefix (all four staleness clocks drop that prefix —
``task_workflow.stage_dispatch_should_skip``,
``autonomous_session_watch._latest_progress_ts`` /
``._latest_nonwatcher_event_ts``, ``tick_triage.latest_event_ts``), and the
pair must carry the two ``reason=`` tokens. The exact-string code tests copy
the notes into Python literals and cannot go red when the PROSE drifts; this
test reads the skill text itself, so a rewording that drops the prefix or
the ``--by`` identity fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILL_MD = Path(__file__).resolve().parent.parent / ".claude" / "skills" / "issue" / "SKILL.md"

# A prescribed guard-exit command block: `post-marker <N> epm:progress
# --by issue-session-guard` followed (optionally via a `\`-newline shell
# continuation) by `--note "..."`. The note payload carries no double quote.
_BLOCK_RE = re.compile(
    r"post-marker <N> epm:progress --by issue-session-guard\s*(?:\\\s*)?--note \"([^\"]+)\""
)


def _note_payloads() -> list[str]:
    """Extract the ``--note`` payloads of every prescribed guard-exit block."""
    return _BLOCK_RE.findall(SKILL_MD.read_text(encoding="utf-8"))


def test_exactly_two_guard_exit_breadcrumb_blocks():
    # One block per exit shape: the Step 0 collision exit + the stale-wake
    # YIELD. A third block (or a dropped one) is prose drift to review.
    payloads = _note_payloads()
    assert len(payloads) == 2, payloads


def test_each_payload_lstrip_starts_with_deliberate_stop_prefix():
    # The lstripped `deliberate-stop ` PREFIX is what all four staleness
    # clocks key their exclusion on — losing it makes the breadcrumb refresh
    # the OWNER's freshness windows.
    payloads = _note_payloads()
    assert payloads, "no guard-exit blocks found in SKILL.md"
    for payload in payloads:
        assert payload.lstrip().startswith("deliberate-stop "), payload


def test_reason_tokens_cover_both_exit_shapes():
    reasons = sorted(
        m.group(1)
        for payload in _note_payloads()
        if (m := re.search(r"reason=([A-Za-z0-9_-]+)", payload)) is not None
    )
    assert reasons == ["stale-wake-yield", "step0-session-collision"], reasons


def test_by_identity_count_matches_block_count():
    # Every `--by issue-session-guard` occurrence in the skill must be one of
    # the two prescribed blocks — a bare mention posting under that identity
    # without the prescribed note shape would evade the payload pins above.
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.count("--by issue-session-guard") == 2, (
        "expected exactly the two prescribed guard-exit command blocks"
    )


def test_stale_wake_registration_probe_is_fail_soft():
    # #1249: the stale-wake re-check's registration-file probe must stay
    # FAIL-SOFT — a missing registration file is the NORMAL case, and a
    # bare `ls`/`cat` on an absent path exits non-zero, cancelling
    # parallel sibling tool calls (5+ sessions, 2026-07-09).
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "~/.eps-autonomous/manual-issue-<N>.json 2>/dev/null || true" in text, (
        "stale-wake registration probe lost its fail-soft form"
    )
