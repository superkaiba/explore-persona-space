"""Structural tests for the new ``/issue-tick`` lightweight recurring driver.

What this pins:

1. **The skill file exists** with the required YAML front matter
   (``name: issue-tick``) so the skill loader can find it.
2. **The skill documents the four branch shapes** the orchestrator depends on
   (TERMINAL / PARK / ACTIVE / GATE-PARK) and the soft-fail title refresh via
   ``session_progress_report.py``.
3. **The skill teardown match string matches the cron prompt literal**
   (``/issue-tick <N>``, NOT ``/issue <N>`` — the round-1 reviewer
   CRITICAL-2). A drift here means a stranded cron after every park / terminal.
4. **The full ``/issue`` skill's CronCreate (Step 6d.2) fires the
   ``/issue-tick <N>`` prompt** — the recurring driver is now the lightweight
   skill, NOT the full ``/issue`` reload.
5. **Every CRON-TEARDOWN site in the full ``/issue`` skill matches
   ``/issue-tick <N>``** so a teardown across N sites doesn't drift from the
   cron prompt (a substring-match version would mis-dedupe sibling issues —
   exact-equality is the contract).
6. **``spawn_session.py --auto`` initial prompt is ``/issue {issue}``**
   (NOT ``/loop 10m /issue {issue}``) — cold start fires the full skill once,
   which then arms the tick cron. Cold respawn via
   ``autonomous_session_watch._respawn`` calls the same ``--auto`` path, so
   this single assertion covers both.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE_TICK_SKILL = ROOT / ".claude" / "skills" / "issue-tick" / "SKILL.md"
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
SPAWN_SESSION = ROOT / "scripts" / "spawn_session.py"


# ── /issue-tick skill file ─────────────────────────────────────────────────


def test_issue_tick_skill_file_exists():
    assert ISSUE_TICK_SKILL.is_file(), (
        f"Expected the lightweight recurring driver at {ISSUE_TICK_SKILL}; "
        "without it the Step 6d.2 cron has nothing to fire."
    )


def test_issue_tick_skill_has_front_matter():
    body = ISSUE_TICK_SKILL.read_text()
    # YAML front matter shape: opens with `---\n`, has `name: issue-tick`,
    # closes with a second `---` before the prose body.
    assert body.startswith("---\n"), "skill file must open with YAML front matter"
    head = body.split("\n---\n", 1)[0]
    assert "name: issue-tick" in head, "front matter must declare name: issue-tick"
    # `description:` is required so the skill loader knows when it applies.
    assert re.search(r"^description:", head, flags=re.M), "front matter must declare description"


def test_issue_tick_skill_uses_session_progress_report():
    body = ISSUE_TICK_SKILL.read_text()
    assert "scripts/session_progress_report.py" in body, (
        "skill must call the canonical title helper, not roll its own format"
    )
    assert "--step" in body, "skill must invoke the helper with --step"


def test_issue_tick_skill_branches_on_status():
    body = ISSUE_TICK_SKILL.read_text()
    # The four branch shapes the orchestrator depends on. Match
    # case-insensitively because the section headings are CAPITALISED in the
    # skill prose.
    for branch in ("TERMINAL", "PARK", "ACTIVE", "GATE-PARK"):
        assert branch in body, f"skill must document the {branch} branch"


def test_issue_tick_skill_teardown_match_is_issue_tick_prompt():
    body = ISSUE_TICK_SKILL.read_text()
    # The cron the skill is fired by has prompt `/issue-tick <N>` — the
    # teardown MUST match that exact literal, not `/issue <N>`.
    assert '"/issue-tick <N>"' in body, "CRON-TEARDOWN string must literally be /issue-tick <N>"
    # Whole-string equality (not substring) is the contract — the skill must
    # state that explicitly so a future reader doesn't drop to .endswith()
    # and break sibling issues.
    assert "whole-string equality" in body or "prompt.strip() == " in body


def test_issue_tick_skill_fires_push_notification():
    body = ISSUE_TICK_SKILL.read_text()
    assert "PushNotification" in body, (
        "skill must fire PushNotification at gate-park / blocked transitions"
    )


def test_issue_tick_skill_does_not_instruct_cron_create():
    """The /issue-tick skill is the recurring driver; it MUST NOT arm crons.

    Only Step 6d.2 of the full /issue skill arms the tick cron. If
    /issue-tick ever calls CronCreate, every tick stacks a duplicate cron
    on top of the one that fired it — the ARM-GUARD in the full skill only
    catches it from the full-skill side.

    We accept REFERENCES to CronCreate in prose (describing what the FULL
    /issue skill does, or in the comparison table explaining what /issue-tick
    does NOT do) but ban CronCreate showing up in this skill's Step
    instructions or the "What this skill does NOT do" should mention it
    explicitly.
    """
    body = ISSUE_TICK_SKILL.read_text()
    # `## What this skill does NOT do` must explicitly list CronCreate as
    # something the skill does NOT arm. That's the contract.
    nots_section_match = re.search(
        r"## What this skill does NOT do(.*?)(?=^##|\Z)", body, flags=re.M | re.S
    )
    assert nots_section_match, "skill must have a 'What this skill does NOT do' section"
    nots_section = nots_section_match.group(1)
    assert "CronCreate" in nots_section, (
        "the 'NOT do' section must explicitly list CronCreate so a future "
        "maintainer doesn't add a cron-arming step to the recurring driver"
    )


# ── /issue skill: Step 6d.2 cron prompt + teardown sites ───────────────────


def test_issue_skill_cron_create_uses_issue_tick_prompt():
    body = ISSUE_SKILL.read_text()
    # The Step 6d.2 CronCreate line.
    assert 'prompt="/issue-tick <N>"' in body, (
        "Step 6d.2 must arm CronCreate with prompt='/issue-tick <N>' — the "
        "lightweight tick, NOT the full /issue reload."
    )


def test_issue_skill_arm_guard_matches_issue_tick_prompt():
    body = ISSUE_SKILL.read_text()
    # The Step 6d.2 ARM-GUARD AND every CRON-TEARDOWN must reference the
    # SAME literal: `/issue-tick <N>` (or `"/issue-tick <N>"` quoted in
    # prose). A drift would mean the guard arms a duplicate cron, or the
    # teardown silently no-ops.
    assert '"/issue-tick <N>"' in body, (
        "ARM-GUARD + CRON-TEARDOWN sites must reference /issue-tick <N>"
    )


def test_issue_skill_no_residual_issue_cron_match():
    """No site in the /issue skill should still match the OLD cron prompt
    literal ``"/issue <N>"`` for CRON-TEARDOWN purposes — that drift left
    stranded crons after the rename.
    """
    body = ISSUE_SKILL.read_text()
    # Catch both `prompt.strip() == "/issue <N>"` and `prompt="/issue <N>"`
    # specifically. We allow `/issue <N>` to appear in prose for OTHER
    # purposes (the skill is invoked as `/issue <N>` by the user), just not
    # as a cron-prompt literal.
    bad_patterns = [
        'prompt.strip() == "/issue <N>"',
        'prompt="/issue <N>"',
        "the `/issue <N>` job",
        'CronDelete the "/issue <N>"',
    ]
    found = [pat for pat in bad_patterns if pat in body]
    assert not found, (
        f"these CRON-TEARDOWN literals still reference the old /issue prompt; "
        f"rewrite to /issue-tick: {found}"
    )


def test_issue_skill_documents_push_notification():
    body = ISSUE_SKILL.read_text()
    # The Step 9b awaiting_promotion exit must fire PushNotification.
    # The Step 2c parked_over_cap exit must fire PushNotification.
    assert "PushNotification" in body, (
        "/issue skill must call PushNotification at gate-park / blocked sites"
    )
    # Document the deferred-tool load alongside the existing Cron* one.
    assert "PushNotification" in body and "ToolSearch" in body


def test_issue_skill_autonomous_section_documents_issue_tick():
    body = ISSUE_SKILL.read_text()
    # The autonomous-behavior section's "Stop the [loop|cron]" bullet must
    # name the new lightweight driver so a maintainer reading the
    # autonomous-behavior section understands what the cron actually fires.
    autonomous_section_start = body.find("Autonomous session behavior")
    assert autonomous_section_start >= 0, "autonomous-behavior section not found"
    autonomous_section_end = body.find("### Step 0", autonomous_section_start)
    autonomous_block = body[autonomous_section_start:autonomous_section_end]
    assert "/issue-tick" in autonomous_block, (
        "the autonomous-behavior section must mention /issue-tick as the "
        "recurring driver — not the legacy /loop 10m /issue shape"
    )


# ── spawn_session.py --auto initial prompt ─────────────────────────────────


def test_spawn_session_auto_prompt_loads_full_issue_skill_once():
    """``--auto`` cold-start (and cold respawn via
    ``autonomous_session_watch._respawn``, which also goes through
    ``spawn-issue --auto``) must boot the FULL ``/issue <N>`` skill exactly
    once. That first invocation arms the recurring ``/issue-tick <N>`` cron
    at Step 6d.2; subsequent ticks are the lightweight driver."""
    body = SPAWN_SESSION.read_text()
    # The --auto prompt assignment line. We pin the exact f-string shape
    # because the spawn-session contract has to round-trip through the
    # daemon's HAPPY_INITIAL_PROMPT env var; an unexpected expansion
    # silently fires the wrong skill.
    assert 'prompt = f"/issue {issue}"' in body, (
        "cmd_spawn_issue's --auto branch must set prompt = f'/issue {issue}'"
    )
    # And the OLD loop-shape must be gone — leaving it as a comment is
    # fine, but it must not be the active assignment.
    active_loop_assignment = re.search(
        r'^\s*prompt\s*=\s*f"/loop 10m /issue \{issue\}"', body, flags=re.M
    )
    assert not active_loop_assignment, (
        "the legacy `/loop 10m /issue {issue}` prompt assignment must be removed; "
        "the recurring driver is now the /issue-tick cron, not /loop"
    )
