r"""Structural tests for the ``/issue-tick`` lightweight recurring driver.

What this pins (guarded-no-op redesign, 2026-06-12):

1. **The skill file exists** with the required YAML front matter
   (``name: issue-tick``) so the skill loader can find it.
2. **The skill's FIRST action is one ``tick_triage.py`` Bash call** and it
   documents the four verdict branches the orchestrator depends on
   (HEALTHY / TERMINAL / GATE-TRANSITION / STALE-REDRIVE) plus the
   fail-toward-coverage rule (non-zero triage exit -> full re-drive).
3. **The skill teardown match string matches the cron prompt literal**
   (``/issue-tick <N>``, NOT ``/issue <N>`` — the round-1 reviewer
   CRITICAL-2), with the hardened assert-after-delete (#501 runaway class).
   Widened 2026-07-05 (#1052): the TEARDOWN sweep additionally matches
   stray one-shot wakeup prompts as a second leg (start-anchored,
   ``(?!\d)``-guarded, resolved from a fresh ``CronList`` at teardown
   time, ``CronDelete`` not-found = success) — the ARM sites still fire
   ONLY the tick prompt.
4. **The full ``/issue`` skill's CronCreate (Step 0 + Step 6d.2) fires the
   ``/issue-tick <N>`` prompt at the ``*/45`` cadence** — the recurring
   driver is the lightweight skill, NOT the full ``/issue`` reload.
5. **Every CRON-TEARDOWN site in the full ``/issue`` skill matches
   ``/issue-tick <N>``** so a teardown across N sites doesn't drift from the
   cron prompt (an unguarded substring-match would mis-dedupe sibling
   issues). Since the 2026-07-05 widening (#1052) every teardown exit site
   also carries the two-leg pointer (stray one-shot wakeups included) —
   pinned per-site by ``test_issue_skill_exit_sites_carry_widened_pointer``.
6. **``spawn_session.py --auto`` initial prompt is ``/issue {issue}``**
   (NOT ``/loop 10m /issue {issue}``) — cold start fires the full skill once,
   which then arms the tick cron. Cold respawn via
   ``autonomous_session_watch._respawn`` calls the same ``--auto`` path, so
   this single assertion covers both.
7. **No stale ``*/20`` tick-cadence references remain** in the tick skills,
   the arm sites, or the campaign twin (interval lengthened 2026-06-12).
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


def test_issue_tick_skill_first_action_is_tick_triage():
    body = ISSUE_TICK_SKILL.read_text()
    assert "scripts/tick_triage.py" in body, (
        "the guarded-no-op tick's FIRST action must be one tick_triage.py "
        "Bash call — that is the entire healthy-path cost model"
    )
    # Fail toward coverage: a broken triage must trigger the full re-drive,
    # never a silent no-op (the alive-stalled-at-PARK class would otherwise
    # be permanently unrecovered).
    assert "Non-zero exit" in body and "STALE-REDRIVE" in body


def test_issue_tick_skill_branches_on_verdict():
    body = ISSUE_TICK_SKILL.read_text()
    # The four triage verdicts the orchestrator branches on.
    for verdict in ("HEALTHY", "TERMINAL", "GATE-TRANSITION", "STALE-REDRIVE"):
        assert verdict in body, f"skill must document the {verdict} verdict branch"


def test_issue_tick_skill_documents_human_active_noop():
    """#1629: the human-activity screen (a human (non-cron) user message in
    this session's transcript converts a would-be STALE-REDRIVE to HEALTHY)
    is documented in the skill AND implemented + kill-switched in
    tick_triage.py. Presence checks only — never counts."""
    body = ISSUE_TICK_SKILL.read_text()
    assert "human-active" in body, "skill must document the human-activity screen (#1629)"
    assert "EPM_TICK_HUMAN_ACTIVE_PROBE" in body, "skill must name the kill switch"
    triage = (ROOT / "scripts" / "tick_triage.py").read_text()
    assert "EPM_TICK_HUMAN_ACTIVE_PROBE" in triage, "tick_triage must carry the kill switch"
    assert "EPM_TICK_HUMAN_PROBE_DEBUG" in triage, "tick_triage must carry the debug telemetry env"


def test_issue_tick_skill_title_refresh_moved_to_watcher():
    body = ISSUE_TICK_SKILL.read_text()
    # The per-tick title refresh moved to the watcher's gate-push pass
    # (2026-06-12); a healthy tick must NOT pay the helper + change_title
    # calls. The skill documents the move so a maintainer doesn't re-add it.
    assert "moved to the watcher" in body.lower(), (
        "skill must document that the title refresh is watcher-owned now"
    )


def test_tick_skills_use_45_min_cadence():
    issue_body = ISSUE_TICK_SKILL.read_text()
    campaign_body = (ISSUE_TICK_SKILL.parent.parent / "campaign-tick" / "SKILL.md").read_text()
    for name, body in (("issue-tick", issue_body), ("campaign-tick", campaign_body)):
        assert "*/45 * * * *" in body, f"{name} must document the */45 cadence"
        assert "*/20 * * * *" not in body, f"{name} still references the stale */20 cadence"


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


def test_tick_skills_digest_only_reads():
    """#1000: every task-state read a tick skill instructs beyond the one
    tick_triage.py call must be a jq-filtered digest (the #866/#906
    refusal-kill prevention — an unpiped view dump pages the task body +
    every marker note into the tick turn's context)."""
    issue_body = ISSUE_TICK_SKILL.read_text()
    campaign_body = (ISSUE_TICK_SKILL.parent.parent / "campaign-tick" / "SKILL.md").read_text()
    assert "## Digest-only task-state reads" in issue_body
    assert ".frontmatter.title" in issue_body  # jq-filtered slug lookup
    assert "latest-marker <N> | jq" in issue_body  # filtered resume probe
    # Every task-state read instruction line in BOTH tick skills must be
    # digest-only. Prohibitive contract prose ("NEVER ..." bans, "does NOT"
    # bullets) is whitelisted — it names the banned forms on purpose.
    # NOTE (tripwire scope): the scan keys on the literal `<N>` placeholder
    # convention; a future line writing a concrete issue number escapes it
    # (accepted — skill prose uses `<N>` throughout; the positive asserts
    # above anchor the contract independently).
    for name, body in (("issue-tick", issue_body), ("campaign-tick", campaign_body)):
        for line in body.splitlines():
            if "NEVER" in line or "does NOT" in line:
                continue  # prohibitive contract prose
            if "view <N> --json" in line:
                # unpiped dump pages body + all notes (reconciler Must-Fix 2a)
                assert "jq" in line, f"{name}: unfiltered view --json instruction: {line!r}"
            elif "task.py view <N>" in line:
                # bare view (no --json) has NO sanctioned instruction form at
                # all (reconciler Must-Fix 2b) — any non-prohibitive line is a FAIL.
                raise AssertionError(f"{name}: bare task.py view <N> instruction: {line!r}")


# ── /issue skill: Step 6d.2 cron prompt + teardown sites ───────────────────


def test_issue_skill_cron_create_uses_issue_tick_prompt():
    body = ISSUE_SKILL.read_text()
    # The Step 6d.2 CronCreate line.
    assert 'prompt="/issue-tick <N>"' in body, (
        "Step 6d.2 must arm CronCreate with prompt='/issue-tick <N>' — the "
        "lightweight tick, NOT the full /issue reload."
    )


def test_issue_skill_arm_sites_use_45_min_cadence():
    body = ISSUE_SKILL.read_text()
    assert 'cron="*/45 * * * *"' in body, (
        "the Step 0 / Step 6d.2 CronCreate sites must arm the */45 cadence "
        "(lengthened from */20 on 2026-06-12 — the 10-min watcher carries "
        "fast detection)"
    )
    assert 'cron="*/20 * * * *"' not in body, (
        "a stale */20 CronCreate site survived the 2026-06-12 cadence change"
    )


def test_issue_tick_skill_teardown_is_hardened():
    body = ISSUE_TICK_SKILL.read_text()
    # The #501 hardening: delete-all-matching with the trailing-digit guard
    # plus the assert-after-delete + one retry.
    assert "ASSERT-AFTER-DELETE" in body, "teardown must re-list and verify deletion"
    assert "(?!\\d)" in body, (
        "the hardened fallback match needs the trailing-digit guard so "
        "'/issue-tick 46' never matches '/issue-tick 467'"
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

    NOTE (2026-07-05, #1052): the banned literals below are the OLD-drift
    *framings* (a teardown that thinks the recurring tick cron's prompt is
    the bare full-skill prompt). The DELIBERATE #1052 one-shot-wakeup leg
    of the teardown sweep uses distinct framing — the phrase
    "stray one-shot wakeups" plus a start-anchored fallback pattern —
    pinned POSITIVELY by ``test_teardown_match_set_includes_one_shot_wakeups``
    and ``test_issue_skill_exit_sites_carry_widened_pointer``; it must
    never be written in any of the banned shapes.
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


def _cron_teardown_procedure_span(issue_skill_body: str) -> str:
    """Return the Step 6d.2 canonical CRON-TEARDOWN procedure span of
    ``issue/SKILL.md`` — from the literal ``**CRON-TEARDOWN procedure``
    heading phrase up to the next bold paragraph heading (the
    ``**Prevention ban`` paragraph that immediately follows it).

    Section-scoping is what makes the #1052 pins non-vacuous: asserting the
    two-leg fragments on the WHOLE file would pass even if the procedure
    paragraph itself were reverted (the fragments also appear at the exit
    sites). ``str.index`` raises ``ValueError`` on a missing anchor, so a
    renamed/removed heading fails LOUDLY rather than silently widening the
    span; the uniqueness asserts keep the first-occurrence binding honest
    (a future DUPLICATE anchor inserted earlier in the file would silently
    rebind the span to the wrong section).
    """
    assert issue_skill_body.count("**CRON-TEARDOWN procedure") == 1, (
        "span start anchor '**CRON-TEARDOWN procedure' is no longer unique in "
        "issue/SKILL.md — str.index would bind the FIRST occurrence, silently "
        "scoping the pins to the wrong section; re-anchor the helper"
    )
    assert issue_skill_body.count("**Prevention ban") == 1, (
        "span end anchor '**Prevention ban' is no longer unique in "
        "issue/SKILL.md — the span could silently end at the wrong paragraph; "
        "re-anchor the helper"
    )
    start = issue_skill_body.index("**CRON-TEARDOWN procedure")
    end = issue_skill_body.index("\n\n**Prevention ban", start)
    return issue_skill_body[start:end]


def test_teardown_match_set_includes_one_shot_wakeups():
    """#1052 widening (incident #980: a live one-shot wakeup with the bare
    full-skill prompt survived teardown on a completed task and re-drove
    it): the CRON-TEARDOWN sweep is TWO-LEG — the recurring tick cron PLUS
    stray one-shot wakeups — and the arm sites still fire ONLY the tick
    prompt (leg 2 is teardown-only, checked by the untouched arm-site
    assertions above)."""
    issue_body = ISSUE_SKILL.read_text()
    span = _cron_teardown_procedure_span(issue_body)
    # (i) leg 2 is named with the distinctive phrase in the canonical
    # procedure span itself, not merely somewhere in the 9000-line file.
    assert "stray one-shot `/issue <N>` wakeups" in span, (
        "the Step 6d.2 canonical CRON-TEARDOWN procedure must name leg 2 "
        "(stray one-shot `/issue <N>` wakeups) — the #980 class"
    )
    # (ii) the leg-2 fallback is START-anchored + trailing-digit-guarded.
    assert r"/issue\s+<N>(?!\d)" in span, (
        "the canonical procedure must carry the start-anchored, "
        r"(?!\d)-guarded leg-2 fallback pattern /issue\s+<N>(?!\d)"
    )
    # The tick skills' canonical pseudocode carries the leg-2 whole-string
    # equality in f-string form (never the banned prompt-literal shapes).
    issue_tick_body = ISSUE_TICK_SKILL.read_text()
    campaign_tick_body = (ISSUE_TICK_SKILL.parent.parent / "campaign-tick" / "SKILL.md").read_text()
    assert 'f"/issue {N}"' in issue_tick_body, (
        "issue-tick's canonical sweep block must match leg 2 by whole-string "
        'equality (f"/issue {N}")'
    )
    assert 'f"/campaign {N}"' in campaign_tick_body, (
        "campaign-tick's mirrored sweep block must match leg 2 by "
        'whole-string equality (f"/campaign {N}")'
    )
    # Prevention ban: an /issue session never schedules its own re-drive
    # (no ScheduleWakeup / one-shot CronCreate), regardless of prompt shape.
    assert "ScheduleWakeup" in issue_body, (
        "issue/SKILL.md must carry the #1052 prevention ban naming "
        "ScheduleWakeup — the sanctioned self-wake is ONLY the tick cron"
    )
    # Campaign twin: whole-file positive pin, kept as plan-prescribed (§4.2
    # File 5 item 4). This pin alone is satisfied by the Step 2 finalize
    # site only (the Step 0 site line-wraps the phrase); per-site coverage
    # of BOTH campaign teardown sites is
    # test_campaign_skill_teardown_sites_carry_widened_pointer.
    campaign_body = (ISSUE_TICK_SKILL.parent.parent / "campaign" / "SKILL.md").read_text()
    assert "stray one-shot `/campaign <N>` wakeups" in campaign_body, (
        "campaign/SKILL.md's CRON-TEARDOWN pointer sites must name the "
        "campaign leg-2 match (stray one-shot `/campaign <N>` wakeups)"
    )


def test_teardown_cron_delete_not_found_is_success():
    """#1052 idempotency (incident #988: a teardown ran CronDelete on a
    recorded job id that was already gone, and the resulting
    'No scheduled job with id ...' error was treated as a failure when the
    job being gone IS the desired end state): every teardown-owning skill
    body must (a) treat a CronDelete not-found error as SUCCESS and (b)
    resolve live ids from a FRESH CronList at teardown time, never a
    recorded id."""
    bodies = {
        "issue-tick": ISSUE_TICK_SKILL.read_text(),
        "campaign-tick": (
            ISSUE_TICK_SKILL.parent.parent / "campaign-tick" / "SKILL.md"
        ).read_text(),
        "issue": ISSUE_SKILL.read_text(),
    }
    for name, body in bodies.items():
        assert "No scheduled job with id" in body, (
            f"{name}: teardown must document the not-found-is-success rule "
            "with the observed error shape 'No scheduled job with id' (#988)"
        )
        assert "FRESH `CronList`" in body, (
            f"{name}: teardown must resolve live job ids from a FRESH "
            "`CronList` at teardown time — recorded ids go stale (#988)"
        )


# The 8 anchored CRON-TEARDOWN edit sites of the #1052 widening. Each anchor
# is a phrase that is (a) unique in issue/SKILL.md (asserted per-site) and
# (b) adjacent to (or part of) the exit-site prose itself, so a missed /
# reverted site FAILS BY NAME instead of hiding behind a whole-file token
# count. Scope per site:
#   "line"   — the anchor's OWN line must carry the widened fragment. Used
#              for the single-line resume-TABLE rows: the two
#              awaiting_promotion rows are ADJACENT lines, so any ±N-line
#              window around either anchor contains the SIBLING row's
#              fragment and a one-row revert would pass silently (round-1
#              reconciler BLOCKER `resume-row-window-hollow`).
#   "window" — the fragment may sit on a nearby line of the same multi-line
#              prose paragraph (±8 lines). Verified per-site: no sibling
#              site's fragment falls inside any window (nearest pair is
#              unrecognised-gate at ~15 lines from park-mode-gate).
_EXIT_SITE_ANCHORS = {
    "cap5-blocked-exit": ("residual at cap-5 — open it", "window"),
    "unrecognised-gate-exit": ("reason: unrecognised_gate_name", "window"),
    # Pinned literal preserved verbatim for
    # tests/test_pv_phase1_done_gate_handler.py::
    # test_section_tail_cron_teardown_scoped_to_park_mode.
    "park-mode-gate": ("run CRON-TEARDOWN before parking", "window"),
    "step9b-awaiting-promotion": ("Run CRON-TEARDOWN now.", "window"),
    "step9c-3x-fail": ("FAIL after 3 rounds — open it", "window"),
    "step10-auto-complete-experiment": (
        "If `epm:merged` is ALREADY present",
        "window",
    ),
    "step10d-terminal-teardown": (
        "The `/issue-tick` backstop stayed armed through the",
        "window",
    ),
    "resume-row-promote-pending-unmerged": ("Step 9b auto-merge was interrupted", "line"),
    "resume-row-promote-pending-merged": ("worktree already merged", "line"),
}


def test_issue_skill_exit_sites_carry_widened_pointer():
    """Per-site completeness for the #1052 widening: every teardown exit
    site (6 inline exit sites + the 2 awaiting_promotion promote-pending
    resume-table rows) carries the widened two-leg pointer. A missed site
    fails BY NAME; a vanished anchor fails LOUDLY (anchor-not-found); a
    duplicated anchor fails LOUDLY (uniqueness) rather than silently
    rebinding the scoped assertion to the wrong occurrence. Table-row
    sites assert the fragment on the anchor's OWN line so an adjacent
    sibling row can never satisfy a reverted row's pin."""
    body = ISSUE_SKILL.read_text()
    lines = body.splitlines()
    for site, (anchor, scope) in _EXIT_SITE_ANCHORS.items():
        hits = [i for i, line in enumerate(lines) if anchor in line]
        assert hits, (
            f"{site}: anchor {anchor!r} not found in issue/SKILL.md — the "
            "exit site moved or was rewritten; re-anchor this test rather "
            "than deleting the per-site pin"
        )
        assert len(hits) == 1, (
            f"{site}: anchor {anchor!r} is not unique in issue/SKILL.md "
            f"({len(hits)} hits) — a duplicate inserted earlier in the file "
            "silently rebinds this site's assertion to the wrong occurrence; "
            "pick a longer anchor"
        )
        i = hits[0]
        if scope == "line":
            assert "stray one-shot" in lines[i], (
                f"{site}: the resume-table row containing anchor {anchor!r} "
                "does not itself carry the widened two-leg fragment "
                "'stray one-shot' — this row missed (or reverted) the #1052 "
                "widening; the adjacent sibling row's fragment does NOT count"
            )
        else:
            window = re.sub(r"\s+", " ", " ".join(lines[max(0, i - 8) : i + 9]))
            assert "stray one-shot" in window, (
                f"{site}: the teardown prose within ±8 lines of anchor "
                f"{anchor!r} does not carry the widened two-leg fragment "
                "'stray one-shot' — this exit site missed the #1052 widening"
            )


# The 2 restated CRON-TEARDOWN pointer sites of campaign/SKILL.md (the #1052
# File-4 edits). The whole-file campaign pin in
# test_teardown_match_set_includes_one_shot_wakeups is satisfied by the
# Step 2 finalize site ALONE — the Step 0 site line-wraps the phrase, so a
# Step 0 revert would pass the whole-file pin silently (round-1 reconciler
# concern `campaign-site-pin-hollow`). Slices are whitespace-normalized
# before asserting because the phrase legitimately wraps across lines
# inside the Step 0 bullet.
_CAMPAIGN_TEARDOWN_SITE_ANCHORS = {
    "step0-terminal-status-branch": "`completed` / `archived` / `blocked` → CRON-TEARDOWN",
    "step2-finalize-teardown": "4. CRON-TEARDOWN (fresh",
}


def test_campaign_skill_teardown_sites_carry_widened_pointer():
    """Per-site completeness for the campaign twin (#1052): BOTH restated
    teardown pointer sites in campaign/SKILL.md (the Step 0 terminal status
    branch and the Step 2 finalize item) carry the campaign leg-2 phrase,
    each asserted on its own whitespace-normalized site slice. A reverted
    site fails BY NAME; a vanished or duplicated anchor fails LOUDLY."""
    campaign_body = (ISSUE_TICK_SKILL.parent.parent / "campaign" / "SKILL.md").read_text()
    lines = campaign_body.splitlines()
    for site, anchor in _CAMPAIGN_TEARDOWN_SITE_ANCHORS.items():
        hits = [i for i, line in enumerate(lines) if anchor in line]
        assert hits, (
            f"{site}: anchor {anchor!r} not found in campaign/SKILL.md — the "
            "teardown site moved or was rewritten; re-anchor this test rather "
            "than deleting the per-site pin"
        )
        assert len(hits) == 1, (
            f"{site}: anchor {anchor!r} is not unique in campaign/SKILL.md "
            f"({len(hits)} hits) — a duplicate silently rebinds this site's "
            "assertion to the wrong occurrence; pick a longer anchor"
        )
        i = hits[0]
        site_slice = re.sub(r"\s+", " ", " ".join(lines[i : i + 6]))
        assert "stray one-shot `/campaign <N>` wakeups" in site_slice, (
            f"{site}: the teardown pointer at anchor {anchor!r} does not "
            "carry the campaign leg-2 phrase (stray one-shot `/campaign <N>` "
            "wakeups, whitespace-normalized) — this site missed (or "
            "reverted) the #1052 widening"
        )


def test_issue_skill_documents_push_notification():
    body = ISSUE_SKILL.read_text()
    # The Step 9b awaiting_promotion exit must fire PushNotification.
    # The Step 2c plan-gate park exit must fire PushNotification.
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
