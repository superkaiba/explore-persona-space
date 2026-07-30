"""Pin the Step 10d KEPT-stash surfacing duty in the /issue SKILL.md (#1751).

Incident #1716 (session 2202031b, 2026-07-27T14:37Z): a post-merge-guard
pre-sync printed `  stash: KEPT stash@{0} (319c2bf16e7c) — apply --check
dirty; manual triage; rescue patch ...` on stderr and the session's
wrap-up said "Post-merge guard clean, no duplicate task folders" — the
manual-triage flag died in a transcript, unowned. #1751 adds a blanket
**KEPT-stash surfacing duty** paragraph to the Step 10d § "Bare push /
merge snippets" subsection (the canonical copy-source every other
`sync_repo_root.py` call site points at), binding every sync the skill
prescribes: on any report line containing `stash: KEPT`, the session (a)
appends one `stash-kept: <ref> (<sha12>) rescue=<rescue-patch path> —
manual triage owed` line PER KEPT entry to the round's durable marker
note (the `epm:merged` note file at merge sites, or one adjacent
`epm:progress` note), (b) carries the same line(s) in the end-of-turn
wrap-up, and NEVER summarizes a KEPT-reporting sync as "clean". Surface
only — the session never pops/drops the stash; triage stays human.

This test pins the duty paragraph's presence + load-bearing content so a
later SKILL.md rewrite cannot silently drop it (#884/#1045/#1134
droppable-prose lineage; the #1713 lost-update class).

NOTE for future SKILL.md editors: a legitimate rewording of the duty
paragraph must update the pinned substrings below IN THE SAME DIFF.
Assertions are whitespace-normalized substring checks (the prose is
line-wrapped in SKILL.md), per the pin-family convention — see e.g.
tests/test_issue_skill_followup_cap_park_note_pin.py.

Paths resolve via ``Path(__file__)`` — NEVER ``task_workflow.repo_root()``,
which reads the MAIN checkout and would miss worktree edits pre-merge.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE_SKILL = ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

REGION_START = "#### Bare push / merge snippets"
REGION_END = "#### Merge safety guards"
DUTY_HEADING = "KEPT-stash surfacing duty"

# The mechanical needle the duty keys on (a report line CONTAINING it — the
# emitted line is two-space-indented `  stash: KEPT ...`, never line-start).
NEEDLE = "stash: KEPT"
# The required disposition-line format (one line PER KEPT entry).
DISPOSITION_FORMAT = "stash-kept: <ref> (<sha12>) rescue=<rescue-patch path> — manual triage owed"
PER_ENTRY_PHRASE = "one line PER KEPT entry"
# Channel (a): the round's durable marker note; channel (b): the wrap-up.
CHANNEL_MERGED = "epm:merged"
CHANNEL_PROGRESS = "epm:progress"
CHANNEL_WRAPUP = "end-of-turn wrap-up"
# The never-"clean" ban (the #1716 swallow shape).
NEVER_CLEAN_BAN = 'NEVER summarize a KEPT-reporting sync as "clean"'
INCIDENT_REF = "#1716"
# Surface-only scope: the session never triages the stash itself.
SURFACE_ONLY = "never pops/drops the stash"
TRIAGE_HUMAN = "triage stays human"


def _normalized(text: str) -> str:
    """Collapse all whitespace to single spaces (wrap-tolerant substring checks)."""
    return " ".join(text.split())


def _snippets_region() -> str:
    """The normalized SKILL.md span between the Bare-push-snippets heading and
    the Merge-safety-guards heading (the duty paragraph's mandated home)."""
    skill_norm = _normalized(ISSUE_SKILL.read_text())
    start = skill_norm.find(REGION_START)
    assert start != -1, (
        f"SKILL.md lost the {REGION_START!r} heading; if the subsection was "
        "renamed, update this pin alongside it."
    )
    end = skill_norm.find(REGION_END, start)
    assert end != -1, (
        f"SKILL.md lost the {REGION_END!r} heading after {REGION_START!r}; "
        "if the subsection was renamed or reordered, update this pin alongside it."
    )
    return skill_norm[start:end]


def test_kept_stash_surfacing_duty_present():
    """The KEPT-stash surfacing duty paragraph lives inside the Step 10d
    § Bare push / merge snippets region (before § Merge safety guards)."""
    region = _snippets_region()
    assert DUTY_HEADING in region, (
        f"SKILL.md's {REGION_START!r} subsection must carry the "
        f"{DUTY_HEADING!r} paragraph (#1751) — every sync_repo_root.py call "
        "site points at this copy-source subsection, so dropping the duty "
        "here silently unbinds ALL of them (incident #1716: a `stash: KEPT` "
        "manual-triage flag was swallowed by a 'clean' wrap-up)."
    )


def test_kept_stash_duty_load_bearing_content():
    """The duty paragraph keeps its load-bearing elements: the `stash: KEPT`
    needle, the `stash-kept:` disposition-line format (one line per KEPT
    entry), BOTH channels (durable marker note + end-of-turn wrap-up), the
    never-"clean" ban, and the surface-only scope."""
    region = _snippets_region()
    required = [
        NEEDLE,
        DISPOSITION_FORMAT,
        PER_ENTRY_PHRASE,
        CHANNEL_MERGED,
        CHANNEL_PROGRESS,
        CHANNEL_WRAPUP,
        NEVER_CLEAN_BAN,
        INCIDENT_REF,
        SURFACE_ONLY,
        TRIAGE_HUMAN,
    ]
    for token in required:
        assert token in region, (
            f"SKILL.md's KEPT-stash surfacing duty (in the {REGION_START!r} "
            f"subsection) must carry {token!r} — the #1751 prescription "
            "(needle, per-entry stash-kept: disposition line, both channels, "
            "never-'clean' ban, surface-only scope) must not be silently "
            "reworded away (incident #1716)."
        )
