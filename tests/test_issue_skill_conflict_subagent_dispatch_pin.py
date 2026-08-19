"""Content-invariant pins for the #1403 residual-conflict subagent dispatch.

Task #1403 replaced the Step-9b/10d merge-conflict recovery's inline
manual-resolution step in `.claude/skills/issue/SKILL.md` with a
context-hygiene branch: the orchestrator materializes the residual
`--diff-filter=U` conflicted-path list to a file (exclusive-arm `if !`
producer, #1184/#1243); a NON-EMPTY list routes to ONE fresh
worktree-scoped `implementer`-class subagent (lean by-reference brief;
verdict + resolution commit sha + per-class counts + path NAMES back —
never conflict hunks), while an EMPTY list commits inline
(`git commit --no-edit`). The orchestrator never reads residual conflict
bodies inline; certification / lint gate / push / merge stay
orchestrator-side, and the dispatch lives inside the one-recovery-attempt
cap.

REGION-SCOPED per the `test_issue_skill_binary_figures_recovery_pin.py`
precedent (slice `#### Merge-conflict recovery` -> `#### The
artifact-confirmed merge procedure`, fail-loud anchors). Prose pins
normalize whitespace AND strip `# ` comment prefixes (rationale prose
wraps across comment lines inside the fenced bash block).

Origin incident: #1338 (2026-07-15, session bc8a308f): 4-file real content
conflicts at a Step 10d squash-merge; the conflict diff was paged inline
into a near-full orchestrator context; "Prompt is too long", no recovery
turn.
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import issue_skill_text

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

_RESIDUAL_LIST_TOKEN = "recovery-residual.txt"
_DISPATCH_HEADING = "##### Residual-conflict subagent dispatch"
# Assert-4 positive token: the fence's non-empty-residual dispatch branch.
_FENCE_BRANCH_TOKEN = "elif [ -s /tmp/issue-<N>-recovery-residual.txt"
# Assert-4 negative token: the removed inline manual-add placeholder
# (unique to the replaced step; NOT `keep main's version of anything
# outside this task's deliverables`, which the subagent brief deliberately
# reuses as its resolution contract).
_REMOVED_MANUAL_ADD = 'git -C "$WT" add <each resolved file>'


def _skill_text() -> str:
    return issue_skill_text()


def _recovery_region(text: str) -> str:
    """The merge-conflict recovery slice (the dispatch branch's home)."""
    start_marker = "#### Merge-conflict recovery"
    end_marker = "#### The artifact-confirmed merge procedure"
    start = text.find(start_marker)
    end = text.find(end_marker)
    assert start != -1, "Merge-conflict recovery heading not found in SKILL.md"
    assert end != -1, "artifact-confirmed merge heading not found in SKILL.md"
    assert start < end, "recovery region must precede the artifact-confirmed merge"
    return text[start:end]


def _normalized_prose(text: str) -> str:
    """Whitespace-collapse AND strip leading `#` comment prefixes per line —
    rationale prose wraps across bash comment lines and numbered-list
    continuation lines, so pins on spanning phrases need the prefixes
    removed and the lines joined."""
    words: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        words.extend(stripped.split())
    return " ".join(words)


def test_recovery_region_carries_residual_dispatch_branch():
    """Assert 1: the recovery region materializes the residual conflicted-path
    list to a file AND carries the residual-conflict subagent dispatch
    subsection (the context-hygiene branch a non-empty list routes to)."""
    recovery = _recovery_region(_skill_text())
    assert _RESIDUAL_LIST_TOKEN in recovery, (
        "the recovery must materialize the residual conflicted-path list "
        "to /tmp/issue-<N>-recovery-residual.txt"
    )
    assert _DISPATCH_HEADING in recovery, (
        "the recovery region must carry the residual-conflict subagent "
        "dispatch subsection (##### heading)"
    )


def test_recovery_prose_bans_inline_conflict_body_reads():
    """Assert 2: the normalized region prose bans inline conflict-body reads
    (the #1338 context bomb) on BOTH sides — the orchestrator's dispatch
    branch (`do NOT read conflict bodies inline`) and the subagent's return
    contract (`NEVER conflict hunks`, bodies, or diff text in the return)."""
    prose = _normalized_prose(_recovery_region(_skill_text()))
    assert "do NOT read conflict bodies inline" in prose, (
        "the dispatch branch must ban inline orchestrator reads of residual "
        "conflict bodies (the #1338 context bomb)"
    )
    assert "NEVER conflict hunks" in prose, (
        "the subagent return contract must ban conflict hunks/bodies/diff "
        "text in the return (an oversized return kills the parent)"
    )


def test_dispatch_lives_inside_one_attempt_cap():
    """Assert 3: the dispatch lives INSIDE the one-recovery-attempt cap
    (never a second dispatch, never an inline fallback read) and the
    subagent pins every resolution to the captured $MAIN_SHA snapshot
    (never re-fetches or re-snapshots — the #1128 shared-ref race)."""
    prose = _normalized_prose(_recovery_region(_skill_text()))
    assert "never a second dispatch" in prose, (
        "the dispatch must live inside the one-recovery-attempt cap (never a second dispatch)"
    )
    assert "never re-fetches or re-snapshots" in prose, (
        "the subagent brief must pin resolutions to the captured $MAIN_SHA "
        "(never re-fetches or re-snapshots, #1128)"
    )


def test_fence_branch_replaces_inline_manual_resolution():
    """Assert 4 (FENCE-DISCRIMINATING, plan-r1 Statistics Must-Fix): without
    this pin the suite is satisfiable with the subsection (Edit B) present
    but the fence branch (Edit A) missing — a vacuous PASS that leaves the
    old inline manual-resolution path executed. Positive: the fence carries
    the non-empty-residual dispatch branch. Negative: the removed inline
    manual-add placeholder is gone from the region."""
    recovery = _recovery_region(_skill_text())
    assert _FENCE_BRANCH_TOKEN in recovery, (
        "the recovery fence must branch on a non-empty residual list "
        "(elif [ -s /tmp/issue-<N>-recovery-residual.txt ...) — Edit A"
    )
    assert _REMOVED_MANUAL_ADD not in recovery, (
        "the inline manual-resolution step (git add <each resolved file>) "
        "must be gone — residual conflicts dispatch to the subagent instead"
    )
