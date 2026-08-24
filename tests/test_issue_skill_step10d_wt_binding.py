"""Pin the #2306 fence-local ``$WT`` bindings in the /issue Step 10d + Step 9c fences.

Fenced blocks are separate shells when extracted and run standalone, and BOTH
``git -C ""`` and ``cd ""`` are silent no-ops (rc=0, cwd unchanged) — an
unbound ``$WT`` retargets the fence at the SHARED repo root with no error
(#2306/#2293). #2306 inserts the D1 fence-local binding block (REPO_ROOT
derivation + ``WT=`` binding + FATAL existence guard) at the top of the four
extraction-prescribed Step 10d fences, and strengthens the three Step 9c gate
fence openings to guard unbound/missing ``$WT`` before the ``cd``.

Pinned invariants (whitespace-normalized substring checks on stable shapes,
per the pin-family convention — tests/test_issue_skill_bare_push_snippets_pin.py):

1. The safe-case auto-merge fence region carries the D1 binding pair AND the
   FATAL guard (plan #2306 acceptance criterion 2).
2. Same for the merge-conflict recovery, rewritten-branch, and
   artifact-confirmed merge fences.
3. Each of the three Step 9c gate fences carries the strengthened
   ``[ -n "$WT" ] && [ -d "$WT" ] && cd "$WT" ||`` guard (count == 3 — the
   guard line appears nowhere else in the composed document).

NOTE for future editors: a legitimate rewording of these fences must update
the pinned substrings below IN THE SAME DIFF. The mechanical companion is
``workflow_lint --check-issue-skill-fence-wt-binding`` (any covered form
passes it); THIS pin holds the specific D1/strengthened-guard text at the
specific fences the #2306 plan names.

Paths resolve via the composed-skill reader (``tests/issue_skill_source``),
which roots at ``Path(__file__)`` — NEVER ``task_workflow.repo_root()``,
which reads the MAIN checkout and would miss worktree edits pre-merge.
"""

from __future__ import annotations

from tests.issue_skill_source import issue_skill_text

# Region anchors (first occurrence is the real heading in every case;
# verified unique across SKILL.md + steps/*.md except the stale-task-folder
# heading, whose second occurrence is a backtick-quoted back-reference AFTER
# the real heading — find() still resolves the heading).
SAFE_CASE_START = "#### The auto-merge procedure (safe case"
SAFE_CASE_END = "#### Merge-conflict recovery (safe case"
MERGE_CONFLICT_END = "##### Residual-conflict subagent dispatch"
REWRITTEN_START = "#### Rewritten-branch landing route"
REWRITTEN_END = "#### The artifact-confirmed merge procedure (unsafe case"
ARTIFACT_END = "#### Post-merge stale-task-folder guard"

# The D1 fence-local binding pair (REPO_ROOT derivation + WT binding).
D1_BINDING = (
    'REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)") '
    'WT="$REPO_ROOT/.claude/worktrees/issue-<N>"'
)
# The D1 FATAL existence guard (exit 1 before any git -C "$WT" / cd "$WT").
D1_FATAL_GUARD = (
    '[ -n "$WT" ] && [ -d "$WT" ] || { echo "FATAL: WT unbound or missing ($WT) '
    r"— refusing; git -C \"\" / cd \"\" silently retarget the SHARED repo root"
    '" >&2; exit 1; }'
)
# The strengthened Step 9c gate opening (guards unbound/missing $WT before cd;
# a bare `cd "$WT" ||` guard is NOT equivalent — cd "" is a silent no-op).
GATE_GUARD = (
    '[ -n "$WT" ] && [ -d "$WT" ] && cd "$WT" || { echo "FATAL: WT unbound/missing '
    r"or cd failed ($WT) — cd \"\" is a silent no-op; gate must never run at the "
    'shared root" >&2; exit 1; }'
)
N_STEP9C_GATE_FENCES = 3


def _normalized(text: str) -> str:
    """Collapse all whitespace to single spaces (wrap-tolerant substring checks)."""
    return " ".join(text.split())


def _region(norm: str, start_anchor: str, end_anchor: str) -> str:
    start = norm.find(start_anchor)
    assert start != -1, (
        f"composed /issue skill text lost the {start_anchor!r} heading; if the "
        "section was renamed, update this pin alongside it."
    )
    end = norm.find(end_anchor, start)
    assert end != -1, (
        f"composed /issue skill text lost the {end_anchor!r} heading after "
        f"{start_anchor!r}; if the section was renamed, update this pin alongside it."
    )
    return norm[start:end]


def _composed_normalized() -> str:
    return _normalized(issue_skill_text())


def test_safe_case_merge_fence_binds_wt() -> None:
    region = _region(_composed_normalized(), SAFE_CASE_START, SAFE_CASE_END)
    assert D1_BINDING in region, (
        "the Step 10d safe-case auto-merge fence lost its #2306 fence-local "
        "REPO_ROOT/WT binding pair"
    )
    assert D1_FATAL_GUARD in region, (
        "the Step 10d safe-case auto-merge fence lost its #2306 FATAL unbound/missing-WT guard"
    )


def test_merge_conflict_recovery_fence_binds_wt() -> None:
    region = _region(_composed_normalized(), SAFE_CASE_END, MERGE_CONFLICT_END)
    assert D1_BINDING in region, (
        "the Step 10d merge-conflict recovery fence lost its #2306 fence-local "
        "REPO_ROOT/WT binding pair"
    )
    assert D1_FATAL_GUARD in region, (
        "the Step 10d merge-conflict recovery fence lost its #2306 FATAL unbound/missing-WT guard"
    )


def test_rewritten_branch_fence_binds_wt() -> None:
    region = _region(_composed_normalized(), REWRITTEN_START, REWRITTEN_END)
    assert D1_BINDING in region, (
        "the Step 10d rewritten-branch landing fence lost its #2306 fence-local "
        "REPO_ROOT/WT binding pair"
    )
    assert D1_FATAL_GUARD in region, (
        "the Step 10d rewritten-branch landing fence lost its #2306 FATAL unbound/missing-WT guard"
    )


def test_artifact_confirmed_fence_binds_wt() -> None:
    region = _region(_composed_normalized(), REWRITTEN_END, ARTIFACT_END)
    assert D1_BINDING in region, (
        "the Step 10d artifact-confirmed merge fence lost its #2306 fence-local "
        "REPO_ROOT/WT binding pair"
    )
    assert D1_FATAL_GUARD in region, (
        "the Step 10d artifact-confirmed merge fence lost its #2306 FATAL unbound/missing-WT guard"
    )


def test_step9c_gate_fences_carry_strengthened_guard() -> None:
    norm = _composed_normalized()
    count = norm.count(GATE_GUARD)
    assert count == N_STEP9C_GATE_FENCES, (
        f"expected the strengthened Step 9c gate guard on exactly "
        f"{N_STEP9C_GATE_FENCES} gate fences, found {count} — a gate fence "
        "was reworded/removed (update this pin in the same diff) or a copy "
        "drifted."
    )
