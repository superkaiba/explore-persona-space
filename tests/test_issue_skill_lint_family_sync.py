"""Pin the #1560 lint/guard-family freshness sync in .claude/skills/issue/SKILL.md.

#1560 (2026-07-20) extended the Step 5a spec-freshness sync to cover the
lint/guard family (scripts/workflow_lint.py, .claude/hooks, the
test_guard_lessons_edit / test_workflow_lint* pin tests), subject-scoped the
sync's per-file branch-side-edit exclusion (the Guard-3 convention), and
added a mandatory pre-gate freshness re-sync against fetched origin/main to
the Step 10d pre-push workflow-lint gate — closing the branch-era-linter
vintage-skew class that red the gate three times on 2026-07-19
(#1489 / #1482 / #1417).

These tests fail the suite if a later SKILL.md editor drops the family
entries, the boundary-paragraph family exception, the pre-gate re-sync
bullet (or reorders it after the gate's stale-verdict rm), the 9a-ter
staleness note, or reintroduces the full-message commit filter the Step 5a
sync and the gate section deliberately avoid.

NOTE for future SKILL.md editors: these assertions pin literal snippet text.
A legitimate rewording of the pinned lines in SKILL.md must update the
matching assertions here IN THE SAME COMMIT, or the suite goes red.
"""

from pathlib import Path

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"

# The banned full-message commit-filter literal, built by CONCATENATION so
# this test file itself never carries the token its negative asserts scan for.
_FULL_MESSAGE_FILTER = "--grep=" + "'spec-freshness'"
_FULL_MESSAGE_INVERT = _FULL_MESSAGE_FILTER + " --invert-grep"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def _step5a_span(text: str) -> str:
    """The Step 5a spec-freshness block + its boundary paragraph.

    From the block's leading comment (unique) to the 429-pacing blockquote
    that follows the boundary paragraph (unique).
    """
    start = text.index("# Step 5a WANTS the WORKTREE root")
    end = text.index("429 pacing at every ensemble fan-out", start)
    return text[start:end]


def _gate_region(text: str) -> str:
    """The pre-push workflow-lint gate section: gate H4 through the auto-merge H4.

    Region-scoped deliberately: the stale-verdict rm line occurs 5x file-wide
    (Step 9c and auto-merge consumers), so whole-file index() would be vacuous
    for the ordering assert; within this region it occurs exactly once.
    """
    start = text.index("#### Pre-push workflow-lint gate")
    end = text.index("#### The auto-merge procedure", start)
    return text[start:end]


# --- (1) Step 5a SPECS family entries -------------------------------------


def test_step5a_specs_include_lint_family():
    assert (
        'SPECS=".claude/agents .claude/skills .claude/rules .claude/workflow.yaml '
        "CLAUDE.md scripts/workflow_lint.py .claude/hooks "
        'tests/test_guard_lessons_edit.py :(glob)tests/test_workflow_lint*.py"'
    ) in _text(), (
        "Step 5a SPECS must carry the #1560 lint/guard family "
        "(workflow_lint.py, .claude/hooks, test_guard_lessons_edit.py, "
        "the :(glob) test_workflow_lint* pin-test family)"
    )


# --- (2) boundary paragraph names the family + retains the exclusion ------


def test_sync_scope_paragraph_names_family_boundary():
    span = _step5a_span(_text())
    assert "spec-coupled lint/guard family" in span, (
        "the sync-scope boundary paragraph must name the family exception"
    )
    assert "do NOT\nextend it further into `scripts/`, `tests/`, or `src/`" in span, (
        "the boundary paragraph must retain the exclusion for everything else"
    )
    assert "main's newer workflow\ntests pin behavior implemented in main's newer" in span, (
        "the boundary paragraph must retain rationale (ii): main's newer tests "
        "pin main's newer scripts/+src/"
    )
    assert "explore_persona_space.workflow" in span, (
        "the boundary paragraph must name the accepted single src seam"
    )
    assert "collection-time ImportError" in span, (
        "the cross-check-at-root operational rule must carry the "
        "collection-time-ImportError staleness symptom"
    )


# --- (3) pre-gate re-sync bullet present + ordered before the verdict rm --


def test_step10d_pregate_resync_present_and_ordered():
    region = _gate_region(_text())
    bullet_idx = region.index("**Pre-gate freshness re-sync (#1560")
    assert "origin/main" in region[bullet_idx:], "re-sync must anchor to origin/main"
    assert 'timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet' in region, (
        "re-sync must start with the bounded fetch"
    )
    assert "[step10d] pre-gate re-sync: synced <n> files (<sha>) | no drift" in region, (
        "the re-sync must end with the ran-vs-never-ran echo breadcrumb"
    )
    rm_idx = region.index("rm -f /tmp/issue-<N>-lint-verdict.txt")
    assert bullet_idx < rm_idx, (
        "the pre-gate re-sync must complete BEFORE the gate's stale-verdict rm "
        "so the verdict sha-binds the synced tip"
    )


# --- (4) 9a-ter inline-gate staleness note ---------------------------------


def test_9ater_staleness_note_present():
    text = _text()
    assert "is the stale-family class" in text, (
        "the 9a-ter inline-gate bullet must carry the #1417 staleness-class diagnostic"
    )
    assert "run the Step 5a sync (now family-inclusive)" in text, (
        "the 9a-ter staleness note must name the Step 5a sync remedy"
    )


# --- (5) no full-message grep-exclusion literal in the gate region ---------


def test_no_grep_specfreshness_in_gate_region():
    """The gate section must never carry the full-message commit-filter form
    (a commit BODY mentioning the token would launder a genuine deliverable
    at the pre-gate re-sync). The existing test_step10d_guard3.py ban covers
    only the Guard-3 span, which stops at the fast-path heading and does NOT
    reach the gate section — hence this region's own negative assert."""
    region = _gate_region(_text())
    assert _FULL_MESSAGE_FILTER not in region, (
        "the pre-push gate section must not carry a full-message "
        "grep-exclusion invocation (subject-scoped filtering only)"
    )


# --- (6) the :(glob)/never-shell-expands comment at the SPECS line ---------


def test_glob_pathspec_comment_present():
    span = _step5a_span(_text())
    assert "`:(glob)` is a git pathspec (never shell-expands" in span, (
        "the SPECS comment must document that :(glob) is a git pathspec that never shell-expands"
    )
    assert "skip grain is PER-ITEM" in span, (
        "the SPECS comment must document the per-item skip grain of the "
        "branch-side-edit guard over the :(glob) family entry"
    )


# --- (7) Step 5a exclusion is subject-scoped (the MF-B pin) -----------------


def test_step5a_exclusion_is_subject_scoped():
    span = _step5a_span(_text())
    assert "--format='%H %s'" in span, (
        "the Step 5a branch-side-edit exclusion must emit '<sha> <subject>' "
        "(subject-scoped, the Guard-3 convention)"
    )
    assert "awk 'index($0, \"spec-freshness\") == 0'" in span, (
        "the Step 5a exclusion must filter via the subject-scoped awk index() form"
    )
    assert _FULL_MESSAGE_INVERT not in span, (
        "the Step 5a exclusion must NOT use the full-message form (it would "
        "wrongly exclude a deliverable whose commit BODY mentions the token)"
    )


# --- (8) the re-sync never re-derives $WT (the MF-A pin) --------------------


def test_pregate_resync_does_not_rederive_wt():
    region = _gate_region(_text())
    bullet_idx = region.index("**Pre-gate freshness re-sync (#1560")
    bullet = region[bullet_idx : region.index("[step10d] pre-gate re-sync:", bullet_idx)]
    assert "ALREADY-BOUND `$WT`" in bullet, (
        "the re-sync bullet must state $WT is already bound by the merge flow"
    )
    assert "WT=$(git rev-parse --show-toplevel)` line" in bullet, (
        "the re-sync bullet must name the Step 5a WT-derivation line to DROP"
    )
    assert "do NOT re-derive `$WT` at Step 10d" in bullet, (
        "the re-sync bullet must ban re-deriving $WT at Step 10d (a repo-root "
        "cwd would rebind it to the shared root)"
    )
