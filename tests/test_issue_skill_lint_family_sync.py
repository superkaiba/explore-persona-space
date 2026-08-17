"""Pin the #1560 → #1714 lint/guard-family freshness sync in .claude/skills/issue/SKILL.md.

#1560 (2026-07-20) extended the Step 5a spec-freshness sync to cover the
lint/guard family (scripts/workflow_lint.py, .claude/hooks, the
test_guard_* / test_workflow_lint* pin tests), subject-scoped the
sync's per-file branch-side-edit exclusion (the Guard-3 convention), and
added a mandatory pre-gate freshness re-sync against fetched origin/main to
the Step 10d pre-push workflow-lint gate — closing the branch-era-linter
vintage-skew class that red the gate three times on 2026-07-19
(#1489 / #1482 / #1417 / #1675→#1682).

#1714 (2026-07-26) makes the sync family-atomic (a branch-side edit on
ANY member of a coupled family — workflow.yaml + markers.md,
scripts/workflow_lint.py + its pin tests, .claude/hooks + its guard
tests — widens the skip to the WHOLE family), moves the Step 10d
re-sync from PRE-gate to POST-gate (right before `gh pr merge`) so
origin/main advancement during the ~30-min gate window can no longer
red the squash merge with CONFLICTING, and adds two new pin tests
(family declaration + Step 5a/§4.6 drift guard).

#1807 (2026-07-29) adds the mechanically-gated verdict RE-BIND to the
Step 10d safe-case block (a post-gate sync commit whose cert-sha..HEAD
delta is provably origin/main-identical A/M-only re-binds the verdict
file's line 2 to the new tip instead of forcing a full gate re-run;
D/R*/C*/T/U rows and non-identical content fail CLOSED behind
REBIND_OK), moves the #1657 head-sync pre-check AFTER the re-sync +
re-bind (it must poll the FINAL tip), and adds a Step 9c step-1a
pre-gate spec-freshness re-sync — a binding REFERENCE to the Step 5a
family-atomic block, never a third inlined FAMILY_OF copy — so a
main-side spec fix landing after the Step 5a sync can no longer red the
9c gate (#1742 class). Pin tests (12) and (13) cover the two additions.

#1883 (2026-08-01) adds the `:(glob)tests/test_issue_skill_*.py`
skill-pin glob to the "workflow" family (both FAMILY_OF copies + both
SPECS lists), closing the #1824 vintage-skew residual: main's SKILL.md
synced without its paired pin test, and 3 test_issue_skill_* pin tests
red the Step 9c gate (~30-min gate re-run + manual reconciliation).
Test (14) additionally pins SPECS <-> SPECS_10D token-set equality
(SPECS_10D was previously unpinned).

#1963 (2026-08-03) adds the guard-script implementation set
`:(glob)scripts/guard_*.sh` to the "guard" family (both FAMILY_OF copies
+ both SPECS lists): the test_guard_* pin tests execute the WORKTREE
copies of scripts/guard_*.sh, so syncing the tests without the scripts
half-syncs the tree and reds main-green guard nodes on pure version
skew (incidents #1860: 3 false-red test_guard_repo_root_branch.py
nodes; #1862: 12 false-red test_1861_exitguard_* nodes). Test (1)'s
SPECS literal and test (9)'s family-membership asserts gain the new
member; tests (10) + (14) mechanically enforce the Step 10d copy +
SPECS_10D parity with no edit.

#1972 (2026-08-03) widens the sync set on three arms (incident basis
2026-07-31: #1776 gate rounds 1-3, #1768 r3/r4/r5 ~40 min extra, #1846
Guard-4 LOST-UPDATE REFUSAL, #1887 lint red from 4 branch-stale
agent-memory files): (arm 1) `.claude/agent-memory` joins SPECS +
SPECS_10D as a singleton, protected by a NEW uncommitted-dirt arm in
pass 1 of BOTH copies — tracked-modified porcelain marks the family
dirty; an untracked (??) path only when it exists at origin/main (the
only case `git checkout origin/main -- <pathspec>` could clobber it);
(arm 2) the Step 9c selector triple — scripts/select_step9c_tests.py,
tests/test_select_step9c_tests.py,
tests/step9c_workflow_invariant_manifest.txt — joins the "lint" family
(the pin test importlib-loads the selector BY PATH and its case 6b pins
WORKFLOW_INVARIANT set-equal to the manifest, so any strict subset is a
half-sync); (arm 3) a per-FILE sibling-issue freshness arm in Step 5a
ONLY (deliberately NOT in the Step 10d post-gate copy — the 10d TG legs
run before it) syncs never-branch-edited scripts/issue<M>_*.py +
tests/test_issue<M>_*.py pairs together. Tests (15) + (16) pin the two
new arms; tests (10) + (14) enforce the copy parity mechanically.

#2208 (2026-08-09) adds an import-satisfiability probe to the #1972
sibling-issue arm (incident #2206: the arm synced a main-NEW
tests/test_issue2038_*.py importing a symbol added to src/ AFTER the
branch point; the worktree src is branch-era, collection ImportErrored,
and `step9c_baseline.py compare` classified the node NEW fail-closed —
~1h gate wall + a manual provenance override). Synced sibling TEST files
now pass a fenced real-collection probe BEFORE the sync commit; a probe
failure reverts the whole same-issue synced pair (branch-era files
restored from HEAD, main-NEW files dropped from index + tree). Section
(16) gains the probe pins + a functional shape repro.

#2303 (2026-08-14) closes the two #2293 defects. Defect 1:
`.claude/config/agent_spec_size_caps.txt` joins the "lint" family in
BOTH copies (SPECS/FAMILY_OF + SPECS_10D) — scripts/workflow_lint.py
reads it at MODULE IMPORT time (_load_agent_spec_caps() raises
FileNotFoundError loud), so syncing the linter without its data file
left a worktree whose every linter-shelling pre-commit hook crashed.
Defect 2: the sync `git commit` return code is now CHECKED in BOTH
copies — a failed commit prints a FATAL line naming the staged paths
and exits non-zero, and the Step 5a success echo fires only AFTER a
verified commit (reporting the committed sha), never an unconditional
staged diffstat; the Step 10d twin reads SYNC_SHA / pushes only after a
verified commit. Tests (1) + (9) gain the caps-file member; section
(17) pins the rc-checked stanzas; section (18) reproduces both defect
shapes under real git.

#2352 (2026-08-17) adds `tests/issue_skill_source.py` — the shared
composed-spec reader (#2155) every test_issue_skill_* pin test imports
(`from tests.issue_skill_source import ...`) — to SPECS + SPECS_10D as a
SINGLETON: its own family, deliberately NO FAMILY_OF entry in either
copy. Incident: the Step 5a sync pulled the current test_issue_skill_*
set into the issue-2333 worktree (fork predates #2155) WITHOUT the
helper; 66 collection errors (ModuleNotFoundError:
tests.issue_skill_source) walled `pytest -k` and would red the Step 9c
gate as NEW. No single-family assignment closes the class: the helper is
imported from MULTIPLE families — the workflow-family skill-pin glob
(x64) AND the lint-family
tests/test_workflow_lint_no_repo_root_worktree_revert.py — plus ~30
unsynced tests, and families dirty-skip INDEPENDENTLY, so a dirty
workflow family (the modal workflow-fix branch) with a clean lint family
syncs the lint importer fresh and recreates the same
ModuleNotFoundError. A singleton syncs whenever it is ITSELF clean,
covering fresh importers in every family and in unsynced tests alike.
Test (1) gains the token; a new negative pin (9b) holds the singleton
disposition (no FAMILY_OF entry in either copy); section (19) adds the
family-aware forward guard: every `tests.<mod>` import in any
family-synced test file must be sync-coverable (same-family glob /
same-family explicit token / singleton token), so the NEXT main-side
helper module a family-synced test imports reds THIS suite on main
instead of red-ing a worktree gate. A runtime import-satisfiability
probe on the FAMILY arm (the #2208 sibling-arm probe's shape) was
considered and DEFERRED: the family sync checkouts+commits atomically
across ALL safe families in one command inside two mirrored fail-closed
blocks — a probe-failure revert would have to unwind whole families
(SKILL.md included) and a bug there reds every Step 5a run and every
merge fleet-wide, while the static guard catches the class EARLIER (on
main, at the PR adding the import).

These tests fail the suite if a later SKILL.md editor drops the family
entries, the boundary-paragraph family exception, the post-gate re-sync
bullet (or reorders it before the gate's stale-verdict rm), the 9a-ter
staleness note, reintroduces the full-message commit filter the Step 5a
sync and the gate section deliberately avoid, drops the family-atomic
declaration in Step 5a, lets the Step 10d inline family-atomic block
drift from Step 5a's family definition, weakens the #1807 re-bind
stanza's fail-closed arms, drops the 9c pre-gate re-sync reference,
drops the #1972 uncommitted-dirt arm from either copy, drops (or
mirrors into the 10d copy) the #1972 sibling-issue per-file arm, drops
the #2208 import-satisfiability probe from the sibling arm, drops the
#2303 caps-file lint-family membership, un-checks the #2303 sync
commit rc in either copy, re-familys the #2352
tests/issue_skill_source.py singleton (a FAMILY_OF entry for it in
either copy), or lets a family-synced test file import a tests.<mod>
helper no SPECS token can sync (guard 19).

NOTE for future SKILL.md editors: these assertions pin literal snippet text.
A legitimate rewording of the pinned lines in SKILL.md must update the
matching assertions here IN THE SAME COMMIT, or the suite goes red.
"""

import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

SKILL = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "issue" / "SKILL.md"

# The banned full-message commit-filter literal, built by CONCATENATION so
# this test file itself never carries the token its negative asserts scan for.
_FULL_MESSAGE_FILTER = "--grep=" + "'spec-freshness'"
_FULL_MESSAGE_INVERT = _FULL_MESSAGE_FILTER + " --invert-grep"


def _text() -> str:
    return issue_skill_text()


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
        'SPECS=".claude/agents .claude/agent-memory .claude/skills .claude/rules '
        ".claude/workflow.yaml "
        "CLAUDE.md scripts/workflow_lint.py .claude/config/agent_spec_size_caps.txt "
        "scripts/select_step9c_tests.py .claude/hooks "
        ":(glob)scripts/guard_*.sh "
        "tests/test_guard_lessons_edit.py "
        "tests/test_workflow_yaml.py tests/test_autonomous_session_watch.py "
        "tests/test_select_step9c_tests.py "
        "tests/step9c_workflow_invariant_manifest.txt "
        ":(glob)tests/test_workflow_lint*.py "
        ":(glob)tests/test_guard_*.py "
        "tests/issue_skill_source.py "
        ':(glob)tests/test_issue_skill_*.py"'
    ) in _text(), (
        "Step 5a SPECS must carry the #1560 lint/guard family "
        "(workflow_lint.py, .claude/hooks, the :(glob) test_workflow_lint* "
        "and :(glob) test_guard_* pin-test families) plus the #1714 "
        "explicit importers tests/test_workflow_yaml.py and "
        "tests/test_autonomous_session_watch.py (workflow_lint symbols "
        "used outside the :(glob) test_workflow_lint* pattern) — the "
        "guard-family widening pinned by #1709 covers all "
        "tests/test_guard_*.py (vintage-skew class "
        "#1489/#1482/#1417/#1675→#1682) — plus the #1883 skill-pin glob "
        ":(glob)tests/test_issue_skill_*.py (prose-pin tests over "
        ".claude/skills content; the #1824 vintage skew) — plus the #1963 "
        "guard-script implementation set :(glob)scripts/guard_*.sh (the "
        "test_guard_* pins execute the worktree copies; syncing tests "
        "without scripts half-syncs the tree — the #1860/#1862 false-red "
        "guard nodes) — plus the #1972 members: .claude/agent-memory "
        "(singleton; dirt-arm-protected) and the Step 9c selector triple "
        "scripts/select_step9c_tests.py + tests/test_select_step9c_tests.py "
        "+ tests/step9c_workflow_invariant_manifest.txt (lint family; the "
        "pin test importlib-loads the selector by path and pins "
        "WORKFLOW_INVARIANT set-equal to the manifest) — plus the #2303 "
        "lint-family data file .claude/config/agent_spec_size_caps.txt "
        "(workflow_lint.py reads it at MODULE IMPORT time; syncing the "
        "linter without it strands a FileNotFoundError-raising linter in "
        "the worktree — the #2293 shape) — plus the #2352 SINGLETON "
        "tests/issue_skill_source.py (the shared composed-spec reader "
        "every test_issue_skill_* pin test imports AND lint-family + "
        "unsynced tests import; cross-family importers mean no single "
        "family covers it — syncing the pin tests without the helper red "
        "66 collection errors in the issue-2333 worktree)"
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
    assert "`:(glob)tests/test_issue_skill_*.py`" in span, (
        "the boundary paragraph must name the #1883 skill-pin glob "
        ":(glob)tests/test_issue_skill_*.py as a workflow-family member "
        "(backtick-wrapped, so this pins the PROSE mention — the bare "
        "bash-block occurrences do not satisfy it)"
    )


# --- (3) post-gate re-sync bullet present + ordered AFTER the verdict rm --


def test_step10d_postgate_resync_present_and_ordered():
    region = _gate_region(_text())
    bullet_idx = region.index("**Post-gate freshness re-sync (#1714")
    assert "origin/main" in region[bullet_idx:], "re-sync must anchor to origin/main"
    assert 'timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet' in region, (
        "re-sync must start with the bounded fetch"
    )
    assert "[step10d] post-gate re-sync: synced <n> files (<sha>) | no drift" in region, (
        "the re-sync must end with the ran-vs-never-ran echo breadcrumb"
    )
    rm_idx = region.index("rm -f /tmp/issue-<N>-lint-verdict.txt")
    assert bullet_idx > rm_idx, (
        "the post-gate re-sync must be positioned AFTER the gate's "
        "stale-verdict rm (the bullet documents a post-gate operation "
        "executed from the auto-merge subsection below, so its prose "
        "must come after the executable gate block containing the rm)"
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
    assert "awk 'index($0, \"sync workflow-surface specs from\") == 0'" in span, (
        "the Step 5a exclusion must filter via the subject-scoped awk index() form "
        "keyed on the prescribed sync-subject anchor (#1789)"
    )
    assert _FULL_MESSAGE_INVERT not in span, (
        "the Step 5a exclusion must NOT use the full-message form (it would "
        "wrongly exclude a deliverable whose commit BODY mentions the token)"
    )


# --- (8) the re-sync never re-derives $WT (the MF-A pin) --------------------


def test_postgate_resync_does_not_rederive_wt():
    region = _gate_region(_text())
    bullet_idx = region.index("**Post-gate freshness re-sync (#1714")
    bullet = region[bullet_idx : region.index("[step10d] post-gate re-sync:", bullet_idx)]
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


# --- (9) family-atomic declaration in Step 5a bash (#1714 new pin) --------


def test_step5a_family_atomicity_declared_in_bash():
    """The Step 5a block must declare family membership as bash associative
    array entries (the #1714 family-atomic skip)."""
    span = _step5a_span(_text())
    assert "declare -A FAMILY_OF" in span, (
        "Step 5a must declare family membership via `declare -A FAMILY_OF` "
        "(the #1714 family-atomic transitive skip)"
    )
    assert 'FAMILY_OF[".claude/workflow.yaml"]="workflow"' in span, (
        "the workflow family must include .claude/workflow.yaml"
    )
    assert 'FAMILY_OF[".claude/skills"]="workflow"' in span, (
        "the workflow family must include .claude/skills (markers.md + SKILL.md derived tables)"
    )
    assert 'FAMILY_OF["scripts/workflow_lint.py"]="lint"' in span, (
        "the lint family must include scripts/workflow_lint.py"
    )
    assert 'FAMILY_OF[":(glob)tests/test_workflow_lint*.py"]="lint"' in span, (
        "the lint family must include :(glob)tests/test_workflow_lint*.py"
    )
    assert 'FAMILY_OF[".claude/hooks"]="guard"' in span, (
        "the guard family must include .claude/hooks"
    )
    assert 'FAMILY_OF[":(glob)scripts/guard_*.sh"]="guard"' in span, (
        "the guard family must include :(glob)scripts/guard_*.sh — the "
        "guard-script implementations the test_guard_* pins execute; "
        "syncing tests without scripts half-syncs the tree (#1860/#1862; #1963)"
    )
    assert 'FAMILY_OF[":(glob)tests/test_guard_*.py"]="guard"' in span, (
        "the guard family must include :(glob)tests/test_guard_*.py"
    )
    assert 'FAMILY_OF["tests/test_workflow_yaml.py"]="workflow"' in span, (
        "the workflow family must include tests/test_workflow_yaml.py "
        "(imports render_*_table from workflow_lint AND reads workflow.yaml data)"
    )
    assert 'FAMILY_OF[":(glob)tests/test_issue_skill_*.py"]="workflow"' in span, (
        "the workflow family must include the skill pin-test glob "
        ":(glob)tests/test_issue_skill_*.py — prose-pin tests over "
        ".claude/skills content; syncing SKILL.md without its paired pin "
        "test reds the Step 9c gate (#1824 vintage skew; #1883)"
    )
    assert 'FAMILY_OF["tests/test_autonomous_session_watch.py"]="lint"' in span, (
        "the lint family must include tests/test_autonomous_session_watch.py "
        "(imports check_asw_docstring_pass_count from workflow_lint)"
    )
    assert 'FAMILY_OF["scripts/select_step9c_tests.py"]="lint"' in span, (
        "the lint family must include scripts/select_step9c_tests.py — the "
        "Step 9c selector; its pin test importlib-loads the WORKTREE copy "
        "by path, so syncing the test without the selector half-syncs the "
        "tree (#1972)"
    )
    assert 'FAMILY_OF["tests/test_select_step9c_tests.py"]="lint"' in span, (
        "the lint family must include tests/test_select_step9c_tests.py — "
        "the selector's by-path importlib pin test (#1972)"
    )
    assert 'FAMILY_OF["tests/step9c_workflow_invariant_manifest.txt"]="lint"' in span, (
        "the lint family must include tests/step9c_workflow_invariant_manifest.txt "
        "— the pin test's case 6b holds WORKFLOW_INVARIANT set-equal to it; "
        "the dominant selector edit updates all three together (#1972)"
    )
    assert 'FAMILY_OF[".claude/config/agent_spec_size_caps.txt"]="lint"' in span, (
        "the lint family must include .claude/config/agent_spec_size_caps.txt "
        "— workflow_lint.py reads it at MODULE IMPORT time "
        "(_load_agent_spec_caps() raises FileNotFoundError loud), so syncing "
        "the linter without its data file strands a crashing linter in the "
        "worktree (#2293; #2303)"
    )
    assert 'FAMILY_OF["tests/test_guard_lessons_edit.py"]="guard"' in span, (
        "the guard family must include the explicit tests/test_guard_lessons_edit.py "
        "entry (it also matches the :(glob) but is declared explicitly for clarity)"
    )
    assert "DIRTY_FAMILIES" in span, (
        "the family-atomic loop must gate the sync on a DIRTY_FAMILIES associative array"
    )


# --- (9b) tests/issue_skill_source.py stays a SINGLETON (#2352 negative pin) --


def test_issue_skill_source_singleton_no_family_assignment():
    """#2352: tests/issue_skill_source.py must have NO FAMILY_OF entry in
    EITHER sync copy — the singleton disposition is the fix. The helper is
    imported from MULTIPLE families (the workflow-family skill-pin glob x64
    AND the lint-family test_workflow_lint_no_repo_root_worktree_revert.py)
    plus ~30 unsynced tests, and families dirty-skip INDEPENDENTLY, so ANY
    single-family assignment reopens the #2352 half-sync class through the
    other families (a dirty workflow family + a clean lint family syncs the
    lint importer fresh against a missing/stale helper). A deliberate
    re-familying must rework guard (19)'s coverage predicate in the same
    commit."""
    text = _text()
    for label, span in (
        ("Step 5a", _step5a_span(text)),
        ("auto-merge", _automerge_span(text)),
    ):
        assert 'FAMILY_OF["tests/issue_skill_source.py"]' not in span, (
            f"the {label} sync copy assigns tests/issue_skill_source.py to a "
            f"family — it must stay a SINGLETON (no FAMILY_OF entry): the "
            f"helper's importers span the workflow AND lint families plus "
            f"unsynced tests, families dirty-skip independently, and any "
            f"single-family assignment recreates the #2352 "
            f"ModuleNotFoundError half-sync through the other families. A "
            f"deliberate re-familying must rework guard (19)'s coverage "
            f"predicate in the same commit."
        )


# --- (10) auto-merge post-gate re-sync matches Step 5a family (#1714 drift guard) --


def test_step10d_family_atomicity_matches_step5a():
    """The auto-merge subsection's inline family-atomic bash block must
    declare the SAME FAMILY_OF entries as the Step 5a block. #1714 §4.6
    inlines the block into `#### The auto-merge procedure` so the
    post-gate re-sync uses the same family definition; a future editor
    that adds a family entry to Step 5a but forgets the auto-merge copy
    would silently produce a divergent post-gate re-sync — this test
    catches that."""
    text = _text()
    # Auto-merge span: from the H4 to the next H4 that follows.
    merge_start = text.index("#### The auto-merge procedure")
    merge_end = text.index("#### ", merge_start + 4)
    merge_span = text[merge_start:merge_end]
    # Every FAMILY_OF entry declared in the Step 5a span must also
    # appear in the auto-merge span (identical string).
    step5a_span = _step5a_span(text)
    for line in step5a_span.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAMILY_OF[") and stripped.endswith(
            ('="workflow"', '="lint"', '="guard"')
        ):
            assert stripped in merge_span, (
                f"Step 5a declares {stripped!r} but the auto-merge "
                f"post-gate re-sync block does not — the two copies must "
                f"stay in sync (§4.6 drift guard, #1714 methodology concern 1)"
            )


# --- (11) Step 5a sources fetched origin/main (#1747 durability pin) --------

# Banned bare-local-main fragments, built by CONCATENATION so this test file
# itself never carries them (the file's _FULL_MESSAGE_FILTER convention).
_BARE_CHECKOUT_MAIN = "checkout " + "main --"
_BARE_CHECKOUT_MAIN_SPECS = _BARE_CHECKOUT_MAIN + " $SAFE_SPECS"
_BARE_DIFF_QUIET_MAIN_SPECS = "diff --quiet " + "main -- $SAFE_SPECS"
_BARE_MERGE_BASE_MAIN = "merge-base HEAD " + "main"


def test_step5a_sources_fetched_origin_main():
    """#1747: the Step 5a spec-freshness sync sources FETCHED origin/main —
    never a possibly-lagging local main (#1724 synced REGRESSED spec bytes
    from a lagging shared-root main) — and skips the ENTIRE sync body on a
    main checkout (the explicit replacement for the old vacuous-local-diff
    self-no-op, which origin/main sourcing removed)."""
    span = _step5a_span(_text())
    # (a) bounded freshness fetch (the #1289/#1714 degrade-never-wedge shape)
    assert 'timeout --kill-after=30s 120s git -C "$WT" fetch origin main --quiet' in span, (
        "Step 5a must run the bounded freshness fetch before the merge-base capture"
    )
    # (b) the surgical checkout sources origin/main
    assert "checkout origin/main -- $SAFE_SPECS" in span, (
        "the Step 5a sync must check out $SAFE_SPECS from origin/main"
    )
    # (c) the merge-base is captured against origin/main
    assert "merge-base HEAD origin/main" in span, (
        "the Step 5a pass-1 scan must merge-base against origin/main"
    )
    # (d) explicit on-main skip guard covering the whole sync body
    assert "rev-parse --abbrev-ref HEAD" in span, (
        "the on-main skip guard must probe the session branch via rev-parse --abbrev-ref HEAD"
    )
    assert '" = "main" ]' in span, (
        "the on-main skip guard must compare the session branch against main"
    )
    assert "[step5a] session on main" in span, (
        "the on-main skip must announce itself with the one-line [step5a] skip echo"
    )
    # (e) NEGATIVE: no bare-local-main sync forms survive in the span
    assert _BARE_CHECKOUT_MAIN_SPECS not in span, (
        "the Step 5a sync must not check out $SAFE_SPECS from bare local main (#1724)"
    )
    assert _BARE_DIFF_QUIET_MAIN_SPECS not in span, (
        "the Step 5a sync condition must not diff against bare local main (#1724)"
    )
    assert _BARE_MERGE_BASE_MAIN not in span, (
        "the Step 5a pass-1 scan must not merge-base against bare local main (#1724)"
    )
    # (f) NEGATIVE, span-scoped: no bare checkout-from-local-main fragment
    # anywhere in the span (catches the prose/comment class — e.g. the
    # in-block :(glob) comment — that the $SAFE_SPECS-anchored bans miss).
    assert _BARE_CHECKOUT_MAIN not in span, (
        "nothing in the Step 5a span may reference a bare checkout from local main"
    )


# --- (12) Step 10d verdict re-bind stanza (#1807 pins) ----------------------


def _automerge_span(text: str) -> str:
    """The auto-merge subsection: its H4 through the next H4 (same
    extraction as test (10))."""
    start = text.index("#### The auto-merge procedure")
    end = text.index("#### ", start + 4)
    return text[start:end]


def test_step10d_verdict_rebind_present():
    """#1807: the safe-case block re-binds the SHA-bound verdict to a
    post-gate sync tip ONLY under the mechanical probe — a --name-status
    enumeration whose every row is A/M and byte-identical to fetched
    origin/main. D/R*/C*/T/U rows fail CLOSED unconditionally (critic
    round-1 Must-Fix: --name-only would read a both-sides-absent deletion
    as main-identical while the certified landing tree CONTAINED the file
    via the own-diff overlay); line 1 is COMPOSED from the existing
    verdict (sed -n 1p), never typed; an unverifiable delta routes to the
    fail-closed BLOCKED arm (stale verdict consumed, no merge), and the
    merge itself is variable-gated on REBIND_OK=yes."""
    span = _automerge_span(_text())
    assert 'git -C "$WT" diff --name-status "$CERT_SHA" HEAD' in span, (
        "the re-bind probe must enumerate the cert-sha..HEAD delta with "
        "--name-status (NOT --name-only — the deletion corner, #1807)"
    )
    assert 'A|M) git -C "$WT" diff --quiet origin/main HEAD -- "$p" || DELTA_OK=no ;;' in span, (
        "A/M rows must keep the fetched-origin/main byte-identity probe"
    )
    assert "*)   DELTA_OK=no ;;   # D / R* / C* / T / U — never sync output" in span, (
        "every non-A/M status row must fail CLOSED unconditionally "
        "(the sync's `checkout origin/main --` can only add/modify)"
    )
    assert "sed -n 1p /tmp/issue-<N>-lint-verdict.txt" in span, (
        "line 1 must be COMPOSED from the existing verdict (sed -n 1p), never typed"
    )
    assert 'if [ "$REBIND_OK" = yes ]; then' in span, (
        "gh pr ready / gh pr merge must be variable-gated on REBIND_OK=yes "
        "(never a bare mid-block false as flow control)"
    )
    fail_echo = span.index("sync delta NOT verifiable as origin/main-identical A/M-only")
    blocked_echo = span.index("BLOCKED: verdict re-bind failed", fail_echo)
    rm_after = span.find("rm -f /tmp/issue-<N>-lint-verdict.txt", blocked_echo)
    assert rm_after != -1, (
        "an unverifiable delta must end in the stale verdict being consumed "
        "(rm -f in the fail-closed BLOCKED arm) — never a merge on an "
        "unverified tip"
    )


# --- (13) Step 9c pre-gate spec-freshness re-sync (#1807 / #1742 pin) -------


def test_step9c_pregate_sync_present():
    """#1807 fix (b): Step 9c step 1a must run the Step 5a family-atomic
    spec-freshness sync — a binding REFERENCE, never a third inlined
    FAMILY_OF copy (which would escape
    test_step10d_family_atomicity_matches_step5a's drift guard) — BEFORE
    the selector computes the gate subset, closing the #1742
    stale-worktree-spec gate-red class."""
    text = _text()
    start = text.index("**9c. Test-verdict gate")
    sel_idx = text.index("select_step9c_tests.py", start)
    region = text[start:sel_idx]
    assert "Pre-gate spec-freshness re-sync (#1742)" in region, (
        "Step 9c step 1a must carry the pre-gate spec-freshness re-sync "
        "BEFORE the first select_step9c_tests.py mention"
    )
    assert "Step 5a family-atomic" in region, (
        "the 9c re-sync must reference the Step 5a family-atomic block"
    )
    assert "never inline a THIRD `FAMILY_OF` copy" in region, (
        "the 9c re-sync must bind as a reference, not a third inlined copy"
    )


# --- (14) SPECS <-> SPECS_10D token-set equality (#1883 pin) -----------------


def test_specs_and_specs_10d_token_sets_match():
    """#1883: the Step 5a SPECS list and the Step 10d auto-merge inline
    copy's SPECS_10D list must carry the SAME pathspec token set. Check
    (10) mechanically pins the FAMILY_OF entries across the two copies;
    SPECS_10D itself was previously unpinned, so an editor adding a spec
    to SPECS but not SPECS_10D would silently produce a divergent
    post-gate re-sync — this test catches that."""
    text = _text()
    m5 = re.search(r'^\s*SPECS="([^"]+)"', _step5a_span(text), flags=re.M)
    assert m5, "Step 5a must declare SPECS as a one-line double-quoted assignment"
    m10 = re.search(r'^\s*SPECS_10D="([^"]+)"', _automerge_span(text), flags=re.M)
    assert m10, (
        "the auto-merge inline block must declare SPECS_10D as a one-line double-quoted assignment"
    )
    assert set(m5.group(1).split()) == set(m10.group(1).split()), (
        "SPECS (Step 5a) and SPECS_10D (Step 10d auto-merge inline copy) "
        "must carry identical pathspec token sets (#1883; the two lists "
        "are the same sync surface — a member added to one but not the "
        "other silently diverges the post-gate re-sync)"
    )


# --- (15) uncommitted-dirt arm in BOTH sync copies (#1972 pin) ---------------


def _dirt_arm_block(span: str) -> str:
    """Extract the #1972 uncommitted-dirt arm from a sync span: from its
    comment lead through the end of the UNCOMMITTED-changes echo line."""
    start = span.index("# Uncommitted-dirt arm (#1972)")
    echo_idx = span.index("carries UNCOMMITTED changes the sync could clobber", start)
    end = span.index("\n", echo_idx)
    return span[start:end]


def test_uncommitted_dirt_arm_in_both_sync_copies():
    """#1972 arm 1: pass 1 of BOTH sync copies (Step 5a + the Step 10d
    auto-merge inline block) must carry the uncommitted-dirt arm.
    Tracked-modified porcelain output (any non-?? line — renames included,
    no path parsing needed) marks the file's family dirty unconditionally;
    an untracked (??) path marks it dirty ONLY when the same path exists at
    fetched origin/main — `git checkout <ref> -- <pathspec>` DOES overwrite
    an untracked file whose path exists at the ref and cannot touch one
    absent from it, so fresh mid-round agent-memory files with no main-side
    name collision never block the sync. Fail-safe direction: dirty ->
    status-quo staleness, never a clobber."""
    text = _text()
    for span_name, span, dirty_marking in (
        ("Step 5a", _step5a_span(text), "DIRTY_FAMILIES[$fam]=1"),
        ("auto-merge", _automerge_span(text), "DIRTY_FAMILIES_10D[$fam]=1"),
    ):
        block = _dirt_arm_block(span)
        assert 'if [ "${line:0:2}" = "??" ]; then' in block, (
            f"the {span_name} dirt arm must branch on the ?? porcelain status prefix"
        )
        assert 'git -C "$WT" cat-file -e "origin/main:$p" 2>/dev/null && DIRT=yes' in block, (
            f"the {span_name} dirt arm's ?? branch must mark dirt ONLY on an "
            "origin/main path collision (cat-file -e existence probe)"
        )
        assert "p=${line:3}; p=${p%/}" in block, (
            f"the {span_name} dirt arm must strip the porcelain prefix + any "
            "trailing slash (a collapsed untracked dir cat-files the tree path)"
        )
        assert block.count("DIRT=yes") == 2, (
            f"the {span_name} dirt arm must be able to mark dirt from BOTH "
            "branches: the ?? collision branch AND the unconditional "
            "tracked-modified else-branch"
        )
        assert dirty_marking in block, (
            f"the {span_name} dirt arm must mark the file's family dirty "
            f"({dirty_marking} — the family-atomic skip)"
        )
        assert "carries UNCOMMITTED changes" in block, (
            f"the {span_name} dirt arm must announce the skip with the "
            "UNCOMMITTED-changes echo lead"
        )


# --- (16) sibling-issue per-FILE arm: Step 5a ONLY (#1972 pin) ---------------


def _sibling_arm_block(span: str) -> str:
    """Extract the #1972 sibling-issue file arm from the Step 5a span: from
    its comment lead through the closing [step5a] echo."""
    start = span.index("# Sibling-issue file freshness (#1972)")
    end = span.index("[step5a] sibling-file sync:", start)
    return span[start:end]


def test_sibling_issue_file_arm_step5a_only():
    """#1972 arm 3: the Step 5a block carries the per-FILE sibling-issue
    freshness arm — never-branch-edited sibling scripts/issue<M>_*.py AND
    their covering tests/test_issue<M>_*.py sync together as a PAIR (the
    arm's own sync commit puts the script into the selector's three-dot
    diff, newly mapping its covering test; syncing the script alone runs a
    fork-era test against a fresh script, the #1824/#1860 half-sync class).
    The Step 10d auto-merge inline copy deliberately does NOT carry the arm
    (the 10d TG legs run before the post-gate re-sync — syncing sibling
    files there moves the tip after certification for zero gate benefit);
    the negative assert anchors on the EXECUTABLE array-init fragment, so
    the 10d copy's prose asymmetry comment cannot trip it. #2116 widens the
    enumeration to sibling scripts/issue<M>_*.sh shell dispatchers: sibling
    tests also INVOKE dispatchers (subprocess / read_text), and a .py-only
    pathspec syncs the test without its .sh (the #1988/#2004 firings)."""
    text = _text()
    arm = _sibling_arm_block(_step5a_span(text))
    assert "':(glob)scripts/issue[0-9]*_*.py'" in arm, (
        "the sibling arm must enumerate sibling-issue scripts via the "
        "numeric-anchored :(glob)scripts/issue[0-9]*_*.py pathspec"
    )
    assert "':(glob)scripts/issue[0-9]*_*.sh'" in arm, (
        "the sibling arm must enumerate sibling-issue shell dispatchers via "
        "the numeric-anchored :(glob)scripts/issue[0-9]*_*.sh pathspec — "
        "sibling tests invoke sibling .sh dispatchers (subprocess / "
        "read_text), and a .py-only pathspec is the #1988/#2004 half-sync "
        "class: the covering test syncs while its dispatcher stays fork-era "
        "or absent (#2116)"
    )
    assert "':(glob)tests/test_issue[0-9]*_*.py'" in arm, (
        "the sibling arm must enumerate the covering tests via the paired "
        ":(glob)tests/test_issue[0-9]*_*.py pathspec (script+test move together)"
    )
    enum_lines = [ln for ln in arm.splitlines() if "diff --name-only origin/main" in ln]
    assert len(enum_lines) == 1, (
        "the sibling arm must carry exactly ONE `diff --name-only origin/main` "
        f"enumeration line (found {len(enum_lines)})"
    )
    for spec in (
        "':(glob)scripts/issue[0-9]*_*.py'",
        "':(glob)scripts/issue[0-9]*_*.sh'",
        "':(glob)tests/test_issue[0-9]*_*.py'",
    ):
        assert spec in enum_lines[0], (
            f"all three sibling pathspecs must co-occur on the enumeration line "
            f"itself (missing {spec}) — the individual substring asserts above "
            "would still pass if a glob moved into a comment while dropped from "
            "the `done < <(git ... diff --name-only origin/main ...)` line (#2116)"
        )
    assert "awk 'index($0, \"sync workflow-surface specs from\") == 0'" in arm, (
        "the sibling arm's branch-side-edit exclusion must reuse the "
        "subject-anchored awk index() form verbatim (the pass-1 / Guard-3 "
        "convention, #1789)"
    )
    assert "scripts/issue<N>_*|tests/test_issue<N>_*" in arm, (
        "the sibling arm must carve out the session's OWN issue scripts and "
        "tests (defense-in-depth beside the bs-commits exclusion)"
    )
    assert 'git -C "$WT" cat-file -e "origin/main:$f" 2>/dev/null' in arm, (
        "the sibling arm must guard the checkout on origin/main existence"
    )
    assert "never deleted" in arm, (
        "the sibling arm must skip files absent on origin/main (never delete)"
    )
    assert 'git -C "$WT" status --porcelain -- "$f" | grep -q .' in arm, (
        "the sibling arm must skip a file with ANY uncommitted dirt "
        "(per-file grain makes the wide skip free)"
    )
    subject = "sync workflow-surface specs from origin/main (spec-freshness; sibling-issue files)"
    assert subject in arm, (
        "the sibling arm's commit subject must carry the sync-subject anchor "
        "phrase (Guard 3 + the arm's own bs-check key on it) with the "
        "sibling-issue qualifier"
    )
    assert "SIBLING_SYNCED=()" not in _automerge_span(text), (
        "the Step 10d auto-merge inline copy must NOT carry the sibling-issue "
        "arm (deliberate asymmetry, #1972 — the 10d TG legs run before the "
        "post-gate re-sync; document it in prose, never mirror the executable arm)"
    )


def test_sibling_sync_import_probe_pins():
    """#2208: the sibling arm probes import-satisfiability of every synced
    sibling TEST file BEFORE the sync commit. The #2206 shape: a main-NEW
    sibling test imports a symbol added to src/ AFTER this branch's fork
    point; the worktree src is branch-era, so pytest COLLECTION ImportErrors
    in the Step 9c gate and `step9c_baseline.py compare` classifies the node
    NEW (fail-closed — the file IS branch-diff-touched via the sync commit,
    and the pristine oracle passes on main), walling the gate (~1h in #2206).
    Pins: the real collection probe command + its timeout fence, the venv
    warm-up outside the per-file fence, the skip-line anchors, the two revert
    branches (branch-era restore vs main-NEW drop), and the
    probe-before-commit ordering."""
    arm = _sibling_arm_block(_step5a_span(_text()))
    assert "pytest --collect-only -q" in arm, (
        "the sibling arm must probe synced test files with a REAL collection "
        "probe (pytest --collect-only -q) — a static module scan cannot see "
        "symbol-level src skew, the #2206 shape"
    )
    assert "timeout --kill-after=15s 180s" in arm, (
        "the per-file probe must be fenced (timeout --kill-after=15s 180s); "
        "a probe timeout counts as failure (fail-safe: revert to staleness)"
    )
    assert "timeout 900s uv run python -c pass" in arm, (
        "the arm must warm the worktree venv OUTSIDE the per-file fence (a "
        "fresh worktree pays a full uv sync on its first uv run, which would "
        "eat the 180s probe fence and revert legitimate syncs)"
    )
    assert "reverting its issue-" in arm, (
        "the probe-failure skip line must announce the pair-atomic revert "
        "(reverting its issue-<M> synced pair)"
    )
    assert "(#2208)" in arm, "the skip line must cite the fix task (#2208)"
    assert 'git -C "$WT" rm -f -q -- "$f"' in arm, (
        "the revert must handle the main-NEW shape (file absent from HEAD — "
        "created by the sync checkout, staged, uncommitted): drop it from "
        "index + working tree via git rm (`checkout HEAD --` would error "
        "there, and main-NEW is exactly the #2206 incident shape)"
    )
    assert 'cat-file -e "HEAD:$f"' in arm, (
        "the revert must branch on HEAD existence (branch-era file -> restore "
        "branch-era content; main-NEW file -> drop from index + tree)"
    )
    subject = "sync workflow-surface specs from origin/main (spec-freshness; sibling-issue files)"
    assert arm.index("pytest --collect-only -q") < arm.index(subject), (
        "the probe must run BEFORE the sync commit (nothing poisoned is ever "
        "committed; a post-commit probe would leave the poisoned file "
        "byte-identical to origin/main and never re-enumerated by the arm's "
        "diff on later rounds)"
    )


_SYNC_SUBJECT_2208 = (
    "sync workflow-surface specs from origin/main (spec-freshness; sibling-issue files)"
)


def _run_git(cwd: Path, *args: str, env: dict) -> str:
    """Run git in the scratch fixture (hermetic identity), failing loud."""
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.email=eps-test@example.com",
            "-c",
            "user.name=EPS Test",
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"git {args} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_sibling_sync_import_probe_repro_2206():
    """#2208 functional repro of the #2206 shape through the SHIPPED arm text.

    Scratch git fixture: origin/main advances past a branch's fork point with
    (i) a poisoned main-NEW tests/test_issue2038_p.py importing a symbol that
    main's src carries but the branch-era src copy lacks (the exact #2206
    symbol-skew shape — the sync deliberately never touches src/), and (ii) a
    legit import-satisfiable pair tests/test_issue1000_ok.py +
    scripts/issue1000_helper.py. The executed block is exactly what
    `_sibling_arm_block` returns (plus the completed closing echo line and the
    literal `<N>` substituted), run under bash with a PATH-shimmed `uv` that
    emulates `uv run pytest --collect-only -q <file>` by exec_module-ing the
    file (REAL import execution; the shim tolerates the warm-up call). Expect:
    the poisoned file reverted (absent from working tree AND index), the legit
    pair synced AND committed under the sync-anchor subject, the #2206/#2208
    skip line emitted, and the [step5a] echo reporting the post-revert count.
    """
    text = _text()
    span = _step5a_span(text)
    arm = _sibling_arm_block(span)
    # _sibling_arm_block's end anchor sits INSIDE the closing echo line (the
    # block ends with `echo "`); re-attach the remainder of that line so the
    # executed script is the shipped prose, count echo included.
    echo_start = span.index("[step5a] sibling-file sync:")
    echo_end = span.index("\n", echo_start)
    script_body = (arm + span[echo_start:echo_end]).replace("<N>", "9999")

    # mkdtemp, not tmp_path: concurrent pytest sessions prune /tmp/pytest-of*
    # numbered roots and can delete live scratch mid-test.
    tmp = Path(tempfile.mkdtemp(prefix="eps2208repro-"))
    try:
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"  # hermetic: no user/system git config
        env["GIT_CONFIG_NOSYSTEM"] = "1"

        origin = tmp / "origin.git"
        _run_git(tmp, "init", "--bare", "-b", "main", str(origin), env=env)

        seed = tmp / "seed"
        _run_git(tmp, "clone", str(origin), str(seed), env=env)
        # Branch-era state: the src module EXISTS but lacks the symbol.
        (seed / "src").mkdir()
        (seed / "src" / "issue2038_srcmod.py").write_text("BRANCH_ERA = True\n")
        _run_git(seed, "add", "src/issue2038_srcmod.py", env=env)
        _run_git(seed, "commit", "-m", "branch-era src", env=env)
        _run_git(seed, "push", "origin", "main", env=env)

        wt = tmp / "wt"
        _run_git(tmp, "clone", str(origin), str(wt), env=env)
        _run_git(wt, "checkout", "-b", "issue-9999", env=env)
        # The arm's own commit runs bare `git -C "$WT" commit`; give the
        # scratch clone a local identity (global config is /dev/null'd).
        _run_git(wt, "config", "user.email", "eps-test@example.com", env=env)
        _run_git(wt, "config", "user.name", "EPS Test", env=env)

        # Advance origin/main PAST the fork point: the poisoned main-NEW test
        # (its import is satisfiable only against main's src) + a legit pair.
        (seed / "src" / "issue2038_srcmod.py").write_text(
            "BRANCH_ERA = True\nSupersededSymbol = object()\n"
        )
        (seed / "tests").mkdir()
        (seed / "tests" / "test_issue2038_p.py").write_text(
            "from issue2038_srcmod import SupersededSymbol\n"
            "\n"
            "\n"
            "def test_symbol():\n"
            "    assert SupersededSymbol is not None\n"
        )
        (seed / "scripts").mkdir()
        (seed / "scripts" / "issue1000_helper.py").write_text("OK = 1\n")
        (seed / "tests" / "test_issue1000_ok.py").write_text("def test_ok():\n    assert True\n")
        _run_git(seed, "add", "-A", env=env)
        _run_git(seed, "commit", "-m", "main-side: poisoned test + legit pair", env=env)
        _run_git(seed, "push", "origin", "main", env=env)
        _run_git(wt, "fetch", "origin", env=env)

        # PATH-shimmed `uv`: the warm-up (`uv run python -c pass`) exits 0; the
        # collection probe exec_module's the target file with $WT/src on
        # sys.path — REAL import execution, hermetic (a scratch repo has no uv
        # project, so the real `uv run pytest` is environment-flaky here; the
        # production probe STRING is pinned by
        # test_sibling_sync_import_probe_pins).
        shim_dir = tmp / "bin"
        shim_dir.mkdir()
        collect_shim = shim_dir / "collect_shim.py"
        collect_shim.write_text(
            "import importlib.util\n"
            "import pathlib\n"
            "import sys\n"
            "\n"
            "target = pathlib.Path(sys.argv[1]).resolve()\n"
            'sys.path.insert(0, str(pathlib.Path.cwd() / "src"))\n'
            "spec = importlib.util.spec_from_file_location(target.stem, target)\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)  # raises on branch-era symbol skew\n"
        )
        uv_shim = shim_dir / "uv"
        uv_shim.write_text(
            "#!/usr/bin/env bash\n"
            "# Hermetic `uv` shim for the #2208 repro (see the test docstring).\n"
            'if [ "$1" = "run" ] && [ "$2" = "python" ]; then\n'
            "  exit 0\n"
            "fi\n"
            'if [ "$1" = "run" ] && [ "$2" = "pytest" ] && [ "$3" = "--collect-only" ] '
            '&& [ "$4" = "-q" ]; then\n'
            f'  exec "{sys.executable}" "{collect_shim}" "$5"\n'
            "fi\n"
            'echo "unexpected uv invocation: $*" >&2\n'
            "exit 97\n"
        )
        uv_shim.chmod(0o755)

        mb = _run_git(wt, "merge-base", "HEAD", "origin/main", env=env).strip()
        script = tmp / "arm.sh"
        script.write_text(script_body)
        env_arm = dict(env)
        env_arm["PATH"] = f"{shim_dir}:{env['PATH']}"
        env_arm["WT"] = str(wt)
        env_arm["MB"] = mb
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=tmp,
            env=env_arm,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"arm run failed:\n{proc.stdout}\n{proc.stderr}"
        out = proc.stdout

        # (1) the poisoned file is ABSENT from working tree AND index post-arm.
        assert not (wt / "tests" / "test_issue2038_p.py").exists(), (
            "the poisoned main-NEW test must be dropped from the working tree"
        )
        ls = _run_git(wt, "ls-files", "--", "tests/test_issue2038_p.py", env=env)
        assert ls.strip() == "", "the poisoned main-NEW test must be dropped from the index"
        # (2) the legit pair is synced AND committed under the sync-anchor subject.
        subj = _run_git(wt, "log", "-1", "--format=%s", env=env).strip()
        assert _SYNC_SUBJECT_2208 in subj, f"sync-anchor subject missing: {subj!r}"
        committed = _run_git(wt, "show", "--name-only", "--format=", "HEAD", env=env)
        assert "tests/test_issue1000_ok.py" in committed, "legit test must be committed"
        assert "scripts/issue1000_helper.py" in committed, "legit paired script must be committed"
        assert "test_issue2038_p.py" not in committed, "the poisoned file must never be committed"
        # (3) the skip line with its #2206/#2208 anchors was emitted.
        assert "reverting its issue-2038 synced pair (#2208)" in out, out
        assert "#2206" in out, out
        # (4) the [step5a] count echo reports the POST-revert survivor count.
        assert "[step5a] sibling-file sync: 2 file(s)" in out, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- (17) rc-checked sync commits in BOTH copies (#2303 static pins) ---------


def test_step5a_sync_commit_rc_checked():
    """#2303 defect 2: the Step 5a family sync must CHECK the sync commit's
    return code — a failed commit (crashed pre-commit hooks, the #2293
    shape) prints a FATAL line naming the staged paths and exits non-zero;
    the success echo fires only AFTER a verified commit and reports the
    committed sha (never an unconditional staged diffstat over a
    staged-but-uncommitted tree)."""
    span = _step5a_span(_text())
    rc_check = (
        'if ! git -C "$WT" commit -m "issue-<N>: sync workflow-surface specs '
        'from origin/main (spec-freshness)" -- $SAFE_SPECS; then'
    )
    assert rc_check in span, (
        "the Step 5a family sync commit must be rc-checked "
        "(`if ! git ... commit ...; then` — #2303 defect 2)"
    )
    fatal = "[step5a] FATAL: family sync commit FAILED (rc != 0)"
    assert fatal in span, (
        "a failed Step 5a sync commit must announce itself with the [step5a] FATAL echo (#2303)"
    )
    assert "git -C \"$WT\" diff --cached --name-only -- $SAFE_SPECS | sed 's/^/  /' >&2" in span, (
        "the FATAL arm must list the staged (failed) paths via "
        "`diff --cached --name-only` on stderr"
    )
    fatal_idx = span.index(fatal)
    committed_idx = span.index("SYNC_COMMITTED=yes")
    assert "exit 1" in span[fatal_idx:committed_idx], (
        "the FATAL arm must exit non-zero BEFORE the success path "
        "(a green success echo over a failed commit is the #2293 defect)"
    )
    assert fatal_idx < committed_idx, (
        "the rc-check FATAL arm must precede the SYNC_COMMITTED=yes success mark"
    )
    success = 'synced from origin/main: commit $(git -C "$WT" rev-parse --short=12 HEAD)'
    assert success in span, (
        "the success echo must report the COMMITTED sha (verified commit), "
        "never a bare staged diffstat (#2303)"
    )
    assert span.index(rc_check) < span.index(success), (
        "the success echo must come AFTER the rc-checked commit"
    )
    assert "no sync commit landed (no family drift, or the checkout errored above)" in span, (
        "the no-sync branch must name BOTH possible causes — no drift OR an "
        "errored (atomic, unmatched-pathspec) checkout — the R6 stale-fetch "
        "degrade prints a git error immediately above (#2303 v2 nit 1)"
    )


def test_step10d_sync_commit_rc_checked():
    """#2303 defect 2, Step 10d twin: SYNC_SHA must be read from `rev-parse
    HEAD` only AFTER a VERIFIED sync commit, and the push must follow the
    SYNC_SHA read — a failed commit aborts the whole merge invocation
    (FATAL + exit 1) BEFORE gh pr ready/merge, leaving the lint verdict
    file intact for the same-tip retry."""
    span = _automerge_span(_text())
    rc_check = (
        'if ! git -C "$WT" commit -m "issue-<N>: sync workflow-surface specs '
        'from origin/main (spec-freshness)" -- $SAFE_SPECS_10D; then'
    )
    assert rc_check in span, (
        "the Step 10d post-gate re-sync commit must be rc-checked "
        "(`if ! git ... commit ...; then` — #2303 defect 2)"
    )
    fatal = "[step10d] FATAL: post-gate re-sync commit FAILED (rc != 0)"
    assert fatal in span, (
        "a failed 10d re-sync commit must announce itself with the "
        "[step10d] FATAL echo and abort the merge attempt (#2303)"
    )
    assert (
        "git -C \"$WT\" diff --cached --name-only -- $SAFE_SPECS_10D | sed 's/^/  /' >&2" in span
    ), "the 10d FATAL arm must list the staged (failed) paths on stderr"
    sync_sha_read = 'SYNC_SHA=$(git -C "$WT" rev-parse HEAD | head -c 12)'
    assert sync_sha_read in span, (
        "the 10d twin must still read SYNC_SHA from rev-parse HEAD "
        "(now only after a verified commit)"
    )
    fatal_idx = span.index(fatal)
    sha_idx = span.index(sync_sha_read)
    assert "exit 1" in span[fatal_idx:sha_idx], (
        "the 10d FATAL arm must exit non-zero BEFORE the SYNC_SHA read — a "
        "failed commit previously yielded the PRE-commit sha, then pushed, "
        "then fed the verdict re-bind stanza (#2303 defect 2)"
    )
    assert fatal_idx < sha_idx, (
        "ordering must be FATAL rc-check < SYNC_SHA read (SYNC_SHA is read "
        "only after a VERIFIED commit)"
    )
    # The stanza's own push (searched AFTER the SYNC_SHA read — the span
    # carries an unrelated earlier `push origin issue-<N>` in the pre-PR
    # section) must sit between the SYNC_SHA read and the SYNC_COUNT echo:
    # the push ships only a verified sync commit.
    push_idx = span.index("push origin issue-<N>", sha_idx)
    count_idx = span.index("SYNC_COUNT=", sha_idx)
    assert sha_idx < push_idx < count_idx, (
        "the stanza's push must follow the verified-commit SYNC_SHA read "
        "and precede the SYNC_COUNT echo (never push an unverified sync)"
    )


# --- (18) behavioral repros of both #2293 defect shapes (#2303) --------------


def _family_arm_block(span: str) -> str:
    """Extract the Step 5a FAMILY arm from the Step 5a span: FAMILY_OF +
    SPECS + the bounded fetch + MB derivation + pass 1 (incl. the #1972
    dirt arm) + pass 2 + the rc-checked sync stanza + the observability
    echo — everything from the family declaration up to the sibling-issue
    arm. The extracted text is the SHIPPED block, executable under bash
    with $WT supplied via env (same convention as the #2208 repro)."""
    start = span.index("declare -A FAMILY_OF")
    end = span.index("# Sibling-issue file freshness (#1972)", start)
    return span[start:end]


# Post-#2303 SPECS pathspecs, one fork-era stub file per pathspec EXCEPT the
# caps file: `git checkout <ref> -- <pathspecs>` is ATOMIC and exits non-zero
# checking out NOTHING when any pathspec matches nothing at the ref, so every
# other member must resolve at origin/main. The caps file is deliberately
# main-NEW (added only in the advance commit) — the exact #2293 topology.
_FORK_STUBS_2303 = (
    ".claude/agents/x.md",
    ".claude/agent-memory/x/MEMORY.md",
    ".claude/skills/issue/SKILL.md",
    ".claude/rules/x.md",
    ".claude/workflow.yaml",
    "CLAUDE.md",
    "scripts/workflow_lint.py",
    "scripts/select_step9c_tests.py",
    ".claude/hooks/x.sh",
    "scripts/guard_x.sh",
    "tests/test_guard_lessons_edit.py",
    "tests/test_workflow_yaml.py",
    "tests/test_autonomous_session_watch.py",
    "tests/test_select_step9c_tests.py",
    "tests/step9c_workflow_invariant_manifest.txt",
    "tests/test_workflow_lint_x.py",
    "tests/test_guard_x.py",
    "tests/issue_skill_source.py",
    "tests/test_issue_skill_x.py",
)

_SYNC_SUBJECT_2303 = "issue-9999: sync workflow-surface specs from origin/main (spec-freshness)"


def _family_sync_fixture(tmp: Path, env: dict) -> Path:
    """Scratch bare origin + a wt clone on issue-9999 whose origin/main has
    advanced past the fork point by a scripts/workflow_lint.py edit + the
    main-NEW .claude/config/agent_spec_size_caps.txt (the #2293 topology).
    Returns the wt path; the wt has already fetched origin."""
    origin = tmp / "origin.git"
    _run_git(tmp, "init", "--bare", "-b", "main", str(origin), env=env)
    seed = tmp / "seed"
    _run_git(tmp, "clone", str(origin), str(seed), env=env)
    for rel in _FORK_STUBS_2303:
        p = seed / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"fork-era stub: {rel}\n")
    _run_git(seed, "add", "-A", env=env)
    _run_git(seed, "commit", "-m", "fork-era stubs", env=env)
    _run_git(seed, "push", "origin", "main", env=env)

    wt = tmp / "wt"
    _run_git(tmp, "clone", str(origin), str(wt), env=env)
    _run_git(wt, "checkout", "-b", "issue-9999", env=env)
    # The arm's own commit runs bare `git -C "$WT" commit`; give the scratch
    # clone a local identity (global config is /dev/null'd).
    _run_git(wt, "config", "user.email", "eps-test@example.com", env=env)
    _run_git(wt, "config", "user.name", "EPS Test", env=env)

    # Advance origin/main PAST the fork point: modify the linter + ADD its
    # import-time data file (the pair the family sync must carry together).
    (seed / "scripts" / "workflow_lint.py").write_text(
        "fork-era stub: scripts/workflow_lint.py\nMAIN_SIDE_FIX = True\n"
    )
    (seed / ".claude" / "config").mkdir(parents=True)
    (seed / ".claude" / "config" / "agent_spec_size_caps.txt").write_text("x.md 1_000\n")
    _run_git(seed, "add", "-A", env=env)
    _run_git(seed, "commit", "-m", "main-side: linter fix + main-NEW caps data file", env=env)
    _run_git(seed, "push", "origin", "main", env=env)
    _run_git(wt, "fetch", "origin", env=env)
    return wt


def test_family_sync_data_dep_repro_2303():
    """#2303 defect 1 repro (shape a — the #2293 strand) through the SHIPPED
    family arm under real git: origin/main advances with a linter edit + the
    main-NEW caps data file; the family sync must carry BOTH in, in ONE sync
    commit, and the success echo must report the verified commit's sha.
    Against the pre-fix block (no caps token in SPECS) the caps file never
    reaches the worktree — the synced linter would raise FileNotFoundError
    at import in every hook that shells it."""
    text = _text()
    script_body = _family_arm_block(_step5a_span(text)).replace("<N>", "9999")

    # mkdtemp, not tmp_path: concurrent pytest sessions prune /tmp/pytest-of*
    # numbered roots and can delete live scratch mid-test.
    tmp = Path(tempfile.mkdtemp(prefix="eps2303repro-"))
    try:
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"  # hermetic: no user/system git config
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        wt = _family_sync_fixture(tmp, env)

        script = tmp / "familyarm.sh"
        script.write_text(script_body)
        env_arm = dict(env)
        env_arm["WT"] = str(wt)
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=tmp,
            env=env_arm,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"family arm failed:\n{proc.stdout}\n{proc.stderr}"

        # (1) the linter's import-time data file synced in WITH the linter.
        assert (wt / ".claude" / "config" / "agent_spec_size_caps.txt").exists(), (
            "the caps data file must sync in pair-atomically with "
            "scripts/workflow_lint.py — a synced linter without it raises "
            "FileNotFoundError in every hook that shells it (#2293)"
        )
        # (2) the pair landed in ONE sync commit under the sync-anchor subject.
        subj = _run_git(wt, "log", "-1", "--format=%s", env=env).strip()
        assert subj == _SYNC_SUBJECT_2303, f"sync-anchor subject missing: {subj!r}"
        committed = _run_git(wt, "show", "--name-only", "--format=", "HEAD", env=env)
        assert "scripts/workflow_lint.py" in committed, "the linter must be in the sync commit"
        assert ".claude/config/agent_spec_size_caps.txt" in committed, (
            "the caps data file must be in the SAME sync commit as the linter"
        )
        # (3) the success echo reports the verified commit's sha.
        short = _run_git(wt, "rev-parse", "--short=12", "HEAD", env=env).strip()
        assert f"[step5a] synced from origin/main: commit {short}" in proc.stdout, proc.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_family_sync_commit_failure_fatal_repro_2303():
    """#2303 defect 2 repro (shape b) through the SHIPPED family arm under
    real git: a failing pre-commit hook (the faithful rc-level proxy for the
    incident's three crashed linter-shelling hooks) makes the sync commit
    fail — the block must exit non-zero with a FATAL stderr line naming the
    staged paths, print NO success line, and leave the synced set staged
    (uncommitted) for inspection. Against the pre-fix block the commit
    failure was swallowed: the success diffstat printed and the block
    exited 0 over a staged-but-uncommitted tree (the #2293 defect)."""
    text = _text()
    script_body = _family_arm_block(_step5a_span(text)).replace("<N>", "9999")

    tmp = Path(tempfile.mkdtemp(prefix="eps2303fatal-"))
    try:
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        wt = _family_sync_fixture(tmp, env)

        # Failing hook: rc-level proxy for the #2293 crashed pre-commit hooks.
        hooks = wt / ".git" / "hooks"
        hooks.mkdir(parents=True, exist_ok=True)
        hook = hooks / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        script = tmp / "familyarm.sh"
        script.write_text(script_body)
        env_arm = dict(env)
        env_arm["WT"] = str(wt)
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=tmp,
            env=env_arm,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode != 0, (
            "a failed sync commit must exit the block non-zero — pre-fix it "
            f"exited 0 over a staged-but-uncommitted tree (#2293)\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
        assert "FATAL" in proc.stderr, f"the failure must be a FATAL stderr line:\n{proc.stderr}"
        assert "scripts/workflow_lint.py" in proc.stderr, (
            "the FATAL arm must name the staged (failed) paths — "
            f"scripts/workflow_lint.py missing from:\n{proc.stderr}"
        )
        combined = proc.stdout + proc.stderr
        assert "[step5a] synced from origin/main:" not in combined, (
            "NO success line may print on a failed sync commit (#2293 defect 2)"
        )
        staged = _run_git(wt, "diff", "--cached", "--name-only", env=env)
        assert staged.strip() != "", "the synced set must be left STAGED for inspection"
        subj = _run_git(wt, "log", "-1", "--format=%s", env=env).strip()
        assert subj != _SYNC_SUBJECT_2303, "no sync commit may land when the hook fails"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- (19) family-synced test files' tests.<mod> imports are sync-coverable (#2352) --

_REPO = Path(__file__).resolve().parents[1]

# Helper modules a family-synced test may import WITHOUT a SPECS token, each
# with a documented rationale. conftest: tests/test_autonomous_session_watch.py
# imports two tests.conftest symbols, but conftest.py is pytest-auto-loaded by
# the WHOLE tests tree and pairs with the branch's OWN tests — syncing it from
# main is exactly the "blind-syncing broader tests/ is actively unsafe" case
# the boundary paragraph bans, so it is a NAMED accepted seam (same remedy
# class as the src seams: a skew fails loud at collection; rebase onto
# origin/main, or cross-check at the repo root).
_EXEMPT_HELPER_MODULES = frozenset({"conftest"})


def _specs_tokens(span: str) -> list[str]:
    """The Step 5a SPECS pathspec tokens (test (14)'s extraction)."""
    m = re.search(r'^\s*SPECS="([^"]+)"', span, flags=re.M)
    assert m, "Step 5a must declare SPECS as a one-line double-quoted assignment"
    return m.group(1).split()


def _family_of_map(span: str) -> dict[str, str]:
    """Every FAMILY_OF["<path>"]="<fam>" assignment in the span, keyed by
    path. Comment-tolerant: only the line prefix through the closing quote
    is matched, so a trailing `# ...` comment cannot hide an assignment."""
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r'^\s*FAMILY_OF\["([^"]+)"\]="(\w+)"', span, flags=re.M)
    }


def _family_synced_test_files(tokens: list[str], family_of: dict[str, str]) -> dict[Path, set[str]]:
    """Enumerate the family-synced tests/*.py files mechanically from the
    SPECS tokens: every existing repo file matched by a `:(glob)tests/...`
    token plus every explicit `tests/*.py` token. Maps each file to the set
    of FAMILIES of its covering tokens (a singleton token — no FAMILY_OF
    entry — contributes no family)."""
    out: dict[Path, set[str]] = {}
    for tok in tokens:
        if tok.startswith(":(glob)"):
            pattern = tok[len(":(glob)") :]
            if not pattern.startswith("tests/"):
                continue
            fam = family_of.get(tok)
            for p in sorted(_REPO.glob(pattern)):
                if p.suffix == ".py" and p.is_file():
                    fams = out.setdefault(p, set())
                    if fam is not None:
                        fams.add(fam)
        elif tok.startswith("tests/") and tok.endswith(".py"):
            p = _REPO / tok
            if p.is_file():
                fams = out.setdefault(p, set())
                fam = family_of.get(tok)
                if fam is not None:
                    fams.add(fam)
    return out


def _tests_module_imports(src: str) -> set[str]:
    """Top-level `tests.<mod>` imports in a test file's source, via three
    line-anchored regexes: `from tests.<mod> import ...`,
    `import tests.<mod>`, and the names-list form
    `from tests import a, b as c` (comma-split, `as`-alias stripped,
    parens tolerated)."""
    mods: set[str] = set()
    for m in re.finditer(r"^\s*from\s+tests\.(\w+)\s+import\b", src, flags=re.M):
        mods.add(m.group(1))
    for m in re.finditer(r"^\s*import\s+tests\.(\w+)", src, flags=re.M):
        mods.add(m.group(1))
    for m in re.finditer(r"^\s*from\s+tests\s+import\s+(.+)$", src, flags=re.M):
        names = m.group(1).split("#")[0].replace("(", " ").replace(")", " ")
        for name in names.split(","):
            name = name.strip().split(" as ")[0].strip()
            if name:
                mods.add(name)
    return mods


def _import_covered(
    mod: str,
    importer_families: set[str],
    tokens: list[str],
    family_of: dict[str, str],
) -> bool:
    """True when tests/<mod>.py is sync-coverable for an importer of the
    given families: (i) matched by a SPECS glob of the SAME family as the
    importer, (ii) an explicit SPECS token assigned to the SAME family, or
    (iii) a SINGLETON SPECS token (no FAMILY_OF entry — always synced when
    itself clean, so no OTHER file's dirt can family-skip it)."""
    target = f"tests/{mod}.py"
    for tok in tokens:
        if tok.startswith(":(glob)"):
            if not fnmatch.fnmatch(target, tok[len(":(glob)") :]):
                continue
            fam = family_of.get(tok)
            if fam is None or fam in importer_families:
                return True
        elif tok == target:
            fam = family_of.get(tok)
            if fam is None or fam in importer_families:
                return True
    return False


def test_family_synced_test_helper_imports_covered():
    """#2352 forward guard: every `tests.<mod>` import in any FAMILY-SYNCED
    test file must be sync-coverable — same-family glob, same-family
    explicit token, or SINGLETON token — so the NEXT main-side helper
    module a family-synced test imports makes THIS suite red on main (the
    skew is unshippable) instead of red-ing a worktree's Step 9c gate with
    a ModuleNotFoundError half-sync (the #2352 incident: 66 collection
    errors in the issue-2333 worktree). Cross-family coverage is
    deliberately NOT sufficient: families dirty-skip independently, so a
    helper reachable only through ANOTHER family's token can still be
    skipped while the importer syncs fresh.

    Known false negatives, disclosed by design: dynamic/importlib imports
    (no static `tests.` import line to match) and multi-level
    `tests.a.b` sub-packages (none exist today — `(\\w+)` matches only the
    first segment) escape the scan; the Step 9c gate remains the runtime
    backstop for those."""
    text = _text()
    span = _step5a_span(text)
    tokens = _specs_tokens(span)
    family_of = _family_of_map(span)
    files = _family_synced_test_files(tokens, family_of)
    # Enumeration sanity: the skill-pin glob alone matches dozens of files;
    # an empty/thin enumeration means the extraction broke, not a clean tree.
    assert len(files) >= 10, (
        f"family-synced test-file enumeration looks broken: only "
        f"{len(files)} files resolved from the SPECS tokens {tokens!r}"
    )
    problems: list[str] = []
    for path in sorted(files):
        importer_families = files[path]
        src = path.read_text(encoding="utf-8")
        for mod in sorted(_tests_module_imports(src)):
            if mod in _EXEMPT_HELPER_MODULES:
                continue
            if not _import_covered(mod, importer_families, tokens, family_of):
                problems.append(
                    f"{path.relative_to(_REPO).as_posix()} "
                    f"(families: {sorted(importer_families) or 'singleton'}) "
                    f"imports tests.{mod}"
                )
    assert not problems, (
        "family-synced test file(s) import a tests.<mod> helper NO Step 5a "
        "SPECS token can sync alongside them — the Step 5a family sync can "
        "pull these tests into a worktree WITHOUT the helper (the #2352 "
        "ModuleNotFoundError half-sync; 66 collection errors). Remedies: "
        "(a) add tests/<mod>.py to SPECS + SPECS_10D as a SINGLETON token "
        "(the #2352 disposition — right when the helper has importers in "
        "more than one family or in unsynced tests), (b) add it as an "
        "explicit token in the importer's OWN family (right only when "
        "every importer shares that family), or (c) add the module to "
        "_EXEMPT_HELPER_MODULES with a documented rationale (the conftest "
        "shape). Uncovered imports:\n  " + "\n  ".join(problems)
    )
