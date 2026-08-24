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
family-synced test file (imports collected via ast.parse — parenthesized
multiline, comma-separated, and aliased forms included; string literals
invisible) must be sync-coverable on EVERY route through which the
importer can arrive fresh — per route a same-family token/glob, or
route-independently the helper itself a SINGLETON token; each matching
SPECS token is an independent per-token sync channel, so
partial-route/cross-family coverage is insufficient — and the NEXT
main-side helper module a family-synced test imports therefore reds THIS
suite on main instead of red-ing a worktree gate. A runtime import-satisfiability
probe on the FAMILY arm (the #2208 sibling-arm probe's shape) was
considered and DEFERRED: the family sync checkouts+commits atomically
across ALL safe families in one command inside two mirrored fail-closed
blocks — a probe-failure revert would have to unwind whole families
(SKILL.md included) and a bug there reds every Step 5a run and every
merge fleet-wide, while the static guard catches the class EARLIER (on
main, at the PR adding the import).

#2260 (2026-08-21) adds the AGENTS prose family: `.claude/agents` + 30
vetted closure-clean agents-prose pin tests join SPECS + SPECS_10D as
FAMILY_agents members (FAMILY_OF entries in BOTH copies), and the
workflow-family cross-reader
tests/test_inline_payload_lint_gate_contract.py joins "workflow" (its
tests.test_issue_skill_inline_gate_pin import forces same-family
admission per guard (19)). Incident #2251: main removed a planner.md
row + its pinning test together; the branch-era test red the
freshly-synced planner.md — a 74-min gate red. Membership is VETTED,
never name-globbed: readers use the literal ".claude/agents" form AND
the quoted path-join form (`/ ".claude" / "agents" /`), and only
closure-clean prose pins join (stdlib / env packages /
tests/issue_skill_source.py / SPECS-synced files only) — behavioral
tests importing unsynced scripts/src stay OUT. A member-existence
containment arm in pass 1 of BOTH copies keeps the ATOMIC checkout
from wedging every family when a literal member is deleted/renamed on
main (the token's family is marked dirty + loud echo; deletion
propagation stays #2385). Tests (1)/(9)/(10) gain the members; a
containment-arm textual pin + a deleted-member functional repro join
section (18); section (20) adds guard (20): the reader-predicate
completeness check (predicate universe minus glob-covered minus
explicitly-familied minus exempt must be EMPTY), existence asserts
with a merge-base vintage-skew discriminator (absent-on-disk +
absent-at-merge-base => SKIP, pure vintage skew; present-at-merge-base
+ absent-on-disk => FAIL, the deleting-PR early warning), a
genuine/incidental exempt split with an AST shape check on the
incidental class, and non-vacuity pins.

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
either copy), lets a family-synced test file import a tests.<mod>
helper not sync-coverable on EVERY of its sync routes (guard 19),
drops a #2260 FAMILY_agents member or the member-existence containment
arm from either copy, lets a new agents-prose reader land
undispositioned, or lets an incidental agents exemption acquire a
genuine agents-path read construct (guard 20).

NOTE for future SKILL.md editors: these assertions pin literal snippet text.
A legitimate rewording of the pinned lines in SKILL.md must update the
matching assertions here IN THE SAME COMMIT, or the suite goes red.
"""

import ast
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

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
        ":(glob)tests/test_issue_skill_*.py "
        "scripts/step5a_sibling_probe.py "
        "tests/test_step5a_sibling_probe.py "
        "tests/test_adversarial_planner_factchecker_grain_pin.py "
        "tests/test_adversarial_planner_lens_brief_headings.py "
        "tests/test_analyzer_language_intrusion_duty.py "
        "tests/test_battery_basis_prose_pins.py "
        "tests/test_code_reviewer_phase_idempotency_gate.py "
        "tests/test_codex_code_reviewer_step09_tag_parity.py "
        "tests/test_codex_critic_numeric_grounding.py "
        "tests/test_consistency_checker_parentless_infra_skip.py "
        "tests/test_cross_issue_protocol_comparability_prose.py "
        "tests/test_daily_three_route_classifier_doc.py "
        "tests/test_diff_base_origin_main_pin.py "
        "tests/test_downwidth_split_prose_pins.py "
        "tests/test_experimenter_md.py "
        "tests/test_fit_loop_batching_review_pin.py "
        "tests/test_implementer_spec_deleted_literal_substep.py "
        "tests/test_implementer_spec_mechanical_pin_sweep.py "
        "tests/test_implementer_spec_names_invariant_local_union.py "
        "tests/test_implementer_spec_names_ruff_policy_pin.py "
        "tests/test_inline_payload_lint_gate_contract.py "
        "tests/test_interp_critic_degenerate_series_lens.py "
        "tests/test_issue_v2_skill_figure_pin_contract.py "
        "tests/test_lean_twin_registration_pin.py "
        "tests/test_mapping_baselines_wiring_pins.py "
        "tests/test_off_pod_phase_slot_pin.py "
        "tests/test_outroot_residue_prose_pins.py "
        "tests/test_plan_handoff_path_convention.py "
        "tests/test_planner_incident_trace_guidance.py "
        "tests/test_planner_phase_outputs_declaration.py "
        "tests/test_realized_rows_prose_pins.py "
        "tests/test_selection_symmetric_nulls_pointers.py "
        'tests/test_v2_composer_plan_path_brief.py"'
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
        "66 collection errors in the issue-2333 worktree) — plus the #2412 "
        "sibling-probe pair scripts/step5a_sibling_probe.py + "
        "tests/test_step5a_sibling_probe.py (the Step 5a "
        "import-satisfiability helper and its unit tests; the "
        "test_issue_skill_* repros execute the worktree helper copy, so "
        "helper-vs-pin sync must be family-atomic — the #1963 precedent) — "
        "plus the #2260 FAMILY_agents members (vetted closure-clean "
        "prose-pin tests over `.claude/agents/*.md` outside the coupled "
        "globs, literal-form AND quoted path-join-form readers alike; the "
        "#2251 half-sync: fresh planner.md vs branch-era pin test, 74-min "
        "gate red) and the workflow-family cross-reader "
        "tests/test_inline_payload_lint_gate_contract.py (its helper import "
        "forces same-family admission per guard (19))"
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
    assert 'FAMILY_OF[".claude/agents"]="agents"' in span, (
        "the agents family must include the .claude/agents dir itself — the "
        "#2260 FAMILY_agents coupling: refreshing agents prose without its "
        "vetted pin tests reds the Step 9c gate on pure vintage skew (#2251: "
        "main removed a planner.md row + its pinning test together; the "
        "branch-era test red the freshly-synced planner.md, a 74-min gate red)"
    )
    assert 'FAMILY_OF["tests/test_mapping_baselines_wiring_pins.py"]="agents"' in span, (
        "the agents family must include tests/test_mapping_baselines_wiring_pins.py "
        "— the #2251 incident file (a LITERAL-form agents reader; #2260)"
    )
    assert 'FAMILY_OF["tests/test_planner_phase_outputs_declaration.py"]="agents"' in span, (
        "the agents family must include tests/test_planner_phase_outputs_declaration.py "
        "— a quoted path-join-form agents reader, so the join class is "
        "represented in the presence pins (#2260)"
    )
    assert 'FAMILY_OF["tests/test_inline_payload_lint_gate_contract.py"]="workflow"' in span, (
        "the workflow family must include the #2260 cross-family reader "
        "tests/test_inline_payload_lint_gate_contract.py — its "
        "tests.test_issue_skill_inline_gate_pin import forces same-family "
        "admission (guard (19) universal-route coverage); assigning it "
        '"agents" would red guard (19)'
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
            ('="workflow"', '="lint"', '="guard"', '="agents"')
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
    pathspec syncs the test without its .sh (the #1988/#2004 firings).
    #2412 widens the pair to issue-namespaced src — the
    src/explore_persona_space/experiments/issue<M> / issue_<M> dirs join
    the globs (a synced main-NEW test importing fork-era issue src is the
    #2204 post-collection escape), with the own-issue carve-out extended
    to match."""
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
    assert "':(glob)src/explore_persona_space/experiments/issue[0-9]*/**'" in arm, (
        "the sibling arm must enumerate issue-namespaced sibling src dirs "
        "(the issue<N> convention) — the #2412 closure widening: syncing a "
        "main-NEW test without its issue-namespaced src re-creates the "
        "#2204 half-sync (main-era test + fork-era issue src)"
    )
    assert "':(glob)src/explore_persona_space/experiments/issue_[0-9]*/**'" in arm, (
        "the sibling arm must enumerate issue-namespaced sibling src dirs "
        "(the issue_<N> convention) — the #2412 closure widening's second "
        "measured dir convention (e.g. issue_1739, the #2204 incident dir)"
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
        "':(glob)src/explore_persona_space/experiments/issue[0-9]*/**'",
        "':(glob)src/explore_persona_space/experiments/issue_[0-9]*/**'",
    ):
        assert spec in enum_lines[0], (
            f"all five sibling pathspecs must co-occur on the enumeration line "
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
    assert (
        "src/explore_persona_space/experiments/issue<N>/*"
        "|src/explore_persona_space/experiments/issue_<N>/*"
    ) in arm, (
        "the own-issue carve-out must extend to the session's OWN "
        "issue-namespaced src dirs (both conventions) — the #2412 widening "
        "must never sync the branch's own experiment src back to main tip"
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


_SIBLING_PROBE_HELPER = Path(__file__).resolve().parents[1] / "scripts" / "step5a_sibling_probe.py"


def _executable_only(source: str) -> str:
    """Source with comments + docstrings stripped (ast round-trip).

    A string pin on the returned text can only be satisfied by EXECUTABLE
    code — prose in a module/function docstring (or a comment) cannot shadow
    a removed call (#2412 r2 NIT process-fence-pin-docstring-shadow). NOTE:
    ``ast.unparse`` renders string literals with SINGLE quotes — pins that
    match verbatim double-quoted source text must keep asserting on the raw
    source instead.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_sibling_sync_import_probe_pins():
    """#2208, hardened #2412: the sibling arm probes import-satisfiability of
    every synced sibling TEST file BEFORE the sync commit, via
    scripts/step5a_sibling_probe.py resolved from the MAIN checkout
    (git-common-dir — the worktree copy is fork-era by construction, the
    very staleness class being probed).

    Arm-side pins: the helper invocation + ROOT resolution + the
    probe-before-commit ordering, and the MF2 crash branch — helper
    rc != 0 reverts EVERY synced path via the checkout-HEAD /
    rm --ignore-unmatch loop INSIDE the else branch, then FATAL + exit 1,
    all BEFORE the commit subject (index-based ordering asserts).

    Helper-side pins (the #2412 durability half): the retained #2208
    collection-probe values (`--collect-only -q`, 180 s fence, 900 s
    warm-up), the MF1 process-GROUP kill fence (start_new_session=True +
    os.killpg + SIGKILL escalation + --kill-after default 15 — a later
    refactor must not silently swap in a naive child-only kill: pytest
    grandchildren inherit the stdout pipe and the post-kill communicate()
    HANGS instead of reverting), the verbatim #2208 revert branches, the
    skip-line anchors, and the #2412 issue-namespaced classifier
    (issue_?\\d+ fullmatch + the experiments-scoped trailing-_\\d+ arm +
    the DIRECTORY-grain strict diff, MF3)."""
    arm = _sibling_arm_block(_step5a_span(_text()))
    helper = _SIBLING_PROBE_HELPER.read_text()

    # --- arm-side: invocation + ROOT resolution ---------------------------
    # The pinned literal INCLUDES the `(cd "$ROOT" && ...)` subshell linkage:
    # pinning the invocation and the ROOT line as independent substrings lets
    # a refactor drop the cd and run the fork-era WORKTREE helper copy while
    # both pins still pass — scratch fixtures cannot distinguish, since a
    # standalone clone's git-common-dir resolves to itself, ROOT == WT there
    # (#2412 r2 pin-cd-root-linkage-unpinned).
    invocation = (
        '(cd "$ROOT" && uv run python scripts/step5a_sibling_probe.py '
        '--worktree "$WT" --kept-out "$KEPT_OUT" -- "${SIBLING_SYNCED[@]}")'
    )
    assert invocation in arm, (
        "the sibling arm must invoke the factored probe helper on the synced "
        'set FROM THE MAIN CHECKOUT — inside the `(cd "$ROOT" && ...)` '
        "subshell (kept-list plumbed back via --kept-out; #2412)"
    )
    root = 'ROOT="$(dirname "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)")"'
    assert root in arm, (
        "the helper must resolve from the MAIN checkout via git-common-dir — "
        "a relative scripts/ path executes the branch's fork-era helper copy, "
        "the exact staleness class this probe exists to catch (#2412)"
    )
    assert 'mapfile -t SIBLING_SYNCED < "$KEPT_OUT"' in arm, (
        "the success branch must rebind SIBLING_SYNCED to the helper's kept "
        "list (survivors only — reverted issues must not reach the commit)"
    )

    # --- arm-side: the MF2 crash branch (full revert, FATAL, exit 1) ------
    fatal = "[step5a] FATAL: sibling probe helper failed (rc != 0)"
    assert fatal in arm, (
        "a helper crash (or ABSENT helper — uv run python exits nonzero on a "
        "missing script, N1) must announce itself with the [step5a] FATAL "
        "echo (#2412 MF2)"
    )
    inv_idx = arm.index(invocation)
    mapfile_idx = arm.index('mapfile -t SIBLING_SYNCED < "$KEPT_OUT"')
    loop_idx = arm.index('for f in "${SIBLING_SYNCED[@]}"; do', inv_idx)
    checkout_idx = arm.index('git -C "$WT" checkout HEAD -- "$f"', loop_idx)
    rm_idx = arm.index('git -C "$WT" rm -f -q --ignore-unmatch -- "$f"', loop_idx)
    assert 'cat-file -e "HEAD:$f"' in arm[loop_idx:], (
        "the crash-branch revert must branch on HEAD existence (branch-era "
        "file -> restore branch-era content; main-NEW file -> drop from "
        "index + tree)"
    )
    fatal_idx = arm.index(fatal)
    exit_idx = arm.index("exit 1", fatal_idx)
    subject = "sync workflow-surface specs from origin/main (spec-freshness; sibling-issue files)"
    subject_idx = arm.index(subject)
    assert inv_idx < mapfile_idx < loop_idx < checkout_idx < rm_idx < fatal_idx < exit_idx, (
        "the crash branch must run the FULL revert loop (checkout-HEAD + "
        "rm --ignore-unmatch, INSIDE the else branch after the success "
        "branch's mapfile) BEFORE the FATAL echo + exit 1 — v2's "
        "leave-staged-for-inspection shape inverted the fail-safe: the "
        "#1972 dirt arm reads staged synced files as dirt and preserves "
        "them into every later round (#2412 MF2)"
    )
    assert exit_idx < subject_idx, (
        "the crash branch's exit 1 must precede the sync-commit subject — "
        "nothing unprobed is ever committed (probe-before-commit, #2208)"
    )
    assert inv_idx < subject_idx, (
        "the probe must run BEFORE the sync commit (nothing poisoned is ever "
        "committed; a post-commit probe would leave the poisoned file "
        "byte-identical to origin/main and never re-enumerated by the arm's "
        "diff on later rounds)"
    )
    assert "--ignore-unmatch" in arm, (
        "the bash crash-branch rm must carry --ignore-unmatch (idempotent "
        "under partial helper-side reverts — a bare git rm rc-fails on the "
        "already-dropped pathspec and aborts the loop mid-revert)"
    )

    # --- helper-side: retained #2208 probe values -------------------------
    assert 'parser.add_argument("--collect-cmd", default="uv run pytest --collect-only -q")' in (
        helper
    ), (
        "the helper must default to the REAL collection probe "
        "(uv run pytest --collect-only -q) — defaults ARE the pinned "
        "production values (#2208 retained)"
    )
    assert 'parser.add_argument("--collect-timeout", type=float, default=180)' in helper, (
        "the per-file probe fence must default to 180 s (verbatim #2208 "
        "value; a probe timeout counts as failure — fail-safe revert)"
    )
    assert 'parser.add_argument("--warmup-timeout", type=float, default=900)' in helper, (
        "the venv warm-up must default to 900 s OUTSIDE the per-file fence "
        "(a fresh worktree pays a full uv sync on its first uv run, which "
        "would eat the 180 s probe fence and revert legitimate syncs)"
    )
    assert 'parser.add_argument("--warmup-cmd", default="uv run python -c pass")' in helper, (
        "the warm-up command must stay `uv run python -c pass` (#2208)"
    )

    # --- helper-side: the MF1 process-GROUP kill fence ---------------------
    # The fence pins assert on EXECUTABLE code only (docstrings + comments
    # stripped): the helper's module docstring quotes the same literals as
    # prose, so a whole-file substring pin would survive a refactor that
    # removes the killpg calls while keeping the documentation (#2412 r2 NIT
    # process-fence-pin-docstring-shadow).
    helper_code = _executable_only(helper)
    assert 'parser.add_argument("--kill-after", type=float, default=15)' in helper, (
        "the SIGKILL escalation delay must default to 15 s (migrating the "
        "retired arm's `timeout --kill-after=15s`; #2412 MF1)"
    )
    assert "start_new_session=True" in helper_code, (
        "probe subprocesses must run with start_new_session=True so the "
        "child's pgid == its pid and the whole process GROUP is signalable "
        "(#2412 MF1; executable code, not prose)"
    )
    assert "os.killpg" in helper_code, (
        "fence expiry must signal the process GROUP (os.killpg) — a naive "
        "single-process kill terminates only the immediate child (uv); the "
        "pytest GRANDCHILDREN inherit the stdout pipe and the post-kill "
        "communicate() blocks on it, so the helper HANGS instead of "
        "reverting (the MF1 wedge, the #2409 timeout class; executable "
        "code, not prose)"
    )
    assert "signal.SIGKILL" in helper_code, (
        "the fence must escalate to SIGKILL-to-group — SIGKILL closes every "
        "inherited pipe end so the final communicate() returns (#2412 MF1; "
        "executable code, not prose)"
    )

    # --- helper-side: verbatim #2208 revert branches + skip-line anchors ---
    assert '"checkout", "HEAD", "--", f' in helper, (
        "the helper revert must restore branch-era files via "
        "git checkout HEAD -- <f> (verbatim #2208 semantics)"
    )
    assert '"rm", "-f", "-q", "--", f' in helper, (
        "the helper revert must drop main-NEW files via git rm -f -q -- <f> "
        "(index + tree; main-NEW is exactly the #2206 incident shape)"
    )
    assert "reverting its issue-" in helper, (
        "the probe-failure skip line must announce the pair-atomic revert "
        "(reverting its issue-<M> synced pair)"
    )
    assert "(#2208)." in helper, "the skip line must cite the fix task (#2208)"
    assert "#2206" in helper, "the skip line must cite the incident class (#2206)"

    # --- helper-side: the #2412 issue-namespaced classifier (MF3) ----------
    assert 're.compile(r"issue_?\\d+")' in helper, (
        "the classifier's issue-stem arm must fullmatch issue_?\\d+ on "
        "component stems (N8 anchoring: issue_763_cofit does NOT fullmatch, "
        "so loose files route lenient)"
    )
    assert 're.compile(r".*_\\d+")' in helper, (
        "the classifier must carry the trailing-issue-number slug arm "
        "(.*_\\d+ — behavior_testbed_545 etc.; the C2/#2412 extension)"
    )
    assert "_ISSUE_STEM_RE.fullmatch(stem)" in helper, (
        "classifier anchoring is FULLMATCH on component stems, never search "
        "(N8 — pinned so the choice cannot drift silently)"
    )
    assert "_TRAILING_SLUG_STEM_RE.fullmatch(stem)" in helper, (
        "the trailing-slug arm is fullmatch-anchored too (N8)"
    )
    assert '_EXPERIMENTS_PREFIX = ("src", "explore_persona_space", "experiments")' in helper, (
        "the trailing-slug arm must be scoped to components under "
        "src/explore_persona_space/experiments/ only (C2 resolution (b))"
    )
    assert '"diff", "--quiet", "origin/main", "--", unit' in helper, (
        "strict identity must diff the OWNING issue-namespaced unit — the "
        "whole DIRECTORY when the owning component is a dir — so a skewed "
        "submodule can never hide behind a byte-identical __init__.py "
        "(#2412 MF3, measured live on the issue-699 worktree)"
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


def _scratch_env() -> dict:
    """Hermetic env for the scratch git fixtures (no user/system git config)."""
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = "/dev/null"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _sibling_arm_script(text: str) -> str:
    """The SHIPPED sibling-arm text as an executable bash script body.

    `_sibling_arm_block`'s end anchor sits INSIDE the closing echo line (the
    block ends with `echo "`); re-attach the remainder of that line so the
    executed script is the shipped prose, count echo included. The literal
    `<N>` is substituted with the scratch issue number 9999.
    """
    span = _step5a_span(text)
    arm = _sibling_arm_block(span)
    echo_start = span.index("[step5a] sibling-file sync:")
    echo_end = span.index("\n", echo_start)
    return (arm + span[echo_start:echo_end]).replace("<N>", "9999")


def _write_uv_shim(tmp: Path) -> Path:
    """PATH-shimmed `uv` for the sibling-arm repros; returns the bin dir.

    Three arms (C4, #2412): (1) `uv run python -c ...` — the venv warm-up —
    STAYS exit-0-swallowed; (2) `uv run python <path> ...` — the #2412
    helper invocation — exec's the REAL interpreter on the path, so the
    planted helper runs its REAL body (the pre-#2412 shim's first arm
    swallowed EVERY `uv run python`, silently no-op'ing the helper: exit 0,
    EMPTY kept-out, SIBLING_SYNCED emptied, and the arm kept + committed
    NOTHING — the mandatory-for-green harness update); (3) `uv run pytest
    --collect-only -q <file>` exec_module's the target file with $WT/src on
    sys.path — REAL import execution, hermetic (a scratch repo has no uv
    project, so the real `uv run pytest` is environment-flaky here; the
    production probe STRING is pinned by
    test_sibling_sync_import_probe_pins).
    """
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
        "# Hermetic `uv` shim for the #2208/#2412 repros (see _write_uv_shim).\n"
        'if [ "$1" = "run" ] && [ "$2" = "python" ]; then\n'
        '  if [ "$3" = "-c" ]; then\n'
        "    exit 0\n"
        "  fi\n"
        f'  exec "{sys.executable}" "${{@:3}}"\n'
        "fi\n"
        'if [ "$1" = "run" ] && [ "$2" = "pytest" ] && [ "$3" = "--collect-only" ] '
        '&& [ "$4" = "-q" ]; then\n'
        f'  exec "{sys.executable}" "{collect_shim}" "$5"\n'
        "fi\n"
        'echo "unexpected uv invocation: $*" >&2\n'
        "exit 97\n"
    )
    uv_shim.chmod(0o755)
    return shim_dir


def _plant_helper(wt: Path, source: Path | str = _SIBLING_PROBE_HELPER) -> None:
    """Copy the (real or stub) probe helper into the scratch worktree.

    The arm's ROOT resolution (`git-common-dir`) resolves a STANDALONE clone
    to itself, so the helper is looked up at
    <wt>/scripts/step5a_sibling_probe.py — the harness plants a copy there
    (the real helper by default; a str plants stub CONTENT instead).
    """
    (wt / "scripts").mkdir(exist_ok=True)
    dest = wt / "scripts" / "step5a_sibling_probe.py"
    if isinstance(source, Path):
        shutil.copyfile(source, dest)
    else:
        dest.write_text(source)


def _run_sibling_arm(
    tmp: Path, wt: Path, env: dict, script_body: str
) -> subprocess.CompletedProcess:
    """Run the shipped sibling-arm text under bash against the scratch wt."""
    shim_dir = tmp / "bin"
    if not shim_dir.exists():
        shim_dir = _write_uv_shim(tmp)
    mb = _run_git(wt, "merge-base", "HEAD", "origin/main", env=env).strip()
    script = tmp / "arm.sh"
    script.write_text(script_body)
    env_arm = dict(env)
    env_arm["PATH"] = f"{shim_dir}:{env['PATH']}"
    env_arm["WT"] = str(wt)
    env_arm["MB"] = mb
    return subprocess.run(
        ["bash", str(script)],
        cwd=tmp,
        env=env_arm,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _scratch_origin_and_wt(tmp: Path, env: dict) -> tuple[Path, Path]:
    """Bare origin + seed clone; returns (seed, wt-placeholder path).

    Callers commit the fork-era state into `seed`, push, then call
    `_clone_issue_wt` to cut the issue-9999 clone at the fork point.
    """
    origin = tmp / "origin.git"
    _run_git(tmp, "init", "--bare", "-b", "main", str(origin), env=env)
    seed = tmp / "seed"
    _run_git(tmp, "clone", str(origin), str(seed), env=env)
    return seed, tmp / "wt"


def _clone_issue_wt(tmp: Path, env: dict) -> Path:
    """Clone the scratch origin at its CURRENT tip as branch issue-9999."""
    wt = tmp / "wt"
    _run_git(tmp, "clone", str(tmp / "origin.git"), str(wt), env=env)
    _run_git(wt, "checkout", "-b", "issue-9999", env=env)
    # The arm's own commit runs bare `git -C "$WT" commit`; give the scratch
    # clone a local identity (global config is /dev/null'd).
    _run_git(wt, "config", "user.email", "eps-test@example.com", env=env)
    _run_git(wt, "config", "user.name", "EPS Test", env=env)
    return wt


def test_sibling_sync_import_probe_repro_2206():
    """#2208 functional repro of the #2206 shape through the SHIPPED arm text.

    Scratch git fixture: origin/main advances past a branch's fork point with
    (i) a poisoned main-NEW tests/test_issue2038_p.py importing a symbol that
    main's src carries but the branch-era src copy lacks (the exact #2206
    symbol-skew shape — the sync deliberately never touches SHARED src), and
    (ii) a legit import-satisfiable pair tests/test_issue1000_ok.py +
    scripts/issue1000_helper.py. The executed block is exactly what
    `_sibling_arm_block` returns (plus the completed closing echo line and the
    literal `<N>` substituted), run under the C4-updated `uv` shim with the
    REAL probe helper planted in the scratch worktree (its git-common-dir is
    its own root). Expect: the poisoned file reverted (absent from working
    tree AND index — now caught by the helper's STATIC arm, module-scope
    import + shared-src symbol miss), the legit pair synced AND committed
    under the sync-anchor subject, the #2206/#2208 skip line emitted, and the
    [step5a] echo reporting the post-revert count.
    """
    script_body = _sibling_arm_script(_text())

    # mkdtemp, not tmp_path: concurrent pytest sessions prune /tmp/pytest-of*
    # numbered roots and can delete live scratch mid-test.
    tmp = Path(tempfile.mkdtemp(prefix="eps2208repro-"))
    try:
        env = _scratch_env()
        seed, _ = _scratch_origin_and_wt(tmp, env)
        # Branch-era state: the src module EXISTS but lacks the symbol.
        (seed / "src").mkdir()
        (seed / "src" / "issue2038_srcmod.py").write_text("BRANCH_ERA = True\n")
        _run_git(seed, "add", "src/issue2038_srcmod.py", env=env)
        _run_git(seed, "commit", "-m", "branch-era src", env=env)
        _run_git(seed, "push", "origin", "main", env=env)

        wt = _clone_issue_wt(tmp, env)

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
        _plant_helper(wt)  # the REAL helper; ROOT == the standalone clone (C4)

        proc = _run_sibling_arm(tmp, wt, env, script_body)
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


def test_sibling_sync_import_probe_repro_2412():
    """#2412 acceptance fixture: TWO cells through the SHIPPED arm text.

    Cell 1 (detection): a poisoned main-NEW sibling test whose ghost import
    sits INSIDE a test function (invisible to `pytest --collect-only` —
    the #2204 escape) and targets a trailing-slug experiments package
    (`testbed_99`) that sits DELIBERATELY OUTSIDE D1's issue-prefixed sync
    closure (N10), so the sync cannot repair the skew before the probe
    runs. The slug package's `__init__.py` is byte-identical to
    scratch-origin/main while `corpora.py` differs and lacks the symbol
    (the MF3 silent-KEEP bait): the static arm's DIRECTORY-grain strict
    rule fires, the whole issue-99 synced pair (test + script) reverts from
    tree AND index, and the legit issue-1000 pair still commits.

    Cell 2 (prevention, positive): a main-NEW sibling test whose
    function-body import needs a symbol PRESENT on scratch-origin/main's
    issue_99 src but ABSENT from the fork-era worktree copy — D1's widened
    globs sync the src to main tip, the strict identity arm reads the
    owning dir as identical, and the pair KEEPs + commits (D1's prevention
    role made explicit; proves cell 1 isn't passing vacuously)."""
    script_body = _sibling_arm_script(_text())
    slug_rel = "src/explore_persona_space/experiments/testbed_99"

    # ---- cell 1: detection (slug-dir skew OUTSIDE D1's closure, N10) -----
    tmp = Path(tempfile.mkdtemp(prefix="eps2412cell1-"))
    try:
        env = _scratch_env()
        seed, _ = _scratch_origin_and_wt(tmp, env)
        slug = seed / slug_rel
        slug.mkdir(parents=True)
        (slug / "__init__.py").write_text("")
        (slug / "corpora.py").write_text("BRANCH_ERA = True\n")
        _run_git(seed, "add", "-A", env=env)
        _run_git(seed, "commit", "-m", "fork-era slug package", env=env)
        _run_git(seed, "push", "origin", "main", env=env)

        wt = _clone_issue_wt(tmp, env)

        # Advance origin/main: the slug package's SUBMODULE gains the ghost
        # symbol (__init__.py untouched — byte-identical both sides, the MF3
        # bait), plus the poisoned main-NEW test + its same-issue script,
        # plus a legit import-satisfiable pair.
        (slug / "corpora.py").write_text("BRANCH_ERA = True\n\n\ndef ghost_fn():\n    return 1\n")
        (seed / "tests").mkdir()
        (seed / "tests" / "test_issue99_p.py").write_text(
            "def test_ghost():\n"
            "    from explore_persona_space.experiments.testbed_99.corpora import ghost_fn\n"
            "    assert ghost_fn() == 1\n"
        )
        (seed / "scripts").mkdir()
        (seed / "scripts" / "issue99_helper.py").write_text("PAIRED = 1\n")
        (seed / "scripts" / "issue1000_helper.py").write_text("OK = 1\n")
        (seed / "tests" / "test_issue1000_ok.py").write_text("def test_ok():\n    assert True\n")
        _run_git(seed, "add", "-A", env=env)
        _run_git(seed, "commit", "-m", "main-side: slug skew + poisoned pair + legit pair", env=env)
        _run_git(seed, "push", "origin", "main", env=env)
        _run_git(wt, "fetch", "origin", env=env)
        _plant_helper(wt)

        # (a) old-probe blindness, demonstrated: module-scope exec of the
        # poisoned file raises nothing — the ghost import is function-body,
        # so collection would PASS this file; only the static arm sees it.
        poisoned_src = (seed / "tests" / "test_issue99_p.py").read_text()
        exec(compile(poisoned_src, "test_issue99_p.py", "exec"), {"__name__": "poisoned_2412"})

        proc = _run_sibling_arm(tmp, wt, env, script_body)
        assert proc.returncode == 0, f"arm run failed:\n{proc.stdout}\n{proc.stderr}"
        out = proc.stdout
        # (b) the WHOLE issue-99 pair reverted from tree AND index.
        for rel in ("tests/test_issue99_p.py", "scripts/issue99_helper.py"):
            assert not (wt / rel).exists(), f"{rel} must be dropped from the working tree"
            assert _run_git(wt, "ls-files", "--", rel, env=env).strip() == "", (
                f"{rel} must be dropped from the index (pair-atomic revert)"
            )
        # ... and the slug dir itself was never synced (OUTSIDE D1, N10).
        assert (wt / slug_rel / "corpora.py").read_text() == "BRANCH_ERA = True\n", (
            "the slug dir sits OUTSIDE D1's closure — the sync must not advance it"
        )
        # (c) the STATIC arm fired, naming the differing owning unit.
        assert "static import scan" in out, out
        assert slug_rel in out, (
            f"the static-refusal line must name the differing unit {slug_rel}:\n{out}"
        )
        assert "reverting its issue-99 synced pair (#2208)." in out, out
        # (d) the legit pair still commits under the sync-anchor subject.
        subj = _run_git(wt, "log", "-1", "--format=%s", env=env).strip()
        assert _SYNC_SUBJECT_2208 in subj, f"sync-anchor subject missing: {subj!r}"
        committed = _run_git(wt, "show", "--name-only", "--format=", "HEAD", env=env)
        assert "tests/test_issue1000_ok.py" in committed, "legit test must be committed"
        assert "scripts/issue1000_helper.py" in committed, "legit paired script must be committed"
        assert "test_issue99_p.py" not in committed, "the poisoned test must never be committed"
        assert "issue99_helper.py" not in committed, "the poisoned pair must never be committed"
        # (e) the count echo reports the POST-revert survivor count.
        assert "[step5a] sibling-file sync: 2 file(s)" in out, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- cell 2: prevention (D1 syncs issue-namespaced src; probe KEEPs) --
    tmp = Path(tempfile.mkdtemp(prefix="eps2412cell2-"))
    try:
        env = _scratch_env()
        seed, _ = _scratch_origin_and_wt(tmp, env)
        pkg = seed / "src" / "explore_persona_space" / "experiments" / "issue_99"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "helpers.py").write_text("FORK_ERA = True\n")
        _run_git(seed, "add", "-A", env=env)
        _run_git(seed, "commit", "-m", "fork-era issue_99 package", env=env)
        _run_git(seed, "push", "origin", "main", env=env)

        wt = _clone_issue_wt(tmp, env)

        # Advance origin/main: issue_99 src gains the symbol; a main-NEW test
        # needs it from INSIDE a test function.
        (pkg / "helpers.py").write_text("FORK_ERA = True\n\n\ndef new_fn():\n    return 1\n")
        (seed / "tests").mkdir()
        (seed / "tests" / "test_issue99_ok.py").write_text(
            "def test_new():\n"
            "    from explore_persona_space.experiments.issue_99.helpers import new_fn\n"
            "    assert new_fn() == 1\n"
        )
        _run_git(seed, "add", "-A", env=env)
        _run_git(seed, "commit", "-m", "main-side: issue_99 symbol + main-NEW test", env=env)
        _run_git(seed, "push", "origin", "main", env=env)
        _run_git(wt, "fetch", "origin", env=env)
        _plant_helper(wt)

        proc = _run_sibling_arm(tmp, wt, env, script_body)
        assert proc.returncode == 0, f"arm run failed:\n{proc.stdout}\n{proc.stderr}"
        out = proc.stdout
        helpers_rel = "src/explore_persona_space/experiments/issue_99/helpers.py"
        # D1 synced the issue-namespaced src to main tip ...
        assert "def new_fn" in (wt / helpers_rel).read_text(), (
            "D1's widened globs must sync the issue-namespaced src file to "
            "origin/main tip BEFORE the probe runs (the prevention half)"
        )
        # ... the probe KEPT (no static refusal, no revert) ...
        assert "static import scan" not in out, out
        assert "reverting its issue-" not in out, out
        # ... and the pair committed together under the sync-anchor subject.
        subj = _run_git(wt, "log", "-1", "--format=%s", env=env).strip()
        assert _SYNC_SUBJECT_2208 in subj, f"sync-anchor subject missing: {subj!r}"
        committed = _run_git(wt, "show", "--name-only", "--format=", "HEAD", env=env)
        assert "tests/test_issue99_ok.py" in committed, "the kept test must be committed"
        assert helpers_rel in committed, "the synced src must be in the SAME sync commit"
        assert "[step5a] sibling-file sync: 2 file(s)" in out, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize("variant", ["stub-exit-97", "absent-helper"])
def test_sibling_sync_helper_crash_reverts(variant):
    """#2412 MF2 + N1: helper rc != 0 => the ARM fully reverts every synced
    path (tree AND index clean), lands NO sync commit, and exits nonzero.

    stub-exit-97: a planted helper stub exits 97 AFTER the sync ran (the
    arm syncs before invoking the helper, so the staged synced set is live
    when the stub dies) — the undecidable-probe shape. absent-helper: no
    helper file at ROOT (the N1 presence/revision divergence class) —
    `uv run python <missing path>` exits nonzero and takes the SAME else
    branch. Both must leave ZERO sync residue: v2's
    leave-staged-for-inspection shape INVERTED the fail-safe (the #1972
    dirt arm reads staged synced files as dirt and preserves them into
    every later round). The synced set carries BOTH revert shapes: a
    branch-era file main modified (checkout-HEAD restore) and a main-NEW
    file (rm --ignore-unmatch drop)."""
    script_body = _sibling_arm_script(_text())
    tmp = Path(tempfile.mkdtemp(prefix="eps2412crash-"))
    try:
        env = _scratch_env()
        seed, _ = _scratch_origin_and_wt(tmp, env)
        (seed / "scripts").mkdir()
        (seed / "scripts" / "issue1000_helper.py").write_text("FORK_ERA = 1\n")
        _run_git(seed, "add", "-A", env=env)
        _run_git(seed, "commit", "-m", "fork-era script", env=env)
        _run_git(seed, "push", "origin", "main", env=env)

        wt = _clone_issue_wt(tmp, env)

        # Advance origin/main: modify the branch-era script (checkout-HEAD
        # revert shape) + add a main-NEW test (git-rm revert shape).
        (seed / "scripts" / "issue1000_helper.py").write_text("MAIN_SIDE = 1\n")
        (seed / "tests").mkdir()
        (seed / "tests" / "test_issue1000_ok.py").write_text("def test_ok():\n    assert True\n")
        _run_git(seed, "add", "-A", env=env)
        _run_git(seed, "commit", "-m", "main-side: modified script + main-NEW test", env=env)
        _run_git(seed, "push", "origin", "main", env=env)
        _run_git(wt, "fetch", "origin", env=env)
        if variant == "stub-exit-97":
            _plant_helper(wt, source="import sys\n\nsys.exit(97)\n")
        head_before = _run_git(wt, "rev-parse", "HEAD", env=env).strip()

        proc = _run_sibling_arm(tmp, wt, env, script_body)
        assert proc.returncode != 0, (
            "a helper crash must exit the arm non-zero (fail-safe #2208/#2412: "
            f"an undecidable probe reverts, never keeps)\n{proc.stdout}\n{proc.stderr}"
        )
        assert "[step5a] FATAL: sibling probe helper failed" in proc.stderr, proc.stderr

        synced = ["scripts/issue1000_helper.py", "tests/test_issue1000_ok.py"]
        tree = subprocess.run(
            ["git", "-C", str(wt), "diff", "--quiet", "HEAD", "--", *synced],
            env=env,
            capture_output=True,
        )
        assert tree.returncode == 0, (
            "the crash branch must leave a clean WORKING TREE on every synced "
            "path (MF2 full revert)"
        )
        idx = subprocess.run(
            ["git", "-C", str(wt), "diff", "--cached", "--quiet", "HEAD", "--", *synced],
            env=env,
            capture_output=True,
        )
        assert idx.returncode == 0, (
            "the crash branch must leave a clean INDEX on every synced path — "
            "staged sync residue is the #1972-dirt-arm inversion (MF2)"
        )
        assert (wt / "scripts" / "issue1000_helper.py").read_text() == "FORK_ERA = 1\n", (
            "the branch-era file must be restored to HEAD content"
        )
        assert not (wt / "tests" / "test_issue1000_ok.py").exists(), (
            "the main-NEW file must be dropped from the working tree"
        )
        assert (
            _run_git(wt, "ls-files", "--", "tests/test_issue1000_ok.py", env=env).strip() == ""
        ), "the main-NEW file must be dropped from the index"
        head_after = _run_git(wt, "rev-parse", "HEAD", env=env).strip()
        assert head_after == head_before, "NO sync commit may land on the crash path"
        subj = _run_git(wt, "log", "-1", "--format=%s", env=env).strip()
        assert _SYNC_SUBJECT_2208 not in subj, "no anchor-subject commit may land on the crash path"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_manual_pair_revert_recovery_documented():
    """#2412 D4 durability pin: the manual pair-revert recovery prose stays.

    Presence checks, not counts: the mechanism heading, the
    anchor-phrase-omitting commit subject (the bs-check then treats the
    reverted paths as deliberate branch edits and skips them every later
    round), the #2204 end-to-end verification sentence, and the N4
    consumed-sibling-src pin-back sentence (a branch that deliberately
    consumes a sibling issue's fork-era src commits the pin-back so the
    widened sync cannot advance the consumed dir mid-experiment)."""
    span = _step5a_span(_text())
    assert "**Manual pair-revert recovery (#2204, #2412).**" in span, (
        "the manual pair-revert recovery prose must stay first-class in the Step 5a doc (#2412 D4)"
    )
    assert "OMITS the anchor phrase `sync workflow-surface specs from`" in span, (
        "the recovery must name the exact mechanism: a hand-revert commit "
        "subject OMITTING the sync-anchor phrase, so the arm's bs-check "
        "treats the paths as deliberate branch edits on every later round"
    )
    assert "`sibling-file sync: 0 file(s)` while still" in span, (
        "the recovery must cite the #2204 end-to-end verification (next "
        "round: 0 sibling files re-synced)"
    )
    assert "syncing 31 legitimate spec files" in span, (
        "the recovery must cite the #2204 end-to-end verification (31 legit "
        "spec files still syncing)"
    )
    assert "deliberately CONSUMES a sibling" in span, (
        "the N4 consumed-sibling-src use case must be named (scripts on main "
        "import other issues' experiments.issue_<M> packages)"
    )
    assert "committing the pin-back keeps the widened" in span, (
        "the N4 sentence must name the pin-back remedy (freeze the consumed "
        "dir at the content the experiment depends on)"
    )


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
# The #2412 sibling-probe pair (scripts/step5a_sibling_probe.py +
# tests/test_step5a_sibling_probe.py) joined SPECS, so it needs fork stubs
# too — without them the atomic checkout errors and syncs NOTHING.
# The #2260 FAMILY_agents members (30 agents prose pins + the workflow
# cross-reader test_inline_payload_lint_gate_contract.py) are literal SPECS
# tokens too — one stub each, same atomic-checkout requirement; guard (20)'s
# completeness failure message names this stub-update duty for future
# members. (A derive-stubs-from-live-SPECS refactor was considered and
# deferred — minimal-diff discipline for a reviewed infra fix.)
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
    "scripts/step5a_sibling_probe.py",
    "tests/test_step5a_sibling_probe.py",
    "tests/test_adversarial_planner_factchecker_grain_pin.py",
    "tests/test_adversarial_planner_lens_brief_headings.py",
    "tests/test_analyzer_language_intrusion_duty.py",
    "tests/test_battery_basis_prose_pins.py",
    "tests/test_code_reviewer_phase_idempotency_gate.py",
    "tests/test_codex_code_reviewer_step09_tag_parity.py",
    "tests/test_codex_critic_numeric_grounding.py",
    "tests/test_consistency_checker_parentless_infra_skip.py",
    "tests/test_cross_issue_protocol_comparability_prose.py",
    "tests/test_daily_three_route_classifier_doc.py",
    "tests/test_diff_base_origin_main_pin.py",
    "tests/test_downwidth_split_prose_pins.py",
    "tests/test_experimenter_md.py",
    "tests/test_fit_loop_batching_review_pin.py",
    "tests/test_implementer_spec_deleted_literal_substep.py",
    "tests/test_implementer_spec_mechanical_pin_sweep.py",
    "tests/test_implementer_spec_names_invariant_local_union.py",
    "tests/test_implementer_spec_names_ruff_policy_pin.py",
    "tests/test_inline_payload_lint_gate_contract.py",
    "tests/test_interp_critic_degenerate_series_lens.py",
    "tests/test_issue_v2_skill_figure_pin_contract.py",
    "tests/test_lean_twin_registration_pin.py",
    "tests/test_mapping_baselines_wiring_pins.py",
    "tests/test_off_pod_phase_slot_pin.py",
    "tests/test_outroot_residue_prose_pins.py",
    "tests/test_plan_handoff_path_convention.py",
    "tests/test_planner_incident_trace_guidance.py",
    "tests/test_planner_phase_outputs_declaration.py",
    "tests/test_realized_rows_prose_pins.py",
    "tests/test_selection_symmetric_nulls_pointers.py",
    "tests/test_v2_composer_plan_path_brief.py",
)

_SYNC_SUBJECT_2303 = "issue-9999: sync workflow-surface specs from origin/main (spec-freshness)"


def _family_sync_fixture(tmp: Path, env: dict, *, delete_member: str | None = None) -> Path:
    """Scratch bare origin + a wt clone on issue-9999 whose origin/main has
    advanced past the fork point by a scripts/workflow_lint.py edit + the
    main-NEW .claude/config/agent_spec_size_caps.txt (the #2293 topology).
    With delete_member set (the #2260 containment topology), the SAME
    advance commit additionally modifies .claude/agents/x.md (an
    agents-family edit the sync would normally carry) and DELETES the named
    member stub (a main-side rename/retire). Returns the wt path; the wt
    has already fetched origin."""
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
    if delete_member is not None:
        # #2260 containment topology: main ALSO edits an agents-family file
        # and deletes one agents member stub (rename/retire on main).
        (seed / ".claude" / "agents" / "x.md").write_text(
            "fork-era stub: .claude/agents/x.md\nmain-side agents edit\n"
        )
        (seed / delete_member).unlink()
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


# --- (18b/#2260) member-existence containment arm: textual pin + repro -------


def _containment_arm_block(span: str) -> str:
    """Extract the #2260 member-existence containment arm from a sync span:
    from its comment lead through the closing esac (the arm is the FIRST
    case statement in the pass-1 loop, inserted before the bs_commits
    stanza in both copies)."""
    start = span.index("# Member-existence containment (#2260")
    end = span.index("esac", start)
    return span[start:end]


def test_member_existence_arm_in_both_sync_copies():
    """#2260: pass 1 of BOTH sync copies (Step 5a + the Step 10d auto-merge
    inline block) must carry the member-existence containment arm.
    `git checkout <ref> -- <pathspecs>` is ATOMIC — a single literal token
    absent at origin/main (deleted/renamed on main) errors the whole
    checkout and syncs NOTHING, wedging every family until manual reconcile
    — so an absent literal member marks ITS family dirty
    (vintage-consistent skip; other families keep syncing). Glob tokens are
    excluded (cat-file -e takes no glob); deletion PROPAGATION (removing
    the stale worktree twin) stays #2385."""
    text = _text()
    for span_name, span, dirty_marking in (
        ("Step 5a", _step5a_span(text), "DIRTY_FAMILIES[$fam]=1"),
        ("auto-merge", _automerge_span(text), "DIRTY_FAMILIES_10D[$fam]=1"),
    ):
        block = _containment_arm_block(span)
        assert 'case "$f" in' in block, (
            f"the {span_name} containment arm must be a per-token case guard"
        )
        assert '":(glob)"*) : ;;' in block, (
            f"the {span_name} containment arm must exclude :(glob) tokens "
            f"(cat-file -e takes no glob)"
        )
        assert 'if ! git -C "$WT" cat-file -e "origin/main:$f" 2>/dev/null; then' in block, (
            f"the {span_name} containment arm must probe literal-member "
            f"existence at origin/main via cat-file -e"
        )
        assert dirty_marking in block, (
            f"the {span_name} containment arm must mark the absent member's "
            f"family dirty ({dirty_marking} — the family-atomic skip)"
        )
        assert "is ABSENT at origin/main" in block, (
            f"the {span_name} containment arm must announce the containment "
            f"with the ABSENT-at-origin/main echo"
        )
        assert "continue" in block, (
            f"the {span_name} containment arm must `continue` past the "
            f"dirty-scan for a token that cannot be synced anyway"
        )


def test_family_sync_deleted_member_contained():
    """#2260 containment repro (§3 condition 4) through the SHIPPED family
    arm under real git: origin/main advances by (a) modifying
    .claude/agents/x.md, (b) DELETING one agents-member stub (a main-side
    rename/retire), (c) modifying scripts/workflow_lint.py (+ the main-NEW
    caps data file, as in the base fixture). The arm must exit 0, echo the
    ABSENT-at-origin/main containment line for the deleted member, skip the
    WHOLE agents family (the main-side agents edit does NOT arrive —
    vintage-consistent, never a half-refresh), still sync the clean lint
    family, and leave the deleted member's stale worktree twin on disk
    (deletion propagation is #2385). Pre-containment, the atomic checkout
    would have errored on the absent pathspec and synced NOTHING for every
    family — the whole-sync wedge this arm exists to contain."""
    text = _text()
    script_body = _family_arm_block(_step5a_span(text)).replace("<N>", "9999")

    deleted = "tests/test_mapping_baselines_wiring_pins.py"
    tmp = Path(tempfile.mkdtemp(prefix="eps2260contain-"))
    try:
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        wt = _family_sync_fixture(tmp, env, delete_member=deleted)

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
        assert f"spec-freshness: {deleted} is ABSENT at origin/main" in proc.stdout, (
            f"the containment echo must name the deleted member:\n{proc.stdout}"
        )
        # The agents family is skipped WHOLE: main's agents edit did NOT arrive.
        assert (wt / ".claude" / "agents" / "x.md").read_text() == (
            "fork-era stub: .claude/agents/x.md\n"
        ), (
            "the agents family must be SKIPPED whole on a deleted member "
            "(vintage-consistent — a fresh .claude/agents/x.md against "
            "branch-era pin tests would be exactly the #2251 half-sync)"
        )
        # The clean lint family still synced (containment, not a whole-sync wedge).
        assert "MAIN_SIDE_FIX = True" in (wt / "scripts" / "workflow_lint.py").read_text(), (
            "the lint family must keep syncing while the agents family is contained"
        )
        # Stale-twin removal is #2385 — the sync never deletes.
        assert (wt / deleted).exists(), (
            "the deleted member's stale worktree twin must survive (deletion "
            "propagation is #2385; the sync never deletes)"
        )
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


def _family_synced_test_files(
    tokens: list[str], family_of: dict[str, str]
) -> dict[Path, list[tuple[str, str | None]]]:
    """Enumerate the family-synced tests/*.py files mechanically from the
    SPECS tokens: every existing repo file matched by a `:(glob)tests/...`
    token plus every explicit `tests/*.py` token. Maps each file to its
    sync ROUTES — one (token, family) tuple per SPECS token matching it,
    family = FAMILY_OF[token] or None for a SINGLETON token (no FAMILY_OF
    entry — the token is its own family). Each SPECS token syncs iff ITS
    OWN family is clean (pass-2 per-token semantics), so every route is an
    INDEPENDENT freshness channel: a multi-route file can arrive fresh
    through ANY one of its routes."""
    out: dict[Path, list[tuple[str, str | None]]] = {}
    for tok in tokens:
        if tok.startswith(":(glob)"):
            pattern = tok[len(":(glob)") :]
            if not pattern.startswith("tests/"):
                continue
            fam = family_of.get(tok)
            for p in sorted(_REPO.glob(pattern)):
                if p.suffix == ".py" and p.is_file():
                    out.setdefault(p, []).append((tok, fam))
        elif tok.startswith("tests/") and tok.endswith(".py"):
            p = _REPO / tok
            if p.is_file():
                out.setdefault(p, []).append((tok, family_of.get(tok)))
    return out


def _tests_module_imports(src: str) -> set[str]:
    """First-level `tests.<mod>` modules imported by a test file's source,
    collected via `ast.parse` over the WHOLE module (module-level and
    nested/function-body import statements alike):

    - `from tests import a, b as c` -> {"a", "b"} (each alias name; a
      Black/ruff-style parenthesized multiline form is the same ImportFrom
      node, so it parses identically),
    - `from tests.a import x` / `from tests.a.b import x` -> {"a"} (the
      first component after "tests."),
    - `import tests.a, tests.b` / `import tests.a as t` -> {"a", "b"}.

    The AST never contains string literals as statements, so import-looking
    lines inside docstrings / triple-quoted fixtures can never
    false-positive. A multi-level package import (`tests.a.b`) is
    deliberately REDUCED to its first component: the coverage target
    becomes `tests/a.py`, a FILE path a package DIRECTORY can never
    satisfy, so a new multi-level import FAILS guard (19) until it gets an
    explicit disposition — fail-loud, never a silent escape (none exist
    today). Known false negatives, disclosed by design: dynamic imports
    (`importlib.import_module` / `__import__` on a string) have no static
    import node, and RELATIVE imports (`from .mod import x` — level >= 1)
    are not collected (no family-synced test uses them; the repo
    convention is absolute `tests.` imports)."""
    mods: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "tests" and len(parts) >= 2:
                    mods.add(parts[1])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "tests":
                for alias in node.names:
                    mods.add(alias.name)
            elif node.module.startswith("tests."):
                mods.add(node.module.split(".")[1])
    return mods


def _helper_coverage(
    mod: str, tokens: list[str], family_of: dict[str, str]
) -> tuple[bool, set[str]]:
    """Coverage facts for tests/<mod>.py against the SPECS tokens, returned
    as (singleton_covered, families). singleton_covered is True when the
    helper's exact path is an EXPLICIT SPECS token with NO FAMILY_OF entry:
    a singleton syncs whenever it is ITSELF clean, so no other file's dirt
    can family-skip it — route-independent coverage (the #2352
    disposition). families is the set of FAMILY_OF families of tokens
    (glob or explicit) matching the helper. A singleton GLOB matching the
    helper deliberately contributes NEITHER: its dirt scope spans every
    file the glob matches, so another matched file's dirt can still skip
    the helper (conservative; no such token exists today)."""
    target = f"tests/{mod}.py"
    singleton = False
    fams: set[str] = set()
    for tok in tokens:
        if tok.startswith(":(glob)"):
            if fnmatch.fnmatch(target, tok[len(":(glob)") :]):
                fam = family_of.get(tok)
                if fam is not None:
                    fams.add(fam)
        elif tok == target:
            fam = family_of.get(tok)
            if fam is None:
                singleton = True
            else:
                fams.add(fam)
    return singleton, fams


def _import_covered(
    mod: str,
    importer_routes: list[tuple[str, str | None]],
    tokens: list[str],
    family_of: dict[str, str],
) -> bool:
    """True when tests/<mod>.py is guaranteed sync-coverable alongside an
    importer that can arrive fresh through ANY of importer_routes. Because
    each SPECS token syncs iff ITS OWN family is clean, every route is an
    independent freshness channel, so coverage quantifies UNIVERSALLY over
    the importer's routes:

    - the helper is itself a SINGLETON SPECS token (route-independent —
      syncs whenever itself clean), OR
    - on EVERY route: the route carries a family F (a singleton importer
      token carries none — only the singleton-helper arm can cover that
      route) and some SPECS token matching the helper is assigned that
      SAME family F.

    An EXISTENTIAL read (covered through at least one of the importer's
    families) is exactly the round-1 defect: an importer matched by tokens
    in TWO families syncs fresh through EITHER, so helper coverage in only
    one leaves the other route recreating the #2352 ModuleNotFoundError
    half-sync."""
    singleton, helper_fams = _helper_coverage(mod, tokens, family_of)
    if singleton:
        return True
    if not importer_routes:
        return False
    return all(fam is not None and fam in helper_fams for _tok, fam in importer_routes)


def _uncovered_imports(
    files: dict[Path, list[tuple[str, str | None]]],
    tokens: list[str],
    family_of: dict[str, str],
) -> list[str]:
    """Guard (19)'s problem collector: one line per (family-synced test
    file, uncovered tests.<mod> import) pair, naming the importer's sync
    routes. Factored out so the pre-#2352-spec-shape non-vacuity test can
    run the identical predicate against a mutated token list."""
    problems: list[str] = []
    for path in sorted(files):
        routes = files[path]
        src = path.read_text(encoding="utf-8")
        for mod in sorted(_tests_module_imports(src)):
            if mod in _EXEMPT_HELPER_MODULES:
                continue
            if not _import_covered(mod, routes, tokens, family_of):
                route_desc = ", ".join(
                    f"{tok} [{fam if fam is not None else 'singleton'}]" for tok, fam in routes
                )
                problems.append(
                    f"{path.relative_to(_REPO).as_posix()} (routes: {route_desc}) "
                    f"imports tests.{mod}"
                )
    return problems


def test_family_synced_test_helper_imports_covered():
    """#2352 forward guard: every `tests.<mod>` import in any FAMILY-SYNCED
    test file must be sync-coverable on EVERY route through which the
    importer can arrive fresh — per route, a same-family token/glob for the
    helper; route-independently, the helper itself a SINGLETON token — so
    the NEXT main-side helper module a family-synced test imports makes
    THIS suite red on main (the skew is unshippable) instead of red-ing a
    worktree's Step 9c gate with a ModuleNotFoundError half-sync (the
    #2352 incident: 66 collection errors in the issue-2333 worktree).
    Cross-family / partial-route coverage is deliberately NOT sufficient:
    families dirty-skip independently and each matching SPECS token is its
    own sync channel, so a helper covered on only SOME of an importer's
    routes can still be skipped while the importer syncs fresh through
    another. Imports are collected via `ast.parse` (parenthesized
    multiline, comma-separated, and aliased forms included; string
    literals invisible); the collector's disclosed false negatives —
    dynamic/importlib imports, relative imports — keep the Step 9c gate as
    the runtime backstop, and a multi-level `tests.a.b` import fails this
    guard by construction (see _tests_module_imports)."""
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
    problems = _uncovered_imports(files, tokens, family_of)
    assert not problems, (
        "family-synced test file(s) import a tests.<mod> helper the Step 5a "
        "SPECS tokens cannot guarantee to sync alongside them on EVERY sync "
        "route — the Step 5a family sync can pull these tests into a "
        "worktree WITHOUT the helper (the #2352 ModuleNotFoundError "
        "half-sync; 66 collection errors). Remedies: (a) add tests/<mod>.py "
        "to SPECS + SPECS_10D as a SINGLETON token (the #2352 disposition — "
        "required when the helper's importers span families, sync through a "
        "singleton token, or live in unsynced tests), (b) add it as a "
        "same-family token for EVERY family named by every importer's "
        "routes (right only when all routes carry families and the helper "
        "is assigned each of them), or (c) add the module to "
        "_EXEMPT_HELPER_MODULES with a documented rationale (the conftest "
        "shape). Uncovered imports:\n  " + "\n  ".join(problems)
    )


def test_tests_module_imports_collector_fixtures():
    """#2352 round 2: the AST import collector handles ordinary valid
    import syntax the round-1 line regexes missed — exact expected module
    sets per fixture (the two BLOCKER-verified gaps plus the documented
    reductions)."""
    # (i) Black/ruff-style parenthesized multiline import-from: the regex
    # matched only "(" -> empty set; AST sees one ImportFrom node.
    src = "from tests import (\n    new_helper,\n    other_helper,\n)\n"
    assert _tests_module_imports(src) == {"new_helper", "other_helper"}
    # (ii) comma-separated plain import: the regex recorded only the first.
    assert _tests_module_imports("import tests.a, tests.b\n") == {"a", "b"}
    # (iii) aliased forms in all three shapes.
    assert _tests_module_imports("from tests import helper_a as h\n") == {"helper_a"}
    assert _tests_module_imports("from tests.foo import bar as baz\n") == {"foo"}
    assert _tests_module_imports("import tests.mod_x as mx\n") == {"mod_x"}
    # (iv) import-looking lines inside a triple-quoted string / docstring:
    # must detect NOTHING (the regex false-positive class).
    src = '"""\nfrom tests.phantom import x\nimport tests.ghost\n"""\nX = 1\n'
    assert _tests_module_imports(src) == set()
    # Documented multi-level reduction: tests.a.b records the FIRST
    # component only (whose tests/a.py coverage target then fails guard 19
    # by construction — a package dir is never a SPECS file token).
    assert _tests_module_imports("from tests.pkg.sub import x\n") == {"pkg"}
    assert _tests_module_imports("import tests.pkg.sub\n") == {"pkg"}
    # Nested (function-body) imports are collected; non-tests imports are not.
    src = "def f():\n    from tests.lazy_helper import thing\n    import os\n"
    assert _tests_module_imports(src) == {"lazy_helper"}
    assert _tests_module_imports("import os\nfrom pathlib import Path\n") == set()


def test_import_covered_route_quantifier_fixtures():
    """#2352 round 2: coverage quantifies UNIVERSALLY over an importer's
    sync routes (synthetic-token fixtures per the reconciler's mechanizable
    spec). The round-1 existential predicate returned covered on fixtures
    (a) and (b); this pin holds the universal semantics."""
    wf_glob = ":(glob)tests/test_wf_*.py"
    tokens = [
        wf_glob,
        "tests/test_wf_lintish.py",
        "tests/helper_wf.py",
        "tests/helper_single.py",
    ]
    family_of = {
        wf_glob: "workflow",
        "tests/test_wf_lintish.py": "lint",
        "tests/helper_wf.py": "workflow",
        # tests/helper_single.py deliberately has NO entry — singleton.
    }
    # (a) importer syncs through a workflow-family glob AND a lint-family
    # explicit token; helper covered only in the workflow family ->
    # UNCOVERED (a dirty workflow family + clean lint family syncs the
    # importer fresh through the lint route while the helper is skipped).
    two_family_routes = [(wf_glob, "workflow"), ("tests/test_wf_lintish.py", "lint")]
    assert not _import_covered("helper_wf", two_family_routes, tokens, family_of)
    # (b) importer with a singleton route + a family route; helper only
    # family-covered -> UNCOVERED (nothing guarantees co-sync with the
    # importer's own singleton token, which syncs whenever ITSELF clean).
    singleton_routes = [("tests/test_wf_selfsync.py", None), (wf_glob, "workflow")]
    assert not _import_covered("helper_wf", singleton_routes, tokens, family_of)
    # (c) a SINGLETON helper is route-independent: covered in both cases.
    assert _import_covered("helper_single", two_family_routes, tokens, family_of)
    assert _import_covered("helper_single", singleton_routes, tokens, family_of)
    # Same-family multi-route overlap (the live test_guard_lessons_edit.py
    # shape: guard glob + explicit guard token) stays covered when the
    # helper is assigned that one shared family.
    family_of_same = dict(family_of)
    family_of_same["tests/test_wf_extra.py"] = "workflow"
    same_family_routes = [(wf_glob, "workflow"), ("tests/test_wf_extra.py", "workflow")]
    assert _import_covered(
        "helper_wf", same_family_routes, [*tokens, "tests/test_wf_extra.py"], family_of_same
    )
    # A helper matched ONLY by a SINGLETON GLOB is deliberately NOT
    # route-independent coverage (the glob's dirt scope spans every file it
    # matches, so another matched file's dirt can still skip the helper).
    tokens_glob = [":(glob)tests/helper_g*.py", wf_glob]
    family_of_glob = {wf_glob: "workflow"}
    assert not _import_covered("helper_glob", [(wf_glob, "workflow")], tokens_glob, family_of_glob)


def test_guard_19_fails_on_pre_2352_spec_shape():
    """Non-vacuity re-pin: with the #2352 singleton token removed from the
    LIVE SPECS token list (the pre-fix spec shape), guard (19)'s predicate
    reports the incident — every test_issue_skill_* pin test imports
    tests.issue_skill_source, which no remaining token can sync. Run
    against the real composed spec + real repo tree, so the check tracks
    the live configuration rather than a frozen fixture."""
    text = _text()
    span = _step5a_span(text)
    tokens = [t for t in _specs_tokens(span) if t != "tests/issue_skill_source.py"]
    assert "tests/issue_skill_source.py" not in tokens
    family_of = _family_of_map(span)
    files = _family_synced_test_files(tokens, family_of)
    problems = _uncovered_imports(files, tokens, family_of)
    assert any("imports tests.issue_skill_source" in p for p in problems), (
        "guard (19) must FAIL on the pre-#2352 spec shape (helper token "
        f"absent) — the guard would be vacuous; got problems={problems!r}"
    )


# --- (20) agents-prose pin tests are family-coupled (#2260 guard) -------------

# Reader predicate (#2260): matches BOTH forms the corpus uses — the literal
# ".claude/agents" string AND the quoted path-join form `".claude" / "agents"`
# (the join alternative requires a closing quote immediately after `agents`,
# so `.claude/agent-memory` can match NEITHER alternative). 67 matching
# tests/test_*.py files at #2260 landing; guard (20) recomputes live, so
# corpus drift self-reconciles.
_AGENTS_READER_RE = re.compile(r'\.claude/agents|["\']\.claude["\']\s*/\s*["\']agents["\']')

# Read-verb receivers the incidental shape check scans (part C below).
_AGENTS_READ_VERBS = frozenset({"read_text", "read_bytes", "open", "glob", "rglob", "iterdir"})

# Member residuals, documented (#2260 §5 — members carry no rationale dict,
# so the named residuals live here): (a) tests/test_mapping_baselines_
# wiring_pins.py (the #2251 incident file) importlib-loads
# src/.../analysis/mapping_baselines.py BY PATH from the worktree — dynamic
# symbol enumeration, partially self-adapting; admitted because excluding
# the incident file would defeat the coupling's purpose. (b)
# tests/test_inline_payload_lint_gate_contract.py is guard-19-FORCED into
# "workflow" (its tests.test_issue_skill_inline_gate_pin import); on a
# workflow-dirty branch, agents can refresh while it stays branch-era —
# strictly narrower than today, where it never syncs at all. (c)
# tests/test_diff_base_origin_main_pin.py importlib-loads
# scripts/select_step9c_tests.py, a SPECS SINGLETON (closure-clean under
# the #2260 rule-1 clarification); an agents-dirty branch can leave the
# test branch-era while the selector singleton syncs. (d)
# tests/test_code_reviewer_phase_idempotency_gate.py imports
# scripts/workflow_lint.py (a lint-family SPECS member — SPECS-synced,
# closure-clean; cross-family residual: agents clean + lint dirty leaves
# the fresh test pinning a branch-era linter's caps map). (e) ~15 members
# also pin prose in .claude/rules / CLAUDE.md / .claude/skills —
# cross-surface reads; those singletons/families sync independently, so a
# member can sync fresh while a rules file it also pins stays branch-era.
# NET exposure still shrinks for every member (today these files never
# sync at all); the improvement is not strict for the cross-surface subset.

# Files with a REAL agents-prose assertion that cannot join a family
# (closure violations — membership would sync them against branch-era
# scripts/src, the sync-scope boundary rationale (ii)). Accepted residuals;
# NOT shape-checked (they legitimately read agents prose). Relocation of
# the movable asserts into a closure-clean pin file is FILED as #2454.
_AGENTS_PROSE_EXEMPT_GENUINE: dict[str, str] = {
    "tests/test_matched_support.py": (
        "behavioral: numpy + importlib-by-path of the matched_support "
        "implementation (unsynced src). GENUINE prose asserts (~:350/:358/"
        ":391) over statistics-critic.md / interpretation-critic.md / "
        "critic.md accepted as residual; relocation filed as #2454."
    ),
    "tests/test_pod_audit.py": (
        "behavioral: pod_audit / pod_config / runpod_api scripts (unsynced). "
        "GENUINE research-pm.md triage-protocol assert "
        "(test_pm_triage_protocol_present, ~:721) accepted as residual; "
        "relocation filed as #2454."
    ),
    "tests/test_bootstrap_pod_git_credentials.py": (
        "behavioral: subprocess-EXECUTES scripts/bootstrap_pod.sh + reads "
        "unsynced scripts/. GENUINE #1271 no-tokenized-remote-URL negative "
        "pin over ALL agent specs (agents-dir glob ~:291) accepted as "
        "residual; relocation filed as #2454."
    ),
    "tests/test_verify_plan.py": (
        "behavioral: imports scripts/verify_plan.py (unsynced). GENUINE "
        "planner.md prose asserts (~:6815 predicate-anchor literals; ~:7372 "
        "durability-pin bullet) — the #2260 implement-time vet re-routed "
        "this file from incidental (the plan's provisional table saw only "
        "its fixture strings); relocation filed as #2454."
    ),
    "tests/test_ensemble_review_cap.py": (
        "behavioral: unsynced src imports (orchestrate.ensemble_strip, "
        "explore_persona_space.workflow) + issue_skill_source. GENUINE "
        "agents-dir glob inside the spelled-cap scan over the whole "
        "workflow doc surface, accepted as residual. #2420 OWNS this "
        "file's workflow-prose coupling decision — #2260 dispositions it "
        "exempt so guard (20) is coherent; any future promotion is #2420's "
        "call. Deliberately EXCLUDED from the #2454 relocation filing."
    ),
}

# Files whose reader-pattern match is INCIDENTAL (docstring / fixture-string
# / non-read literal; no agents read at all). Shape-checked (part C): an
# incidental exemption that later gains a genuine agents-path read construct
# reds mechanically instead of staying silently green forever.
_AGENTS_PROSE_INCIDENTAL_EXEMPT: dict[str, str] = {
    "tests/test_daily_drive_filings.py": (
        "fixture strings only (~:1952); behavioral file (daily_drive_filings "
        "/ file_infra_task scripts) — no agents read construct."
    ),
    "tests/test_step10d_guard3.py": (
        "the SPECS literal itself inside an index() string arg (~:373) — a "
        "workflow-prose pin, no agents read construct (a #2420-class "
        "member, noted for that session)."
    ),
    "tests/test_subprocess_env_explicit.py": (
        "docstring mention (~:5) only; its read_text is a repo-wide *.py "
        "AST sweep — no agents read construct."
    ),
    "tests/test_sweep_parked_wf_candidates.py": (
        "fixture strings only; behavioral (scripts + task_workflow src) — no agents read construct."
    ),
    "tests/test_task_workflow.py": (
        "docstring mention (~:272) only; behavioral (task_workflow src) — no agents read construct."
    ),
}


def _agents_reader_files() -> list[Path]:
    """tests/test_*.py files whose RAW TEXT matches the agents reader
    pattern — guard (20)'s universe. LEXICAL, not dependency-aware;
    documented accepted misses (part F): helper-module indirection (a
    conftest helper reading agents on a test's behalf), f-string /
    variable-built paths, and subprocess-mediated reads remain forward
    escape routes — a bounded grep found none in today's corpus (the same
    limitation class as the Step 9c selector's accepted misses)."""
    return [
        p
        for p in sorted(_REPO.glob("tests/test_*.py"))
        if _AGENTS_READER_RE.search(p.read_text(encoding="utf-8"))
    ]


def _existence_verdict(on_disk: bool, at_merge_base: bool) -> str:
    """Vintage-skew discriminator (#2260, part B): 'check' when the file is
    on disk; for an absent file, 'skip' when it is ALSO absent at
    merge-base(HEAD, origin/main) — main-new relative to this tree, pure
    vintage skew (a freshly-synced guard naming a member/exempt file the
    branch's vintage predates; an unconditional existence assert would
    re-create the #1824/#1860 false-red class fleet-wide) — and 'fail' when
    present at merge-base (THIS tree deleted it: on main, and on any branch
    cut before the deletion, merge-base still holds the file, so the
    deleting PR's own gate reds). Named residual: a deletion MERGED without
    cleaning its FAMILY_OF/exempt entry goes guard-silent post-merge
    (merge-base then tolerates); the runtime containment arm keeps echoing
    at every sync until reconciled — the pressure survives."""
    if on_disk:
        return "check"
    return "fail" if at_merge_base else "skip"


def _merge_base_origin_main() -> str:
    """merge-base(HEAD, origin/main) sha — the suite's existing convention
    (the Step 5a block itself computes the same ref, and a repo without
    origin/main cannot run the sync either). Fails LOUD on a git error
    (check=True), never a silent skip."""
    proc = subprocess.run(
        ["git", "-C", str(_REPO), "merge-base", "HEAD", "origin/main"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _present_at_ref(ref: str, rel: str) -> bool:
    """True iff <ref>:<rel> resolves (git cat-file -e). A missing path
    exits non-zero with 'Not a valid object name' / 'does not exist'; any
    OTHER failure raises — fail-loud, never a silent skip."""
    proc = subprocess.run(
        ["git", "-C", str(_REPO), "cat-file", "-e", f"{ref}:{rel}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True
    if "Not a valid object name" in proc.stderr or "does not exist" in proc.stderr:
        return False
    raise RuntimeError(
        f"git cat-file -e {ref}:{rel} failed unexpectedly (rc={proc.returncode}): {proc.stderr}"
    )


def _agents_read_constructs(src: str) -> list[str]:
    """Agents-path READ CONSTRUCTS in a module's source — the part-C shape
    detector: a BinOp div-chain or a Path()/open() call whose source
    segment (ast.get_source_segment) matches the reader pattern
    (module-level path constants and inline constructs alike), or a
    read-verb call (read_text / read_bytes / open / glob / rglob /
    iterdir) whose RECEIVER's source segment matches it. A bare string
    constant (fixture text, docstring, an index() argument) never
    triggers. Documented accepted misses: a subprocess-mediated read
    (subprocess.run(["cat", ".claude/agents/..."])) and reads built from
    f-string/variable fragments that never place the joined fragment in
    one source segment."""
    hits: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            seg = ast.get_source_segment(src, node)
            if seg and _AGENTS_READER_RE.search(seg):
                hits.append(seg)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"Path", "open"}:
                seg = ast.get_source_segment(src, node)
                if seg and _AGENTS_READER_RE.search(seg):
                    hits.append(seg)
            elif isinstance(func, ast.Attribute) and func.attr in _AGENTS_READ_VERBS:
                recv = ast.get_source_segment(src, func.value)
                if recv and _AGENTS_READER_RE.search(recv):
                    hits.append(recv)
    return hits


def _agents_uncovered_readers(tokens: list[str], family_of: dict[str, str]) -> list[str]:
    """Guard (20)'s completeness collector: reader-universe files with NO
    disposition — not matched by a `:(glob)tests/...` SPECS token, not an
    explicit tests/*.py SPECS token WITH a FAMILY_OF entry (any family),
    and not a key of either exempt dict. Factored out so the non-vacuity
    pin can run the identical predicate against a mutated token list. Each
    uncovered entry carries its origin/main-presence annotation (the
    failure-message arm (d) probe — a disk-present but origin/main-absent
    file is a stale main-deleted twin, not a new reader)."""
    glob_pats = [t[len(":(glob)") :] for t in tokens if t.startswith(":(glob)tests/")]
    explicit = {t for t in tokens if t.startswith("tests/") and t.endswith(".py")}
    problems: list[str] = []
    for p in _agents_reader_files():
        rel = p.relative_to(_REPO).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in glob_pats):
            continue
        if rel in explicit and rel in family_of:
            continue
        if rel in _AGENTS_PROSE_EXEMPT_GENUINE or rel in _AGENTS_PROSE_INCIDENTAL_EXEMPT:
            continue
        marker = (
            ""
            if _present_at_ref("origin/main", rel)
            else " [ABSENT at origin/main — arm (d): stale main-deleted twin]"
        )
        problems.append(rel + marker)
    return problems


def test_agents_prose_pin_tests_family_coupled():
    """#2260 guard (20), completeness arm (part A): every tests/test_*.py
    whose raw text matches the agents reader pattern (the literal
    `.claude/agents` form OR the quoted path-join form) must be
    dispositioned — matched by a coupled `:(glob)tests/...` SPECS token, an
    explicit tests/*.py SPECS token WITH a FAMILY_OF entry (any family), or
    a key of one of the two exempt dicts — so the FAMILY_agents membership
    cannot silently re-rot (the #1883/#1963/#2352 recurrence class;
    incident #2251: a fresh main-side planner.md against a branch-era pin
    test red the Step 9c gate for 74 min). Over-matching (a comment mention
    joins the universe) is the SAFE direction: the remedy is a one-line
    exempt entry, mirroring the selector's over-selection convention."""
    span = _step5a_span(_text())
    tokens = _specs_tokens(span)
    family_of = _family_of_map(span)
    readers = _agents_reader_files()
    # Enumeration sanity: 67 matches at #2260 landing; an empty/thin sweep
    # means the extraction or pattern broke, not a clean tree.
    assert len(readers) >= 40, (
        f"agents-reader enumeration looks broken: only {len(readers)} "
        f"tests/test_*.py files match the reader pattern (67 at #2260 landing)"
    )
    problems = _agents_uncovered_readers(tokens, family_of)
    assert not problems, (
        "tests/test_*.py file(s) match the agents reader pattern with NO "
        "disposition — refreshing .claude/agents without its pin tests reds "
        "the Step 9c gate on pure vintage skew (#2251). Decision procedure, "
        "one arm per file: (a) closure-clean agents-prose pin (imports "
        "limited to stdlib / env packages / tests/issue_skill_source.py / "
        "SPECS-synced files) => add "
        'FAMILY_OF["tests/<f>.py"]="agents" + the SPECS AND SPECS_10D '
        "tokens in BOTH sync copies + a _FORK_STUBS_2303 stub + the test "
        "(1) SPECS-literal update; (b) a tests.<mod> import covered only in "
        "ANOTHER synced family forces THAT family (guard (19) "
        "universal-route coverage — the "
        "tests/test_inline_payload_lint_gate_contract.py shape); (c) an "
        "incidental mention (docstring / fixture string / non-read literal) "
        "or a behavioral file importing unsynced scripts/src => the "
        "matching exempt dict with a per-file rationale (genuine-residual "
        "entries also name their relocation plan — #2454); (d) a file "
        "present on disk but ABSENT at origin/main (annotated below) is a "
        "stale main-deleted twin the sync never deletes => remove the local "
        "twin (deletion propagation is #2385). Undispositioned:\n  " + "\n  ".join(problems)
    )


def test_agents_guard_member_and_exempt_existence():
    """#2260 guard (20), part B: every explicit tests/*.py FAMILY_OF key
    and every exempt-dict key must exist on disk — discriminated by
    vintage, never asserted unconditionally: a key absent from disk AND
    absent at merge-base(HEAD, origin/main) is main-new relative to this
    tree (pure vintage skew — SKIPPED, the exact false-red class #2260
    exists to close), while present-at-merge-base + absent-on-disk FAILs
    (this tree deleted it without cleaning the coupling — the deleting-PR
    early warning). See _existence_verdict for the named post-merge
    residual."""
    span = _step5a_span(_text())
    family_of = _family_of_map(span)
    keys = sorted(
        {k for k in family_of if k.startswith("tests/") and k.endswith(".py")}
        | set(_AGENTS_PROSE_EXEMPT_GENUINE)
        | set(_AGENTS_PROSE_INCIDENTAL_EXEMPT)
    )
    mb: str | None = None
    deleted: list[str] = []
    for rel in keys:
        if (_REPO / rel).is_file():
            continue
        if mb is None:
            mb = _merge_base_origin_main()  # lazy: zero git calls when all exist
        if _existence_verdict(on_disk=False, at_merge_base=_present_at_ref(mb, rel)) == "fail":
            deleted.append(rel)
    assert not deleted, (
        "FAMILY_OF / exempt-dict key(s) present at merge-base(HEAD, "
        "origin/main) but ABSENT on disk — this tree deleted them without "
        "cleaning the coupling. Remove, in the deleting commit: the "
        "FAMILY_OF entries + SPECS/SPECS_10D tokens + _FORK_STUBS_2303 stub "
        "(members) or the exempt entry (exemptions): " + ", ".join(deleted)
    )


def test_agents_guard_exempt_split_hygiene_and_shape():
    """#2260 guard (20), part C: the two exempt dicts are mutually disjoint
    and disjoint from familied members; every on-disk exempt file still
    matches the reader pattern (a stale exemption fails loud); and every
    INCIDENTAL exempt file passes the AST shape check — NO agents-path read
    construct. Bare string constants never trigger, so fixture text and
    docstrings stay green; an incidental exemption that later gains a
    genuine agents read reds mechanically instead of staying silently green
    forever (what makes the exempt half of the guard falsifiable — the
    pre-#2260 single-dict hygiene checked existence/pattern/disjointness
    only and could not detect semantic growth). GENUINE entries are
    deliberately NOT shape-checked — they legitimately read agents prose
    (accepted residuals; relocation #2454, except
    tests/test_ensemble_review_cap.py, which #2420 owns)."""
    span = _step5a_span(_text())
    family_of = _family_of_map(span)
    gen = set(_AGENTS_PROSE_EXEMPT_GENUINE)
    inc = set(_AGENTS_PROSE_INCIDENTAL_EXEMPT)
    assert not gen & inc, f"exempt dicts must be disjoint; overlap: {sorted(gen & inc)}"
    familied = {k for k in family_of if k.startswith("tests/") and k.endswith(".py")}
    assert not (gen | inc) & familied, (
        f"exempt entries are also FAMILY_OF members — pick ONE disposition: "
        f"{sorted((gen | inc) & familied)}"
    )
    for rel in sorted(gen | inc):
        p = _REPO / rel
        if not p.is_file():
            continue  # absence is part B's job (merge-base discriminated)
        src = p.read_text(encoding="utf-8")
        assert _AGENTS_READER_RE.search(src), (
            f"{rel}: exempt entry no longer matches the agents reader "
            f"pattern — stale exemption; remove it"
        )
        if rel in inc:
            hits = _agents_read_constructs(src)
            assert not hits, (
                f"{rel}: INCIDENTAL exemption acquired a genuine agents-path "
                f"read construct — re-route it per guard (20)'s decision "
                f"procedure (FAMILY_agents membership if closure-clean, else "
                f"the GENUINE dict with a rationale + relocation plan). "
                f"Constructs:\n  "
                + "\n  ".join(h.strip().replace(chr(10), " ")[:100] for h in hits)
            )


def test_agents_guard_nonvacuity_pins():
    """#2260 guard (20), part E(i)-(iii) — the guard-19 non-vacuity
    pattern: (i) completeness — with one member token removed from the
    LIVE token list, the predicate reports it; (ii) join-form — the reader
    pattern matches the literal path-join source shape (a fixture string
    here, so the join class can never silently fall out of the pattern)
    and `.claude/agent-memory` matches NEITHER alternative; (iii) shape
    check — the AST detector FIRES on a real member file's module-level
    div-chain constant and PASSES on a synthetic fixture-string-only
    source."""
    span = _step5a_span(_text())
    tokens = _specs_tokens(span)
    family_of = _family_of_map(span)
    # (i) completeness non-vacuity, against the LIVE token list.
    victim = "tests/test_mapping_baselines_wiring_pins.py"
    assert victim in tokens, "the #2251 incident file must be a live SPECS token"
    problems = _agents_uncovered_readers([t for t in tokens if t != victim], family_of)
    assert any(p.startswith(victim) for p in problems), (
        f"guard (20) must report {victim} once its SPECS token is removed — "
        f"the completeness predicate would be vacuous; got {problems!r}"
    )
    # (ii) the join-form alternative + the agent-memory non-match.
    assert _AGENTS_READER_RE.search('(_REPO / ".claude" / "agents" / "analyzer.md")'), (
        "the reader pattern must match the quoted path-join form"
    )
    assert not _AGENTS_READER_RE.search('(_REPO / ".claude" / "agent-memory" / "x.md")'), (
        "`.claude/agent-memory` must not match the join alternative "
        "(closing quote required immediately after `agents`)"
    )
    assert not _AGENTS_READER_RE.search(".claude/agent-memory/implementer/MEMORY.md"), (
        "`.claude/agent-memory` must not match the literal alternative"
    )
    # (iii) shape-detector positive + negative controls.
    member_src = (_REPO / "tests" / "test_planner_phase_outputs_declaration.py").read_text(
        encoding="utf-8"
    )
    assert _agents_read_constructs(member_src), (
        "the shape detector must FIRE on "
        "tests/test_planner_phase_outputs_declaration.py's module-level "
        'PLANNER_MD div-chain (REPO_ROOT / ".claude" / "agents" / ...)'
    )
    synthetic = 'X = """fixture: .claude/agents/foo.md"""\nY = s.index(\'SPECS=".claude/agents\')\n'
    assert _agents_read_constructs(synthetic) == [], (
        "bare string constants (fixture text, index() args) must NOT trigger the shape detector"
    )


def test_agents_guard_vintage_discriminator_topology():
    """#2260 guard (20), part E(iv): the discriminator's verdict topology
    as a pure-function test (no live-repo mutation) — absent-on-disk +
    absent-at-merge-base => skip (pure vintage skew, green);
    absent-on-disk + present-at-merge-base => fail (this tree deleted it —
    the deleting-PR early warning); on-disk => check (the normal arms
    run)."""
    assert _existence_verdict(on_disk=False, at_merge_base=False) == "skip"
    assert _existence_verdict(on_disk=False, at_merge_base=True) == "fail"
    assert _existence_verdict(on_disk=True, at_merge_base=False) == "check"
    assert _existence_verdict(on_disk=True, at_merge_base=True) == "check"
