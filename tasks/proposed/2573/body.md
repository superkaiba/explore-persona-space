---
title: 'Step 5a/10d FAMILY_OF lint family: couple tests/test_step9c_*.py with the
  synced selector — family sync itself creates the WORKFLOW_INVARIANT hybrid-tree
  skew (issue-2342 gate round burned)'
kind: infra
tags: []
created_at: '2026-08-25T11:01:57Z'
has_clean_result: false
workflow: v1
---
## Gap

The Step 5a family-atomic spec-freshness sync (and its Step 10d inline copy in `.claude/skills/issue/steps/18-step-10d.md`) puts `scripts/select_step9c_tests.py`, `scripts/step9c_baseline.py`-adjacent pins, `tests/test_select_step9c_tests.py`, and `tests/step9c_workflow_invariant_manifest.txt` in the `lint` family — but NOT the step9c-coupled test files themselves (`tests/test_step9c_*.py`, e.g. `tests/test_step9c_base_identity.py`, and any test file a main-side selector edit ADDS to `WORKFLOW_INVARIANT`, e.g. `tests/test_step9c_constructed_path_consumers.py`).

## Incident (issue #2342 Step 10d, 2026-08-25)

The pre-gate Step 5a re-sync imported main's `select_step9c_tests.py`, whose `WORKFLOW_INVARIANT` literal names `tests/test_step9c_constructed_path_consumers.py`. That test file (added on main after the branch forked) and the updated `tests/test_step9c_base_identity.py` sit OUTSIDE the sync family, so the branch tree went hybrid: synced selector + stale/missing coupled tests. The Step 10d mapped-invariant gated leg then failed 3 NEW nodes (`test_pinned_invariant_list_matches_live_tree` on the missing file; 2 `test_step9c_base_identity` nodes on the stale copy) → verdict `block` → one full ~5h gate round burned + a manual 2-file `git checkout origin/main --` repair (commit 2e4e566865 on issue-2342) + a second full gate round. The residual is documented in 18-step-10d.md as "(α) non-family rules-pin tests", but here the trigger was the family sync ITSELF creating the skew — the remedy is family-membership, not documentation.

## Asked change

Add the step9c-coupled test files to the `lint` family in BOTH inlined FAMILY_OF copies (Step 5a in `.claude/skills/issue/SKILL.md` / its step file, and the Step 10d post-gate inline copy) and the `SPECS`/`SPECS_10D` pathspec lists: `:(glob)tests/test_step9c_*.py` (covers base_identity, baseline, constructed_path_consumers, and future additions), family `lint`. Rationale: any main-side `WORKFLOW_INVARIANT`/selector/step9c_baseline change is co-committed with its test files on main, so syncing the scripts without the tests is exactly the hybrid-tree shape the family-atomic design exists to prevent. Update the lint family drift pins (`tests/test_issue_skill_lint_family_sync.py` or equivalent) accordingly.

## Provenance

Found by the /issue 2342 orchestrator at Step 10d (pre-push lint gate round 1 block, adjudicated as sync skew; repair commit 2e4e566865). Not a duplicate of #2374 (LESSONS.md coupling), #2423 (script-only sync satisfiability probe), or #2424 (sibling-script probe input set) — same class, distinct file coupling.

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/steps/18-step-10d.md
wf_fix: true
