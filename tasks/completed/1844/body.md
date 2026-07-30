---
title: 'workflow-fix: add the expected pair line `tests/test_issue_skill_lint_fami'
kind: infra
tags:
- wf-fix
- wf-fix-fp:577984a32f98
- urgent-main-red
created_at: '2026-07-30T04:03:41Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: tests/test_select_step9c_tests.py\n\
  bug_observed: `tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree`\
  \ fails on origin/main (verified rc==1 at the main checkout): #1807's merge (9906468844)\
  \ added tests/test_issue_skill_lint_family_sync.py, whose text references select_step9c_tests.py\
  \ (~line 432), so the --map-files dependency arm now emits a 7th pair for the scripts/select_step9c_tests.py\
  \ payload while the exact-set live-tree pin (tests/test_select_step9c_tests.py:2183)\
  \ still expects 6.\nwhy_workflow_gap: a red live-tree selector pin forces every\
  \ intervening session's Step 9c gate + gate-scope duty to re-classify a failure\
  \ it did not cause (the #1643/#1681 fleet-wide-per-hour-cost class); the pin's own\
  \ docstring prescribes the deliberate 1-line update when a new consumer legitimately\
  \ appears.\nproposed_change: add the expected pair line `tests/test_issue_skill_lint_family_sync.py\\\
  tscripts/select_step9c_tests.py` (sorted position 3) to the test's expected list\
  \ and update its \"6 pairs\" docstring to 7.\ndiff_sketch: |\n  In tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree\
  \ (~L2183):\n  +    \"tests/test_issue_skill_lint_family_sync.py\\tscripts/select_step9c_tests.py\"\
  ,\n  (sorted position 3 in the expected list; docstring \"6 pairs\" -> \"7 pairs\"\
  )\nconfidence: high\nrelated_task: #1810\nurgency: main-red\nfailing_test: tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree\n\
  wf_fix: true\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1810. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree` red on origin/main: add the expected pair line `tests/test_issue_skill_lint_family_sync.py\tscripts/select_step9c_tests.py` (sorted position 3) to the test's expected list and update its "6 pairs" docstring to 7.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** `tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree` fails on origin/main (verified rc==1 at the main checkout): #1807's merge (9906468844) added tests/test_issue_skill_lint_family_sync.py, whose text references select_step9c_tests.py (~line 432), so the --map-files dependency arm now emits a 7th pair for the scripts/select_step9c_tests.py payload while the exact-set live-tree pin (tests/test_select_step9c_tests.py:2183) still expects 6.
- **Failing node (router-verified):** `tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree -q` -> rc=1 at main @ 74358a0786 (2026-07-30T04:03:16Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `tests/test_select_step9c_tests.py`
- Failing node: `tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session carries a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion
  guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: tests/test_select_step9c_tests.py
- fingerprint: 577984a32f98
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_select_step9c_tests.py
bug_observed: `tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree` fails on origin/main (verified rc==1 at the main checkout): #1807's merge (9906468844) added tests/test_issue_skill_lint_family_sync.py, whose text references select_step9c_tests.py (~line 432), so the --map-files dependency arm now emits a 7th pair for the scripts/select_step9c_tests.py payload while the exact-set live-tree pin (tests/test_select_step9c_tests.py:2183) still expects 6.
why_workflow_gap: a red live-tree selector pin forces every intervening session's Step 9c gate + gate-scope duty to re-classify a failure it did not cause (the #1643/#1681 fleet-wide-per-hour-cost class); the pin's own docstring prescribes the deliberate 1-line update when a new consumer legitimately appears.
proposed_change: add the expected pair line `tests/test_issue_skill_lint_family_sync.py\tscripts/select_step9c_tests.py` (sorted position 3) to the test's expected list and update its "6 pairs" docstring to 7.
diff_sketch: |
  In tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree (~L2183):
  +    "tests/test_issue_skill_lint_family_sync.py\tscripts/select_step9c_tests.py",
  (sorted position 3 in the expected list; docstring "6 pairs" -> "7 pairs")
confidence: high
related_task: #1810
urgency: main-red
failing_test: tests/test_select_step9c_tests.py::test_cli_map_files_transitive_pairs_live_tree
wf_fix: true
<!-- /workflow-fix-candidate -->
