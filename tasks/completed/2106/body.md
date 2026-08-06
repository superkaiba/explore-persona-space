---
title: 'daily-fix: fix red main: test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass
  — capture each helper''s return and raise o'
kind: infra
tags:
- urgent-main-red
created_at: '2026-08-06T01:35:24Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue1739_sycoood_regen.py\n\
  bug_observed: three discarded returns of fail-soft upload helpers (_upload_folder_filtered\
  \ at lines 472 and 504, _upload at line 520) fail the no-flags workflow_lint upload-return-discard\
  \ check on pristine origin/main, making tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass\
  \ red fleet-wide\nwhy_workflow_gap: a fail-soft upload helper whose return is discarded\
  \ exits 0 on silent durability loss (upload-policy: 'upload returned no path' is\
  \ a TRACKED GAP, never warning-and-continue), and every intervening Step 9c/10d\
  \ gate must re-classify this pre-existing red until it is fixed\nproposed_change:\
  \ capture each helper's return and raise on empty (the hub.upload_raw_completions_to_data_repo\
  \ shape), or waive each call with '# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>' (>=10\
  \ chars) if the discard is deliberate\ndiff_sketch: |\n  - _upload_folder_filtered(...)\n\
  \  + base_url = _upload_folder_filtered(...)\n  + if not base_url:\n  +     raise\
  \ RuntimeError(\"upload returned no path (issue1739 raw completions)\")\n  (same\
  \ shape at lines 472 and 504; _upload(...) at line 520)\nconfidence: medium\nrelated_task:\
  \ #2085\nurgency: main-red\nfailing_test: tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass\n\
  wf_fix: false\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#2085. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass` red on origin/main: capture each helper's return and raise on empty (the hub.upload_raw_completions_to_data_repo shape), or waive each call with '# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>' (>=10 chars) if the discard is deliberate

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** three discarded returns of fail-soft upload helpers (_upload_folder_filtered at lines 472 and 504, _upload at line 520) fail the no-flags workflow_lint upload-return-discard check on pristine origin/main, making tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass red fleet-wide
- **Failing node (router-verified):** `tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass`
- **Confidence (emitter):** medium
- verified-at-filing: `uv run pytest tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass -q` -> rc=1 at main @ b2e2b32b97 (2026-08-06T01:35:20Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/issue1739_sycoood_regen.py`
- Failing node: `tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: 66efdea261e3
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1739_sycoood_regen.py
bug_observed: three discarded returns of fail-soft upload helpers (_upload_folder_filtered at lines 472 and 504, _upload at line 520) fail the no-flags workflow_lint upload-return-discard check on pristine origin/main, making tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass red fleet-wide
why_workflow_gap: a fail-soft upload helper whose return is discarded exits 0 on silent durability loss (upload-policy: 'upload returned no path' is a TRACKED GAP, never warning-and-continue), and every intervening Step 9c/10d gate must re-classify this pre-existing red until it is fixed
proposed_change: capture each helper's return and raise on empty (the hub.upload_raw_completions_to_data_repo shape), or waive each call with '# UPLOAD_RETURN_DISCARD_EXEMPT: <reason>' (>=10 chars) if the discard is deliberate
diff_sketch: |
  - _upload_folder_filtered(...)
  + base_url = _upload_folder_filtered(...)
  + if not base_url:
  +     raise RuntimeError("upload returned no path (issue1739 raw completions)")
  (same shape at lines 472 and 504; _upload(...) at line 520)
confidence: medium
related_task: #2085
urgency: main-red
failing_test: tests/test_workflow_lint.py::test_check_upload_return_discard_live_trees_pass
wf_fix: false
<!-- /workflow-fix-candidate -->
