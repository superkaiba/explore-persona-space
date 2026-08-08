---
title: 'workflow-fix: sidecar canary tolerant of foreign concurrent '
kind: infra
tags:
- wf-fix
- wf-fix-fp:253371f2c59b
- daily-auto-filed
- trigger-dense
created_at: '2026-08-02T07:03:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): test_zz_production_sidecar_untouched_by_suite
  false-FAILed a Step 9c gate on #1876 (assert 56632 == 56332): two concurrent sessions''
  REAL production denies (guard-deny-events.jsonl rows at 2026-08-01T09:59:58Z and
  10:13:13Z, guard=repo_root_branch) appended during the 36-min 126-file gate window,
  moving the sidecar size between module-import snapshot and the end-of-module check.'
workflow: v1
---
# workflow-fix: sidecar canary tolerant of foreign concurrent denies

## Overview / Motivation

Auto-filed by the /daily 2026-08-01 Step C parked-candidate sweep from a workflow-fix candidate parked on task #1876 (emitting agent: the #1876 session, recursion-guarded; formal candidate block, fingerprint 253371f2c59b).

## Goal

Make `test_zz_production_sidecar_untouched_by_suite` concurrent-append-tolerant: fail only on rows the suite itself wrote, tolerate valid production deny rows appended by foreign concurrent sessions during the gate window.

## Workflow gap

- **Bug observed:** `test_zz_production_sidecar_untouched_by_suite` false-FAILed a Step 9c gate on #1876 (`assert 56632 == 56332`): two concurrent sessions' REAL production denies (`guard-deny-events.jsonl` rows at 2026-08-01T09:59:58Z and 10:13:13Z, guard=repo_root_branch) appended during the 36-min 126-file gate window, moving the sidecar size between module-import snapshot and the end-of-module check.
- **Why it is a workflow gap:** the canary's own docstring names this false-fail as an accepted caveat ("denies are rare exception events"), but on a 30+-session fleet running multi-file Step 9c gates the import-to-execution window is minutes long, so the accepted-rare event now burns full gate re-runs (36 min each) and reads as a NEW failure in the step9c_baseline compare (the pristine single-file oracle never sees a growth window, so it can never strip it).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n '_PROD_SIDECAR_SIZE_AT_IMPORT\|test_zz_production_sidecar_untouched_by_suite' tests/test_guard_repo_root_branch.py` → 3 hits in the named target (size snapshot at L116, test def at L4044, raw-size assert `assert current == _PROD_SIDECAR_SIZE_AT_IMPORT` at L4055) (2026-08-02 UTC). Landed-fix check: `git log --oneline --since='7 days ago' -- tests/test_guard_repo_root_branch.py` → 3 commits (#1861/#1859/#1710 guard-mask work), none touching the sidecar canary's size-equality shape.

## Proposed change (candidate diff sketch — refine in planning)

```diff
- _PROD_SIDECAR_SIZE_AT_IMPORT = _PROD_SIDECAR.stat().st_size if ...
+ _PROD_SIDECAR_ROWS_AT_IMPORT = _read_rows(_PROD_SIDECAR)
  ...
- assert current == _PROD_SIDECAR_SIZE_AT_IMPORT
+ new_rows = _read_rows(_PROD_SIDECAR)[len(_PROD_SIDECAR_ROWS_AT_IMPORT):]
+ suite_rows = [r for r in new_rows if _written_by_this_suite(r)]
+ assert suite_rows == [], suite_rows   # foreign concurrent denies tolerated
```

Alternative from the candidate: tolerate growth whose appended rows parse as valid production deny events from foreign pids; keep the hard fail for rows the suite itself wrote.

## Scope / surfaces

- Primary target: `tests/test_guard_repo_root_branch.py`
- Grep the workflow surface for sibling sidecar-size canaries before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/test_guard_repo_root_branch.py
- fingerprint: 253371f2c59b
- origin: parked candidate on task #1876, ts 2026-08-01T10:43:05Z, routed by /daily 2026-08-01 Step C.

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_guard_repo_root_branch.py
bug_observed: test_zz_production_sidecar_untouched_by_suite false-FAILed a Step 9c gate on #1876 (assert 56632 == 56332): two concurrent sessions' REAL production denies (guard-deny-events.jsonl rows at 2026-08-01T09:59:58Z and 10:13:13Z, guard=repo_root_branch) appended during the 36-min 126-file gate window, moving the sidecar size between module-import snapshot and the end-of-module check.
why_workflow_gap: the canary's own docstring names this false-fail as an accepted caveat ("denies are rare exception events"), but on a 30+-session fleet running multi-file Step 9c gates the import-to-execution window is minutes long, so the accepted-rare event now burns full gate re-runs (36 min each) and reads as a NEW failure in step9c_baseline compare (the pristine single-file oracle never sees a growth window, so it can never strip it).
proposed_change: make the canary concurrent-append-tolerant: compare COUNT of suite-attributable rows instead of raw size — e.g. snapshot the sidecar rows at import and assert that no NEW row carries this run's pid/session (or a suite-marker env value), OR tolerate growth whose appended rows parse as valid production deny events from foreign pids; keep the hard fail for rows the suite itself wrote.
confidence: medium
related_task: #1876
<!-- /workflow-fix-candidate -->
