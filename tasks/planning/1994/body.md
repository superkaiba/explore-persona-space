---
title: 'workflow-fix: phase-done lint test hangs at interpreter exit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7afc39759a27
- daily-auto-filed
created_at: '2026-08-02T07:06:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): A standalone pytest session
  of tests/test_workflow_lint_phase_done_check.py hangs at interpreter exit (>240s,
  timeout-killed) after all 30 tests PASS — reproduced 3x on the pristine main checkout;
  a bare `import workflow_lint` exits cleanly, implicating test-triggered non-daemon
  thread residue (workflow_lint''s probe ThreadPoolExecutor, scripts/workflow_lint.py:10050),
  the same slow-exit family as'
workflow: v1
---
# workflow-fix: phase-done test file hangs at interpreter exit — join/daemonize probe threads or register surcharge

## Overview / Motivation

Auto-filed by the /daily 2026-08-01 Step C parked-candidate sweep from a workflow-fix candidate parked on task #1880 (emitting agent: implementer round 1, recursion-guarded; formal candidate block, fingerprint 7afc39759a27).

## Goal

Stop `tests/test_workflow_lint_phase_done_check.py` from wedging implementers' gate-matched local pytest runs: fix the interpreter-exit hang (thread residue from workflow_lint's probe ThreadPoolExecutor) or register the file in the Step 9c slow-test surcharge registry.

## Workflow gap

- **Bug observed:** a standalone pytest session of `tests/test_workflow_lint_phase_done_check.py` hangs at interpreter exit (>240s, timeout-killed) after all 30 tests PASS — reproduced 3x on the pristine main checkout by the emitter; a bare `import workflow_lint` exits cleanly, implicating test-triggered non-daemon thread residue (workflow_lint's probe ThreadPoolExecutor, `scripts/workflow_lint.py` ~L10050), the same slow-exit family as `tests/test_workflow_lint.py`'s registered 2400s surcharge. (Repro count is the emitter's own probe — `probed` per the miner-field contract; re-verify the hang at plan time.)
- **Why it is a workflow gap:** the selector's slow-test surcharge registry does not cover this file, so every implementer's gate-matched local run wedges on it (the #1880 round burned three timeout-killed union chunks isolating it) while the full-suite Step 9c gate absorbs it invisibly.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'test_workflow_lint_phase_done_check' scripts/select_step9c_tests.py` → 0 hits (absence-of-registration claim confirmed in-target; the sibling `tests/test_workflow_lint.py` IS registered at select_step9c_tests.py:330, confirming the registry exists and covers the same slow-exit family); `sed -n '10048,10052p' scripts/workflow_lint.py` → `with ThreadPoolExecutor(max_workers=max_workers) as ex:` present at the claimed site (2026-08-02 UTC). Landed-fix check: `git log --oneline --since='7 days ago' -- scripts/select_step9c_tests.py` → 4 commits (#1910/#1897/#1876/#1875), none registering this file. NOTE: the hang itself (>240s, 3x repro) is the emitter's measurement — `unverified hypothesis — verify at plan time: the hang reproduces on the current main checkout` (the ThreadPoolExecutor is context-managed, so the leak mechanism needs confirmation).

## Proposed change (candidate diff sketch — refine in planning)

```diff
+ # select_step9c_tests.py slow-surcharge registry:
+ "tests/test_workflow_lint_phase_done_check.py": <measured>s surcharge,
+   # 30/30 tests pass in ~40s; session EXIT hangs >240s standalone (thread residue,
+   # workflow_lint.py:10050 ThreadPoolExecutor family — same driver as test_workflow_lint.py)
```

Alternative (preferred if the leak confirms): daemonize/join the probe threads at workflow_lint module scope so the interpreter exits cleanly.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`, `tests/test_workflow_lint_phase_done_check.py`, `scripts/workflow_lint.py`
- Grep for other ThreadPoolExecutor probe sites in workflow_lint.py before editing; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py, tests/test_workflow_lint_phase_done_check.py, scripts/workflow_lint.py
- fingerprint: 7afc39759a27
- origin: parked candidate on task #1880, ts 2026-08-01T13:08:59Z, routed by /daily 2026-08-01 Step C.

<!-- workflow-fix-candidate v1 -->
target_file: scripts/select_step9c_tests.py, tests/test_workflow_lint_phase_done_check.py, scripts/workflow_lint.py
bug_observed: A standalone pytest session of tests/test_workflow_lint_phase_done_check.py hangs at interpreter exit (>240s, timeout-killed) after all 30 tests PASS — reproduced 3x on the pristine main checkout; a bare `import workflow_lint` exits cleanly, implicating test-triggered non-daemon thread residue (workflow_lint's probe ThreadPoolExecutor, scripts/workflow_lint.py:10050), the same slow-exit family as tests/test_workflow_lint.py's registered 2400s surcharge (1188.62s standalone).
why_workflow_gap: The selector's slow-test surcharge registry does not cover this file, so every implementer's gate-matched local run wedges on it (this round burned three timeout-killed union chunks isolating it) while the full-suite Step 9c gate absorbs it invisibly.
proposed_change: Fix the interpreter-exit hang (ensure workflow_lint's probe executor threads are joined/daemonized, or the tests shut them down) OR register tests/test_workflow_lint_phase_done_check.py in select_step9c_tests.py's slow-test surcharge registry so local gate-matched runs pre-emptively defer it.
confidence: medium
related_task: #1880
<!-- /workflow-fix-candidate -->
