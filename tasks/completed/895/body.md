---
title: 'workflow-fix: Step 9c selector misses glob-scanning tests for scripts/*.py
  diffs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:99aa3ec12cbb
created_at: '2026-07-02T21:58:28Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #877 Phase-2 Alternatives critic + orchestrator
  Step 9c observation: select_step9c_tests.py maps tests by import only; glob-scanning
  guard tests (test_shared_vm_thread_caps.py) never enter the touched subset for scripts/*.py
  diffs'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #877 (emitting agents: the Phase-2 Alternatives critic + the orchestrator's own Step 9c run).

## Goal

Make `scripts/select_step9c_tests.py` include glob-scanning workflow-invariant tests (e.g. `tests/test_shared_vm_thread_caps.py`) when a diff adds or touches `scripts/*.py` entrypoints.

## Workflow gap

- **Bug observed:** the Step 9c touched-file selector maps tests by import only, so tests that cover touched files via directory glob scans (`test_shared_vm_thread_caps.py` walks `scripts/` itself) are omitted; new `scripts/*.py` offenders slip through Step 9c until someone runs the full suite on main.
- **Why it is a workflow gap:** two live instances — (a) 12 thread-caps offenders accreted AFTER the #847 freeze because per-task Step 9c `touched` runs never selected the guard test (#658/#722/#779 scripts); (b) on #877 itself the orchestrator had to HAND-APPEND `tests/test_shared_vm_thread_caps.py` to the selector's printed command because the selector missed the very acceptance test of the fix.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

Option A (simplest): add `tests/test_shared_vm_thread_caps.py` to `WORKFLOW_INVARIANT` in `scripts/select_step9c_tests.py` (~line 85).
Option B (targeted): a `GLOB_SCAN_TESTS = {"scripts/": ("tests/test_shared_vm_thread_caps.py",)}` map — include when any touched path matches the glob root. Planner picks; B avoids a fixed ~3s cost on non-script diffs but A is one line.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Grep before editing: `grep -rln 'select_step9c_tests' .claude/ CLAUDE.md scripts/ tests/` and update the paired test file (`tests/` pin for the selector, if present) + the /issue SKILL.md Step 9c text only if the contract wording changes.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: 99aa3ec12cbb

Surfaced prose (verbatim, Alternatives critic on #877): "12 offenders accreted AFTER the #847 freeze — meaning the guard test detects on full-suite runs but Step 9c touched-scope test runs let new scripts slip through until someone runs the suite on main. [...] the orchestrator may want to consider whether test_shared_vm_thread_caps.py should join the always-run set for any diff adding scripts/*.py". Orchestrator's own instance (#877 Step 9c): the selector's printed subset omitted tests/test_shared_vm_thread_caps.py for a diff touching 3 scripts/*.py files; appended by hand.
