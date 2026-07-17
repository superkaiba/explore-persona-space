---
title: 'daily-fix: allowlist 952 dashboard in pod-shellout test'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b2bc17cbebce
- daily-auto-filed
created_at: '2026-07-17T06:50:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): workflow-invariant test
  test_no_pod_side_task_py_shellout fails on pristine origin/main — scripts/issue952_behavior_dashboard.py:484,505
  shell out to scripts/task.py find with neither a _LOCAL_VM_ONLY_PATHS entry nor
  an epm-lint waiver'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from four independent parked candidates on tasks #1392, #1402 (formal, fp fa90bebf7734), #1407, #1433 (formal, fp 339f00bf1756), plus prose mentions on #1397/#1399/#1424.

## Goal

Restore tests/test_no_pod_side_task_py_shellout.py to green on pristine main by allowlisting (or refactoring) scripts/issue952_behavior_dashboard.py's two task.py find shellouts.

## Workflow gap

- **Bug observed:** workflow-invariant test test_no_pod_side_task_py_shellout fails on pristine origin/main — scripts/issue952_behavior_dashboard.py:484,505 shell out to scripts/task.py find with neither a _LOCAL_VM_ONLY_PATHS entry nor an epm-lint waiver
- **Why it is a workflow gap:** A workflow-invariant test red on main means the invariant no longer blocks NEW pod-side shellouts crisply; every task's Step 9c gate must baseline-strip it and every session re-proves pre-existence.
- **Confidence (emitter):** high (4 independent raises in one day)
- verified-at-filing: `uv run pytest tests/test_no_pod_side_task_py_shellout.py -x -q` -> FAILED on current main (2026-07-17 UTC run at filing time); `grep -n issue952_behavior_dashboard tests/test_no_pod_side_task_py_shellout.py` -> 0 hits (allowlist entry absent — the absence claim); `grep -n 'scripts/task.py' scripts/issue952_behavior_dashboard.py` -> 2 hits (L484, L505)

## Proposed change (candidate diff sketch — refine in planning)

Add `"scripts/issue952_behavior_dashboard.py"` to `_LOCAL_VM_ONLY_PATHS` with a comment (VM-only dashboard builder, #952 inline rounds; never pod-invoked), OR replace the two `subprocess.run([..., "scripts/task.py", "find", "952"])` sites with `explore_persona_space.task_workflow.tasks_dir()`-based resolution. Planner picks.

## Scope / surfaces

- Primary target: `tests/test_no_pod_side_task_py_shellout.py`
- Secondary: scripts/issue952_behavior_dashboard.py (if the refactor path is chosen)
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b2bc17cbebce


Verbatim formal candidate blocks live on tasks #1402 and #1433 events.jsonl (epm:workflow-fix-candidate, 2026-07-16T19:57:45Z / 2026-07-17T03:03:01Z).
