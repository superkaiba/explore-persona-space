---
title: 'workflow-fix: sandbox test_vm_disk_guard against live tasks'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:3894bdb23b5b
created_at: '2026-07-04T07:10:56Z'
has_clean_result: false
origin_prompt: 'source: prose-followup (implementer r1, (d) section). Candidate: tests/test_vm_disk_guard.py
  (7 tests) + tests/test_vm_disk_guard_data_disk.py (1 test) exercise clean_issue_downloads(658,...)
  without rebinding task_workflow.tasks_dir, so they fail on clean main whenever a
  live active task declares data/issue_658/ as input (currently #661/#666/#742). Proposed
  change: apply the parity file''s fake_repo sandbox (or a synthetic issue number).
  routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this is a workflow-fix
  session; candidate logged + surfaced, not auto-filed; see .claude/rules/workflow-fix-on-bug.md
  § Recursion guard).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

apply the parity file's fake_repo sandbox (or a synthetic issue number) to the test_vm_disk_guard tests

## Workflow gap

- **Bug observed:** 8 tests exercise clean_issue_downloads(658,...) without rebinding task_workflow.tasks_dir, so they fail on clean main whenever a live active task declares data/issue_658/ as input (currently #661/#666/#742)
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

apply the parity file's fake_repo sandbox (or a synthetic issue number) to the test_vm_disk_guard tests

## Scope / surfaces

- Primary target: `tests/test_vm_disk_guard.py, tests/test_vm_disk_guard_data_disk.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: tests/test_vm_disk_guard.py, tests/test_vm_disk_guard_data_disk.py
- fingerprint: 3894bdb23b5b

source: prose-followup (implementer r1, (d) section). Candidate: tests/test_vm_disk_guard.py (7 tests) + tests/test_vm_disk_guard_data_disk.py (1 test) exercise clean_issue_downloads(658,...) without rebinding task_workflow.tasks_dir, so they fail on clean main whenever a live active task declares data/issue_658/ as input (currently #661/#666/#742). Proposed change: apply the parity file's fake_repo sandbox (or a synthetic issue number). routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this is a workflow-fix session; candidate logged + surfaced, not auto-filed; see .claude/rules/workflow-fix-on-bug.md § Recursion guard).
