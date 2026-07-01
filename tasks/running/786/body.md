---
title: 'daily-fix: task-workflow API atomicity (git-mv, post-marker, view)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ba9fdef5836a
- daily-auto-filed
created_at: '2026-07-01T06:53:12Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: Task #722 was genuinely corrupted
  this day: a non-atomic status-transition `git mv` split the task across two folders
  (events.jsonl+artifacts moved, body.md+plans did not) and left REGISTRY with no
  #7'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

(a) Make the status-transition move whole-task-dir + REGISTRY update atomic under one flock+commit, rolling back on partial failure before touching REGISTRY; (b) in get_task/_read_body raise a typed error naming the stale path + the `task.py audit` remedy; (c) make post-marker's events.jsonl append atomic (temp+rename, or hold flock across the full write) so a mid-write kill cannot leave a partial line; (d) wrap cmd_view's stdout print in a SIGPIPE-safe guard.

## Workflow gap

- **Bug observed:** Task #722 was genuinely corrupted this day: a non-atomic status-transition `git mv` split the task across two folders (events.jsonl+artifacts moved, body.md+plans did not) and left REGISTRY with no #722 entry; `get_task`/`_read_body` then crash with a bare FileNotFoundError instead of a typed 'registry stale, run task.py audit' error; a Bash-timeout-killed `post-marker` left a partial/corrupt events.jsonl append needing manual byte-offset truncation; and `task.py view --json` piped to a consumer that exits early tracebacks with BrokenPipeError.
- **Evidence:** living_docs check reports #696/#722/#750/#764/#766 registry/filesystem inconsistent today. Sources: /daily miners (batches 04/05/06/08).
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: ba9fdef5836a
- source: /daily route-2 (2026-06-30)
