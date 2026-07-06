---
title: 'daily-fix: task.py bounded wait on live rebase detached HEAD'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:a5a7c17106c7
created_at: '2026-07-04T07:13:17Z'
has_clean_result: false
origin_prompt: 'daily finding 3 (2026-07-03): task.py hard-refused mutations during
  concurrent rebases (detached HEAD), >=7 sessions hit.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a /daily 2026-07-03 finding.

## Goal

detect live .git/rebase-merge / rebase-apply state and bounded-wait (e.g. up to 120s) + retry before refusing

## Workflow gap

- **Bug observed:** during a concurrent session's `git pull --rebase` the shared repo root sits detached for ~minutes and task.py mutations hard-refuse ('main worktree HEAD is detached; re-attach to main') — hit >=7 sessions on 2026-07-03
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

detect live .git/rebase-merge / rebase-apply state and bounded-wait (e.g. up to 120s) + retry before refusing

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py, scripts/task.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py, scripts/task.py
- fingerprint: a5a7c17106c7

daily finding 3 (2026-07-03): task.py hard-refused mutations during concurrent rebases (detached HEAD), >=7 sessions hit.
