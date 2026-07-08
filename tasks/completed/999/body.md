---
title: 'daily-fix: fix session_progress_report write_self_report crash'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:fd532a21ffdb
created_at: '2026-07-04T07:13:32Z'
has_clean_result: false
origin_prompt: 'daily finding 6 (2026-07-03): session_progress_report.py Traceback
  at line 362 in write_self_report (#906 session).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a /daily 2026-07-03 finding.

## Goal

fix the write_self_report crash and add a regression test

## Workflow gap

- **Bug observed:** crashed with a Traceback at line 362 in write_self_report during a #906 session on 2026-07-03
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

fix the write_self_report crash and add a regression test

## Scope / surfaces

- Primary target: `scripts/session_progress_report.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/session_progress_report.py
- fingerprint: fd532a21ffdb

daily finding 6 (2026-07-03): session_progress_report.py Traceback at line 362 in write_self_report (#906 session).
