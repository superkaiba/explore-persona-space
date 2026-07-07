---
title: 'daily-fix: guard_log_dump exempts code files from size heuri'
kind: infra
tags:
- wf-fix
- wf-fix-fp:487270b89d93
- daily-auto-filed
created_at: '2026-07-05T07:03:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): On 2026-07-04 (14:45 UTC)
  the guard_log_dump PreToolUse hook blocked #986''s legitimate sed range read of
  scripts/workflow_lint.py - a source file classified as ''log-like'' purely by whole-file
  size; a retry was blocked too and the agent had to work around with grep.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Exempt code/doc extensions (.py/.md/.sh/.yaml) from the log-dump heuristic, or gate on the size of the REQUESTED RANGE rather than whole-file size, so bounded range reads of large source files pass.

## Workflow gap

- **Bug observed:** On 2026-07-04 (14:45 UTC) the guard_log_dump PreToolUse hook blocked #986's legitimate sed range read of scripts/workflow_lint.py - a source file classified as 'log-like' purely by whole-file size; a retry was blocked too and the agent had to work around with grep.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/hooks/guard_log_dump.sh`
- Session: b5030d38 (#986).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/hooks/guard_log_dump.sh
- source: /daily 2026-07-04 problem sweep (transcript-mined)
