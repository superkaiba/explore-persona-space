---
title: 'daily-fix: mechanically block piped git push (mask-prone)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d28b1fab68c5
- daily-auto-filed
created_at: '2026-07-05T07:02:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): The CLAUDE.md ''never pipe
  a git push'' rule keeps failing open: on 2026-07-04 the #957 session piped its Step
  10d push through a filter, the pipe masked the rejected push, and the session proceeded
  believing the merge landed until the PR state contradicted it (4 sessions hit the
  same on 07-02).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Add a mechanical guard: a PreToolUse Bash hook (or workflow_lint check for committed recipes) that blocks git push piped into tail/grep/head, pointing at the bare-push + exit-code rule.

## Workflow gap

- **Bug observed:** The CLAUDE.md 'never pipe a git push' rule keeps failing open: on 2026-07-04 the #957 session piped its Step 10d push through a filter, the pipe masked the rejected push, and the session proceeded believing the merge landed until the PR state contradicted it (4 sessions hit the same on 07-02).
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/hooks/, scripts/workflow_lint.py`
- Session: 4e0b440a (#957, 06:34 UTC). The prose rule alone is demonstrably not holding; the sleep-poll guard is the precedent for mechanizing it.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/hooks/, scripts/workflow_lint.py
- source: /daily 2026-07-04 problem sweep (transcript-mined)
