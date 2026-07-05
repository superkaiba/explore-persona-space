---
title: 'daily-fix: Step 0 detect-and-exit posts an audit marker'
kind: infra
tags:
- wf-fix
- wf-fix-fp:657f7311793c
- daily-auto-filed
created_at: '2026-07-05T07:03:03Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): A /issue 958 session died
  ~1 minute after spawn on 2026-07-04 (04:06-04:07 UTC, 23 transcript lines) with
  no recorded reason - no detect-and-exit note, no error. A Step 0 single-orchestrator
  collision exit is indistinguishable from a wrapper crash in the audit trail.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Step 0 detect-and-exit posts one epm:progress note ('exiting: session collision with <id>' or the concrete guard reason) BEFORE ending, so a silent corpse is distinguishable from a crash.

## Workflow gap

- **Bug observed:** A /issue 958 session died ~1 minute after spawn on 2026-07-04 (04:06-04:07 UTC, 23 transcript lines) with no recorded reason - no detect-and-exit note, no error. A Step 0 single-orchestrator collision exit is indistinguishable from a wrapper crash in the audit trail.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Session: 21180785 (#958).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
