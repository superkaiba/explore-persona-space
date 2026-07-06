---
title: 'daily-fix: both-arms mapping default applied at capture time'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3e82fe87700f
- daily-auto-filed
created_at: '2026-07-05T07:04:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): On 2026-07-04 (04:04-04:10
  UTC) Thomas filed a mapping experiment (#958) and had to correct the capture 6 minutes
  later: ''wait say also that we want to do context -> answer but also prefix -> answer
  mapping'' - despite the standing 2026-07-03 CLAUDE.md rule that every representation-mapping
  experiment runs BOTH prefix-based and context-based arms. The capture path applied
  the rule only at plan time'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

The task-capture path (chat capture + /issue Step 1 clarifier checklist) applies the prefix+context both-arms default at CAPTURE time for mapping/geometry tasks - one clarifier checklist line.

## Workflow gap

- **Bug observed:** On 2026-07-04 (04:04-04:10 UTC) Thomas filed a mapping experiment (#958) and had to correct the capture 6 minutes later: 'wait say also that we want to do context -> answer but also prefix -> answer mapping' - despite the standing 2026-07-03 CLAUDE.md rule that every representation-mapping experiment runs BOTH prefix-based and context-based arms. The capture path applied the rule only at plan time.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Session: f337f146 (interactive chat).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)
