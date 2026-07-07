---
title: 'daily-fix: refusal-wedged orchestrator ticks on harmful-cont'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b19229eccc7f
- daily-auto-filed
created_at: '2026-07-07T06:49:42Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): #1074''s orchestrator session
  (6f682c18, 2026-07-06) was refusal-killed on EVERY wake turn 20:36-22:33 UTC (~9
  subagent refusal-kills earlier in the day, then 38 orchestrator-turn refusals) —
  zero successful actions for 2h while Phase D Batch-API judge results sat unharvested;
  the watcher respawn (a2343e65) eventually recovered. CLAUDE.md''s spurious-refusals
  ladder covers subagent kills but has no'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Keep tick wake prompts vocabulary-thin for harmful-content-family tasks (tick_triage/issue-tick should not inline trigger-dense task titles/notes into the wake prompt), and document an orchestrator-refusal-wedge recovery step (e.g. fresh respawn with a thinned initial prompt) in the refusals ladder.

## Workflow gap

- **Bug observed:** #1074's orchestrator session (6f682c18, 2026-07-06) was refusal-killed on EVERY wake turn 20:36-22:33 UTC (~9 subagent refusal-kills earlier in the day, then 38 orchestrator-turn refusals) — zero successful actions for 2h while Phase D Batch-API judge results sat unharvested; the watcher respawn (a2343e65) eventually recovered. CLAUDE.md's spurious-refusals ladder covers subagent kills but has no answer for the ORCHESTRATOR itself being refused on every tick wake.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue-tick/SKILL.md, scripts/tick_triage.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: .claude/skills/issue-tick/SKILL.md, scripts/tick_triage.py
- source: /daily 2026-07-06 problem sweep (transcript-mined)
