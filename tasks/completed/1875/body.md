---
title: 'daily-fix: /issue Step 0 preloads TaskOutput/Monitor schemas'
kind: infra
tags:
- wf-fix
- wf-fix-fp:92ce13e78071
- daily-auto-filed
created_at: '2026-07-30T07:13:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): Three sessions across three
  miner groups burned turns calling TaskOutput (or Monitor) with unloaded deferred
  schemas / string-typed params (InputValidationError) — autonomous /issue sessions
  always need these tools'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miners F-P6 (#1812), G-P9 (#1738), H-P7 — 3 independent sessions in one day).

## Goal

The deferred-schema trap on always-needed tools is cheap to remove at boot.

## Workflow gap

- **Bug observed:** Each incident: a wasted call + retry after loading the schema; CLAUDE.md documents the trap but sessions keep paying it at first use under time pressure.
- **Why it is a workflow gap:** Step 0 already preloads other tool schemas selectively (e.g. change_title, PushNotification, Cron*) — TaskOutput/Monitor are more universally needed than those.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n ToolSearch .claude/skills/issue/SKILL.md` -> L418/L890/L1229 (existing selective preloads; TaskOutput/Monitor absent) (2026-07-30, this run).

## Proposed change (refine in planning)

One line in the Step 0 boot sequence; keep it lazy for interactive sessions if schema-load cost matters.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 92ce13e78071
