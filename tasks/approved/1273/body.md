---
title: 'daily-fix: driver guards the daily-fix title prefix'
kind: infra
tags:
- wf-fix
- wf-fix-fp:848772dc0b24
- daily-auto-filed
created_at: '2026-07-11T06:54:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): the 2026-07-09 driver run
  filed 26 route-2 tasks (#1221-#1246) with BARE manifest titles (no ''daily-fix:''
  prefix) - the /daily SKILL route-2 contract puts the prefix inside --title, and
  task_workflow.is_open_workflow_fix_task''s dedup predicate REQUIRES a workflow-fix:/daily-fix:
  title prefix, so those filings are invisible to the title-prefix dedup surface'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

daily_drive_filings.py prepends 'daily-fix: ' to a route-2 manifest title that lacks a WF_FIX_TITLE_PREFIXES prefix (before the [:60] truncation), so a manifest composed without the prefix still satisfies the dedup contract

## Workflow gap

- **Bug observed:** the 2026-07-09 driver run filed 26 route-2 tasks (#1221-#1246) with BARE manifest titles (no 'daily-fix:' prefix) - the /daily SKILL route-2 contract puts the prefix inside --title, and task_workflow.is_open_workflow_fix_task's dedup predicate REQUIRES a workflow-fix:/daily-fix: title prefix, so those filings are invisible to the title-prefix dedup surface
- **Provenance / evidence:** /daily 2026-07-10 run observation: #1221 frontmatter title is bare while the SKILL.md route-2 command block specifies the prefixed form; the 07-10 manifest carried the prefix manually.

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 848772dc0b24

- workflow_fix_target: scripts/daily_drive_filings.py
