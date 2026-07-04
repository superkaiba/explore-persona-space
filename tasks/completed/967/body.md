---
title: 'workflow-fix: watcher post-hoc external-marker triage observer'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:3a064def88df
created_at: '2026-07-04T07:09:59Z'
has_clean_result: false
origin_prompt: 'routed: parked — running under workflow_fix_target Provenance (recursion
  guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Two concrete follow-ups
  surfaced by the plan (sec12) + Phase-2 critics, logged for the parent orchestrator
  / next PM pass, NOT auto-filed from this session:

  1. target_file: .claude/agents/research-pm.md — emitter-side --by convention for
  cross-session advisory posts (PM-chat/watcher advisory markers SHOULD set a distinctive
  --by, e.g. pm-chat / watcher), sharpening triage_candidates_since_last_dispatch''s
  mechanical layer. confidence: medium.

  2. target_file: scripts/autonomous_session_watch.py (or scripts/tick_triage.py)
  — post-hoc compliance observer: a non-gating pass re-running triage_candidates_since_last_dispatch
  at recent compute breadcrumbs and flagging an ''external-markers triaged: none''/missing
  line against a non-empty candidate set (Statistics-lens S6 + Alternatives-lens concern
  1; mechanizable). confidence: medium.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add a non-gating watcher pass re-running triage_candidates_since_last_dispatch at recent compute breadcrumbs and flagging a missing/none triage line against a non-empty candidate set

## Workflow gap

- **Bug observed:** no post-hoc check that a dispatch actually triaged external advisory markers; an 'external-markers triaged: none' line against a non-empty candidate set goes unnoticed
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add a non-gating watcher pass re-running triage_candidates_since_last_dispatch at recent compute breadcrumbs and flagging a missing/none triage line against a non-empty candidate set

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py, scripts/tick_triage.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py, scripts/tick_triage.py
- fingerprint: 3a064def88df

routed: parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Two concrete follow-ups surfaced by the plan (sec12) + Phase-2 critics, logged for the parent orchestrator / next PM pass, NOT auto-filed from this session:
1. target_file: .claude/agents/research-pm.md — emitter-side --by convention for cross-session advisory posts (PM-chat/watcher advisory markers SHOULD set a distinctive --by, e.g. pm-chat / watcher), sharpening triage_candidates_since_last_dispatch's mechanical layer. confidence: medium.
2. target_file: scripts/autonomous_session_watch.py (or scripts/tick_triage.py) — post-hoc compliance observer: a non-gating pass re-running triage_candidates_since_last_dispatch at recent compute breadcrumbs and flagging an 'external-markers triaged: none'/missing line against a non-empty candidate set (Statistics-lens S6 + Alternatives-lens concern 1; mechanizable). confidence: medium.
