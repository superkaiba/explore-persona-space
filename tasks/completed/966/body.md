---
title: 'workflow-fix: distinctive --by for PM/watcher advisory markers'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:1c8dd6ddf671
created_at: '2026-07-04T07:09:44Z'
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

emitter-side --by convention for cross-session advisory posts (PM-chat/watcher advisory markers SHOULD set a distinctive --by, e.g. pm-chat / watcher), sharpening the mechanical triage layer

## Workflow gap

- **Bug observed:** PM-chat/watcher advisory markers post with a generic --by, so triage_candidates_since_last_dispatch cannot mechanically distinguish cross-session advisory posts
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

emitter-side --by convention for cross-session advisory posts (PM-chat/watcher advisory markers SHOULD set a distinctive --by, e.g. pm-chat / watcher), sharpening the mechanical triage layer

## Scope / surfaces

- Primary target: `.claude/agents/research-pm.md`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/agents/research-pm.md
- fingerprint: 1c8dd6ddf671

routed: parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Two concrete follow-ups surfaced by the plan (sec12) + Phase-2 critics, logged for the parent orchestrator / next PM pass, NOT auto-filed from this session:
1. target_file: .claude/agents/research-pm.md — emitter-side --by convention for cross-session advisory posts (PM-chat/watcher advisory markers SHOULD set a distinctive --by, e.g. pm-chat / watcher), sharpening triage_candidates_since_last_dispatch's mechanical layer. confidence: medium.
2. target_file: scripts/autonomous_session_watch.py (or scripts/tick_triage.py) — post-hoc compliance observer: a non-gating pass re-running triage_candidates_since_last_dispatch at recent compute breadcrumbs and flagging an 'external-markers triaged: none'/missing line against a non-empty candidate set (Statistics-lens S6 + Alternatives-lens concern 1; mechanizable). confidence: medium.
