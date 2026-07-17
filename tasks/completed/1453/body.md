---
title: 'daily-fix: instant wedge lane for Prompt-is-too-long'
kind: infra
tags:
- wf-fix
- wf-fix-fp:89073e126181
- daily-auto-filed
created_at: '2026-07-17T06:56:53Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): a session whose turn dies
  with the synthetic ''Prompt is too long'' API error is unrecoverable in-session
  by definition (every future turn fails identically), but the watcher''s prompt-wedge
  lane waits for the generic >=3-failed-wake accumulation across 45-min tick spacing
  — #1335 lost ~65 min wedged (4 dead tick turns 12:48-13:50Z) with a one-line fix
  unapplied'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1335, ebff95d1: context-ceiling wedge 12:48-13:53Z, no compact_boundary fired, 4 consecutive dead tick turns).

## Goal

Cut context-ceiling wedge recovery from ~65+ min (3-wake accumulation) to one watcher tick.

## Workflow gap

- **Bug observed:** a session whose turn dies with the synthetic 'Prompt is too long' API error is unrecoverable in-session by definition (every future turn fails identically), but the watcher's prompt-wedge lane waits for the generic >=3-failed-wake accumulation across 45-min tick spacing — #1335 lost ~65 min wedged (4 dead tick turns 12:48-13:50Z) with a one-line fix unapplied
- **Why it is a workflow gap:** The wedge lane exists to bound dead-session time; a deterministic, textually-identifiable unrecoverable failure should not wait for statistical accumulation.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'Prompt is too long' scripts/autonomous_session_watch.py` -> 0 hits (no subclass handling — absence claim); wedge-lane counters exist (EPM_TICK_WEDGE_MIN_FAILED_TURNS lineage, #1127/#1209/#1241)

## Proposed change (candidate diff sketch — refine in planning)

treat 'Prompt is too long' as an instant prompt-wedge subclass: on the FIRST failed wake turn whose api-error text matches it, stop + force-respawn (existing episode-belt + day caps apply)

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 89073e126181

