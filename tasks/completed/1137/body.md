---
title: 'daily-fix: suppress stalled-alert on fresh poll-tick marker'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fd4b4f2ffea8
- daily-auto-filed
created_at: '2026-07-08T06:59:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): ALIVE-BUT-STALLED alerts
  fired on a by-design-parked blocked session (#1092, 15:33Z) and twice on a demonstrably
  healthy mid-stream session (21:03/21:23Z) whose long bg-poll gaps froze the Happy
  self-report; debounce prevented false respawns but one alert prompted a user query.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

suppress the stalled alert when the task's latest non-watcher marker (poll-tick, breadcrumb, long-phase-heartbeat) is fresher than the staleness threshold even if the Happy self-report title is stale

## Workflow gap

- **Bug observed:** ALIVE-BUT-STALLED alerts fired on a by-design-parked blocked session (#1092, 15:33Z) and twice on a demonstrably healthy mid-stream session (21:03/21:23Z) whose long bg-poll gaps froze the Happy self-report; debounce prevented false respawns but one alert prompted a user query.
- **Why it is a workflow gap:** Alert noise trains the reader to ignore the channel; the durable marker stream already distinguishes healthy-quiet from wedged.

## Proposed change

In the watcher stalled-session pass: before alerting, check the owning task latest non-watcher marker; a marker fresher than the staleness threshold suppresses the alert. Keep the alert for genuinely marker-silent sessions. Respawn predicates untouched. Low priority.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: cb3b6065 + 0bbf1182 (#1092) 15:33Z, 21:03Z, 21:23Z.
