---
title: 'daily-fix: refusal-wedge lane misses live wedges'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4843aad346e1
- daily-auto-filed
created_at: '2026-07-08T06:58:32Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): #1104 merged 10:34Z mid-incident
  yet the #1098 wedge (5bdae5b8, 08:33-11:13Z) ran 40 more minutes and was recovered
  only by the generic ALIVE-BUT-STALLED lane at 189min; the #1090 afternoon wedge
  (5e464f3d, 16:31-19:06Z, ~3.4h) also missed the fast lane; partial tick turns that
  post a heartbeat before dying refresh the self-report and defeat the >=1h-stale
  precondition.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

verify decide_prompt_wedge against the real #1098/#1090 transcript tails and fix the predicate: count a turn ENDING in isApiErrorMessage as a failed wake even when it posted a mid-turn heartbeat; check the 64KB tail window given multi-KB refusal rows

## Workflow gap

- **Bug observed:** #1104 merged 10:34Z mid-incident yet the #1098 wedge (5bdae5b8, 08:33-11:13Z) ran 40 more minutes and was recovered only by the generic ALIVE-BUT-STALLED lane at 189min; the #1090 afternoon wedge (5e464f3d, 16:31-19:06Z, ~3.4h) also missed the fast lane; partial tick turns that post a heartbeat before dying refresh the self-report and defeat the >=1h-stale precondition.
- **Why it is a workflow gap:** The refusal-aware lane exists precisely to cut these multi-hour wedges to ~30min; two same-day misses show the predicate/tail-window does not match production evidence shapes.

## Proposed change

Replay the #1098 (5bdae5b8, 08:52-11:13Z rows) and #1090 (5e464f3d, 16:31-19:06Z) transcript tails through decide_prompt_wedge with the production tail-window; fix whichever element prevented firing: (a) a turn that ends in isApiErrorMessage counts as a failed wake even when it posted a heartbeat mid-turn (do not let a dying turn refresh the staleness clock); (b) widen the tail window / row classification if multi-KB refusal + skill-inject rows push evidence out of the window. Consider (planner decides) a proactive fresh-session respawn once refusals exceed N/hour even when some wakes still land (c16b10ca lost ~every other wake 00:26-06:07Z while never tripping the lane). Regression test pinned to the real transcript shapes.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: 5bdae5b8 (#1098) 08:33-11:13Z; 5e464f3d (#1090) 16:31-19:06Z; c16b10ca (#1090) refusal storm 00:26-06:07Z.
