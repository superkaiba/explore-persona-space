---
title: 'daily-fix: watcher auto-recovers ALIVE-BUT-STALLED sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:92b5ee2d9672
- daily-auto-filed
created_at: '2026-07-04T22:38:19Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — watcher-stalled-autorecover (fp 92b5ee2d9672)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Add a bounded auto-recovery arm to the stall pass: a REGISTERED session at an ACTIVE status whose inner loop is provably dead past a threshold (transcript mtime / last-API-turn age / marker recency, ~60-90 min) is stopped + respawned, reusing the existing duplicate-respawn guard (#759) and stall detection (#845).

## Workflow gap

- **Bug observed:** 2026-07-03: #810's fit wedged 98 min (2% CPU, session frozen at 'Running tick_triage') and #816 idled 169 min — the watcher posted session-stalled ALERTS but never recovered; 9 more live-wrapper/dead-loop sessions in the evening outage were invisible to the orphan-respawn pass because they were still REGISTERED.
- **Why it matters:** Detection landed in #845 but recovery stayed manual; alert-only stalls cost hours per incident.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 92b5ee2d9672
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
