---
title: 'daily-fix: watcher stall detection — marker-recency cross-check, fencing,
  registration hygiene'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:b4b5688ebd51
created_at: '2026-07-02T07:12:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-01 problem sweep route-2: (a) #763: watcher respawned
  a new /issue driver keyed on stale Happy self-report while the old session was ALIVE
  and polling — two drivers overlapped ~4h posting markers on the same task. (b) #812:
  PM-side recovery respawned while the ''dead'' session was mid-implementer (test
  file edited 57s earlier) — new session had to process-hunt and stop it. (c) #811:
  two consecutive ALIVE-BUT-STALLED sessions'
---
## Overview / Motivation

Auto-filed by the nightly /daily problem sweep (2026-07-01) — route 2 (behavior/logic change requiring independent review through the full /issue pipeline).

## Goal

Before declaring a session stalled: cross-check the task's events.

## Bug observed (2026-07-01 sessions)

(a) #763: watcher respawned a new /issue driver keyed on stale Happy self-report while the old session was ALIVE and polling — two drivers overlapped ~4h posting markers on the same task. (b) #812: PM-side recovery respawned while the 'dead' session was mid-implementer (test file edited 57s earlier) — new session had to process-hunt and stop it. (c) #811: two consecutive ALIVE-BUT-STALLED sessions could not be auto-respawned because the Happy daemon was unreachable at alert time — GPU idled until manual PM recovery. (d) #665: a 16h-transcript-idle registered session held the Step 0 guard against a legitimate re-drive. (e) #779: tick prompts were enqueued AND dequeued 5x with no turn executed (wedged session) — ~90 min lag before respawn killed an in-flight implementer.

All five sub-cases from 2026-07-01 transcripts; each cost duplicated work, killed subagents, or idle GPU.

## Proposed change (refine in planning)

Before declaring a session stalled: cross-check the task's events.jsonl marker recency (a heartbeat under ~2h means not stalled) AND probe process liveness + worktree file mtimes; on respawn, stop/fence the superseded session-id first; retry-with-backoff (or queue for next tick) when the daemon is unreachable during a respawn; unregister issue-mappings whose session is transcript-idle past the reconcile threshold; detect N consecutive dequeued tick prompts with no new transcript rows as a faster wedge trigger.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` no-flags default run passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: b4b5688ebd51
- source: /daily 2026-07-01 problem sweep (transcript miners)
