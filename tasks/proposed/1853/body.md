---
title: 'daily-fix: watcher leaves filed infra tasks undispatched for'
kind: infra
tags:
- wf-fix
- wf-fix-fp:77c1ec5a75f0
- daily-auto-filed
created_at: '2026-07-30T07:02:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): 10 filed kind:infra tasks
  (#1823 urgent-main-red 22h, #1824, #1835, #1836, #1837, #1840, #1841, #1842, #1845,
  #1846) sat at proposed with only epm:created while later filings got sessions; the
  urgent-park router''s dedup branch posts a routed-record without dispatching the
  never-dispatched dedup target'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miners A-P1, E-P1, G-P1, J-P1 (4 independent groups), verified live by the orchestrator).

## Goal

A filed, ripe, auto-dispatchable infra task must actually get dispatched — the watcher's proposed_infra_sweep is the documented ~10-min backstop, and an urgent-main-red task must not sit behind a day-long queue.

## Workflow gap

- **Bug observed:** Ten filed infra tasks sat at `proposed` with only `epm:created` for 8-22h (#1823 filed 07:57Z by the urgent-park router with session_spawned:false; the router's later dedup hit against it posted a routed-record — 'deduped against #1823 ... session_spawned: false', #1800 events 2026-07-30T00:23:18Z — without dispatching it). Meanwhile main stayed red 40+h and every session's Step 9c gate re-classified the red (fixed tonight by /daily route-1). Later-created tasks (#1829, #1838, #1844) got sessions via their filers' direct spawn.
- **Why it is a workflow gap:** the sweep is the backstop for filers whose spawn no-ops (cap full / daemon unreachable); if it does not drain — or drains oldest-first with a permanently full cap and no urgency ordering (probed: ids.sort() oldest-first, miner A) — filed==dispatched breaks and urgent reds live for days.
- **Confidence (emitter):** medium
- verified-at-filing: `task.py view` on the 10 ids -> each status proposed, 1 event (2026-07-30, this run); E-P1 probed the dedup branch at scripts/autonomous_session_watch.py ~L10212 (`dedup_hit is not None` posts the record, never dispatches) and `~/.eps-autonomous/infra-drain-queue.json` last updated 2026-07-17.

## Proposed change (refine in planning)

Three legs, planner to scope: (a) urgent-router dedup branch best-effort dispatches a never-dispatched proposed kind:infra dedup target (same daemon/cap gates as _urgent_wf_park_file_and_dispatch); (b) urgent-main-red tag gets priority (or a fast lane past the cap) in proposed_infra_sweep; (c) root-cause why the sweep never dispatched any of the 10 (cap accounting during the 07-29 wave of ~38 sessions? predicate miss on router-filed tasks? the stale infra-drain-queue.json?).

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 77c1ec5a75f0
