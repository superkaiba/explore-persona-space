---
title: 'daily-fix: wedge-failover checks live owner before terminate'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fcf425d14bc3
- daily-auto-filed
created_at: '2026-07-25T06:47:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): The watcher pod-safety
  wedge arm terminated pod-1586 mid-crash-fix-round at 2026-07-24T05:33Z while the
  owning /issue 1586 session was live and actively fixing, destroying local run state
  and forcing a from-p0 relaunch; no failover record was posted on the issue''s events
  so the owner misdiagnosed it as a provider re-rent for ~90 min'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session dd0af0ae, task #1586).

## Goal

The watcher's pod-safety wedge-failover arm must not terminate a pod out from under a live owning session mid-crash-fix-round, and must leave a durable record of the failover on the owning issue's events.

## Workflow gap

- **Bug observed:** at 2026-07-24T05:33Z the #770-lane wedge arm terminated + re-provisioned pod-1586 while the owning autonomous `/issue 1586` session was mid-crash-fix-round (relaunches 3-6 with active implementer/review cycles). All local run state (checkpoints, done phases) was destroyed, forcing a from-p0 "adopted" relaunch 7. No marker was posted on #1586's events, so the owner misdiagnosed the event as a provider-side re-rent/volume wipe (#1112 class) and corrected the record only ~90 min later (07:06Z "RECORD CORRECTION + false-dead veto" marker on #1586).
- **Why it is a workflow gap:** the wedge arm's provably-safe predicate (#770) reads pod/input state but not OWNING-SESSION liveness/activity; and the failover leaves no issue-side record, violating the durable-record norm every other watcher arm follows.
- **Confidence (emitter):** high on the incident; medium on the exact predicate design (planner decides escalate-vs-terminate criteria).
- verified-at-filing: `grep -n 'wedge' scripts/autonomous_session_watch.py` → wedge arm present (multiple hits incl. the #770/#1582 lanes); `git log --oneline --since='7 days ago' -- scripts/autonomous_session_watch.py` → 5 commits (#1653/#1582/#1564/#1532/#1519), none touching wedge-arm owner-liveness or issue-side failover records (2026-07-25). Incident evidence is #1586 events 05:33-07:06Z (marker record), not grep-verifiable — labeled from session records.

## Proposed change (candidate diff sketch — refine in planning)

Add to the wedge-failover arm: (a) an owner-liveness probe (live registered session for the issue + recent crash-fix-round markers, e.g. epm:failure/epm:progress newer than ~1-2h) that converts auto-terminate into ALERT/escalate; (b) unconditional posting of a failover marker (pod id, old/new, reason) on the owning issue's events.jsonl at action time.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (pod-safety wedge arm)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: fcf425d14bc3

- workflow_fix_target: scripts/autonomous_session_watch.py
