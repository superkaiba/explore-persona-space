---
title: 'daily-fix: stop releases registration + dispatch lease'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7a1af9675caa
- daily-auto-filed
created_at: '2026-07-17T06:57:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): spawn_session.py stop --session-id
  leaves the crash-recovery registration + per-issue dispatch lease in place, so a
  deliberate stop->respawn gets the replacement killed as a duplicate (registration
  still names the stopped session, 690s into a 900s window) — the #1090 scope-broadening
  tonight took a manual unregister + two lease clears + 3 spawn attempts'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (chat a996d49a, 20:56-20:58Z, #1090 follow-up session churn).

## Goal

Make deliberate stop-then-respawn a one-shot operation.

## Workflow gap

- **Bug observed:** spawn_session.py stop --session-id leaves the crash-recovery registration + per-issue dispatch lease in place, so a deliberate stop->respawn gets the replacement killed as a duplicate (registration still names the stopped session, 690s into a 900s window) — the #1090 scope-broadening tonight took a manual unregister + two lease clears + 3 spawn attempts
- **Why it is a workflow gap:** stop is the sanctioned manual path; leaving stale registration/lease state behind turns every deliberate restart into a guard fight.
- **Confidence (emitter):** high (reproduced tonight: 3 spawn attempts + manual state clears)
- verified-at-filing: `grep -n 'dispatch.lease\|def.*stop' scripts/spawn_session.py` -> lease machinery at L93-L237 (#843 M1) with no stop-side release call visible; incident: replacement spawn killed as duplicate at 20:57Z (transcript a996d49a)

## Proposed change (candidate diff sketch — refine in planning)

on stop with the process confirmed dead, atomically unregister the crash-recovery registration and release the issue dispatch lease so an immediate deliberate stop->respawn is one-shot

## Scope / surfaces

- Primary target: `scripts/spawn_session.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 7a1af9675caa

