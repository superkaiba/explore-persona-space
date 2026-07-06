---
title: 'daily-fix: watcher detects API-auth outage, stops respawn churn'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7c7600216452
- daily-auto-filed
created_at: '2026-07-04T22:38:06Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — watcher-auth-outage (fp 7c7600216452)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Add an auth-outage detection pass: N instant-freeze respawns / consecutive API-400 auth errors across sessions => suppress per-session respawns (backoff), send ONE loud Telegram push naming the auth failure, and resume respawns only after a cheap probe call succeeds.

## Workflow gap

- **Bug observed:** 2026-07-03 19:40-21:36Z fleet outage: every session died on API 400 'Not a valid API key for this workspace'; the watcher respawn-looped all evening while each fresh session froze on the same poisoned key (and the Happy daemon itself was unreachable part of the time); recovery was Thomas-initiated (/login + manual broadcast), ~2-3.5h of fleet stall across ~14+ sessions.
- **Why it matters:** The watcher's respawn machinery amplified the outage (churn, duplicate registrations) instead of surfacing it; a single alert + backoff converts hours of churn into one actionable ping.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 7c7600216452
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
