---
title: 'daily-fix: watcher lane for zero-assistant-row boot deaths'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bd634a3b6f11
- daily-auto-filed
created_at: '2026-07-11T06:52:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): a dispatched /issue session
  that dies at skill load produces ZERO assistant rows, accumulates no failed wake
  turns, and so trips no wedge lane; the watcher loops dispatch -> boot-death -> 12h
  stale-registration-unregister -> re-dispatch silently (observed 2026-07-10T14:33Z
  through 07-11T03:14Z on tasks #1251-#1256, >=8 dead sessions; the #1209 freeze-below-threshold
  family one level earlier)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

extend the watcher wedge detection with a boot-death lane: a dispatched session whose transcript has zero assistant rows after N minutes is stopped + surfaced (push) instead of waiting 12h for the stale-registration pass

## Workflow gap

- **Bug observed:** a dispatched /issue session that dies at skill load produces ZERO assistant rows, accumulates no failed wake turns, and so trips no wedge lane; the watcher loops dispatch -> boot-death -> 12h stale-registration-unregister -> re-dispatch silently (observed 2026-07-10T14:33Z through 07-11T03:14Z on tasks #1251-#1256, >=8 dead sessions; the #1209 freeze-below-threshold family one level earlier)
- **Provenance / evidence:** miner-01 P1 part 3, /daily 2026-07-10 transcript sweep (evidence: repeated dispatch/unregister cycles on #1251/#1252/#1253 events.jsonl).

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: bd634a3b6f11

- workflow_fix_target: scripts/autonomous_session_watch.py
