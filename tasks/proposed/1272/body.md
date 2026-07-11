---
title: 'daily-fix: verified-at-filing grep line in wf-fix template'
kind: infra
tags:
- wf-fix
- wf-fix-fp:110bd03ac6be
- daily-auto-filed
created_at: '2026-07-11T06:52:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): three tasks filed 2026-07-09/10
  carried stale/overcounted claims (#1221: 3 of 4 target scripts never had the claimed
  calls; #1229: ''16 unguarded sites'' were mostly already-guarded; #1249: the claimed
  registration path was a transcript improvisation) - each spawned session burned
  verification rounds proving no-change; the ''grep the surface before emitting''
  rule exists but candidate synthesis off tr'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

require a 'verified-at-filing: <grep cmd + hit count>' line in the auto-filed body template (workflow-fix-on-bug.md section Body-file template) so filers must run the grep at synthesis time; planner should weigh the added filing friction (the emitting miner suggested route 3 for exactly that tradeoff) and may deflect with a reasoned no-change

## Workflow gap

- **Bug observed:** three tasks filed 2026-07-09/10 carried stale/overcounted claims (#1221: 3 of 4 target scripts never had the claimed calls; #1229: '16 unguarded sites' were mostly already-guarded; #1249: the claimed registration path was a transcript improvisation) - each spawned session burned verification rounds proving no-change; the 'grep the surface before emitting' rule exists but candidate synthesis off transcript prose skips it
- **Provenance / evidence:** miner-01 P7, /daily 2026-07-10 transcript sweep (3 wasted verification rounds in one day).

## Scope / surfaces

- Primary target: `.claude/rules/workflow-fix-on-bug.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 110bd03ac6be

- workflow_fix_target: .claude/rules/workflow-fix-on-bug.md
