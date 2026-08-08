---
title: 'daily-fix: clamp composed poll sleeps below the Bash cap'
kind: infra
tags:
- wf-fix
- wf-fix-fp:94e2eddee78a
- daily-auto-filed
created_at: '2026-07-29T07:18:11Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1768''s poll tick composed
  an 1800s sleep into a bg-Bash whose tool cap is 600s — a doomed call plus stale-poll
  confusion afterward'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-I P10.

## Goal

Stop poll recipes composing sleep intervals the Bash tool cap kills.

## Workflow gap

- **Bug observed:** #1768's session composed an 1800 s wait into a background Bash call; the tool's 600 000 ms ceiling killed it, and the dead call produced stale-poll confusion on the next wake. (unverified hypothesis — verify at plan time: which recipe emitted the 1800 s figure — the miner did not locate the emitting line.)
- **Why it is a workflow gap:** poll recipes state intervals without a clamp against the documented Bash cap, so any long-cadence workload invites an over-cap sleep composition.
- **Confidence (emitter):** low-medium
- verified-at-filing: the 600 s Bash cap is documented (CLAUDE.md/tool docs); emitting-line unlocated (labeled).

## Proposed change (candidate diff sketch — refine in planning)

One clamp sentence at the poll-recipe interval guidance (<=540 s per call; chain ticks for longer cadences).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (poll/tick recipes)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 94e2eddee78a

- workflow_fix_target: .claude/skills/issue/SKILL.md

