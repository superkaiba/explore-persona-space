---
title: 'daily-fix: probe real-corpus plan assumptions at smoke time'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d07223b4f074
- daily-auto-filed
created_at: '2026-07-29T07:17:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1768''s production crash
  2: plan assumption 4 (n_distinct_prefix==1) was hard-asserted and false on the real
  corpus — ~55 min of production time lost to a mid-run assert a 1-row manifest read
  at smoke time would have caught'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-I P2.

## Goal

Probe arm-gating real-corpus structural assumptions in the smoke slice so their fail-loud asserts fire before production.

## Workflow gap

- **Bug observed:** #1768's production run crashed mid-flight on plan assumption 4 (`n_distinct_prefix==1`), which is false on the real corpus (n=2) — ~55 min lost, hot-fixed in-session. unverified hypothesis — verify at plan time: the exact assert location + the claim that a 1-row manifest read at smoke time exercises it (read from the transcript narrative + the session's hot-fix marker v18, not re-run).
- **Why it is a workflow gap:** §-assumptions entries are verified at plan time by the fact-checker where checkable, but REAL-corpus structural claims are only checkable against the corpus — the smoke slice is the first place that data exists, and no duty routes assumption probes there.
- **Confidence (emitter):** medium
- verified-at-filing: mechanism labeled above; `verify at plan time` markers included (the planner spec's §-assumptions section is the named target — re-read it before wording the duty).

## Proposed change (candidate diff sketch — refine in planning)

One §-assumptions clause in planner.md (+ possibly a smoke-checklist mirror in the implementer spec).

## Scope / surfaces

- Primary target: `.claude/agents/planner.md` (§-assumptions)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: d07223b4f074

- workflow_fix_target: .claude/agents/planner.md

