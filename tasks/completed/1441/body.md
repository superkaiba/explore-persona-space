---
title: 'daily-fix: add #1383 context clause to yaml BINDING summary'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a6683eb3c015
- daily-auto-filed
created_at: '2026-07-17T06:51:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the line-3255 orchestrator
  filing-channel ''BINDING (#1307)'' summary states the per-target-hits, relocation-grep,
  and semantic-probe clauses but not the #1383 context-consistency clause (a presence
  hit whose surrounding context already implements the change = landed fix -> dedup,
  never file)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked prose candidate on task #1383 (Methodology critic, plan round 1). #1383 is completed (landed 2026-07-16), so the premise ('after #1383 lands') is ripe.

## Goal

Bring the workflow.yaml orchestrator filing-channel BINDING summary to full 3-of-3 clause coverage.

## Workflow gap

- **Bug observed:** the line-3255 orchestrator filing-channel 'BINDING (#1307)' summary states the per-target-hits, relocation-grep, and semantic-probe clauses but not the #1383 context-consistency clause (a presence hit whose surrounding context already implements the change = landed fix -> dedup, never file)
- **Why it is a workflow gap:** The yaml summary is the orchestrator-channel quick reference; a 2-of-3 clause summary invites exactly the #1330-style landed-fix filing the third clause exists to stop (no contradiction — line ~3259 defers to the rule doc — but a real coverage gap).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'BINDING (#1307)' .claude/workflow.yaml` -> 1 hit (L3255, states per-target-hits + relocation-grep + semantic-probe clauses); `grep -n 'context-consistency' .claude/workflow.yaml` -> 0 hits (absence claim); #1383 status: completed (task.py view)

## Proposed change (candidate diff sketch — refine in planning)

Append the context-consistency clause to the L3255 summary string: a presence hit binds only after its surrounding lines are read — context already implementing the proposed change = landed fix, dedup not file (#1330/#1309) — and name the pin test.

## Scope / surfaces

- Primary target: `.claude/workflow.yaml`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: a6683eb3c015



