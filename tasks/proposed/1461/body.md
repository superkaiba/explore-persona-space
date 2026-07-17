---
title: 'daily-fix: neutralize guard-task revision briefs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:eef2a20ad1b2
- daily-auto-filed
created_at: '2026-07-17T06:57:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the rung-(e) first-pass
  neutral-vocabulary discipline covers initial briefs, but a #1413 planner-REVISION
  spawn was refusal-killed and a second spawn dispatched with its brief truncated
  mid-sentence by a mid-stream refusal — Must-Fix lists on guard/security tasks were
  inlined into the revision brief with the trigger-dense payload text'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1413 session 80298992, 22:43-23:47Z: one refusal-killed revision spawn, one truncated-brief dispatch, three refusal-killed tick turns).

## Goal

Extend the trigger-dense brief discipline from first-pass briefs to revision-round briefs.

## Workflow gap

- **Bug observed:** the rung-(e) first-pass neutral-vocabulary discipline covers initial briefs, but a #1413 planner-REVISION spawn was refusal-killed and a second spawn dispatched with its brief truncated mid-sentence by a mid-stream refusal — Must-Fix lists on guard/security tasks were inlined into the revision brief with the trigger-dense payload text
- **Why it is a workflow gap:** Revision rounds re-inline the very findings text the first-pass discipline kept out; the refusal surface returns exactly when the fix loop needs to run.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'revision' .claude/rules/trigger-dense-review.md` -> 0 hits (absence claim: no revision-brief clause); incident rows in transcript 80298992 (refusal-killed spawns + truncated brief)

## Proposed change (candidate diff sketch — refine in planning)

extend the neutralization duty explicitly to planner-revision briefs on guard/security tasks: Must-Fix details passed by file reference, never inlined; note the truncated-spawn verify duty applies to revision spawns too. SECONDARY: extend CLAUDE.md refusal-ladder rung (e)'s neutral-vocabulary list with steering / causal-intervention phrasing — 'causal steering' briefs tripped the same filter 3x on #1415 (2026-07-16 ~11:08-12:32Z) before a sonnet-pin recovered

## Scope / surfaces

- Primary target: `.claude/rules/trigger-dense-review.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: eef2a20ad1b2

