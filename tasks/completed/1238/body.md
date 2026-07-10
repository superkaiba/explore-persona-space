---
title: tolerate wrapped N/A escape declarations in verify_plan
kind: infra
tags:
- wf-fix
- wf-fix-fp:3945c7b4fb8e
- daily-auto-filed
created_at: '2026-07-10T06:54:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): _standalone_na_declared''s
  lstrip set ('' \t>*-'' at scripts/verify_plan.py:1453) rejects a backtick-wrapped
  or single-quote-wrapped escape declaration at bullet start (live shape in #1090
  v1-v4:369) — a'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1203.

## Goal
Consider tolerating a leading backtick/quote in the shared helper's strip set (one shared change + fixtures); the planner must weigh this against the anti-paste rationale before widening.

## Workflow gap
- **Bug observed:** _standalone_na_declared's lstrip set (' \t>*-' at scripts/verify_plan.py:1453) rejects a backtick-wrapped or single-quote-wrapped escape declaration at bullet start (live shape in #1090 v1-v4:369) — a literal-minded planner pasting the remedy's quoted form on its own line still FAILs; shared behavior across all call sites.
- **Why it is a workflow gap:** The escape mechanism exists so a legitimately-exempt plan can satisfy a check; rejecting the natural quoted paste form costs a mechanical bounce round.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)
(none)

## Scope / surfaces
- Primary target: `scripts/verify_plan.py`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/verify_plan.py
- fingerprint: n/a (prose park)

routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — logged, not auto-filed).

source: prose-followup (alternatives critic, #1203 plan v1 review)
target_file: scripts/verify_plan.py
bug_observed: _standalone_na_declared's lstrip set (' \t>*-') rejects a backtick-wrapped or single-quote-wrapped escape declaration at bullet start (live shape in #1090 v1-v4:369) — a literal-minded planner pasting the remedy's quoted form on its own line still FAILs; shared behavior across all 10 call sites.
proposed_change: consider tolerating leading backtick/quote in the shared helper's strip set (one 10-check shared change + fixtures); weigh against the anti-paste rationale before widening.
confidence: low
related_task: #1203
