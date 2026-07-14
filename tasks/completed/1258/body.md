---
title: 'daily-fix: audit c20 co-occurrence WARN across N/A escapes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c30c2819b09f
- daily-auto-filed
created_at: '2026-07-11T06:51:18Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): the c20 escape-masking
  co-occurrence WARN exists at one check only; other _standalone_na_declared sites
  may pass silently when their escape co-occurs with detected live content (c12/c24
  differ by design - their escapes suppress WARN-grade vocabulary heuristics)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

audit the other _standalone_na_declared call sites for escapes whose only live case is post-detection; decide per-check whether the #1223 c20 co-occurrence-WARN treatment ports

## Workflow gap

- **Bug observed:** the c20 escape-masking co-occurrence WARN exists at one check only; other _standalone_na_declared sites may pass silently when their escape co-occurs with detected live content (c12/c24 differ by design - their escapes suppress WARN-grade vocabulary heuristics)
- **Provenance / evidence:** Alternatives critic prose follow-up, #1223 plan v1 (parked 2026-07-10T07:26:37Z). NOT covered by #1237 (recognizer migration) or #1238 (wrapped-declaration pin); ~22 _standalone_na_declared sites in current tree - planner must re-enumerate (candidate line numbers drifted post-#1237).

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: c30c2819b09f

- workflow_fix_target: scripts/verify_plan.py
