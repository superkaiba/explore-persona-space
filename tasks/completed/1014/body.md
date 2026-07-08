---
title: 'workflow-fix: check 17 lacks a Context-row lineage-token requirement'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d0cc834e3853
created_at: '2026-07-04T17:28:02Z'
has_clean_result: false
origin_prompt: 'prose follow-up from #958 clean-result-critic r1 (fp d0cc834e3853):
  Add a check-17 requirement that the **Context:** footer row carries a lineage token
  (parent/follow-up lineage, or an explicit ''fresh (no parent)'' clause).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the clean-result-critic on task #958.

## Goal

Add a check-17 requirement that the **Context:** footer row carries a lineage token (parent/follow-up lineage, or an explicit 'fresh (no parent)' clause).

## Workflow gap

- **Bug observed:** Task #958's **Context:** footer row lacked any lineage clause; only the Claude clean-result critic caught it as a procedural finding.
- **Why it is a workflow gap:** check 17 already parses the Context row; the lineage token is a one-clause extension the critic marked mechanizable.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing; update tests/test_verify_task_body.py alongside.

## Constraints / invariants

- Workflow-surface only; forward-only v4 rule (never newly hard-FAIL grandfathered v3/v2 bodies); tests pass.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 and carries a workflow_fix_target: Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: d0cc834e3853

(surfaced prose from clean-result-critic verdict, task #958 round 1)
