---
title: 'workflow-fix: v4 no-issue-refs check misses markdown-linked refs in Methodology
  prose'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9c29e5ce595c
created_at: '2026-07-04T17:27:44Z'
has_clean_result: false
origin_prompt: 'prose follow-up from #958 clean-result-critic r1 (fp 9c29e5ce595c):
  Extend verify_task_body.py''s v4 no-issue-refs check to catch markdown-linked [#K](...)
  references outside ## Goal / the footer (e.g. inside ## Methodology prose), not
  only bare #K tokens.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the clean-result-critic on task #958.

## Goal

Extend verify_task_body.py's v4 no-issue-refs check to catch markdown-linked [#K](...) references outside ## Goal / the footer (e.g. inside ## Methodology prose), not only bare #K tokens.

## Workflow gap

- **Bug observed:** The v4 no-issue-refs check passed a markdown-linked [#778](...) inside ## Methodology -> **Evaluation:** on task #958; both clean-result critics had to catch it manually as a lens blocker.
- **Why it is a workflow gap:** A mechanizable check the critics both re-derived by hand; the verifier owns the no-issue-refs gate but only matches bare #K tokens.
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
- fingerprint: 9c29e5ce595c

(surfaced prose from clean-result-critic verdict, task #958 round 1)
