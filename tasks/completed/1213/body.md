---
title: 'daily-fix: verify_plan warns on ckpt ladder without retentio'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1b2ecd7fa9f4
- daily-auto-filed
created_at: '2026-07-09T07:01:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): #1112 kept all 30 full-FT
  dose-ladder rungs -> ENOSPC (07-07); #1133 shipped the sizing RULE but the planner
  deferred a mechanical check as fragile — leaving the class reviewer-enforced only.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Decide (in-pipeline) whether a WARN-grade ladder-without-retention check is worth mechanizing; ship it if yes.

## Workflow gap

- **Bug observed:** Transcript ef9063d8 (issue-1133) 09:21:43Z: the planner surfaced an optional verify_plan check for ladder-without-retention but judged a regex detector fragile ("plans phrase ladders many ways") and deferred it — the /daily classifier routes deferred-but-concrete candidates for the pipeline planner to decide with the file open.
- **Why it is a workflow gap:** The #1112 ENOSPC class is now rule-covered but has no mechanical backstop; verify_plan checks are WARN-capable, so a low-precision detector can still be net-positive.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

A c-series WARN check: plan mentions a per-rung/checkpoint ladder token set AND no retention/disk-bound token within the compute-sizing section => WARN with the #1133 rule pointer. The spawned planner may deflect with a reasoned no-change report.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-C U2 (ef9063d8 09:21:43Z)
