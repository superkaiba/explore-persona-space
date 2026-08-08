---
title: 'daily-fix: runbook markers carry verified-by ran|read'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b97661b37590
- daily-auto-filed
created_at: '2026-08-03T07:04:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): The durable Option-A dispatch
  runbook marker (epm:progress v235 on #1345) claimed ''no fits change needed'' --
  verified only for the fits-side ALLOWLIST by reading, not by running; the arm enumeration
  path was wrong and two jobs (incl. 17929, 28m44s) died on it before the v246 AMENDMENT
  (positive incident: epm:progress v246 on #1345, 2026-08-02T19:21:44Z, ''v235 said
  no fits change needed. That was W'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner6, session a0400dd4, task #1345).

## Goal

Durable dispatch runbooks distinguish executed verification from static reads, so successors don't burn jobs on statically-verified-only claims.

## Workflow gap

- **Bug observed:** The durable Option-A dispatch runbook marker (epm:progress v235 on #1345) claimed 'no fits change needed' -- verified only for the fits-side ALLOWLIST by reading, not by running; the arm enumeration path was wrong and two jobs (incl. 17929, 28m44s) died on it before the v246 AMENDMENT (positive incident: epm:progress v246 on #1345, 2026-08-02T19:21:44Z, 'v235 said no fits change needed. That was WRONG. The claim was verified only for the fits-side ALLOWLIST').
- **Why it is a workflow gap:** The detached-handoff conventions require recording re-attach handles and state, but nothing distinguishes 'I ran this' from 'I read this and believe it' -- exactly the distinction that killed two jobs today.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'verified-by' .claude/rules/pod-side-reporting.md CLAUDE.md` -> 0 in both. Incident markers cited: #1345 v235 (18:5xZ) refuted by v246 (2026-08-02T19:21:44Z).

## Proposed change (refine in planning)

continuation-runbook / ready-to-dispatch markers state, per load-bearing claim, whether it was verified by RUNNING (the command executed) or by READING (static inspection) -- a `verified-by: ran|read` tag -- so a successor session knows which premises need a smoke before dispatching on them.

## Scope / surfaces

- Primary target: `.claude/rules/pod-side-reporting.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b97661b37590

- workflow_fix_target: .claude/rules/pod-side-reporting.md

