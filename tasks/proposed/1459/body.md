---
title: 'daily-fix: address-concern summary-cap UX'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7ef65039fb5a
- daily-auto-filed
created_at: '2026-07-17T06:57:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): task.py address-concern
  hard-raises ValueError on summary >200 chars with no cap named in the CLI help —
  two sessions burned round-trips today (#1398: 324 chars; #1090: argparse bounce
  then 203 chars, a 3-char overage costing two retries)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (two independent hits: #1398 session f628e7aa 17:13Z; #1090 session e6e79656 ~00:10Z).

## Goal

Stop the 200-char address-concern cap from costing agent round-trips.

## Workflow gap

- **Bug observed:** task.py address-concern hard-raises ValueError on summary >200 chars with no cap named in the CLI help — two sessions burned round-trips today (#1398: 324 chars; #1090: argparse bounce then 203 chars, a 3-char overage costing two retries)
- **Why it is a workflow gap:** A validation cap that is discoverable only by tripping it is a recurring per-session tax.
- **Confidence (emitter):** medium (recurrence 2 in one day)
- verified-at-filing: `grep -n 'summary too long\|> 200' src/explore_persona_space/task_workflow.py` -> L6451 (doc) / L6460 (raise); the argparse help for address-concern does not name the cap (absence claim)

## Proposed change (candidate diff sketch — refine in planning)

name the 200-char cap in the CLI help/error path and/or soft-truncate at a word boundary with a warning for small overages (planner picks; the hard cap at the storage layer may stay)

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 7ef65039fb5a

