---
title: 'daily-fix: inline lint gate defers under high VM load'
kind: infra
tags:
- wf-fix
- wf-fix-fp:517ebd1cb2be
- daily-auto-filed
created_at: '2026-08-03T07:02:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): A certified inline-gate
  run FAILed exit 3 under VM load1~31 (1 failed + 3 error mapped tests, 8m34s pytest)
  and PASSed clean ~35 min later once load dropped -- the session improvised a load-drop
  Monitor watch and retried (session a0400dd4, 17:08-18:25Z, task #1345). unverified
  hypothesis -- verify at plan time: the failures were load-flake (timeout-sensitive
  tests), inferred from the clean retry a'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner6, session a0400dd4, task #1345).

## Goal

The inline payload lint gate does not false-block payloads because the shared VM happened to be at load 30 when the mapped tests ran.

## Workflow gap

- **Bug observed:** A certified inline-gate run FAILed exit 3 under VM load1~31 (1 failed + 3 error mapped tests, 8m34s pytest) and PASSed clean ~35 min later once load dropped -- the session improvised a load-drop Monitor watch and retried (session a0400dd4, 17:08-18:25Z, task #1345). unverified hypothesis -- verify at plan time: the failures were load-flake (timeout-sensitive tests), inferred from the clean retry at low load.
- **Why it is a workflow gap:** The gate's mapped-pytest leg is timing-sensitive; fleet-busy hours make gate verdicts load-dependent, and each false BLOCK costs a manual retry loop the session must improvise.
- **Confidence (emitter):** low
- verified-at-filing: `grep -c -iE 'getloadavg|loadavg|load1' scripts/inline_lint_gate.py` -> 0 (no load awareness).

## Proposed change (refine in planning)

check os.getloadavg() before the mapped-pytest leg; above a threshold (~20 on this 32-core VM) either wait bounded for load to drop or return a distinct INCONCLUSIVE-under-load verdict, so a loaded fleet doesn't convert to false BLOCKs.

## Scope / surfaces

- Primary target: `scripts/inline_lint_gate.py`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 517ebd1cb2be

- workflow_fix_target: scripts/inline_lint_gate.py

