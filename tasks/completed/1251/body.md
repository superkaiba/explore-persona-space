---
title: 'step9c baseline oracle tolerates unrelated concurrent dirty '
kind: infra
tags:
- wf-fix
- wf-fix-fp:4585a630978d
- daily-auto-filed
created_at: '2026-07-10T06:55:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): baseline oracle refused
  rc=2 on 1190 because a concurrent session''s dirty src/ (issue_779) sat at the shared
  repo root'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-09 from the nightly transcript problem sweep (miner 00).

## Goal

Make the Step 9c baseline oracle (`scripts/step9c_baseline.py`) tolerate UNRELATED concurrent dirty paths at the shared repo root instead of refusing (rc=2) when another session's in-flight experiment files are present.

## Workflow gap

- **Bug observed:** #1190's Step 9c test-verdict (2026-07-09T20:51Z) had its baseline oracle refuse with rc=2 because a CONCURRENT session's dirty `src/` files (issue_779 work) sat at the shared repo root; the session recovered only via a solo re-run window.
- **Why it is a workflow gap:** The repo root is shared by design (many parallel sessions); a baseline gate that requires a fully clean tree fails stochastically under normal fleet concurrency — the same class as the #1160 trunk-red interference — and costs a full re-run round each time.

## Proposed change (refine in planning)

Scope the oracle's dirty-tree refusal to paths RELEVANT to the tests being verified (the touched files / selected test set), ignoring unrelated dirty paths (with a logged WARN naming them); check first whether #1212's archived-tree gate already covers this and, if so, close as duplicate with evidence.

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py`
- Secondary: `scripts/select_step9c_tests.py` (path-relevance helper, if shared)

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 4585a630978d

- workflow_fix_target: scripts/step9c_baseline.py
