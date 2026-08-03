---
title: 'daily-fix: deferred-commit ERROR keeps the failing hook line'
kind: infra
tags:
- wf-fix
- wf-fix-fp:553e940d13c1
- daily-auto-filed
created_at: '2026-07-29T07:17:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): the deferred-commit ERROR
  surface truncates stderr to a blind [-500:] tail that on 2026-07-28 showed only
  Skipped/Passed pre-commit hook lines — the actual failing hook was invisible (2
  firings, #1482''s marker stream held hostage alongside #1092''s)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-F P9 (miner-probed; re-verified).

## Goal

Make the deferred-commit ERROR name the failing pre-commit hook instead of a blind 500-byte tail.

## Workflow gap

- **Bug observed:** When marker commits deferred on 2026-07-28 (the gitleaks-false-positive episode holding #1092/#1482 marker streams), the deferred-commit ERROR's `stderr_tail[-500:]` showed only Skipped/Passed hook lines — the root cause was invisible from the surfaced text (2 firings; diagnosis needed a manual re-run). The gitleaks `Fingerprint:` special-case extraction shows the right pattern; every other failing hook gets the blind tail.
- **Why it is a workflow gap:** the deferred-commit handler is the ONE surface an orchestrator reads when a marker commit defers; a tail that crops the failure line defeats its purpose.
- **Confidence (emitter):** high (probed)
- verified-at-filing: `grep -n 'stderr_tail' src/explore_persona_space/task_workflow.py` → the blind `[-500:]` at 6798 (surrounding handler 6770-6850 read; only the gitleaks Fingerprint extraction is targeted) (2026-07-29 UTC). Sibling demarcation: #1780 (landed 2026-07-29) covers the gitleaks-MESSAGE leg only — this filing is the general failing-hook-line extraction; the spawned session verifies scope against #1780's merged diff before implementing.

## Proposed change (candidate diff sketch — refine in planning)

Failure-line extraction over the full captured streams with tail fallback; unit test with a synthetic multi-hook stderr.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (deferred-commit handler ~L6798)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 553e940d13c1

- workflow_fix_target: src/explore_persona_space/task_workflow.py

