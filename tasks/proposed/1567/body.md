---
title: 'daily-fix: pin git context in guard deny-shape tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6c41a942adc9
- daily-auto-filed
created_at: '2026-07-20T06:48:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): 23 deny-shape tests transiently
  failed on concurrent shared-root off-main state'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Make the guard deny-shape tests (tests/test_guard_repo_root_branch.py family) robust to a concurrent shared-root off-main state — pin/snapshot their git context or run against the worktree — so a transient fleet state cannot red 23 tests in a Step 9c gate run.

## Workflow gap

- **Bug observed:** a Step 9c gate first run read 25F/4789P; 23 of the failures were guard deny-shape tests that all passed on immediate re-run (28/28 in 8.6s) — the cause was a transient shared-root off-main state from a concurrent session, and the flake cost a full ~28-min gate re-run.
- **Why it is a workflow gap:** gate-selected tests that shell against the SHARED repo root inherit fleet git-state races; a deterministic gate should not depend on concurrent sessions' momentary branch state.
- **Confidence (emitter):** medium
- verified-at-filing: incident-anchored: session b295683f (task #1528) @ 09:16 UTC 2026-07-19, heartbeat note 'first run read 25F/4789P but all 23 guard deny-shape failures passed on immediate re-run (28/28 in 8.6s) — transient shared-root off-main'. Target-file presence: `tests/test_guard_repo_root_branch.py` exists (2000+ lines); the plan should locate which fixtures resolve the shared root vs a pinned tmp repo (no count claim made).

## Proposed change (candidate diff sketch — refine in planning)

(none — sketch: fixtures create/pin a scratch git context (tmp repo or the session worktree) instead of reading the live shared-root HEAD; or the selector marks the family flake-retry-once)

## Scope / surfaces

- Primary target: `tests/test_guard_repo_root_branch.py` (and sibling guard-test fixtures)
- Alternative: `scripts/select_step9c_tests.py` flake-retry note (the plan picks one).

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: tests/test_guard_repo_root_branch.py
- fingerprint: 59a22833ab15

Mined evidence: session b295683f (task #1528), 2026-07-19: 23-test transient red, ~28-min gate re-run.
