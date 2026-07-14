---
title: 'daily-fix: conftest autouse hermeticity guard, watcher tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fac2ecbd34bd
- daily-auto-filed
created_at: '2026-07-11T06:51:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-10 problem sweep (route 2): the #1247 fail-loud hermeticity
  guards are duplicated per-file across the three watcher test files; a new watcher
  test file (or deletion of one per-file fixture) silently loses the invariant'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-10 problem sweep (route 2 - behavior/logic change, independent review required).

## Goal

commit the issue1247 hermeticity-probe shape as a shared conftest/plugin autouse guard covering the watcher test files, replacing the per-file copies

## Workflow gap

- **Bug observed:** the #1247 fail-loud hermeticity guards are duplicated per-file across the three watcher test files; a new watcher test file (or deletion of one per-file fixture) silently loses the invariant
- **Provenance / evidence:** Round-3 code-reviewer standing recommendation, #1247 (parked 2026-07-10T10:15:10Z). Distinct residual - the rec POST-dates the per-file guards merged in PR #984.

## Scope / surfaces

- Primary target: `tests/test_autonomous_session_watch.py, tests/test_stalled_detector_and_gc.py, tests/test_auth_outage_guard.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a workflow-fix Provenance line - it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: fac2ecbd34bd

- workflow_fix_target: tests/test_autonomous_session_watch.py, tests/test_stalled_detector_and_gc.py, tests/test_auth_outage_guard.py
