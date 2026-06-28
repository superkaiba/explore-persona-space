---
title: fix full-suite test-isolation pollution (module-singleton leak across 4 tests)
kind: infra
tags:
- daily-held
created_at: '2026-06-28T07:13:32Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog: four tests pass in isolation but fail under full-suite ordering due to a module-singleton leak.

## Goal

Full pytest suite green without special-casing, so the /issue Step 9c test-verdict gate is trustworthy.

## Problem (from /daily 2026-06-27)

These PASS in isolation but FAIL under full-suite ordering (global-state leak across tests):

- `test_autonomous_session_watch.py::test_vm_reclaim_hf_hub_cache_evicts_through_bounded_worker`
- two `test_issue642_analyze_decision_rule` tests
- `test_issue672_validation.py::test_loop_poller_detects_second_failover_in_quiet_period`

Step 9c currently special-cases these, which masks a real ordering-dependent pollution and weakens the gate.

## Proposed change

Add autouse teardown fixtures that reset the polluted module-level singleton(s) so the full suite passes without special-casing. Then remove the Step 9c special-case so the gate runs the unmodified full suite.

## Scope / target files

- `tests/` (the four test files above + the autouse fixture location)
- the module(s) owning the leaked singleton (identify via the failing tests)

## Constraints

- Workflow-surface / test-surface only.
- Lint gate green; ruff clean on touched files.
- Do NOT weaken the assertions to make tests pass — fix the isolation; the singleton reset is the fix.
