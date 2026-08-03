---
title: 'daily-fix: runpod_api UP037 reds ruff-policy test on main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e3a326a6b83f
- daily-auto-filed
created_at: '2026-07-24T06:46:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-23 problem sweep (route 2): scripts/runpod_api.py:488
  UP037 quoted type annotation fails tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset
  on pristine main'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-23 parked-candidate routing pass (Step C). Raised as recursion-guard-parked prose follow-ups on tasks #1618, #1624, #1629, #1638 (four independent implementer/planner reports, all reproducing from pristine main).

## Goal

Fix the ruff UP037 (quoted type annotation) at `scripts/runpod_api.py:488` so `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset` is green on main.

## Workflow gap

- **Bug observed:** `scripts/runpod_api.py:488:26: UP037 [*] Remove quotes from type annotation` fails `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset` on pristine main (introduced by the #1112 merge, commit `b66910d748`), redding every Step 9c full-suite gate that maps this test.
- **Why it is a workflow gap:** `runpod_api.py` is a workflow-helper script; a pristine-main-red policy test poisons every issue session's gate verdict fleet-wide.
- **Confidence (emitter):** high (4 independent parks)
- verified-at-filing: `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -x -q` → 1 failed, the failure body naming EXACTLY `scripts/runpod_api.py:488:26: UP037` ("Found 1 error. 1 fixable with --fix") (2026-07-24 UTC). Bare `uv run ruff check scripts/runpod_api.py` passes (project config differs from the test's full ruleset) — the test invocation is the binding oracle.

## Proposed change (candidate diff sketch — refine in planning)

One-line fix: unquote the type annotation at `scripts/runpod_api.py:488` (equivalently `ruff check --fix` scoped to that rule), then confirm the policy test passes.

## Scope / surfaces

- Primary target: `scripts/runpod_api.py`
- The policy test names exactly one error; no sibling hits.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: e3a326a6b83f

- workflow_fix_target: scripts/runpod_api.py

Origin: four parked `epm:workflow-fix-candidate` notes on #1618 (2026-07-23T09:32:21Z), #1624 (12:37:19Z), #1629 (15:40:50Z), #1638 (18:25:52Z) — e.g. "#1629 r1: line ~488 UP037 fails tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset on PRISTINE main (introduced by the #1112 merge); proposed_change: fix the UP037 (unquote the annotation)".
