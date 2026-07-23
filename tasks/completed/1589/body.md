---
title: 'daily-fix: step9c selector-module diff selects zero tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:58f5c3cd919e
- daily-auto-filed
created_at: '2026-07-22T06:44:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): a selector-module diff
  maps to zero tests; test_step9c_baseline references the selector by symbol only'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1579 (emitting agent: code-reviewer, #1579 round 1).

## Goal

Make a diff touching `scripts/select_step9c_tests.py` itself select its own pin/consumer tests (at minimum `tests/test_step9c_baseline.py`), instead of mapping to zero tests.

## Workflow gap

- **Bug observed:** a selector-module diff never selects `tests/test_step9c_baseline.py` because that consumer references the selector by SYMBOL name only — no import of the module file path, no stem match, no literal path — a miss class adjacent to the docstring's documented residuals.
- **Why it is a workflow gap:** the Step-9c test gate's own selector is invisible to itself; a regression in the selector ships without its consumer tests running on the mapped-tests leg.
- **Confidence (emitter):** medium (empirically confirmed at filing — see below).
- verified-at-filing: `uv run python scripts/select_step9c_tests.py --map-files scripts/select_step9c_tests.py` → ZERO pairs printed (2026-07-22; the claimed miss reproduces, and is broader than the candidate stated: not even a selector self-test is selected); `grep -n 'test_step9c_baseline' scripts/select_step9c_tests.py` → 0 registration hits. NOT a duplicate of open #865 (`on_hold`, wf-fix-fp:d2551dac39f5 — worktree-blind diffing, a different bug on the same file; the open-sibling advisory will list it — verified distinct at filing).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up) Register `tests/test_step9c_baseline.py` (and the selector's own test suite, if any) in `WORKFLOW_INVARIANT`, or add a literal/dependency mapping so selector-module diffs select their pin tests.

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Grep `tests/` for other symbol-only consumers of the selector before editing; list them in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` default run passes; ruff passes; the selector's safe-by-direction over-select contract is preserved.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 58f5c3cd919e

- workflow_fix_target: scripts/select_step9c_tests.py

Verbatim parked candidate (task #1579 events, 2026-07-21T10:18:11Z): "parked — running under workflow_fix_target recursion guard (see .claude/rules/workflow-fix-on-bug.md § Recursion guard); surfaced by code-reviewer r1 (source: prose-followup): a selector-module diff never selects tests/test_step9c_baseline.py because that consumer references the selector by SYMBOL name only (no import of the module file path, no stem match, no literal path) — a miss class adjacent to the docstring's documented residuals. target_file: scripts/select_step9c_tests.py; confidence: medium; related_task: #1579."
