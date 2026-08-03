---
title: 'workflow-fix: c37 requires bundled_in_no_flags pin test'
kind: infra
tags:
- wf-fix
- wf-fix-fp:60dae7e47a81
- daily-auto-filed
created_at: '2026-07-25T06:50:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): A plan targeting a workflow_lint.py
  check addition can claim no-flags bundling while its test list names no bundled_in_no_flags
  pin test - the silent-unbundling class recurred twice (1385 and the 1648 plan v2)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 Step C parked-candidate routing (parked on #1648 at 2026-07-24T10:14:20Z by the statistics-critic, plan v2 review; recursion guard).

## Goal

Close the silent-unbundling class: a plan that adds a workflow_lint check and claims no-flags bundling must name the pin test that proves the bundling.

## Workflow gap

- **Bug observed:** #1648's plan v2 claimed no-flags bundling with no `*bundled_in_no_flags*` pin test named; the class previously recurred in #1385.
- **Why it is a workflow gap:** c37 verifies the bundling CLAIM against workflow_lint's dispatch source but does not require the plan's test enumeration to carry the pin test that keeps the bundling true after later refactors.
- **Confidence (emitter):** medium.
- verified-at-filing: `grep -n 'c37\|bundled_in_no_flags' scripts/verify_plan.py` → c37 present (:96, :6202-6236 claim-verb anchoring); `bundled_in_no_flags` test-enumeration requirement ABSENT (0 hits; absence bind); `git log --oneline --since='7 days ago' -- scripts/verify_plan.py` → 5 commits (#1573/#1557/#1551/#1550/#1535), none adding this extension (2026-07-25).

## Proposed change (candidate diff sketch — refine in planning)

Extend check 37: when the plan's diff scope includes a new `--check-*` in `scripts/workflow_lint.py` AND the plan claims no-flags bundling, WARN unless the plan's test enumeration names a `*bundled_in_no_flags*` test (escape string: `N/A — not bundled into no-flags`).

## Scope / surfaces

- Primary target: `scripts/verify_plan.py` (+ its test file)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 60dae7e47a81

- workflow_fix_target: scripts/verify_plan.py
