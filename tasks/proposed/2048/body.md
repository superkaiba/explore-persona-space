---
title: 'daily-fix: per-family pilot basis for fan-out sizing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bf6ae72a01a7
- daily-auto-filed
created_at: '2026-08-03T07:06:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): The #1739 core-family walls
  were undersized ~2-3x: three compute-deviation relaunch cycles in one morning (coreevil
  rc=9 -> rc=7 -> 13.6h fence; corehall re-fenced 17h; coresyc measured fence), and
  the a5syc relaunch was sized 30h against an original 10h budget grounded in a single
  arm-level pilot. unverified hypothesis -- verify at plan time: the pilot basis was
  formally followed but the one arm-'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner1, session f98a12ed, task #1739).

## Goal

Fan-out wall/fence sizing scales with the family whose cells it fences, so designed halts fire on real deviations instead of systematic under-sizing.

## Workflow gap

- **Bug observed:** The #1739 core-family walls were undersized ~2-3x: three compute-deviation relaunch cycles in one morning (coreevil rc=9 -> rc=7 -> 13.6h fence; corehall re-fenced 17h; coresyc measured fence), and the a5syc relaunch was sized 30h against an original 10h budget grounded in a single arm-level pilot. unverified hypothesis -- verify at plan time: the pilot basis was formally followed but the one arm-level reference did not transfer across box families whose budgets differ 250->16000 (miner-inferred; the plan s9 sizing basis was reconstructed from transcript prose, not re-read).
- **Why it is a workflow gap:** The MEASURED-pilot mandate exists but is silent on how many pilots a heterogeneous fan-out needs; one reference cell under-sizes every larger family and each under-size costs a deviation-relaunch cycle.
- **Confidence (emitter):** low
- verified-at-filing: `grep -c -iE 'per-family|box famil' .claude/rules/plan-compute-sizing.md` -> 0 (single-pilot basis assumed).

## Proposed change (refine in planning)

when a fan-out's families differ in budget/grid size by more than ~4x, require a PER-FAMILY (behavior x budget band) measured pilot extrapolation as the sizing basis, not one arm-level reference cell.

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: bf6ae72a01a7

- workflow_fix_target: .claude/rules/plan-compute-sizing.md

