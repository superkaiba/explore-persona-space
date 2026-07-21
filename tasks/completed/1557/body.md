---
title: 'daily-fix: check 31 - new pin test must be selector-visible'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5094de40ff2d
- daily-auto-filed
created_at: '2026-07-20T06:46:49Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): check 31 accepts new pin
  tests on workflow-surface globs without selector registration'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-19 parked-candidate sweep (Step C) from a workflow-fix candidate parked on task #1546 (emitting agent: Statistics critic, follow-up + mechanizable sketch; parked under the recursion guard).

## Goal

Extend `scripts/verify_plan.py` check 31 (Durability pin) so a NEW pin test whose asserted surface matches a WORKFLOW_SURFACE_GLOBS skip-glob (.claude/skills/**, .claude/agents/*) must also be named in the plan as registered in `select_step9c_tests.py` WORKFLOW_INVARIANT (or be rules-pin-discoverable).

## Workflow gap

- **Bug observed:** check 31 accepts a NEW pin test on a workflow-surface glob without requiring its Step 9c selector registration; an unregistered pin never runs at the gate.
- **Why it is a workflow gap:** an unregistered pin is a protection illusion at the Step 9c gate — family precedent: #1242/#1268 both needed manual registration after the fact.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'WORKFLOW_INVARIANT\|WORKFLOW_SURFACE_GLOBS' scripts/verify_plan.py` → 0 hits (no registration cross-check exists — absence claim, in-target 0-hit is the evidence); `grep -n 'Durability pin' scripts/verify_plan.py` → present (:152-154, :5087, :5134-5136 — check 31 exists and accepts a NEW pin test with no registration requirement). `git log --oneline --since='2026-07-17' -- scripts/verify_plan.py` shows no commit adding such a cross-check (2026-07-19).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up with mechanizable sketch: check-31 satisfier for a NEW pin test on a workflow-surface glob additionally requires a plan line naming the select_step9c_tests.py WORKFLOW_INVARIANT registration, or rules-pin discoverability)

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Also inspect `scripts/select_step9c_tests.py` (WORKFLOW_INVARIANT registry) and `tests/test_verify_plan.py` for the check-31 tests.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `tests/test_verify_plan.py` extended.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 8bf7c3e1f415

Verbatim parked candidate (epm:workflow-fix-candidate on #1546, 2026-07-19T16:16:04Z): "source: prose-followup (Statistics critic Follow-up + mechanizable sketch). target_file: scripts/verify_plan.py. proposed_change: extend check 31 (Durability pin) so a NEW pin test whose asserted surface matches a WORKFLOW_SURFACE_GLOBS skip-glob (.claude/skills/**, .claude/agents/*) must also be named in the plan as registered in select_step9c_tests.py WORKFLOW_INVARIANT (or be rules-pin-discoverable) — an unregistered pin is a protection illusion at the Step 9c gate (family precedent: #1242/#1268 both needed manual registration)."
