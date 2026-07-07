---
title: 'workflow-fix: WARN-only check_ood_folds in verify_plan'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:e757c9c76a81
created_at: '2026-07-04T07:11:07Z'
has_clean_result: false
origin_prompt: 'source: prose-followup (implementer report, plan §11 Decision 6).
  target_file: scripts/verify_plan.py. proposed_change: add a WARN-only check_ood_folds
  keyed on the ''Required: OOD generalization folds'' block heading / the ''N/A —
  no held-out predictive DV'' escape line, mirroring check_measurement_validity''s
  shape, with tests/test_verify_plan.py coverage. confidence: medium. routed: parked
  — running under workflow_fix_target recursion guard (.claude/rules/workflow-fix-on-bug.md
  § Recursion guard); logged for the next human/orchestrator pass, not auto-filed.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add a WARN-only check_ood_folds keyed on the 'Required: OOD generalization folds' block heading / the 'N/A - no held-out predictive DV' escape line, mirroring check_measurement_validity's shape, with tests/test_verify_plan.py coverage

## Workflow gap

- **Bug observed:** verify_plan has no check for the 'Required: OOD generalization folds' block, so a plan can silently omit the group-level fold declaration
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add a WARN-only check_ood_folds keyed on the 'Required: OOD generalization folds' block heading / the 'N/A - no held-out predictive DV' escape line, mirroring check_measurement_validity's shape, with tests/test_verify_plan.py coverage

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: e757c9c76a81

source: prose-followup (implementer report, plan §11 Decision 6). target_file: scripts/verify_plan.py. proposed_change: add a WARN-only check_ood_folds keyed on the 'Required: OOD generalization folds' block heading / the 'N/A — no held-out predictive DV' escape line, mirroring check_measurement_validity's shape, with tests/test_verify_plan.py coverage. confidence: medium. routed: parked — running under workflow_fix_target recursion guard (.claude/rules/workflow-fix-on-bug.md § Recursion guard); logged for the next human/orchestrator pass, not auto-filed.
