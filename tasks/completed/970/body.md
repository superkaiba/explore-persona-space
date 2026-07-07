---
title: 'workflow-fix: add SUCCESS|FAILURE to audit verdict_caps'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:87df9f2a3432
created_at: '2026-07-04T07:10:34Z'
has_clean_result: false
origin_prompt: 'routed: parked — running under workflow_fix_target Provenance (recursion
  guard); NOT auto-filed. source: prose-followup (planner Observation 2). target_file:
  scripts/audit_clean_results_body_discipline.py. bug_observed: verdict_caps (line
  ~38) covers REJECTED|INDETERMINATE|PASSED|EXCEEDING but not the #763 incident sentence''s
  ''SUCCESS was not met'' caps-verdict. proposed_change: consider adding SUCCESS|FAILURE
  to verdict_caps after enumerating legitimate ALL-CAPS ''SUCCESS'' usages. confidence:
  low. related_task: #892.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

consider adding SUCCESS|FAILURE to verdict_caps after enumerating legitimate ALL-CAPS 'SUCCESS' usages

## Workflow gap

- **Bug observed:** verdict_caps (~line 38) covers REJECTED|INDETERMINATE|PASSED|EXCEEDING but not the #763 incident sentence's 'SUCCESS was not met' caps-verdict
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** low

## Proposed change (candidate sketch — refine in planning)

consider adding SUCCESS|FAILURE to verdict_caps after enumerating legitimate ALL-CAPS 'SUCCESS' usages

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 87df9f2a3432

routed: parked — running under workflow_fix_target Provenance (recursion guard); NOT auto-filed. source: prose-followup (planner Observation 2). target_file: scripts/audit_clean_results_body_discipline.py. bug_observed: verdict_caps (line ~38) covers REJECTED|INDETERMINATE|PASSED|EXCEEDING but not the #763 incident sentence's 'SUCCESS was not met' caps-verdict. proposed_change: consider adding SUCCESS|FAILURE to verdict_caps after enumerating legitimate ALL-CAPS 'SUCCESS' usages. confidence: low. related_task: #892.
