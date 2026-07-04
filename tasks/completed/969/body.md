---
title: 'workflow-fix: v4-scope pre_reg prose restriction in body audit'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:afba20681c32
created_at: '2026-07-04T07:10:23Z'
has_clean_result: false
origin_prompt: 'routed: parked — running under workflow_fix_target Provenance (recursion
  guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard); NOT auto-filed.
  source: prose-followup (planner Observation 1 + both methodology critics). target_file:
  scripts/audit_clean_results_body_discipline.py. bug_observed: _restrict_pre_reg_to_prose_sections
  gates on the v3 sentinel only (line ~548), so v4 bodies get a WHOLE-BODY pre_reg
  scan — stricter than Lens 7''s prose-section intent mapped onto v4 names; a spec-permitted
  procedural pre-reg mention in v4 ## Methodology prose would false-positive (the
  v4 analogue of incident #623). proposed_change: extend the function with a v4-sentinel
  branch scoping to (''takeaways'',''results''). confidence: medium (no observed v4
  false-positive incident yet). related_task: #892.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

extend _restrict_pre_reg_to_prose_sections with a v4-sentinel branch scoping to ('takeaways','results')

## Workflow gap

- **Bug observed:** _restrict_pre_reg_to_prose_sections gates on the v3 sentinel only (~line 548), so v4 bodies get a WHOLE-BODY pre_reg scan; a spec-permitted procedural pre-reg mention in v4 ## Methodology prose would false-positive (v4 analogue of incident #623)
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

extend _restrict_pre_reg_to_prose_sections with a v4-sentinel branch scoping to ('takeaways','results')

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: afba20681c32

routed: parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard); NOT auto-filed. source: prose-followup (planner Observation 1 + both methodology critics). target_file: scripts/audit_clean_results_body_discipline.py. bug_observed: _restrict_pre_reg_to_prose_sections gates on the v3 sentinel only (line ~548), so v4 bodies get a WHOLE-BODY pre_reg scan — stricter than Lens 7's prose-section intent mapped onto v4 names; a spec-permitted procedural pre-reg mention in v4 ## Methodology prose would false-positive (the v4 analogue of incident #623). proposed_change: extend the function with a v4-sentinel branch scoping to ('takeaways','results'). confidence: medium (no observed v4 false-positive incident yet). related_task: #892.
