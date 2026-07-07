---
title: 'daily-fix: verify_plan c12 ASCII-x boundary + c18 negation-r'
kind: infra
tags:
- wf-fix
- wf-fix-fp:269f86978474
- daily-auto-filed
created_at: '2026-07-07T06:49:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-06 problem sweep (route 2): Two concrete residuals
  parked under the recursion guard by the #1086 workflow-fix session (28d40897, 2026-07-06),
  with fix sketches already written: (1) the c12 ASCII ''x'' multiplication token
  matches word-split artifacts (needs a whitespace/boundary requirement); (2) _C18_NEG_DEFER_RE
  misses common negations — doesn''t/cannot/can''t/''fails to''/until — and its n''t
  token is dead code.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-06 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

(a) require whitespace/word-boundary around the ASCII x multiplication token and add 2 negative fixtures; (b) extend _C18_NEG_DEFER_RE with the missing negation forms and fix the dead n't alternative; tests for both.

## Workflow gap

- **Bug observed:** Two concrete residuals parked under the recursion guard by the #1086 workflow-fix session (28d40897, 2026-07-06), with fix sketches already written: (1) the c12 ASCII 'x' multiplication token matches word-split artifacts (needs a whitespace/boundary requirement); (2) _C18_NEG_DEFER_RE misses common negations — doesn't/cannot/can't/'fails to'/until — and its n't token is dead code.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- source: /daily 2026-07-06 problem sweep (transcript-mined)
