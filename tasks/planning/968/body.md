---
title: 'workflow-fix: tag-parity lint for hollow-verification-gate'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:70dad07d790d
created_at: '2026-07-04T07:10:10Z'
has_clean_result: false
origin_prompt: 'source: prose-followup (code-reviewer r1). routed: parked — running
  under workflow_fix_target Provenance / EPM_WORKFLOW_FIX_SESSION recursion guard
  (workflow-fix-on-bug.md § Recursion guard). Candidate: add a tag-parity lint for
  hollow-verification-gate to scripts/workflow_lint.py, mirroring the existing compute-shape-mismatch
  parity check at workflow_lint.py:3954-3996 (asserts the tag appears in both reviewer
  files'' **Blocker tags:** template lines + step prose). Concrete + mechanizable;
  the #890 plan explicitly deferred it as a separate task. Next human/orchestrator
  pass may file it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add a tag-parity lint for hollow-verification-gate to workflow_lint.py, mirroring the compute-shape-mismatch parity check (tag appears in both reviewer files' Blocker tags template lines + step prose)

## Workflow gap

- **Bug observed:** the hollow-verification-gate blocker tag has no parity lint; the existing compute-shape-mismatch parity check (workflow_lint.py:3954-3996) covers only that tag
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add a tag-parity lint for hollow-verification-gate to workflow_lint.py, mirroring the compute-shape-mismatch parity check (tag appears in both reviewer files' Blocker tags template lines + step prose)

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 70dad07d790d

source: prose-followup (code-reviewer r1). routed: parked — running under workflow_fix_target Provenance / EPM_WORKFLOW_FIX_SESSION recursion guard (workflow-fix-on-bug.md § Recursion guard). Candidate: add a tag-parity lint for hollow-verification-gate to scripts/workflow_lint.py, mirroring the existing compute-shape-mismatch parity check at workflow_lint.py:3954-3996 (asserts the tag appears in both reviewer files' **Blocker tags:** template lines + step prose). Concrete + mechanizable; the #890 plan explicitly deferred it as a separate task. Next human/orchestrator pass may file it.
