---
title: 'workflow-fix: poll guard on phase lines after done-detection'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:de29c7f7947c
created_at: '2026-07-04T07:12:14Z'
has_clean_result: false
origin_prompt: 'PARKED — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target
  (recursion guard, workflow-fix-on-bug.md § Recursion guard): alternatives critic
  prose follow-up (source: prose-followup): (1) poll_pipeline.py post-hoc consistency
  guard — if a poll after done-detection observes NEW [phase=...] lines after the
  matched done line, escalate/alert rather than remain terminal; covers the .py-fan-out
  FN class (§4.6(i)) at runtime, complementing this lint. Target: scripts/poll_pipeline.py.
  Not routed by this session; logged for the next orchestrator/human pass.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add a poll_pipeline.py post-hoc consistency guard: if a poll after done-detection observes NEW [phase=...] lines after the matched done line, escalate/alert rather than remain terminal

## Workflow gap

- **Bug observed:** a poll after done-detection can observe NEW [phase=...] lines after the matched done line (the .py-fan-out FN class, #942 plan 4.6(i)) and remain terminal
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add a poll_pipeline.py post-hoc consistency guard: if a poll after done-detection observes NEW [phase=...] lines after the matched done line, escalate/alert rather than remain terminal

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: de29c7f7947c

PARKED — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target (recursion guard, workflow-fix-on-bug.md § Recursion guard): alternatives critic prose follow-up (source: prose-followup): (1) poll_pipeline.py post-hoc consistency guard — if a poll after done-detection observes NEW [phase=...] lines after the matched done line, escalate/alert rather than remain terminal; covers the .py-fan-out FN class (§4.6(i)) at runtime, complementing this lint. Target: scripts/poll_pipeline.py. Not routed by this session; logged for the next orchestrator/human pass.
