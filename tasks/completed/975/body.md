---
title: 'workflow-fix: poll_pipeline synthesized envelope version max+1'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:5ff6ef7fb425
created_at: '2026-07-04T07:11:20Z'
has_clean_result: false
origin_prompt: "routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this session\
  \ is a workflow-fix session; candidate LOGGED for the next orchestrator pass, NOT\
  \ auto-filed)\nsource: prose-followup (planner § Follow-ups + Claude code-reviewer\
  \ bug-class sweep, both confirmed the line)\n\n<!-- workflow-fix-candidate v1 -->\n\
  target_file: scripts/poll_pipeline.py\nbug_observed: The synthesized-envelope fallback\
  \ (~lines 846-856) pins \"kind\": \"epm:results\", \"version\": 1 in code; on a\
  \ follow-up round's re-run or a re-drained sentinel this reproduces the same version-collision\
  \ class (#389/#825) in the poller path.\nwhy_workflow_gap: The prose surfaces now\
  \ defer to max+1 (#917), but this code path still hardcodes version 1 — the last\
  \ checked-in literal for a round-versioned kind, outside #917's declared scope (poll_pipeline.py\
  \ was must-not-touch).\nproposed_change: The synthesized envelope should omit the\
  \ version (let task_workflow.post_event derive max+1) or compute max(existing)+1\
  \ for the kind; needs its own analysis of multipart/pointer-marker + sentinel-schema\
  \ interactions.\ndiff_sketch: |\n  - envelope = {\"kind\": \"epm:results\", \"version\"\
  : 1, ...}\n  + envelope = {\"kind\": \"epm:results\", ...}  # version omitted ->\
  \ post_event derives max+1\nconfidence: medium\nrelated_task: #917\n<!-- /workflow-fix-candidate\
  \ -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

the synthesized envelope should omit the version (let task_workflow.post_event derive max+1) or compute max(existing)+1 for the kind; needs its own analysis of multipart/pointer-marker + sentinel-schema interactions

## Workflow gap

- **Bug observed:** the synthesized-envelope fallback (~lines 846-856) pins 'kind': 'epm:results', 'version': 1 in code; on a follow-up round's re-run or a re-drained sentinel this reproduces the version-collision class (#389/#825) in the poller path
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

the synthesized envelope should omit the version (let task_workflow.post_event derive max+1) or compute max(existing)+1 for the kind; needs its own analysis of multipart/pointer-marker + sentinel-schema interactions

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: 5ff6ef7fb425

routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — this session is a workflow-fix session; candidate LOGGED for the next orchestrator pass, NOT auto-filed)
source: prose-followup (planner § Follow-ups + Claude code-reviewer bug-class sweep, both confirmed the line)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/poll_pipeline.py
bug_observed: The synthesized-envelope fallback (~lines 846-856) pins "kind": "epm:results", "version": 1 in code; on a follow-up round's re-run or a re-drained sentinel this reproduces the same version-collision class (#389/#825) in the poller path.
why_workflow_gap: The prose surfaces now defer to max+1 (#917), but this code path still hardcodes version 1 — the last checked-in literal for a round-versioned kind, outside #917's declared scope (poll_pipeline.py was must-not-touch).
proposed_change: The synthesized envelope should omit the version (let task_workflow.post_event derive max+1) or compute max(existing)+1 for the kind; needs its own analysis of multipart/pointer-marker + sentinel-schema interactions.
diff_sketch: |
  - envelope = {"kind": "epm:results", "version": 1, ...}
  + envelope = {"kind": "epm:results", ...}  # version omitted -> post_event derives max+1
confidence: medium
related_task: #917
<!-- /workflow-fix-candidate -->

