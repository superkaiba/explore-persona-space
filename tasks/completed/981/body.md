---
title: 'workflow-fix: rescope BPE-delimiter memory immunity claim'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:1bb37601134f
created_at: '2026-07-04T07:12:05Z'
has_clean_result: false
origin_prompt: 'source: prose-followup (code-reviewer round 1). target_file: .claude/agent-memory/experiment-implementer/feedback_zero_width_span_bpe_delimiter_merge.md.
  bug_observed: memory file lines 18-19 still carry the unqualified FORMAT-scoped
  ''Chat-template formats are immune'' claim that #929 MF2 rescoped to special-token
  BOUNDARIES in the gotchas bullet; an experiment-implementer loading only its memory
  reads the over-broad claim. proposed_change: one-line rescope of the memory file''s
  Why/How-to-apply immunity sentence to boundary-scoped phrasing. routed: parked —
  EPM workflow-fix session recursion guard (candidates logged, never auto-routed from
  a wf-fix session).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

one-line rescope of the memory file's Why/How-to-apply immunity sentence to boundary-scoped phrasing

## Workflow gap

- **Bug observed:** memory file lines 18-19 still carry the unqualified FORMAT-scoped 'Chat-template formats are immune' claim that #929 MF2 rescoped to special-token BOUNDARIES in the gotchas bullet; an experiment-implementer loading only its memory reads the over-broad claim
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

one-line rescope of the memory file's Why/How-to-apply immunity sentence to boundary-scoped phrasing

## Scope / surfaces

- Primary target: `.claude/agent-memory/experiment-implementer/feedback_zero_width_span_bpe_delimiter_merge.md`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/agent-memory/experiment-implementer/feedback_zero_width_span_bpe_delimiter_merge.md
- fingerprint: 1bb37601134f

source: prose-followup (code-reviewer round 1). target_file: .claude/agent-memory/experiment-implementer/feedback_zero_width_span_bpe_delimiter_merge.md. bug_observed: memory file lines 18-19 still carry the unqualified FORMAT-scoped 'Chat-template formats are immune' claim that #929 MF2 rescoped to special-token BOUNDARIES in the gotchas bullet; an experiment-implementer loading only its memory reads the over-broad claim. proposed_change: one-line rescope of the memory file's Why/How-to-apply immunity sentence to boundary-scoped phrasing. routed: parked — EPM workflow-fix session recursion guard (candidates logged, never auto-routed from a wf-fix session).
