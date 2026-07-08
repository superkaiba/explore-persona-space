---
title: 'workflow-fix: add c16 N/A escape phrase to adv-planner list'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:4c2a960e48a0
created_at: '2026-07-04T07:12:09Z'
has_clean_result: false
origin_prompt: 'parked — running under workflow_fix_target Provenance (recursion guard,
  .claude/rules/workflow-fix-on-bug.md § Recursion guard). Candidate (prose, from
  implementer round 1):

  target_file: .claude/skills/adversarial-planner/SKILL.md

  bug_observed: the Phase 1.5.0 canonical N/A escape-phrase list (lines ~113-121)
  enumerates per-check escape phrases but is now missing c16''s phrase after task
  #937 lands.

  proposed_change: add one line — `N/A — no re-extracted reference arms` (check 16)
  — to that list.

  confidence: high

  related_task: #937

  routed: parked: EPM_WORKFLOW_FIX_SESSION'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add one line — 'N/A - no re-extracted reference arms' (check 16) — to the Phase 1.5.0 canonical N/A escape-phrase list

## Workflow gap

- **Bug observed:** the Phase 1.5.0 canonical N/A escape-phrase list (lines ~113-121) enumerates per-check escape phrases but is missing c16's phrase after task #937 lands
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** high

## Proposed change (candidate sketch — refine in planning)

add one line — 'N/A - no re-extracted reference arms' (check 16) — to the Phase 1.5.0 canonical N/A escape-phrase list

## Scope / surfaces

- Primary target: `.claude/skills/adversarial-planner/SKILL.md`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/skills/adversarial-planner/SKILL.md
- fingerprint: 4c2a960e48a0

parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard). Candidate (prose, from implementer round 1):
target_file: .claude/skills/adversarial-planner/SKILL.md
bug_observed: the Phase 1.5.0 canonical N/A escape-phrase list (lines ~113-121) enumerates per-check escape phrases but is now missing c16's phrase after task #937 lands.
proposed_change: add one line — `N/A — no re-extracted reference arms` (check 16) — to that list.
confidence: high
related_task: #937
routed: parked: EPM_WORKFLOW_FIX_SESSION
