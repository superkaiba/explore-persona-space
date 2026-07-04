---
title: 'workflow-fix: lint Step 0 stale-label fresh-label-execute clause'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:6310d2419ffa
created_at: '2026-07-04T07:09:21Z'
has_clean_result: false
origin_prompt: 'parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target (recursion
  guard, workflow-fix-on-bug.md § Recursion guard). Candidate surfaced by round-2
  implementer + Codex reviewer prose: add a workflow_lint.py mechanical assertion
  that the SKILL.md Step 0 stale-label disposition paragraph contains an explicit
  fresh-label-execute clause and no unconditional ''On None ... skip the label'' branch.
  target_file: scripts/workflow_lint.py. Not auto-routed from this session; next human/orchestrator
  pass may file it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

add a workflow_lint.py mechanical assertion that the SKILL.md Step 0 stale-label disposition paragraph contains an explicit fresh-label-execute clause and no unconditional skip-on-None branch

## Workflow gap

- **Bug observed:** the /issue SKILL.md Step 0 stale-label disposition paragraph could regress to an unconditional 'On None ... skip the label' branch without any mechanical assertion
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

add a workflow_lint.py mechanical assertion that the SKILL.md Step 0 stale-label disposition paragraph contains an explicit fresh-label-execute clause and no unconditional skip-on-None branch

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: 6310d2419ffa

parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target (recursion guard, workflow-fix-on-bug.md § Recursion guard). Candidate surfaced by round-2 implementer + Codex reviewer prose: add a workflow_lint.py mechanical assertion that the SKILL.md Step 0 stale-label disposition paragraph contains an explicit fresh-label-execute clause and no unconditional 'On None ... skip the label' branch. target_file: scripts/workflow_lint.py. Not auto-routed from this session; next human/orchestrator pass may file it.
