---
title: 'daily-fix: register step 5a in workflow.yaml steps enum'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0d8a713f1962
- daily-auto-filed
created_at: '2026-07-30T07:12:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): post_step_completed.py
  rejects step ''5a'' while issue SKILL.md names Step 5a 24 times — every session
  posting 5a hits the error branch and the step record coarsens to ''5'''
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner H-P6 (probed)).

## Goal

The step registry should accept every step id the SKILL documents; the mismatch degrades step records and burns an error branch per posting.

## Workflow gap

- **Bug observed:** `post_step_completed.py` enumerates the known-steps set on rejection — no 5a; the fallback coarsens the record to '5'.
- **Why it is a workflow gap:** SKILL.md and workflow.yaml drifted when Step 5a was introduced; the lint that pins SKILL-vs-yaml step consistency evidently does not cover sub-steps.
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: `grep -c 'Step 5a' .claude/skills/issue/SKILL.md` -> 24; `grep -c 5a .claude/workflow.yaml` -> 0 (re-confirmed this run).

## Proposed change (refine in planning)

Add the enum entry; check post_step_completed.py needs no other change; consider extending the consistency lint to sub-step ids.

## Scope / surfaces

- Primary target: `.claude/workflow.yaml`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/workflow.yaml
- fingerprint: 0d8a713f1962
