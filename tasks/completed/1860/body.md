---
title: 'daily-fix: humanize loop verifies candidate body before set-'
kind: infra
tags:
- wf-fix
- wf-fix-fp:345cf12944ce
- daily-auto-filed
created_at: '2026-07-30T07:07:55Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): #1775''s humanize pass
  applied the body via task.py set-body BEFORE verifying; the live body carried a
  v4 conciseness FAIL until an extra edit/verify cycle'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner G-P4 (session 3693ac94, #1775; probed)).

## Goal

The live task body should never carry a verifier FAIL the pipeline itself is about to catch — verify the candidate file first, then apply.

## Workflow gap

- **Bug observed:** The humanize loop set-body'd a candidate that then FAILed verify_task_body (v4 conciseness), costing an extra edit/set-body/verify cycle with a briefly-live non-compliant body.
- **Why it is a workflow gap:** the SKILL's humanize step orders apply-then-verify; the verifier's --file mode exists precisely for pre-apply checks.
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: `grep -n add_argument scripts/verify_task_body.py` -> `--file` ('path to a body.md to verify') exists (re-confirmed L14646 this run).

## Proposed change (refine in planning)

Reorder the step: verify the candidate file, apply on PASS only; on FAIL iterate on the candidate file.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 345cf12944ce
