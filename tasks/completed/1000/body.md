---
title: 'daily-fix: thin trigger-dense wording from issue-tick surfaces'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:f2c24494d689
created_at: '2026-07-04T07:13:37Z'
has_clean_result: false
origin_prompt: 'daily finding 7 (2026-07-03): issue-tick turns refusal-killed 3x on
  #906; tick snapshot carries trigger-dense wording.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a /daily 2026-07-03 finding.

## Goal

thin trigger-dense vocabulary from tick prompts / triage snapshot content for harmful-content tasks (digest-only references), per the #866 prevention rule

## Workflow gap

- **Bug observed:** /issue-tick turns on #906 (insecure-code task) were killed 3x by Usage Policy false-positives, ending the session on a refusal — the tick snapshot/title surface carries trigger-dense wording
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

thin trigger-dense vocabulary from tick prompts / triage snapshot content for harmful-content tasks (digest-only references), per the #866 prevention rule

## Scope / surfaces

- Primary target: `.claude/skills/issue-tick/SKILL.md, scripts/tick_triage.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/skills/issue-tick/SKILL.md, scripts/tick_triage.py
- fingerprint: f2c24494d689

daily finding 7 (2026-07-03): issue-tick turns refusal-killed 3x on #906; tick snapshot carries trigger-dense wording.
