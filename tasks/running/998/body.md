---
title: 'daily-fix: exempt verbatim sample lines from humanize ban scan'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:5f6e60651aed
created_at: '2026-07-04T07:13:27Z'
has_clean_result: false
origin_prompt: 'daily finding 5 (2026-07-03): humanize absolute-ban check flags verbatim
  sample completions (#923).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a /daily 2026-07-03 finding.

## Goal

scope the humanize/absolute-ban scan to prose, exempting blockquoted **Completion:** / sample-data lines

## Workflow gap

- **Bug observed:** the Step 9a humanize absolute-ban check FAILed #923's clean-result because a VERBATIM sample completion (required by SPEC) contained 'Certainly!'
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

scope the humanize/absolute-ban scan to prose, exempting blockquoted **Completion:** / sample-data lines

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 5f6e60651aed

daily finding 5 (2026-07-03): humanize absolute-ban check flags verbatim sample completions (#923).
