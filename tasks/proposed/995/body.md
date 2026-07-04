---
title: 'daily-fix: batch_judge retry read-after-write 404 post-create'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-04T07:13:12Z'
has_clean_result: false
origin_prompt: 'daily finding 2 (2026-07-03): batch_judge treated an immediate post-create
  404 as terminal (#742 5th judge-rerun launch killed).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a /daily 2026-07-03 finding.

## Goal

treat a 404 within ~60s of batches.create as retryable with backoff in the batch_judge poller

## Workflow gap

- **Bug observed:** Anthropic Batch API read-after-write 404 — polling a msgbatch id 67ms after batches.create returned it raised anthropic.NotFoundError and was treated as terminal, killing #742's 5th judge-rerun launch
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

treat a 404 within ~60s of batches.create as retryable with backoff in the batch_judge poller

## Scope / surfaces

- Primary target: `src/explore_persona_space/eval/batch_judge.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: src/explore_persona_space/eval/batch_judge.py
- fingerprint: 5d28cf7715a9

daily finding 2 (2026-07-03): batch_judge treated an immediate post-create 404 as terminal (#742 5th judge-rerun launch killed).
