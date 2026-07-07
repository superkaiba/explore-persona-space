---
title: 'daily-fix: batch_judge tolerates transient 404 on poll retrieve'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-04T23:01:38Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — batchjudge-poll-404 (fp 5051f513acb4)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

Bounded transient-404 retry (backoff, N attempts over a few minutes) on the poll retrieve beyond the create grace, before declaring a batch lost.

## Workflow gap

- **Bug observed:** 2026-07-03 #742: a transient batches.retrieve 404 MID-POLL crashed the driver (epm:failure infra); the batch was alive server-side and the 6th launch later harvested it 10/10 — a full relaunch cycle lost. #995's landed grace covers only BATCH_CREATE_404_GRACE_S after create, not a 404 hours into polling.
- **Why it matters:** The Anthropic batch API demonstrably returns transient 404s for live batches; a single-shot read is too fragile for a multi-hour poll loop.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/eval/batch_judge.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: src/explore_persona_space/eval/batch_judge.py
- fingerprint: 5051f513acb4
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)
