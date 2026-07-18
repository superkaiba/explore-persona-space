---
title: 'daily-held: settle #779-line mapping terminology + writeup b'
kind: infra
tags:
- daily-held
created_at: '2026-07-05T07:04:43Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 3): Thomas re-steered the #779
  writeup across ~19 messages on 2026-07-04 (00:48-04:07 UTC): context-vector/query-vector
  naming (4 messages), per-example specificity, eval-distribution coverage, why PCA=64,
  whether KRR answers the nonlinearity question. The draft used unsettled terminology
  and did not pre-justify method choices.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 3: judgment call -> needs-human), from the nightly transcript problem sweep.

## Goal

Held for your call (taste/scientific-meaning carve-out): confirm the settled terminology (context vector / query vector / prefix definitions) so it can be captured in the #779-line docs, and confirm the writeup convention that analysis choices (PCA dim, kernel choice) ship with their grounding by default.

## Workflow gap

- **Bug observed:** Thomas re-steered the #779 writeup across ~19 messages on 2026-07-04 (00:48-04:07 UTC): context-vector/query-vector naming (4 messages), per-example specificity, eval-distribution coverage, why PCA=64, whether KRR answers the nonlinearity question. The draft used unsettled terminology and did not pre-justify method choices.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `docs/`
- Session: 9caa26ea (interactive).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: docs/
- source: /daily 2026-07-04 problem sweep (transcript-mined)
