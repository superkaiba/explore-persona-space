---
title: 'daily-fix: HF upload_dataset 504 verify-side false-fail'
kind: infra
tags:
- wf-fix
- wf-fix-fp:49195ef9fae5
- daily-auto-filed
created_at: '2026-07-01T06:54:28Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: upload_dataset uploads the file
  successfully but the post-upload tree-listing verification 504s (Gateway Time-out
  on /tree/main?recursive=true) and the helper raises RuntimeError(''upload_dataset
  retur'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Retry/tolerate a 504 on the post-upload tree-listing verification (the upload already succeeded); do not map a verify-side gateway timeout to 'upload failed silently'.

## Workflow gap

- **Bug observed:** upload_dataset uploads the file successfully but the post-upload tree-listing verification 504s (Gateway Time-out on /tree/main?recursive=true) and the helper raises RuntimeError('upload_dataset returned ... HF Hub upload failed silently'), forcing a hand HfApi upload workaround (a reproducible '504 verification storm').
- **Evidence:** issue 666 on 2026-06-30 (#749 documented the gotcha; this is the code fix). Source: /daily miner batch 09.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/hub.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: src/explore_persona_space/orchestrate/hub.py
- fingerprint: 49195ef9fae5
- source: /daily route-2 (2026-06-30)
