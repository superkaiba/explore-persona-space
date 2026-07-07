---
title: 'daily-fix: HF upload/verify transient 429/500 retry robustness'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-04T07:13:22Z'
has_clean_result: false
origin_prompt: 'daily finding 4 (2026-07-03): HF 429 storms + transient 500 on verify
  leg killed #931 staging and crashed #920 post-upload.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a /daily 2026-07-03 finding.

## Goal

longer/backoff-aware retry ceiling on tree-listing + xet-token paths, and retry on the _upload_tensors verify leg

## Workflow gap

- **Bug observed:** (a) org-wide HF 429 storms on xet-read-token / /tree listing killed #931's staging via retry exhaustion and slowed #811/#813; (b) a transient HF 500 on the paginated list_repo_files verify leg of _upload_tensors crashed #920 AFTER all uploads succeeded (GCE EXIT trap then deleted the instance)
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

longer/backoff-aware retry ceiling on tree-listing + xet-token paths, and retry on the _upload_tensors verify leg

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/hub.py (upload/staging paths)`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: src/explore_persona_space/orchestrate/hub.py (upload/staging paths)
- fingerprint: 970e18ecb679

daily finding 4 (2026-07-03): HF 429 storms + transient 500 on verify leg killed #931 staging and crashed #920 post-upload.
