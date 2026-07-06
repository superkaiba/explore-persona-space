---
title: 'daily-fix: poll_pipeline sentinel consume-then-post crash-sa'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b0b7150e42e8
- daily-auto-filed
created_at: '2026-07-06T06:58:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-05 problem sweep (route 2): #952''s post-restart sentinel
  poll consumed the results sentinel (marked .processed) and THEN hung on SSH (hostkey
  drift after the GCP instance restart), leaving no marker posted and forcing a manual
  drain plus a hand-check that epm:results v2 had not double-posted. Session cfbd36d3,
  task #952, ~2026-07-06T00:02Z.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-05 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Make the sentinel consume->post sequence crash-safe: post the marker (idempotent, keyed on sentinel content/attempt id) BEFORE marking .processed, or write a drain-in-progress breadcrumb the next poll can recover from - so a poller killed/hung between the two steps never strands an un-posted result.

## Workflow gap

- **Bug observed:** #952's post-restart sentinel poll consumed the results sentinel (marked .processed) and THEN hung on SSH (hostkey drift after the GCP instance restart), leaving no marker posted and forcing a manual drain plus a hand-check that epm:results v2 had not double-posted. Session cfbd36d3, task #952, ~2026-07-06T00:02Z.
- **Why it is a workflow gap:** the failure originates in the workflow surface / shared helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Pick the ordering with an idempotency key so neither duplicate markers nor lost markers are possible; the planner should weigh both orderings.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files; relevant tests pass.

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- source: /daily 2026-07-05 problem sweep (transcript-mined)
