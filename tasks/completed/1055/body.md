---
title: 'daily-fix: GCP finalize false-failed when deliverables compl'
kind: infra
tags:
- wf-fix
- wf-fix-fp:015c846696b4
- daily-auto-filed
created_at: '2026-07-05T07:03:13Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): Twice on 2026-07-04 a GCE
  workload ended with eps/phase=failed from a tail/finalize non-zero exit even though
  every declared deliverable was verified complete on HF: #811''s A100 at ~7.75h (''finalize
  hiccup, all deliverables complete'') and #811''s instance-stage tail crash after
  the 10.86 GB store landed. Each cost a manual diagnosis cycle to establish nothing
  was lost.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

When the finalize/tail step exits non-zero AFTER the workload's declared deliverables verify uploaded, publish a distinct phase (e.g. finalize_failed_artifacts_ok) instead of the terminal failed that triggers crash-response machinery.

## Workflow gap

- **Bug observed:** Twice on 2026-07-04 a GCE workload ended with eps/phase=failed from a tail/finalize non-zero exit even though every declared deliverable was verified complete on HF: #811's A100 at ~7.75h ('finalize hiccup, all deliverables complete') and #811's instance-stage tail crash after the 10.86 GB store landed. Each cost a manual diagnosis cycle to establish nothing was lost.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Sessions: f2ef929f (PM loop, 00:27-01:28 UTC), 86923fc5 (#811, ~00:30-01:30 UTC). Related: #1004 (rc-masking false phase=done) is the OPPOSITE defect (done on crash); keep the two coherent.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- source: /daily 2026-07-04 problem sweep (transcript-mined)
