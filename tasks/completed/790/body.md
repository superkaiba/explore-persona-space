---
title: 'daily-fix: GCP backend fetch_results/log Permission denied'
kind: infra
tags:
- wf-fix
- wf-fix-fp:65809272ed66
- daily-auto-filed
created_at: '2026-07-01T06:53:51Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: GCP-lane best-effort scp of eval_results/issue_N
  + figures/issue_N fails with `Permission denied` (workload writes as root, gcloud
  scp runs as a user without read perms), and confirm_artifacts then FA'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Have GCP fetch_results + the log-tail probe read via sudo (or chmod artifacts/logs group-readable on the pod before teardown, or run the workload as the scp user), and stop hard-FAILing confirm_artifacts on figures/ the analyzer (not the workload) generates.

## Workflow gap

- **Bug observed:** GCP-lane best-effort scp of eval_results/issue_N + figures/issue_N fails with `Permission denied` (workload writes as root, gcloud scp runs as a user without read perms), and confirm_artifacts then FAILs on missing figures/, forcing a manual sudo-SSH fetch + --skip-confirm-artifacts finalize; monitoring log reads (`wc /workspace/logs/issue-N.log`) hit the same Permission denied and silently return empty.
- **Evidence:** issues 722/744 on 2026-06-30. Sources: /daily miners batches 04/07/09/10.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/gcp.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/gcp.py
- fingerprint: 65809272ed66
- source: /daily route-2 (2026-06-30)
