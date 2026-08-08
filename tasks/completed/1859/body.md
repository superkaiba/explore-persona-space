---
title: 'daily-fix: repo-root guard allows timeout-prefixed ssh paylo'
kind: infra
tags:
- wf-fix
- wf-fix-fp:59b21ad06827
- daily-auto-filed
- trigger-dense
created_at: '2026-07-30T07:07:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): guard_repo_root_branch.sh
  false-blocked a `timeout ... ssh` payload twice on the #1769 failover critical path;
  the gcloud arm already has the #1463 timeout allowance, the ssh arm does not'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner I-P4 (probed)).

## Goal

A timeout-wrapped remote ssh command is the sanctioned bounded-invocation shape; the guard should not fail-closed on it when the bare ssh form is waived.

## Workflow gap

- **Bug observed:** Two false blocks on the failover critical path; the guard's own header lists 'wrapped/variable/abs-path ssh (timeout N ssh ...)' as a known fail-closed residual.
- **Why it is a workflow gap:** the #1463 allowance was added to the gcloud arm only; the identical composition on the ssh arm still blocks.
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: `grep -n 'ssh|timeout' scripts/guard_repo_root_branch.sh` -> L90/149-150 gcloud timeout allowance; L344 names the ssh residual (2026-07-30).

## Proposed change (refine in planning)

Mirror the gcloud-arm timeout-prefix acceptance in the ssh waiver arm; add a hook test for `timeout 30 ssh host 'git ...'`.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: 59b21ad06827
