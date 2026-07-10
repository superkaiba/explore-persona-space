---
title: env-reading git credential helper for pod bootstrap
kind: infra
tags:
- wf-fix
- wf-fix-fp:8120e7804493
- daily-auto-filed
created_at: '2026-07-10T06:54:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): bootstrap_pod.sh step 4
  still embeds the GitHub token in the git remote URL at rest (verified on main, lines
  ~254-255), unlike the GCE lane which #1205 migrated to an env-reading git credential
  helper'
workflow: v1
---
## Overview / Motivation
Auto-filed by the /daily Step-C parked-candidate routing pass (2026-07-09) from a recursion-guard-parked workflow-fix candidate raised on task #1205.

## Goal
Migrate the pod-side tokenized-remote-at-rest pattern to the same env-reading git credential helper #1205 gave the GCE lane, for log-leak parity (token out of .git/config + remote URLs).

## Workflow gap
- **Bug observed:** bootstrap_pod.sh step 4 still embeds the GitHub token in the git remote URL at rest (verified on main, lines ~254-255), unlike the GCE lane which #1205 migrated to an env-reading git credential helper.
- **Why it is a workflow gap:** Token-at-rest in .git/config + remote URLs leaks into logs and process listings; the GCE lane already has the fixed pattern — the pod lane lacks parity.
- **Confidence (emitter):** low

## Proposed change (candidate diff sketch — refine in planning)
(none — mirror the #1205 GCE credential-helper leg)

## Scope / surfaces
- Primary target: `scripts/bootstrap_pod.sh`

## Constraints / invariants
- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance
- workflow_fix_target: scripts/bootstrap_pod.sh
- fingerprint: n/a (prose park)

parked — running under workflow_fix_target Provenance (recursion guard, workflow-fix-on-bug.md § Recursion guard). source: prose-followup (planner §12). target_file: scripts/bootstrap_pod.sh. proposed_change: migrate the pod-side tokenized-remote-at-rest pattern (bootstrap_pod.sh step 4) to the same env-reading git credential helper #1205 gave the GCE lane, for log-leak parity (token out of .git/config + remote URLs). confidence: low. For the nightly /daily parked-candidate routing pass.
