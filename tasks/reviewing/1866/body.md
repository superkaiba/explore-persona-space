---
title: 'daily-fix: SLURM completed-work-but-FAILED state disambiguat'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3a6ca9e3ca35
- daily-auto-filed
created_at: '2026-07-30T07:09:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): Fellows job 15194 completed
  all its work but reported SLURM state FAILED 0:0 and the poller read ''dead''; manual
  verification was needed. Plausibly the same #1836 status-writer tmp race polluting
  the final status write'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner H-P4 residual (#1689 fellows lane; the other two crashes are already filed as #1835/#1836)).

## Goal

A job whose workload finished must not read as bare 'dead' to the poller on a status-writer artifact.

## Workflow gap

- **Bug observed:** unverified hypothesis — verify at plan time: the FAILED 0:0 despite-complete state is the #1836 tmp race polluting the final write (miner hypothesis; #1836 covers the heartbeat/phase writes — whether it covers the exit-status path is the open question).
- **Why it is a workflow gap:** the poller maps SLURM FAILED to dead unconditionally; an artifacts-present disambiguation avoids false crash-fix rounds.
- **Confidence (emitter):** medium
- verified-at-filing: n/a — reason: incident is cluster-side; #1835/#1836 exist at proposed (task.py view, 2026-07-30) — this filing is ONLY the residual disambiguation.

## Proposed change (refine in planning)

Add the artifacts-present disambiguation to the SLURM poll path; verify #1836 fix coverage of the final status write first (dedup: if covered, deflect no-change).

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py
- fingerprint: 3a6ca9e3ca35
