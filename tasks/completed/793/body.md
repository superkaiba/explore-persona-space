---
title: 'daily-fix: SLURM lane rsync from worktree branch not main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:847c5ea8a8ba
- daily-auto-filed
created_at: '2026-07-01T06:54:19Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: The SLURM lane rsyncs from the
  repo-root install (always on `main`) instead of the invoking worktree branch, so
  it cannot honor repo_branch=''issue-N'' and raises ValueError(''SLURM lane cannot
  honor rep'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Have the SLURM lane rsync from the invoking worktree path (which is on the issue branch), or stage the branch tree explicitly before rsync.

## Workflow gap

- **Bug observed:** The SLURM lane rsyncs from the repo-root install (always on `main`) instead of the invoking worktree branch, so it cannot honor repo_branch='issue-N' and raises ValueError('SLURM lane cannot honor repo_branch=...').
- **Evidence:** issue 773 on 2026-06-30. Source: /daily miner batch 10.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py
- fingerprint: 847c5ea8a8ba
- source: /daily route-2 (2026-06-30)
