---
title: 'daily-fix: verify_plan flags bare lane env vars in s10'
kind: infra
tags:
- wf-fix
- wf-fix-fp:57d5fe5c8b17
- daily-auto-filed
created_at: '2026-08-03T07:06:05Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): The #1979 f1g capture-leg
  launch attempt 1 died at spawn: the plan s10 command used the GCE-only bare `$WORKLOAD_ROOT`
  on the fellows lane, whose custom stage runs `set -u` -> ''WORKLOAD_ROOT: unbound
  variable''; a full dispatch burned before a set-u-safe relaunch (session ed4cd9a6,
  05:18:58Z log tail + 05:19:28Z marker; the session''s own note named the missing
  bash -n + set-u expansion dry-run).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miner5, session ed4cd9a6, task #1979).

## Goal

A plan s10 workload command cannot carry a lane-specific bare env var that dies under another lane's set -u stage.

## Workflow gap

- **Bug observed:** The #1979 f1g capture-leg launch attempt 1 died at spawn: the plan s10 command used the GCE-only bare `$WORKLOAD_ROOT` on the fellows lane, whose custom stage runs `set -u` -> 'WORKLOAD_ROOT: unbound variable'; a full dispatch burned before a set-u-safe relaunch (session ed4cd9a6, 05:18:58Z log tail + 05:19:28Z marker; the session's own note named the missing bash -n + set-u expansion dry-run).
- **Why it is a workflow gap:** The auto lane walks fellows FIRST, so a GCE-composed command routinely executes on SLURM; the env contract differs per lane and nothing lints s10 commands against it (c43 covers sentinel declarations, not env-var expansion).
- **Confidence (emitter):** medium (incident probed by miner: transcript rows; lint absence probed at compose time)
- verified-at-filing: `grep -c -iE 'WORKLOAD_ROOT|unbound|set -u' scripts/verify_plan.py` -> 0 (no such check).

## Proposed change (refine in planning)

add a verify_plan.py check flagging bare lane-specific env vars (e.g. $WORKLOAD_ROOT) in s10 workload commands on unpinned/auto-lane plans -- require the set-u-safe `${VAR:-<default>}` form or an explicit lane pin.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 57d5fe5c8b17

- workflow_fix_target: scripts/verify_plan.py

