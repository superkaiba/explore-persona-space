---
title: 'daily-fix: failover re-provision carries launch env pins'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2520f66e6d68
- daily-auto-filed
created_at: '2026-07-25T06:48:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): The watcher wedge-failover
  replacement pod for issue 1586 lost the original launch''s env pins - WANDB_PROJECT
  was unset, so 3 runs synced to the wrong default WandB project and 4 marker-FT runs
  stayed pod-local, causing upload-verify round-1 FAIL'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session dd0af0ae, task #1586).

## Goal

A watcher-provisioned failover pod must inherit the original launch's environment pins so runs land in the right WandB project / upload destinations.

## Workflow gap

- **Bug observed:** the failover pod for #1586 (05:33Z re-provision) lost `WANDB_PROJECT` — 3 runs synced to the wrong default WandB project and 4 mk-FT runs stayed pod-local; upload-verification FAILed round 1 (10:45Z: "the failover pod lost the WANDB_PROJECT pin") and needed a manual pod-side sync + re-verify to PASS.
- **Why it is a workflow gap:** the wedge-failover re-provision path boots the replacement pod without threading the original dispatch's env (the handle sidecar `.claude/cache/issue-<N>-handle.json` holds the launch context), so any env-pinned workload silently degrades after failover.
- **Confidence (emitter):** high on the incident; medium on where the pins should be persisted (handle sidecar vs dispatch env snapshot).
- verified-at-filing: `git log --oneline --since='7 days ago' -- scripts/autonomous_session_watch.py` → 5 commits, none touching failover env threading (2026-07-25). Incident evidence is #1586's upload-verification round-1 FAIL record — session records, not grep-verifiable.

## Proposed change (candidate diff sketch — refine in planning)

On the wedge-failover re-provision path, read the original launch env pins (WANDB_PROJECT et al.) from the issue handle sidecar (or persist them there at dispatch if absent) and export them into the replacement pod's bootstrap.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (failover core); possibly `scripts/pod_lifecycle.py` / dispatch handle writer

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 2520f66e6d68

- workflow_fix_target: scripts/autonomous_session_watch.py
