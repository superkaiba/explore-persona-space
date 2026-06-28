---
title: Dedicated GCP persistent disk OR ext4 project quotas — bounded-by-construction
  disk isolation for active tasks
kind: infra
tags: []
created_at: '2026-06-27T02:08:01Z'
has_clean_result: false
parent_id: 679
---
## Goal

Make active-task VM disk consumption **bounded by construction** so no single
task can exhaust the shared CPU-VM root disk (`/`, ~500 GB ext4, fleet-shared),
and so a disk-full condition is recoverable **without deleting any active
task's data**.

## Why

Parent task #679 wired detection + escalation + a defensive parity guard for
the active-task disk-pressure class, but those are *mitigations*: an active
task can still hold a large re-downloadable / generated cache the
terminal-status-gated cleaner cannot reap, and #679's escalation only *warns*
(it never deletes active data, by design). The load-bearing structural fix
#679 deliberately deferred to this follow-up: isolate each active task's
footprint so one task's growth cannot starve the shared `/`, and so recovery
never requires deleting in-use data.

Two candidate mechanisms (pick one in planning, grounded against cost +
operational fit):

1. **Dedicated GCP persistent disk** mounted for `.claude/worktrees/` + the
   per-issue download/store caches, separate from the boot disk holding `/`.
   A runaway task fills its own volume, not the shared root; the boot disk
   stays healthy and every concurrent session keeps working.
2. **ext4 project quotas** (`prjquota`) per issue under `data/issue_<N>/` +
   `.claude/worktrees/issue-<N>/`, so a task that exceeds its quota fails its
   own writes (a clean, attributable, fail-loud signal) instead of silently
   consuming shared headroom and triggering the silent-Bash-failure regime
   (#552).

Either way the invariant is: **one task cannot exhaust `/`, and a disk-full
condition is bounded + recoverable without deleting active data.**

## Scope

Infra / fleet-operations change to the orchestration VM's storage layout +
the worktree/cache placement. Not an experiment. Parent: #679.
