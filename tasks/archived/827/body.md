---
title: 'workflow-fix: liveness-veto the #664 zombie-GPU stall flag (2-tick persistence)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0ae098bbb021
created_at: '2026-07-02T00:46:01Z'
has_clean_result: false
origin_prompt: 'poll_pipeline.py falsely flagged stalled/vllm_worker_dead_zombie_gpu
  on a healthy #778 followup run during a between-phase vLLM engine cycle; add a log-freshness/cpu-advancing
  veto + two-tick persistence to the zombie override.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed detector false positive on task #778 (follow-up round poll tick 1).

## Goal

Guard the #664 zombie-GPU stall override in `poll_pipeline.py` with run-liveness evidence: do not flag `vllm_worker_dead_zombie_gpu` when the log mtime is fresh (< ~60s) and session CPU is advancing; require the zombie condition to persist across two consecutive ticks before overriding status to `stalled`.

## Workflow gap

- **Bug observed:** a `poll_pipeline.py` tick flagged `stalled` / `vllm_worker_dead_zombie_gpu` on a HEALTHY #778 follow-up run (`last_log_mtime_sec_ago: 8`, `cpu_advancing: true`, phase advancing through `manyshot_regen`) because a transient host-namespace compute PID (313516) held ≥1 GiB VRAM during a vLLM engine teardown/spin-up cycle between phases (the driver cycles engines per phase; nvidia-smi briefly attributes VRAM to a dying host-namespace pid that is absent from the container's `/proc` by construction). A direct probe 2 minutes later found the dispatcher alive, a fresh EngineCore loading shards, judge calls returning 200, and NO compute pid holding VRAM.
- **Why it is a workflow gap:** the #664 zombie heuristic ("compute pid holds ≥1024 MiB but is absent from /proc → stalled") is namespace-blind and instant-sampled; on multi-phase drivers that repeatedly tear down + respawn vLLM engines, the teardown window reliably produces this transient state, so every such run risks a false `stalled` verdict that routes to an unnecessary experimenter respawn (which would `kill -KILL` a healthy engine per the zombie-reaper brief). The orchestrator caught it by manual probe this time; the detector should carry the evidence guard itself.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

In `poll_pipeline.py`, at the zombie-GPU override site (the #664 block):

```
+ # Evidence guard: a genuinely dead worker cannot be appending to the log.
+ zombie_candidate = <existing condition>
+ if zombie_candidate and (last_log_mtime_sec_ago < 60 or cpu_advancing):
+     log WARNING "zombie-GPU signature with FRESH liveness evidence — deferring one tick"
+     record zombie_candidate_ts in the poll state sidecar
+ elif zombie_candidate and prior tick also recorded zombie_candidate:
+     status = "stalled"; stall_reason = "vllm_worker_dead_zombie_gpu"   # unchanged
```

Two-consecutive-ticks persistence + a log-freshness/cpu-advancing veto. Keep the existing behavior for the true-positive case (#664's actual incident had a stale log + no CPU advance).

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'vllm_worker_dead_zombie_gpu' .claude/ CLAUDE.md scripts/`) and update every hit (the gotchas rule text + experimenter memory reference the reaper recipe — those stay; only the DETECTOR's firing condition changes).

## Constraints / invariants

- Workflow-surface only.
- Do not weaken the true-positive path: a stale-log + no-CPU zombie must still flag on the second tick at latest.
- `tests/` pin: add a unit test for the veto (fresh log ⇒ no stall flag on first tick; persistent zombie + stale log ⇒ stall).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: 0ae098bbb021

<!-- workflow-fix-candidate v1 -->
target_file: scripts/poll_pipeline.py
bug_observed: poll_pipeline.py tick flagged stalled/vllm_worker_dead_zombie_gpu on a healthy #778 followup run (log mtime 8s, cpu_advancing true, phase advancing) because a transient host-namespace compute pid held VRAM during a vLLM engine teardown/spin-up cycle between phases.
why_workflow_gap: The #664 zombie heuristic is namespace-blind and instant-sampled; multi-phase drivers that cycle vLLM engines reliably produce this transient state, so healthy runs risk false stalled verdicts that route to unnecessary (and destructive — kill -KILL) experimenter respawns.
proposed_change: Guard the zombie override with run-liveness evidence (skip when log mtime < ~60s or CPU advancing) and require the zombie condition to persist across two consecutive ticks before flagging stalled.
diff_sketch: |
  + if zombie_candidate and (last_log_mtime_sec_ago < 60 or cpu_advancing):
  +     defer one tick (record candidate in poll state)
  + elif zombie_candidate and prior_tick_candidate:
  +     status = "stalled"  # unchanged true-positive path
confidence: high
related_task: #778
<!-- /workflow-fix-candidate -->
