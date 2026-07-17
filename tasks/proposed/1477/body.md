---
title: 'workflow-fix: poll_pipeline zombie-override CPU-rate arbiter false-stalls
  healthy CPU-bound runs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4b6b5b634e9b
created_at: '2026-07-17T14:10:26Z'
has_clean_result: false
origin_prompt: 'orchestrator observation on #1345 2026-07-17: negative session-CPU
  rate samples flipped the zombie override to stalled/vllm_worker_dead_zombie_gpu
  against a synchronously-verified 2000%+ CPU opcomp worker (3 false stalls in one
  afternoon)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #1345 (emitting agent: /issue orchestrator, 2026-07-17).

## Goal

Make `scripts/poll_pipeline.py`'s zombie-override session-CPU rate arbiter robust: the rate must never read negative on a healthy workload, and the override to `stalled` should confirm with a synchronous per-PID `pcpu` probe before flipping a `running` verdict.

## Workflow gap

- **Bug observed:** on pod-1345 (task #1345, 2026-07-17, relaunch 3), during a healthy ~3h CPU-bound `opcomp` segment (worker pid 184938 at 2012-2064% CPU in synchronous `ps` probes), the zombie-GPU override flipped status `running` -> `stalled` with `stall_reason=vllm_worker_dead_zombie_gpu` at 13:09Z and 14:08Z because the cross-tick session-CPU rate sampled NEGATIVE (`now=-0.88 prev=-0.18`, earlier `now=0.82 prev=-0.56`; veto threshold 0.50). A negative rate on a monotone CPU counter is impossible for a live workload — the sampling scope resets/drifts across poll ticks (each tick is a fresh SSH session), so the #951 liveness veto fails exactly during the log-quiet CPU-bound segments it exists to protect (#518/#658 class). The same tick's `_session_cpu_advancing` INFO line showed `current=9351.0 vs running-max=83824.0` — the "session CPU" total DROPPED by ~74k core-seconds between ticks, confirming scope inconsistency rather than workload death.
- **Why it is a workflow gap:** `poll_pipeline.py` is the fleet-wide run-liveness arbiter; a false `stalled`/`vllm_worker_dead_zombie_gpu` verdict routes orchestrators toward the zombie-reaper/failure path against healthy runs (the #664 recovery brief prescribes kill -KILL on the flagged pids). Only a manual synchronous probe prevented mis-routing on #1345.
- **Confidence (emitter):** high (three contradicting probe pairs captured in #1345 events v119/v120 + tick transcripts).
- verified-at-filing: `grep -n "_session_cpu\|veto threshold" scripts/poll_pipeline.py` -> 8 hits in 1 file (scripts/poll_pipeline.py: L603 veto-threshold comment; L3964 `_parse_session_cpu`; L3994 `_session_cpu_advancing`; L4051 `_roll_session_cpu_max`) (2026-07-17).

## Proposed change (candidate diff sketch — refine in planning)

```
  # in the zombie-override rate arbiter (#951 leg):
+ # A negative cross-tick rate is IMPOSSIBLE for a live workload on a
+ # monotone counter — treat it as NO SIGNAL (sampling-scope reset), not
+ # as "CPU idle": fall back to a synchronous per-PID probe before
+ # overriding running -> stalled.
+ if rate_now < 0 or rate_prev < 0:
+     sync_pcpu = _probe_top_workload_pcpu(ssh)   # ps -o pcpu of the launcher tree
+     if sync_pcpu is not None and sync_pcpu >= 50.0:
+         veto (healthy CPU-bound); do not override
```

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '_session_cpu' scripts/`) and update every hit; list them in the plan. Add a regression test beside the existing poll-pipeline tests pinning: negative rate + alive high-pcpu workload => no stalled override.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- Fail-safe direction preserved: when BOTH the rate and the synchronous probe are unavailable, keep today's behavior (2-tick persistence override) — the fix must not make a genuinely dead worker unreapable.

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: 4b6b5b634e9b

Surfaced prose (verbatim): the zombie-override's session-CPU RATE arbiter can compute a negative rate on a healthy 20-core workload (observed now=-0.88/prev=-0.18 and now=0.82/prev=-0.56 on pod-1345, 2026-07-17, task #1345 opcomp segment; running-max 83824 -> current 9351 core-sec drop confirms cross-tick scope reset), flipping the #826/#951 zombie override to status=stalled + stall_reason=vllm_worker_dead_zombie_gpu on healthy log-quiet CPU-bound segments.
