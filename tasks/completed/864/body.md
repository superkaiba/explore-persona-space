---
title: 'workflow-fix: zombie-GPU override veto keys on cpu/proc liveness, not only
  log freshness'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fe870da2e958
created_at: '2026-07-02T10:23:30Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #813: main poller with the #826 fix
  false-stalled twice (vllm_worker_dead_zombie_gpu) on a healthy host-PID-namespace
  run whose log was legitimately quiet 29 min during CPU-bound compression; veto must
  key on cpu_advancing / live workload procs'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #813 (emitting agent: orchestrator).

## Goal

Extend `poll_pipeline.py`'s zombie-GPU override veto beyond log-mtime freshness: suppress the `vllm_worker_dead_zombie_gpu` override when `cpu_advancing` is true or live workload processes (run_cell/dispatch pattern) exist in the container.

## Workflow gap

- **Bug observed:** On pod-813 (issue #813, 8×H100, host-PID-namespace container), the main-checkout poller — already carrying the #826 namespace-robust fix (`4916a60763`, stall-window-coupled liveness veto + 2-tick persistence) — emitted `status=stalled` / `stall_reason=vllm_worker_dead_zombie_gpu` twice (2026-07-02 07:40Z and 10:21Z) while the run was manifestly HEALTHY: 4 `issue813_run_cell.py` processes alive at ~100% CPU each (long CPU-bound NPZ-compression stretch), `cpu_advancing: true` in the tick JSON itself, DONE count advancing 1→4/12, disk healthy. The 4 "dead-in-proc" VRAM holders (~38 GB each) flagged by the zombie probe were these SAME 4 live cells seen under their HOST-namespace PIDs (nvidia-smi reports host PIDs; the container `/proc` shows container PIDs).
- **Why it is a workflow gap:** The #826 liveness veto is coupled to the stall window (log mtime), so it LAPSES exactly when a healthy workload legitimately silences the log for >stall-window during long CPU-only stretches (NPZ compression, Xet hash phases of 20–80 GB batch commits). On a host-PID-namespace pod the raw dead-in-proc signature is then permanently "true" for every live process, so any long quiet stretch converts to a false zombie stall. The poller already computes `cpu_advancing` (it was `true` in both false-positive ticks) but does not consult it in the override; had the orchestrator followed the SKILL.md `stalled` routing mechanically, a healthy 7-cell run would have been failure-routed to an experimenter respawn with a zombie-reap brief (destroying 4 in-flight cells' in-RAM state).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
# scripts/poll_pipeline.py — zombie override site (~line 2657)
- if status == "running" and zombie_gpu_pids:
+ # Veto the zombie override on ANY positive liveness signal, not only a
+ # fresh log: cpu_advancing (session CPU delta) or live workload procs.
+ workload_procs_alive = _probe_live_workload_procs(pod)  # pgrep -c -f 'run_cell|dispatch' via the existing SSH probe
+ if status == "running" and zombie_gpu_pids and not cpu_advancing and not workload_procs_alive:
      ... override to stalled ...
```

Also consider mapping host↔container PIDs via `/proc/<pid>/status` `NSpid:` (or comparing the count of VRAM-holding PIDs to the count of live workload processes) so the probe itself stops mis-flagging live processes on host-PID-namespace pods. Update `tests/test_poll_pipeline_zombie_gpu.py` with a fixture for the "cpu_advancing=true + quiet log + host-PID VRAM holders" shape, and keep the `failure_patterns.md` mirror in sync.

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Secondary: `tests/test_poll_pipeline_zombie_gpu.py`, `.claude/skills/issue/failure_patterns.md` (mirror)
- Grep the workflow surface for the pattern before editing (`grep -rln 'vllm_worker_dead_zombie_gpu' .claude/ CLAUDE.md scripts/ src/explore_persona_space/backends/`) and update every hit; list them in the plan. (Known hits: `scripts/poll_pipeline.py`, `scripts/backend_poll.py`, `src/explore_persona_space/backends/runpod.py`, `src/explore_persona_space/backends/base.py`, `.claude/skills/issue/SKILL.md`, `.claude/skills/issue/failure_patterns.md`, `tests/test_poll_pipeline_zombie_gpu.py` — most carry the stall_reason through unchanged; the override logic lives in `poll_pipeline.py`.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- Do NOT weaken the TRUE-zombie detection (#664: genuinely orphaned `VLLM::EngineCore` holding VRAM with the parent dead AND no CPU advance) — the veto must key on POSITIVE liveness signals, so a genuinely dead run (no CPU delta, no workload procs) still trips the override.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: fe870da2e958

<!-- workflow-fix-candidate v1 -->
target_file: scripts/poll_pipeline.py
bug_observed: main poller with the #826 fix emitted stalled/vllm_worker_dead_zombie_gpu twice on issue #813 while 4 run_cell processes were alive at 100% CPU and cpu_advancing=true — the flagged dead-in-proc VRAM holders were the live cells' host-namespace PIDs, and the log-mtime-coupled liveness veto lapsed during a legitimate 29-min CPU-only quiet stretch
why_workflow_gap: the zombie override's liveness veto keys only on log freshness, so on host-PID-namespace pods every long CPU-bound quiet stretch (NPZ compression, Xet hashing) converts to a false zombie stall despite the poller itself computing cpu_advancing=true
proposed_change: veto the vllm_worker_dead_zombie_gpu override on any positive liveness signal — cpu_advancing OR live workload processes (run_cell/dispatch pgrep) — not only a fresh log; optionally map host↔container PIDs via /proc NSpid
diff_sketch: |
  - if status == "running" and zombie_gpu_pids:
  + if status == "running" and zombie_gpu_pids and not cpu_advancing and not workload_procs_alive:
        ... override to stalled ...
confidence: high
related_task: #813
<!-- /workflow-fix-candidate -->
