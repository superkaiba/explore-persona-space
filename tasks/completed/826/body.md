---
title: 'workflow-fix: zombie-GPU probe namespace-aware (host-PID false positive)'
kind: infra
tags:
- wf-fix
created_at: '2026-07-02T00:39:09Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate v1: poll_pipeline.py zombie probe false-positives
  on host-namespace nvidia-smi PIDs (RunPod), overriding healthy runs to stalled (raised
  on #816 tick 1)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #816 (emitting agent: orchestrator).

## Goal

Make `poll_pipeline.py`'s zombie-GPU probe namespace-aware so healthy runs on host-PID-namespace RunPod containers are not overridden `running -> stalled`.

## Workflow gap

- **Bug observed:** On pod-816 (issue #816, 8×H100), the zombie-GPU probe flagged 3 nvidia-smi compute PIDs holding ~25 GB VRAM as "absent from /proc" and overrode a manifestly HEALTHY run (`phase_log_mtime_sec_ago: 1`, steering cells completing 7/53, 10 live workload processes) to `status=stalled` with `stall_reason: vllm_worker_dead_zombie_gpu`. On this RunPod host, `nvidia-smi --query-compute-apps` reports HOST-namespace PIDs which are never resolvable in the container's `/proc` — so the probe's "PID absent from /proc ⇒ dead worker" inference is structurally false-positive for EVERY healthy process on such hosts.
- **Why it is a workflow gap:** `scripts/poll_pipeline.py` is workflow surface; the #664 zombie predicate assumes container-visible nvidia-smi PIDs. A false `stalled` verdict routes healthy runs into the Step-7 infra-respawn path (experimenter respawn with a kill-by-PID reaper brief) — which would KILL healthy workers mid-run.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
In the zombie-GPU probe (poll_pipeline.py, #664 block):
+ # Namespace-mismatch guard: if NONE of the nvidia-smi compute PIDs resolve
+ # in /proc AND the workload's own log/phase mtime is fresh (< stall window),
+ # nvidia-smi is reporting host-namespace PIDs (RunPod containers) — skip the
+ # zombie override entirely for this tick.
+ if all_absent and phase_log_fresh:
+     log.info("zombie probe: all compute PIDs unresolvable + log fresh -> host PID namespace; skipping override")
+ else: (existing override)
Also consider: only fire the zombie override when log staleness ALSO holds
(the original #664 case had a DEAD log; a fresh log + unresolvable PIDs is
the namespace signature, not a zombie).
```

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'zombie' scripts/ .claude/`) and update every hit; list them in the plan (the recovery-brief prose in `.claude/skills/issue/SKILL.md` Step 7 + `.claude/rules/gotchas.md` may need a one-line "namespace false-positive" caveat).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The genuine #664 zombie case (dead log + unresolvable PIDs holding VRAM) MUST still fire — the fix narrows the predicate, never removes it.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: (computed at filing)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/poll_pipeline.py
bug_observed: zombie-GPU probe treats host-namespace nvidia-smi PIDs (unresolvable in container /proc on RunPod) as dead workers and overrides a healthy advancing run to status=stalled (issue #816 tick 1; log mtime 1s, cells completing).
why_workflow_gap: the #664 predicate assumes container-visible nvidia-smi PIDs; on host-PID-namespace containers it is false-positive for every healthy worker, routing healthy runs into the infra-respawn reaper path.
proposed_change: skip the zombie override when ALL nvidia-smi compute PIDs are unresolvable AND the workload log/phase mtime is fresh; require log-staleness as a conjunct for the override.
diff_sketch: |
  + if all_pids_unresolvable and phase_log_mtime_sec_ago < STALL_WINDOW:
  +     skip zombie override (host PID namespace signature)
  - unconditional override running -> stalled on unresolvable VRAM-holding PIDs
confidence: high
related_task: #816
<!-- /workflow-fix-candidate -->
