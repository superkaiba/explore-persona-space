---
title: 'workflow-fix: thread disk requirement into RunPod CPU fallback (containerDiskInGb
  + refuse-on-infeasible)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2ba06dfd7720
created_at: '2026-07-04T12:06:34Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #958 experimenter: RunPod CPU fallback
  threads no disk-size requirement; deployCpuPod supports containerDiskInGb; GCP-sized
  CPU stages >50GB peak silently dispatched onto infeasible pods (fp 2ba06dfd7720)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #958 (emitting agent: experimenter).

## Goal

Thread a disk-size requirement into the RunPod CPU fallback (containerDiskInGb) and refuse / route away when the mapped RunPod CPU instance cannot satisfy the plan's footprint.

## Workflow gap

- **Bug observed:** The cpu-mid RunPod fallback provisioned a 50G-container-disk / 16 GB-RAM CPU pod for a workload whose plan §9 sized 80 GB disk + 32 GB RAM (GCP e2-standard-8), forcing an experimenter pre-launch abort (#958).
- **Why it is a workflow gap:** The RunPod CPU fallback lane threads no disk-size requirement — deployCpuPod supports containerDiskInGb, but the lane uses a fixed instance/disk regardless of the plan's footprint, so any GCP-sized CPU stage >50 GB peak is silently dispatched onto an infeasible pod.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
# backends/runpod.py (CPU-pod create path)
+ container_disk_gb = max(50, requested_boot_disk_gb or 50)
+ payload["containerDiskInGb"] = container_disk_gb
# backends/router.py (_runpod_terminal_rung / CPU fallback)
+ if intent in RUNPOD_CPU_INSTANCE_FOR_INTENT and requested_boot_disk_gb > RUNPOD_CPU_MAX_DISK.get(instance, 50):
+     raise NoComputeAvailableError("cpu fallback cannot satisfy disk requirement")
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/runpod.py`, `src/explore_persona_space/backends/router.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'RUNPOD_CPU_INSTANCE_FOR_INTENT\|deployCpuPod' src/explore_persona_space/backends/ scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; tests/test_router*.py invariants preserved (RunPod stays the last rung; cpu-bigmem keeps its typed terminal).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/runpod.py, src/explore_persona_space/backends/router.py
- fingerprint: 2ba06dfd7720

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/runpod.py, src/explore_persona_space/backends/router.py
bug_observed: The cpu-mid RunPod fallback provisioned a 50G-container-disk / 16 GB-RAM CPU pod for a workload whose plan §9 sized 80 GB disk + 32 GB RAM (GCP e2-standard-8), forcing an experimenter pre-launch abort (#958).
why_workflow_gap: The RunPod CPU fallback lane threads no disk-size requirement — deployCpuPod supports containerDiskInGb, but the lane uses a fixed instance/disk regardless of the plan's footprint, so any GCP-sized CPU stage >50 GB peak is silently dispatched onto an infeasible pod.
proposed_change: Thread a disk requirement (from the GCP lane's --boot-disk-gb / plan footprint) into the RunPod CPU fallback's containerDiskInGb, and refuse/route to cpu-bigmem when the requirement exceeds what the mapped RunPod CPU instance can provide.
confidence: medium
related_task: #958
<!-- /workflow-fix-candidate -->
