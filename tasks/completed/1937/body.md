---
title: 'workflow-fix: gotchas entry — fellows shared-node GPU width + vLLM util sizing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f06efa34c0f7
created_at: '2026-07-31T13:20:11Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1902 crash-fix round (fellows
  job 16127: shared-node GPU allocation assumptions — width from device count + fixed
  vLLM util 0.6 died on a memory-shared H200 node)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson block raised on task #1902 (emitting agent: experiment-implementer, crash-fix round; orchestrator routed).

## Goal

Add a `.claude/rules/gotchas.md` entry documenting the fellows-cluster shared-node GPU trap: derive fan-out width/device ids from the SLURM allocation env and size vLLM `gpu_memory_utilization` from `torch.cuda.mem_get_info` free bytes — never a device count or a fixed fraction.

## Workflow gap

- **Bug observed:** #1902 fellows job 16127 died at P1 vLLM engine init: the dispatcher detected `ngpu=8` on a 4-GPU allocation (nvidia-smi enumerates the physical node — fellows has NO GPU cgroup isolation) and the fixed `gpu_memory_utilization=0.6` demanded 0.6×139.8=83.9 GiB on a device with 81.2 GiB free (other tenants hold ~58 GiB on EVERY GPU of node-2, live-measured 2026-07-31).
- **Why it is a workflow gap:** gotchas.md carries the fellows lane's launch traps (the CLAUDE.md fellows row + pod-side conventions), but nothing warns that fellows nodes are memory-SHARED — every future GPU-dispatching driver on this lane will re-hit both halves (width overstep + fixed-util engine death) unless the trap is documented where training/eval/orchestrate authors load it.
- **Confidence (emitter):** high
- verified-at-filing: `grep -rn 'fellows' .claude/rules/gotchas.md` → 0 hits for shared-node/GPU-isolation content (2026-07-31); the incident evidence is the #1902 `epm:failure` v1 marker + live `nvidia-smi --query-gpu=index,memory.used,memory.total` on charmander (all 8 devices ~57-59 GiB used); reference fix landed at issue-1902 commit `4b3fafa8dc84` (verified pushed on origin/issue-1902)

## Proposed change (candidate diff sketch — refine in planning)

```
+ ## Fellows SLURM nodes are GPU-SHARED (no isolation) — width + vLLM sizing traps
+ - nvidia-smi / torch.cuda.device_count() enumerate the PHYSICAL node (8x H200),
+   not your allocation; derive width + device ids from SLURM env:
+   CUDA_VISIBLE_DEVICES (if slurm-set) > SLURM_JOB_GPUS/SLURM_STEP_GPUS >
+   SLURM_GPUS_ON_NODE (ids assumed 0..N-1, log it); fail-loud when absent.
+ - Every device carries other tenants' memory (~58 GiB/device observed);
+   vLLM gpu_memory_utilization is a fraction of TOTAL memory — compute
+   min(cap~0.55, (free - margin~6GiB)/total) from torch.cuda.mem_get_info
+   on the leg's device, fail-loud floor ~0.20. Ref: issue1902_common.py::
+   vllm_util_for_free + issue1902_dispatch.sh width block (4b3fafa8dc84;
+   incident #1902 job 16127).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Consider whether `src/explore_persona_space/backends/slurm.py`'s fellows ClusterConfig row or CLAUDE.md's fellows lane bullet should cross-reference the entry (grep `grep -rln 'fellows' .claude/ CLAUDE.md` and cite, don't duplicate).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` no-flags passes (gotchas.md is lint-scanned; keep the LESSONS.md index row's fires-when current if the entry adds a new trigger class).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: f06efa34c0f7

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: #1902 fellows job 16127 died at P1 vLLM engine init: ngpu=8 detected on a 4-GPU allocation and fixed util 0.6 demanded 83.9 GiB on a device with 81.2 GiB free (other tenants hold ~58 GiB on every GPU)
why_workflow_gap: fellows nodes are memory-shared with no GPU isolation and nothing in gotchas.md warns future drivers off device-count width detection or fixed vLLM util fractions on that lane
proposed_change: add a gotchas.md entry: fellows SLURM nodes share GPUs with no isolation — derive width/device ids from the SLURM allocation env and size vLLM gpu_memory_utilization from mem_get_info free bytes, never a fixed fraction or device count
diff_sketch: |
  + gotchas.md § Fellows SLURM nodes are GPU-SHARED: width from SLURM env
  + (CVD > SLURM_JOB_GPUS > SLURM_GPUS_ON_NODE), vLLM util = min(0.55,
  + (free-6GiB)/total) via mem_get_info, floor 0.20; ref 4b3fafa8dc84 / #1902
confidence: high
related_task: #1902
<!-- /workflow-fix-candidate -->
