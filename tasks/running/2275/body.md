---
title: 'workflow-fix: SLURM --mem is GPU-count-derived with no per-dispatch override
  (memory-bound phase OOMs)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-13T13:29:09Z'
has_clean_result: false
origin_prompt: '#1336 fellows job 12643 fit_pool: 3 pooled units OOM-killed (cgroup
  oom_kill 3, memory.peak==memory.max==1TiB); --mem = min(128*gpus,1800) = 1024G with
  no override path, mem_gb_cap=1800 unused'
workflow: v1
---
## Provenance

workflow_fix_target: src/explore_persona_space/backends/slurm.py

Filed by the #1336 orchestrator (fellows SLURM job 12643) after `fit_pool` lost 3 pooled fit units to cgroup OOM.

## Overview / Motivation

Filed from #1336 (fellows SLURM job 12643, 2026-08-13): the `fit_pool` phase lost 3 pooled fit units to cgroup OOM (`memory.events oom_kill 3`, `memory.peak == memory.max == 1 TiB`) because the job's memory allocation is a pure function of GPU count and cannot be raised per dispatch.

## Goal

A memory-bound phase on a SLURM lane can declare a RAM requirement and either get it or fail loud — instead of silently receiving a GPU-count-derived allocation and OOMing in production.

## Workflow gap

- **Bug observed:** `backends/slurm.py:1823` renders `#SBATCH --mem={min(cluster.mem_gb_per_gpu * gpus, cluster.mem_gb_cap)}G`. On fellows (`mem_gb_per_gpu=128`, `mem_gb_cap=1800`) an 8-GPU job gets `min(1024, 1800)` = **1024 G**, leaving 776 G of the cluster's own configured cap unused. #1336's 8-wide pooled fit needed ≈1,550 GiB (measured on-arm units 65–70 GiB RSS; off-arm units carry ~4× the rows) and was OOM-killed.
- **Why it is a workflow gap, and why it is NOT a duplicate:** there is no per-dispatch memory knob on this lane at all — `min_ram_gb` never appears in `slurm.py`, so a stated RAM requirement is silently inert with **no signal**, and `verify_plan` has no SLURM-lane RAM check. Two consequences: (a) buying RAM requires over-requesting GPUs (memory is per-GPU), so a memory-bound phase holds GPUs it cannot use — #1336's fix keeps 8 GPUs solely for the 1 TiB ceiling and will idle 6 of them; (b) requesting FEWER GPUs *reduces* the memory that caused the failure, which is the opposite of the intuitive remedy.
- **Distinct from the two completed siblings, both GCP-scoped:** #1998 (`--min-ram-gb` inert on a **GCP GPU** dispatch → fail loud) states in its own Proposed-change section "keep RunPod-CPU (#1010) and SLURM behavior byte-unchanged"; #2033 (RAM floor on wide fan-out) targets `verify_plan.py` against **GCP ladder rungs** (A100-40 85 GB boxes). Neither touches the SLURM `--mem` formula or its missing override. Different target file, different mechanism, different lane.
- **Confidence:** high — the formula, the absent override, and the OOM counters were all read directly (see verified-at-filing).
- verified-at-filing (2026-08-13): `grep -nE 'mem_gb_per_gpu|mem_gb_cap|--mem' src/explore_persona_space/backends/slurm.py` → the formula at :1823, fellows row `mem_gb_per_gpu=128  # cluster rule 2; node ceiling ~251 G/GPU` at :482 and `mem_gb_cap=1800` at :485; `grep -c min_ram_gb src/explore_persona_space/backends/slurm.py` → 0 (no override path); `dispatch_issue.py launch --help` exposes `--min-ram-gb` with no SLURM consumer. Live job cgroup on node-7: `memory.max 1099511627776`, `memory.peak 1099511627776`, `memory.events → max 81848 / oom 99 / oom_kill 3`, matching the 3 rc=137 units exactly.

## Proposed change (refine in planning)

In `backends/slurm.py`: honor a declared RAM requirement on the SLURM lanes — thread `min_ram_gb` (already parsed by `dispatch_issue.py`) into the `--mem` render as `max(formula, min_ram_gb)` clamped to `mem_gb_cap`, and FAIL LOUD pre-submit when the request exceeds `mem_gb_cap` (naming the cap and the satisfying alternatives) rather than silently submitting the formula value. Consider raising the fellows `mem_gb_per_gpu` toward the recorded ~251 G/GPU node ceiling, which the existing `mem_gb_cap=1800` already bounds — but that is a shared-config change affecting every fellows dispatch, so planning should weigh it against the per-dispatch override, which is the narrower fix. Plan-side twin: a `verify_plan` WARN when a plan §9 RSS estimate × realized fan-out width exceeds the SLURM lane's rendered `--mem`, mirroring #2033's GCP-side check. Keep GCP / RunPod-CPU behavior byte-unchanged.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py`
- Likely also `scripts/verify_plan.py` (plan-side WARN), `scripts/dispatch_issue.py` (thread + refuse), `tests/test_slurm_backend.py` pins, and `.claude/rules/plan-compute-sizing.md` (document that SLURM RAM is GPU-count-derived, so a memory-bound phase sizes its GPU request for RAM).
