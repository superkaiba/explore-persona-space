---
title: 'gotchas.md: fellows CPU analogue — plain-SSH nproc sees the node, inside-job
  nproc sees the allocation (ConstrainCores=yes vs ConstrainDevices=no asymmetry)'
kind: infra
tags: []
created_at: '2026-08-13T09:56:43Z'
has_clean_result: false
parent_id: 1336
workflow: v1
---
## Goal

Extend the existing `.claude/rules/gotchas.md` fellows-cluster entry — "**Fellows SLURM nodes are GPU-SHARED (no GPU cgroup isolation) — never size fan-out width or vLLM memory from device enumeration**" — with its **CPU analogue**, which the current entry does not cover and which is enforced by a DIFFERENT mechanism (so the existing text's reasoning does not carry over).

## The trap

`nproc` run over plain SSH to a fellows node reports the **physical node** (192 on node-2). `nproc` run INSIDE a job allocation reports the **allocation** (64), because the cluster enforces per-job core constraints:

```
TaskPlugin   = task/cgroup
ConstrainCores = yes            # /etc/slurm/cgroup.conf
SelectType   = select/cons_tres  (CR_CORE_MEMORY)
```

Verified live 2026-08-13 on node-2: `CPUTot=192`, `CPUAlloc=64`, against job 12643's `NumCPUs=64` / `MinCPUsNode=64`.

**Note the asymmetry with the GPU half of the entry, which is why this needs saying explicitly:** `ConstrainDevices=no`, so GPUs are NOT cgroup-isolated and `nvidia-smi -L` always shows all 8 — but `ConstrainCores=yes`, so CPUs ARE constrained and `nproc` is allocation-correct *inside* the job. The two resources behave OPPOSITELY on the same cluster. A reader who generalizes the existing GPU entry ("never trust enumeration") to CPUs draws the wrong conclusion in the other direction — that inside-job `nproc` is untrustworthy, when it is in fact the right source.

## Why it matters (issue #1336, 2026-08-13)

A thread-oversubscription fix computed `threads_per_worker = max(1, min(floor(nproc/width), inherited_OMP))`. Two errors followed from probing `nproc` outside the allocation:

1. **A wrong declared fix-engaged signal.** The crash-fix round recorded the expected banner as `nproc=192 threads_per_worker=24`. The correct values are `nproc=64 threads_per_worker=8`. Per `.claude/rules/crash-fix-rounds.md`, absence of the declared signal means "diagnose before any further relaunch" — so a HEALTHY run would have been read as fix-not-engaged, sending the next round to diagnose a non-bug and burning a launch cycle. Caught only because ~9.6 h of queue wait left time to re-derive it.
2. **Oversubscription understated 3x.** The incident was recorded as "512 threads on 192 cores" (2.7x). Against the real 64-core cpuset it is **8x** (8 workers x 64 inherited `OMP_NUM_THREADS` = 512 threads on 64 allocated cores) — which changes the diagnosis's strength, since 8x oversubscription explains a measured 0.27x-of-one-worker throughput far better than 2.7x.

## Requested change

Extend the existing fellows entry (do NOT open a new section — byte-budget pressure from the #2189 relocations) with:

- **CPU rule:** derive CPU/thread counts from the ALLOCATION, and probe from INSIDE the job (`srun`/`sbatch` context) or from `scontrol show job <id>` (`NumCPUs` / `MinCPUsNode`) — never a plain-SSH `nproc`, which sees the whole shared node. Inside-job `nproc` IS allocation-correct here because `ConstrainCores=yes`.
- **The asymmetry, stated in one clause:** `ConstrainCores=yes` (CPUs constrained) vs `ConstrainDevices=no` (GPUs NOT constrained) — do not generalize the GPU-enumeration rule to CPUs, or vice versa.
- **Read-side corollary:** a thread/width value recorded from a plain-SSH probe is UNVERIFIED for in-job purposes; when it is baked into a declared fix-engaged signal, the wrong expectation inverts the signal's meaning.

## Acceptance

- The fellows entry carries the CPU rule, the `ConstrainCores`-vs-`ConstrainDevices` asymmetry, and the allocation-probe forms.
- Cross-reference from `.claude/rules/crash-fix-rounds.md` § "declare the fix-engaged signal" is considered (a declared signal whose expected VALUES come from an out-of-allocation probe is a live failure mode there) — add a short pointer only if it does not bloat that file.
- `workflow_lint.py --check-lessons-index` passes.

## Provenance

Surfaced by the #1336 autonomous session during a queued wait on SLURM 12643 (fellows/charmander); the corrected signal + restated arithmetic are recorded in that task's `epm:progress` v291. Sibling task #2273 targets the same file for an unrelated zsh `nomatch` probe trap — distinct fingerprint, filed separately per the workflow-fix-on-bug dedup rule.
