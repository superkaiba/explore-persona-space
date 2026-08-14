---
title: 'gotchas.md: dispatcher deriving GPU width/pins from nvidia-smi instead of
  the SLURM allocation targets unallocated GPUs on narrow allocations'
kind: infra
tags: []
created_at: '2026-08-12T19:32:33Z'
has_clean_result: false
origin_prompt: 'Surfaced by /issue 1336''s plan-registered 1-GPU precheck (SLURM 11981):
  dispatcher computed NGPU from nvidia-smi --list-gpus (8) on a 1-GPU allocation and
  overrode the correct inherited CUDA_VISIBLE_DEVICES=5 with a literal 0, re-pointing
  compute to physical GPU 0 which another job was using. Latent on whole-node allocations,
  which is why four prior 8-GPU attempts showed no symptom.'
workflow: v1
---
## Goal

Add a `.claude/rules/gotchas.md` entry (GPU/orchestration section) for a demonstrated cross-issue trap: **a dispatcher that derives GPU width or device pins from `nvidia-smi` rather than from the SLURM allocation silently targets GPUs it was never allocated whenever the allocation is narrower than the physical node.** The failure is invisible on whole-node allocations — which is why it can sit latent for many runs and then surface the first time someone runs a narrow smoke/precheck/debug job.

## Observed (task #1336, 2026-08-12, SLURM job 11981 on the fellows/charmander cluster)

A deliberately narrow 1-GPU gate-chain precheck was dispatched (`dispatch_issue.py launch ... --gpus 1`). The per-issue dispatcher computes:

```bash
NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l )     # scripts/issue1336_dispatch.sh:91
```

and then pins devices with a literal index — `CUDA_VISIBLE_DEVICES=0` on its single-GPU paths, and `width=$NGPU` with `CUDA_VISIBLE_DEVICES=$w` for w in 0..width-1 in its work-conserving worker pool.

Ground truth, probed from INSIDE the allocation (`srun --jobid=11981 --overlap --ntasks=1` on the compute node — the only place these values are readable):

- `scontrol show job 11981 -d` → `AllocTRES=...,gres/gpu=1`, `Nodes=node-0 ... GRES=gpu:1(IDX:5)`
- inside the job: `CUDA_VISIBLE_DEVICES=5`, `SLURM_STEP_GPUS=5`, `SLURM_JOB_GPUS=UNSET`
- inside the job: `nvidia-smi --list-gpus | wc -l` → **8**
- node-0 occupancy at probe time: GPUs 0-3 and 6-7 carried other jobs (5.6-51 GB, 7-51% util); GPU 4 free; GPU 5 (ours) 2,135 MiB.

So SLURM did everything correctly — it allocated one GPU and exported the right `CUDA_VISIBLE_DEVICES`. The dispatcher then *overrode* that correct value with a literal `0`, re-pointing the process from our allocated physical GPU 5 to physical GPU 0, a device another job was using. `NGPU=8` would additionally have fanned the worker pool across eight GPUs we did not hold (it did not fire in this run only because none of the four prechecked phases calls the pool — verified by enumerating every call site).

## The two distinct mistakes, worth naming separately

1. **`nvidia-smi` is not an allocation query.** `nvidia-smi --list-gpus` is a driver-level enumeration that ignores `CUDA_VISIBLE_DEVICES`, so it reports the physical device count regardless of what the scheduler granted. A GPU count derived from it is the NODE's width, never the JOB's. (Corollary, which cost this session a wrong marker claim: an `nvidia-smi` count of 8 inside a 1-GPU job is NOT evidence that cgroup isolation is absent — the tool simply does not answer that question.)
2. **Overriding an inherited `CUDA_VISIBLE_DEVICES` with a literal index discards the mapping.** Under `CVD=5`, the process's `cuda:0` already IS physical GPU 5. Setting `CUDA_VISIBLE_DEVICES=0` does not mean "our first GPU" — it re-selects physical GPU 0. The index space of the pin must be the inherited list, not the physical node.

## Why it stays latent

On a whole-node allocation SLURM exports `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, so `nvidia-smi`'s count and the physical indices coincide with the allocated set and every literal pin resolves correctly. Task #1336 ran four prior 8-GPU attempts through this same dispatcher with no symptom. The bug appears only on a narrow allocation — precisely the shape used for cheap smoke/precheck/debug runs, i.e. exactly when someone is trying to de-risk an expensive run.

## Proposed fix (prose-only, one gotchas.md entry)

State the rule and the correct pattern:

- Derive job GPU width from the allocation: parse the inherited `CUDA_VISIBLE_DEVICES` (comma-separated) and take its cardinality; fall back to an `nvidia-smi` count ONLY when `CUDA_VISIBLE_DEVICES` is unset/empty. `SLURM_STEP_GPUS` / `SLURM_JOB_GPUS` and `scontrol show job -d`'s `GRES=gpu:N(IDX:...)` are corroborating sources.
- Pin worker w to the w-th ELEMENT of the inherited list, never to the literal w:
  ```bash
  if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
      IFS=',' read -ra ALLOC <<< "$CUDA_VISIBLE_DEVICES"; NGPU=${#ALLOC[@]}
  else
      NGPU=$( (nvidia-smi --list-gpus 2>/dev/null || true) | wc -l )
      ALLOC=( $(seq 0 $((NGPU > 0 ? NGPU - 1 : 0))) )
  fi
  # worker w uses ${ALLOC[w]}; single-GPU paths use ${ALLOC[0]}
  ```
  This is byte-identical on a whole-node allocation (`ALLOC[w] == w`) and correct on a narrow one.
- Add the probe discipline that made this diagnosable, since it is reusable and this session got it wrong twice first: **a login-node shell is not the compute node.** Any per-node probe — `nvidia-smi`, `pgrep`, `/proc/<pid>/environ`, `free`, node-local `df` — must run INSIDE the allocation (`srun --jobid=<id> --overlap`) or it describes an unrelated machine. `scontrol` / `squeue` are cluster-wide and stay valid from the login node. (In this incident an `ssh <cluster-alias>` shell landed on node-2 while the job ran on node-0, producing a per-GPU memory table for the wrong machine and an empty `pgrep`.)

## Acceptance criteria

- `.claude/rules/gotchas.md` carries the entry in its GPU/orchestration region, naming both mistakes, the correct pattern, the whole-node-latency reason, and the `srun --overlap` probe discipline, with the `#1336` / job-11981 incident cited.
- If the LESSONS.md index row for `gotchas.md` enumerates trigger topics, extend it so the new topic is discoverable at plan time (the index is lint-enforced by `workflow_lint.py --check-lessons-index`).
- `uv run python scripts/workflow_lint.py` (no flags) no worse than its pre-change baseline (~15 pre-existing failures on `main`, unrelated — do not chase them; assert only that none newly names an edited file).

## Non-goals

- No change to `scripts/issue1336_dispatch.sh` (fixed in #1336's own code round v21, already dispatched) and no sweep of other issues' dispatchers in this task — the rule is the deliverable. A separate audit of existing `nvidia-smi`-derived widths across `scripts/issue*_dispatch.sh` may be filed later if wanted.
- No new mechanical lint. "Did this dispatcher derive width from the allocation?" is a shell-semantics question a text heuristic would mostly get wrong, and a false-positive lint on every legitimate `nvidia-smi` call would be worse than the prose rule.
- No change to the backend router or `dispatch_issue.py` — they passed `--gpus 1` correctly and SLURM honoured it; the defect is entirely inside per-issue dispatcher shell.

## Provenance

Surfaced by the `/issue 1336` autonomous session while running the plan-registered 1-GPU gate-chain precheck (SLURM 11981) that exists to de-risk a 210 GPU-h run. The precheck cleared five gates and resolved an offline-unresolvable bound; this was one of two bugs it found. Cost: no wasted compute (~0.3 GPU-h total, and the run's primary value was delivered).
