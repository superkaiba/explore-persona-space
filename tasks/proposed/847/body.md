---
title: 'infra: default BLAS/torch thread caps for shared-VM python launches'
kind: infra
tags: []
created_at: '2026-07-02T07:19:57Z'
has_clean_result: false
origin_prompt: 'PM-chat VM triage 2026-07-02: load 186-226 on 32 cores; zero OMP caps
  repo-wide; 5-6 jobs x 32 threads. Diagnosis /tmp/vm_load_diagnosis.md. Fix: setdefault
  thread caps on shared-VM launches only.'
---
## Overview / Motivation

Auto-filed from the 2026-07-02 shared-VM overload incident (load 186-226 on 32 cores). Read-only diagnosis: /tmp/vm_load_diagnosis.md (copy into artifacts/ at planning).

## Goal

Cap BLAS/torch intra-op threads for python jobs launched on the SHARED VM so N concurrent jobs cannot oversubscribe the box ~6x (each uncapped torch/numpy job spawns nproc=32 intra-op threads; 5-6 concurrent jobs -> run queue ~160-190 -> load 186-226; PSI cpu some avg10=96.5%).

## Workflow gap

- **Bug observed:** repo-wide, NO VM launch surface sets OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS or torch.set_num_threads; /proc/environ of all 6 heavy PIDs during the incident had zero caps. Each job realizes only ~5-6 cores of throughput while holding 32 runnable threads.
- **Why it is a workflow gap:** the shared VM hosts every session's CPU work; uncapped defaults make fleet collisions pathological (64-min layers for a 60s job; earlyoom SIGTERM sweeps at <10% mem-avail killed 4-7GB workers silently).
- **Confidence (emitter):** high

## Proposed change (refine in planning)

- `src/explore_persona_space/orchestrate/env.py` `load_dotenv()`: `os.environ.setdefault` OMP/MKL/OPENBLAS_NUM_THREADS to a VM default (e.g. 8) ONLY when running on the shared VM (not RunPod `/workspace`, not GCE, not SLURM — pods/instances are dedicated and WANT full width). Launch-time explicit values always win (setdefault).
- Add the cap to the VM-side launch guidance in `.claude/rules/code-style.md` (compute-throughput bullet) so ad-hoc / smoke / free-analysis launches inherit it.
- Consider `torch.set_num_threads` equivalent where torch is imported before env (document the ordering constraint).

## Scope / surfaces

- Primary: `src/explore_persona_space/orchestrate/env.py` (library; in the #747 spirit), `.claude/rules/code-style.md`.
- Verify no pod/GCE/SLURM regression: those lanes must keep full threads (grep `bootstrap_pod.sh`, `backends/gcp.py`, `backends/slurm.py`).

## Constraints / invariants

- setdefault-only (explicit env always wins). Detection must fail OPEN (no cap) off-VM.
- ruff passes; tests for the detection predicate.

## Provenance

- origin: PM-chat VM triage 2026-07-02, load-186 diagnosis (/tmp/vm_load_diagnosis.md)
