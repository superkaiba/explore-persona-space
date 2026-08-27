---
title: No check verifies a GPU-lane-routed phase actually allocates a CUDA context
  (P5 ran 4xH100 at 0 MiB)
kind: infra
tags:
- workflow-fix
created_at: '2026-08-27T09:52:03Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'Measured during #2546 arm 2 p5_fits: 8 nvidia-smi samples over 56s
  showed all four H100s at 0 percent AND 0 MiB, so no CUDA context existed, while
  the phase ran healthily CPU-bound at ~4.8 cores. The dispatcher logged alloc=0,1,2,3,
  so the log looks like GPU routing. Arm 3 ran the same phase the same way for 2.11h,
  so it has been silently true across arms.'
workflow: v1
---
---
kind: infra
---

# Nothing verifies that a phase routed to a GPU lane actually uses the GPU

## Goal

Add a check that catches a phase which is planned, sized, and provisioned onto a GPU lane but
never allocates a CUDA context. Today a plan can assert GPU-worthiness, pass the compute-sizing
review, provision a peak-width GPU pod, and run entirely on CPU with nobody noticing until
someone samples `nvidia-smi` by hand.

## Observed (issue #2546 arm 2, `p5_fits`, 2026-08-27)

The P5 fits phase ran on `pod-2546-arm2` (4x H100) with the GPUs completely unused. Sustained
measurement, 8 samples over 56 s:

    gpu 0..3    0 %   0 MiB    on every sample

The `0 MiB` is what makes this conclusive rather than suggestive: no CUDA context is allocated at
all, so it is not a gap between compute spikes. Meanwhile the phase was healthily CPU-bound,
measured on a cumulative basis over a ~30 min window:

    session_cpu_secs   432 -> 9,040   =  ~290 CPU-sec/min  =  ~4.8 cores across 4 workers

The dispatcher even reports a GPU allocation: `[p5] fan-out: 71 registry jobs across 4 worker(s)
(alloc=0,1,2,3)`. So the observable that looks like GPU routing (an allocation string in the log)
is present while the actual GPU use is nil. An operator reading the log would reasonably conclude
the GPUs were in use.

Arm 3 ran the same phase the same way for 2.11h, so this is the established behavior of the
recipe rather than a regression in one arm. That is the point: it has been silently true across
arms.

## Why the existing surfaces did not catch it

The compute-character machinery is well developed in the opposite direction. `.claude/rules/pods.md`
and `.claude/rules/plan-compute-sizing.md` both work hard to route work TO a GPU lane when it
deserves one (the iterative-optimization carve-out, the ~15-30 min GPU-worthiness floor) and to
keep CPU-only phases OFF GPU pods. But every one of those is a PLAN-TIME assertion about what the
phase will do. Nothing compares the assertion against what the phase actually did.

`.claude/rules/pods.md` is also explicit that GPU-width right-sizing is "plan-time routing, NOT a
mid-run gate", which is correct as a guard against thrashing a live run, but it means a wrong
plan-time call has no later correction point at all. In #2546 the honest options mid-run were to
migrate (an unsanctioned deviation risking the round) or to burn ~2h of idle 4x H100. Neither is
good, and both were avoidable at plan time.

## Requested changes (for the plan to choose among; not prescriptive)

1. **A cheap realized-use probe.** Any phase provisioned on a GPU lane could sample
   `nvidia-smi --query-gpu=utilization.gpu,memory.used` a few times early in the phase and warn
   loudly when memory.used stays at 0 MiB across the window. Memory is the better signal than
   utilization: it distinguishes "no CUDA context" from "between kernels" without needing a long
   sample. The poller already reads `gpu_util` every tick and already has a
   `gpu_idle_advisory_posted` field, so most of the plumbing exists; what is missing is treating
   sustained ZERO MEMORY as a distinct, louder verdict than idle utilization.
2. **Make the plan state it.** Have `plan-compute-sizing.md` require each GPU-lane phase to state
   whether it allocates a CUDA context, so there is a falsifiable claim to check the probe
   against. A phase that cannot answer yes is a CPU phase.
3. **Close the mid-run dead end.** Decide, and write down, what a session SHOULD do on discovering
   this mid-run. The current rules leave it stranded between an unsanctioned migration and burning
   the pod. Even a documented "record it, finish the phase, file the plan defect" is better than
   silence, because that is what #2546 did by inference.

## Explicitly NOT to be done

- Do NOT turn this into a mid-run migration gate. The plan-time-only posture of the GPU-width
  carve-out is deliberate and mid-run backend thrash is worse than idle GPUs.
- Do NOT weaken the existing route-TO-GPU carve-outs (the iterative-optimization fit rule, the
  ~15-30 min floor). This adds a verification of the routing decision, not a new bias against GPU
  lanes.
- Do NOT touch the live `pod-2546-arm2`, which is running the phase in question.

## Provenance

Measured by the orchestrator during #2546 arm 2 `p5_fits` after the poller reported
`gpu_util: 0,0,0,0`, which was then confirmed as sustained (8 samples / 56 s, 0 MiB throughout)
rather than instantaneous. Recorded on #2546 as `epm:progress` v184, together with the decision to
hold rather than migrate and the reasoning for it.
