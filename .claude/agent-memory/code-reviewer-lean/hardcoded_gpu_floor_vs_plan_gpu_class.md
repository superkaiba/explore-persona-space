---
name: hardcoded-gpu-floor-vs-plan-gpu-class
description: Check every hardcoded GPU free-MiB floor in a dispatcher against the plan's pinned GPU class TOTAL MiB — an H200-inherited 120000 floor is unsatisfiable on H100 (~81,559 total) and exits "no usable GPUs" on the planned pod (#2479 R1 g8)
metadata:
  type: feedback
---

When a dispatcher/wrapper gates device usability on `nvidia-smi
--query-gpu=memory.free` ≥ a hardcoded MiB floor, verify the floor is
satisfiable on the GPU class the PLAN pins — compare against the class's
TOTAL MiB (H100 80GB ≈ 81,559; H100 NVL ≈ 95,830; H200 ≈ 143,771; A100-80
≈ 81,920). A floor above the class total means EVERY device is skipped and
the wrapper dies "no usable GPUs" on its very first GPU leg — on the
planned hardware, before any work.

**Why:** #2479 R1 g8: the new `issue2479_p1p4_launch.sh` required ≥120,000
MiB free (an H200-era value inherited from the parent #1345 runs) while
plan §9 pinned 4×H100 in the compute table, the dispatch command, and the
wrapper's own header comment. The U1-ported capture launcher in the SAME
pipeline used 60,000 — the tell that the new value was a stale-hardware
constant, not a sized requirement. The wrapper's own --dry-run and all
local smokes pass (no GPU query executes), so only a pod launch exposes it
(#408 burn-a-pod-per-shallow-bug shape).

**How to apply:** for any diff adding/porting a `min_free`/free-MiB
constant: (1) find the plan's pinned GPU type (compute table + dispatch
command + `--gpu-type`); (2) compare the floor to that class's total MiB —
floor > total is an automatic Critical; floor > ~85% of total deserves a
question (boot/driver overhead eats some); (3) diff the floor against the
SAME pipeline's sibling launchers — divergent floors across phases sharing
one pod are a smell; (4) remember dry-runs never execute the nvidia-smi
branch, so smoke evidence cannot clear this class. Prescribed fix shapes:
sibling-parity constant, or derive the floor from `memory.total` (e.g.
≥70%). Related: [[registered-gate-quantity-substituted]] (plan-vs-code
diffing), the Step 0.72 own-device scoping gate (orthogonal — this is
about the THRESHOLD, that one about WHICH devices aggregate).
