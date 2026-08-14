---
name: fanout-cvd-ordinal-not-entry
description: A GPU fan-out that detects n_gpus under the parent's CUDA_VISIBLE_DEVICES then pins children with CVD=str(g) escapes a restricted/reordered parent CVD — pin the g-th ENTRY of the parent's CVD list, not the ordinal (#2225 R1 g2)
metadata:
  type: feedback
---

When reviewing a subprocess GPU fan-out, check the pair: (a) where the slot
count comes from (`torch.cuda.device_count()` respects the LAUNCHER's CVD) and
(b) what the child env pins. `{"CUDA_VISIBLE_DEVICES": str(g)}` for
`g in range(n_gpus)` is an ABSOLUTE ordinal: under a parent
`CUDA_VISIBLE_DEVICES=4,5,6,7` (SLURM partial-node allocation, sibling job on
the pod), children land on GPUs 0..3 — outside the allocation, colliding with
other jobs. Correct: if parent CVD is set, `parent_cvd.split(",")[g]`.

**Why:** #2225 R1 g2 (`scripts/issue2225_train.py` `run_fan_out`): benign on
the plan's dedicated 8×H100 pod (CVD unset) so only a Concern, but the fellows
SLURM lane was the plan's named fallback — exactly where the scheduler sets a
restricted CVD. Distinct from the gotchas.md `+gpu_id=N` Hydra CVD-clobber
(that one OVERWRITES a pin; this one mis-derives it).

**How to apply:** on any fan-out/dispatcher diff, grep for
`CUDA_VISIBLE_DEVICES` writes and trace the index back to its source; flag
ordinal pins whenever the launcher can inherit a non-trivial CVD (SLURM lanes
reachable, shared pods). Also check the companion footgun seen same review: a
mode flag like `--fan-out` that is never read, so a BARE invocation falls
through to the full production fan-out.
