---
name: multi-model-pod-weight-accumulation
description: "Panel plans sharding many models' cells across one pod: HF_HOME weights accumulate monotonically across waves — size the disk row on cumulative weight bytes + max concurrent in-flight capture stores, never 'one model resident' (#2588 v2)"
metadata:
  type: feedback
---

A §9 per-pod disk row for a MULTI-MODEL panel pod (N models' cells sharded
across the pod's GPUs in waves) must be sized on: cumulative HF_HOME weight
bytes across ALL models the pod ever loads (weights are never auto-reaped
between waves) + the MAX concurrent in-flight capture/output stores (one per
GPU running, not "≤2 cells") + venv/caches — or the plan declares a per-model
weight reap once a model's cells complete.

**Why:** #2588 v2 booked pod-2588 (4×H100, 11 cells / 7 distinct models, 3
waves) under a formula written for its single-model big pods ("weights ≤64 GB,
one big model resident + ≤2 cells' captures ≤40 GB + venv 15 ≈ 120 GB").
Actual profile: ~75 GB cumulative small-model weights + up to 4 concurrent
~11-15 GB captures + 15 GB venv ≈ 146 GB vs the RunPod MooseFS ~130 GB per-pod
EDQUOT quota; even the worst single wave landed ~132-134 GB. The
`assert_out_root_headroom` preamble converts the overrun into a mid-run HALT
on the pod carrying 11 of 19 cells — fail-loud but still a dead phase.

**How to apply:** whenever a plan's §9 assigns >1 model's generation/capture
cells to one pod, recompute the pod's peak as
`Σ weight_bytes(models ever loaded) + n_gpus × max_capture_store_gb + venv`
and check it against the pod quota (~130 GB MooseFS on RunPod). REVISE under
Methodology item 16 (fan-out accumulation: retained per-cell INPUTS count too)
unless a per-model weight reap or per-wave arithmetic under quota is stated.
Related: [[regen-trigger-headroom]] (the same plan's other cap-interaction trap).
