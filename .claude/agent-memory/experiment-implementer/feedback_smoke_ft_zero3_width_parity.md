---
name: Smoke keeps ZeRO-3 production process width
description: Narrowing a ZeRO-3 full-FT smoke to --num_processes 1 OOMs deterministically at the first optimizer step; smoke/production parity includes the RESOURCE dimension (process width), and a parent's narrow-smoke pin transfers only if the smoke host transfers too
type: feedback
---

A cloned dispatcher that narrows a ZeRO-3 full-FT smoke to `--num_processes 1`
OOMs deterministically at the FIRST optimizer step — single-process ZeRO-3
shards nothing, so 7B bf16 weights + grads + fp32 Adam moments (~86 GB) land on
one 80 GB GPU (#1315 p1_train smoke, 2026-07-15; traceback: `exp_avg_sq`
alloc fail in `stage3.py _optimizer_step`).

**Why:** smoke = production with fewer steps/cells INCLUDING the process
shape — width is a resource dimension of smoke/production parity (#397
class). #1112's `_ft_num_processes` smoke=1 pin was legitimate ONLY because
its smoke ran on a 1-GPU GCE instance; #1315's smoke runs on the 4x A100-80
ft-7b pod, so the pin must not transfer with the clone.

**How to apply:** when cloning a dispatcher, audit every `if cfg.smoke:`
branch that composes LAUNCH width / CUDA_VISIBLE_DEVICES — keep production
width unless the smoke HOST genuinely differs; pin it with an
arg-composition regression test asserting `--num_processes <N>` + CVD in
BOTH modes (worked example: tests/test_issue1315_dispatch.py
test_ft_launch_width_smoke_invariant).

RECURRED same-day in #1333 (crash-fix r4, 2026-07-15): the #1333 dispatcher
was written BEFORE this memory landed and carried the identical
`if cfg.smoke: return 1` in `_ft_num_processes`; its pod smoke died rc=1
~50 s into p2_train on the SAME 4x A100-80 ft-7b pod. The r4 fix mirrors
#1315 (smoke-invariant width + under-provision guard +
tests/test_issue1333_dispatch.py::test_ft_launch_width_smoke_invariant) and
adds `_run_subprocess` inner-log-tail-on-failure so the next subprocess
crash's traceback lands in the crash-persisted workload.log (the inner
`ft_mk4.log` was outside the GCE trap globs and died with the instance).
Sweep duty: any OTHER in-flight dispatcher cloned from the #1112 family
needs the same audit before its first pod smoke.
