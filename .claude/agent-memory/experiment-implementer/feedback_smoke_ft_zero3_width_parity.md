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
