---
name: cuDNN-TF32 vs fp32-matmul parity gates diverge on H100/A100
description: fp32 parity asserts comparing a cuDNN RNN kernel path (nn.GRU/nn.LSTM) against plain ATen matmuls fail at ~2e-4 with CORRECT math because cudnn.allow_tf32 defaults True while cuda.matmul.allow_tf32 defaults False; run parity gates on a float64 clone with a tight ~1e-8 tol.
type: feedback
---

On H100/A100, `torch.backends.cudnn.allow_tf32` defaults TRUE while
`torch.backends.cuda.matmul.allow_tf32` defaults FALSE — so an fp32 parity
assert comparing a cuDNN RNN kernel path (`nn.GRU`/`nn.LSTM`, even a length-1
unroll) against plain ATen matmuls (`nn.GRUCell` / a hand-rolled reference)
diverges at ~2e-4 and fails any ~1e-4 tolerance with CORRECT math. The
divergence is kernel precision, not implementation parity: at tiny toy dims
true fp32 accumulation noise is ~1e-7.

Fix shape: run the kernel-parity gate on a float64 deep-copy clone (TF32 is
fp32-only, so the kernel gap vanishes — measured noise 1.1e-16) with a TIGHT
~1e-8 tolerance, so a real transposed-weight / wrong-gate bug still fails.
Keep a loose (~1e-2) fp32-vs-fp64 cross-check for the production dtype, and
keep exercising the exact dispatched functions (hollow-gate rule). Do NOT
loosen the fp32 tolerance to ~1e-3 — that usually stops catching real bugs.

Differential diagnosis recipe (seconds, on the failing GPU): run the gate
as-is (TF32-on ≈ 2e-4 FAIL), re-run under `torch.backends.cudnn.allow_tf32 =
False` (≈ 2e-6 PASS), then in float64 (≈ 1e-16) — the three-point signature
confirms TF32 vs a genuine math bug. Distinct from the L4-vs-A100
gauge-origin-hardware trap (gotchas.md, #667): that one is cross-hardware;
this one is two kernel paths on ONE GPU. #841 round 21 (attempt-8 crash-fix).
