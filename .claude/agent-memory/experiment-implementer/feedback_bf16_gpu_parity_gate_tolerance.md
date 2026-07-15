---
name: bf16-GPU parity-gate tolerance (never calibrate on the CPU smoke)
description: A capture determinism/parity cosine bar calibrated on the bit-exact CPU-float32 smoke fails spuriously on bf16 CUDA (~1e-6 cosine noise from batch-composition reduction order); use >= 0.9999, grounded on committed same-surface gates
type: feedback
---

A within-run determinism / capture-parity gate whose threshold was chosen on the
CPU-float32 smoke (bit-exact re-computation) fails SPURIOUSLY in production on
bf16 CUDA: the re-capture batches rows differently, cuBLAS reduction order
shifts, and cosines land ~1e-6 below 1.0 (measured 0.99999887 on #1005,
2026-07-15 — refused an entirely healthy 50-context capture and cost a full
GCP instance cycle).

**Why:** float32 CPU torch is deterministic for these reductions; bf16 CUDA
kernels are batch-shape-dependent. A real failure (dropout left on, wrong
weights/revision, adapter mis-application) reads cosine < 0.99 — three orders
of magnitude away.

**How to apply:** when implementing any GPU capture-parity / determinism
spot-check, set the cosine bar to >= 0.9999 (not 6 nines), ground it on a
committed same-surface reference (parent #928's parity gates: 0.999; #1005 F3
prefix-constancy: 0.9999), and state the bf16 rationale in a comment. Never
promote the CPU smoke's bit-exactness into the production bar.
