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

[SUPERSEDED-IN-PART by feedback_bf16_gpu_parity_gate_tolerance (two-bar update) — see #1005 round 3] **How to apply:** when implementing any GPU capture-parity / determinism
spot-check, set the cosine bar to >= 0.9999 (not 6 nines), ground it on a
committed same-surface reference (parent #928's parity gates: 0.999; #1005 F3
prefix-constancy: 0.9999), and state the bf16 rationale in a comment. Never
promote the CPU smoke's bit-exactness into the production bar.


**Two-bar update (#1005 round 3, supersedes the flat >= 0.9999 advice above):**
a flat 0.9999 bar is ALSO insufficient for near-single-position quantities
(short prefix means, boundary tokens) at DEEP layers — measured worst 0.991745,
layer 27 in all 50 contexts, layer-0 min >= 0.999999 everywhere. Use the #779
two-bar structure from gotchas.md: EARLY layers 0-3 >= 0.999 (real bugs corrupt
layer 0 at cos 0.43-0.84) + flat all-layer >= 0.98 (gross corruption). Span-mean
summaries smooth the noise and may keep tighter flat bars.

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [cuDNN-TF32 parity gates diverge on H100](feedback_cudnn_tf32_fp32_parity_gate.md) — cudnn.allow_tf32 defaults True (matmul TF32 off): fp32 parity asserts of nn.GRU-vs-GRUCell fail at ~2e-4 with correct math; run the gate on a float64 clone at ~1e-8 tol + loose fp32 cross-check. #841 r21.
- [Parity gates on determinate data are blind](feedback_parity_gate_determinate_data_blind.md) — end-to-end R2 parity tests pass even under the WRONG recipe on strong-signal data (#931: pre-fix 0.0006 vs post-fix 0.0008); the fails-pre-fix pin is a bit-level same-init serial-replica test with the early-stop branch asserted to fire; tiny no-signal smoke deltas are init noise.
- [Parity floor: weak writer vs gauge error](feedback_parity_floor_weak_writer_vs_gauge_error.md) — per-adapter-class write-ratio floors; gauge errors are √r-multiplicative, not 10% shortfalls (#813)
- [bf16 single-position equivalence-gate calibration](feedback_bf16_single_position_equivalence_gate_calibration.md) — span-mean-calibrated cosine bars lack headroom for single-position states; layer-27 bf16 jitter breaches 0.999 while layer 0 stays 0.999999; gate early layers per-layer + flattened with measured headroom (#779 r12)
- [numerics-probe thresholds calibrated to deployment dtype](feedback_numerics_probe_thresholds_dtype.md) — bf16 batched forwards read ~1e-3 cos deviations; never assert CPU-fp32-calibrated epsilons on GPU (#923)
- [bf16-GPU parity-gate tolerance](feedback_bf16_gpu_parity_gate_tolerance.md) — never calibrate a capture determinism cosine bar on the CPU smoke; bf16 CUDA noise is ~1e-6, use >= 0.9999 (#1005)
- [apply-parity probe N sizing follows Wilson-CI half-width](feedback_apply_parity_probe_n_sizing.md) — ±tolerance rate-gate must set N from (#667)
