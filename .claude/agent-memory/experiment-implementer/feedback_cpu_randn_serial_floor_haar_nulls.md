---
name: CPU randn is serial — Haar-null device-routing floor
description: torch CPU randn is thread-invariant (~50-55 ns/value); routing QR to GPU leaves a ~0.5 s/matrix CPU floor per 3584^2 fp64 draw when the seeded RNG stream must stay CPU-pinned (#1417 r2)
type: feedback
---

Rule: when device-routing a per-draw Haar-orthogonal null (`randn` on a CPU
generator + `torch.linalg.qr`), moving the QR to GPU removes only the
multithreaded FLOP leg. The CPU `randn` fill is SERIAL (MT19937 + Box-Muller,
measured thread-invariant: 0.665 s @ 1 thread vs 0.611 s @ 8 threads for a
3584x3584 fp64 matrix, ~50-55 ns/value) and becomes the new wall whenever the
seeded stream must stay on the CPU generator for reproducibility.

**Why:** #1417 round-2 — the fix brief projected ~60-120 s/pair post-fix
assuming randn was negligible; production-shape measurement showed randn ~0.6
s/matrix x 800 matrices/pair ≈ ~400 s/pair irreducible floor (vs 961 s
pre-fix). QR itself: ~1.4 s CPU → ~30-60 ms A100 fp64. Sign-fix
(`Q * sign(diag(R))`) canonicalizes Q across LAPACK/cuSOLVER conventions, so
device-routing QR is semantics-safe up to fp64 rounding while the randn stream
is untouched.

**How to apply:** before projecting a device-routing speedup for any
draw-battery, measure BOTH legs at production shape (randn AND the
factorization) — the RNG fill does not thread-scale and cannot move to GPU
without changing the stream (a plan-level reproducibility decision). Halving
draw counts is the only same-stream knob for the randn floor.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [CPU randn serial floor in Haar nulls](feedback_cpu_randn_serial_floor_haar_nulls.md) — torch CPU randn is thread-invariant (~50 ns/val); GPU-routing QR leaves ~0.5 s/matrix stream-pinned floor — measure both legs before projecting (#1417 r2)
