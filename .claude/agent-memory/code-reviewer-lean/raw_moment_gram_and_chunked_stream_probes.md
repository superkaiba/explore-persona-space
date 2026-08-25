---
name: raw-moment-gram-and-chunked-stream-probes
description: certifying a raw-moment Gram substitution (small-n parity gate IS representative — cancellation ratio is n-independent) and a chunked stream-builder equivalence via a monkeypatched-iterator synthetic probe (#1901 R2 g1)
metadata:
  type: feedback
---

Two verification recipes from #1901 R2 g1 (both fixes PASSed on them):

1. **Raw-moment Gram substitution** (centered accumulation `Σ(x−μ)(x−μ)ᵀ`
   replaced by `S_xx − n·μμᵀ` for a shared multi-rung pass): the catastrophic-
   cancellation ratio per dim is `S_xx[i,i]/centered ≈ 1 + (μ_i/σ_i)²` —
   **n-independent** — so a small-n parity gate (rung 50, |ΔR²| ≤ 1e-6 vs the
   two-pass reference) exercises the same cancellation regime as the top rung;
   what it does NOT bound is accumulated summation error, which for blockwise
   fp64 (~tens of block adds) leaves ≥6 orders of headroom vs a 1e-6 tolerance.
   Diff the derived standardizer line-by-line against the parent's exact
   conventions (ddof, epsilon placement — `+eps` AFTER sqrt vs inside — eigh
   clamp, fac dict keys, val-λ loop init/strict-> /isfinite): any one mismatch
   silently breaks parity at every rung the gate doesn't run.

2. **Chunked stream-builder equivalence** (a monolithic corpus build split
   into resumable chunks with a dummy-prefix to preserve global enumeration
   indices): do not settle it by reading alone — monkeypatch the module's
   corpus iterator with a small synthetic corpus and compare monolithic vs
   concatenated-chunk outputs field-by-field (row ids, spans, meta). Preconditions
   to check in the source first: the enumeration index is assigned BEFORE the
   per-article filter, per-article RNG is seeded by that global index (not a
   shared stream), no global selection/cap in keep-all mode, and the filter
   handles the dummy payload ("" tokenizes to an empty (0,2) offsets array)
   without consuming state. Also confirm the resume skip-count granularity ==
   the chunk record granularity (same iterator unit).

**Why:** #1901 r2's two heaviest fixes (shared prefix-Gram, b0 chunked resume)
both rested on claims a diff read cannot certify; the two probes above settled
them in ~2 min each (parity test on synthetic 240×16; 7-article chunk probe,
224/224 pairs identical).

**How to apply:** any diff that (a) swaps centered two-pass statistics for
raw-moment accumulation with a small-n parity gate, or (b) converts a
monolithic stream build into chunk-checkpointed resume claiming byte-identical
output. Related: [[fails-pre-fix-probe-parent-commit]] (parent-blob
extraction certifies the fail-pre-fix half of a fix-round),
[[upload-batch-resume-never-reenqueues]] (the same commit's producer-side
wedge, fixed by entry-time local-vs-Hub reconcile).
