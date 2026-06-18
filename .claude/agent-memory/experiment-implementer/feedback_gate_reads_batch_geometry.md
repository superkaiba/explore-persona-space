---
name: Reproduction gates vs stored bf16 records need matched batch geometry
description: Left-pad batch composition (companions -> max_len -> RoPE offsets) shifts bf16 logp by ~0.16 nat; smoke --limit-* subsets break gate comparability while parent-geometry re-reads are bit-exact. Also align zero-shift/partial-batch re-reads to full main-read batches.
type: feedback
---

A reproduction gate comparing fresh forward-pass numbers against STORED bf16
records (e.g. a <=0.1-nat logp gate vs a parent run's per-row records) is only
valid at the parent's exact batch geometry. Changing a row's batch COMPANIONS
(smoke `--limit-*` subsetting, different batch size, partial trailing batch)
changes `max_len` -> left-pad length -> RoPE absolute offsets -> bf16 jitter
up to ~0.16 nat on a 7B/28-layer model — 0 nat at matched geometry (bit-exact).

**Why:** #597 r8 (2026-06-11): base-side four-float gate FAILed at 0.1634 nat
on a smoke 3x3 subset; re-reading the same row inside its ORIGINAL
full-enumeration sub-batch of 8 reproduced the stored logp to 0.0000. Same
class as a latent production landmine: a zero-shift sanity re-reading
rows[:50] in batches of 8 leaves a 2-row trailing batch whose rows had
different companions in the main read — fails a 1e-3 tol in production while
passing in smoke (smoke-passes/production-crashes inversion).

**How to apply:** (1) subset/smoke gate reads must reconstruct each gated
row's original full-enumeration sub-batch and forward THAT (cheap: only the
needed sub-batches); full runs at the parent batch size reuse main reads.
(2) Never loosen the gate tolerance to absorb geometry jitter — matched
geometry keeps the gate bit-sharp. (3) Any same-model re-read compared at
tight tolerance (zero-shift passes, invariant re-reads) must use IDENTICAL
batch slicing to the read it is compared against (floor partial counts to
full batches). Worked impl: `parent_geometry_fourfloat` +
`aligned_zero_shift_rows` in
`src/explore_persona_space/experiments/leakage_dynamics_597/shift_svd.py`.
