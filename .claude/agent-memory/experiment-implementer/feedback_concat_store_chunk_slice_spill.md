---
name: concat-store-chunk-slice-spill
description: Chunked loops over the FIT PREFIX of a concat(fit,holdout) memmap must clamp the last slice end at n_fit — numpy clamps at ARRAY end, silently spilling holdout rows into fit-side counts (#2476 k200 gc_ii rc=31)
metadata:
  type: feedback
---

Rule: any chunked iteration over a LOGICAL BLOCK PREFIX of a larger array
(`for s in range(0, n_fit, B): x[s : s + B]`) must clamp the slice end at the
block boundary (`x[s : min(s + B, n_fit)]`) whenever the array extends past it
(concat(fit, holdout) stores, fit+val memmaps, padded shards). numpy slicing
clamps at the ARRAY end, not the logical block end, so the final chunk silently
absorbs `(ceil(n_fit/B)*B - n_fit)` rows of the NEXT block.

**Why:** #2476 k200 census (rc=31): the gc_ii self-consistency recount looped
`yc[s : s + 8192]` over `range(0, 120000, 8192)` on a (140000, 2659)
concat(fit, holdout) memmap — the last chunk spilled 2,880 holdout rows,
inflating near-saturated features by up to +2,832 and halting production at
the record-first gate. Mis-attributed TWICE before the code read: r8 blamed
fp16 recount storage; the crash-fix brief blamed inference top-k truncation.
The clamped recount matched the banked counts EXACTLY (max delta 0/2,659
features — even the tol=3 GEMM-flip allowance was unneeded), and
`buggy == clamped + spill-block firings` everywhere, at production AND smoke.

**How to apply:** (1) When writing any block-prefix chunk loop over a wider
store, clamp the end index — and pin it with a fixture where the array is
LONGER than the block and the next block's rows are all-firing (the spill then
shifts counts detectably). (2) Diagnosis heuristics for count-mismatch gates
on such stores: all-positive deltas bounded by `B - n_fit % B` (and by the
next block's row count) = spill signature; and check the sum cap FIRST — a
claimed top-k-truncated leg can never sum past k*n_rows, so
`sum(counts) > k*n` refutes truncation arithmetically before any code read.
(3) Cross-semantics self-consistency arms must compare like with like — but
verify WHICH leg deviates by recomputing both legs from the banked artifacts
(read-only, minutes) before accepting any mechanism theory; exact per-feature
reproduction of the recorded gate numbers is the bar.
