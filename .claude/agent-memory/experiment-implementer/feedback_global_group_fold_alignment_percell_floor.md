---
name: global-group-fold-alignment-percell-floor
description: Re-aligning grouped folds GLOBALLY across cells must balance PER-CELL fold loads — naive global greedy-LPT starves per-cell n_train floors that every per-cell assignment satisfied (#2378 r-interp-1)
metadata:
  type: feedback
---

When fixing cross-cell fold-assignment disagreement (shared grouping keys —
families/conversations — assigned to different folds per cell, leaking every
target's eval groups into sibling cells' pooled training rows) by deriving ONE
global key -> fold assignment: balance the assignment on PER-CELL fold loads
(greedy over keys sorted by -global count, each key to the fold minimizing
(max per-cell resulting load, global load, fold id)), never on global loads
alone.

**Why:** per-cell key counts are heterogeneous, so a globally-balanced
assignment can concentrate a cell's biggest groups in one fold — #2378: naive
global LPT on the realized 25-family fold map produced per-cell min n_train
5110/5115/5116 against the plan-G2b floor of >5120 (=d) on 3/5 story cells,
while the per-cell min-max greedy cleared it at worst 5193. Every REGISTERED
per-cell assignment satisfied the floor, so the violation appears only at
re-alignment time and only per cell.

**How to apply:** any global re-alignment of grouped folds across cells
(pooled fits, LOFO variants, cross-cell transfer) — (1) re-assert the
producer's per-cell per-fold n_train floor on the DERIVED assignment (the
builder's own check re-run), (2) measure the naive variant against the
realized map BEFORE committing to a method (one 20-line script), (3) keep the
derivation rng-free/deterministic from the sorted key list + counts when the
registered convention is (seeding belongs only to branches with a genuine
draw). Sibling fact for #2378 specifically: chat/plain_text/chat_user_real
row ids ARE content keys (mt_+source_hash[:12], pools drawn mutually
disjoint), so zero row-id overlap == content disjointness — check consistency
of SHARED keys' folds, not raw sharing (a paired-user topology legitimately
shares every key with identical folds). Worked impl:
scripts/issue2378_pool.py::_global_family_assignment + derive_global_family_folds.
