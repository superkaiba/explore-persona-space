---
name: Batched PRESS-LOCO ridge twin is exactly reproducible; "batched" plan labels may cover only one loop
description: Nested-CV LOCO ridge (per-fold standardization + PRESS lambda-select + dual solve) batches exactly (rtol<=1e-7) via fold-chunked bmm Grams + ONE batched eigh reused across lambdas; and a round that "batched the fits" may have batched only ONE of the serial loops (floors vs LOCO chain) — enumerate EVERY loop in the phase before attesting throughput. #811 r14.
type: feedback
---

Rule: when batching a nested-CV LOCO ridge (per-fold standardization, PRESS
inner-LOO λ-selection, dual/Woodbury solve — the #658 `_ridge_predict_loco`
family), the EXACT estimator batches cleanly: gather `(b, m, d)` fold designs
(per-fold μ/σ with `.std(correction=0)` on the gathered copy), `bmm` Grams, ONE
batched `eigh` per chunk reused across the λ grid (the PRESS identity), batched
`argmin` (first-min tie-break matches serial), batched `linalg.solve`, held-out
pred as `(Xn x_held)ᵀα`. Equivalence lands at rtol=1e-7/atol=1e-10 on real data
— no need to weaken semantics (per-fold standardization does NOT block batching;
it only blocks sharing one decomposition across folds). Chunk cap from a
MEASURED ru_maxrss factor (closed-form fp64 ≈ 2.9× the `(chunk, m, d)` unit —
far below the autograd path's 26×).

**Why:** #811's fix round 1 batched the floor bootstrap and attested the fits
phase "batched", but the ridge-LOCO chain (5 serial calls × 480 folds/unit)
stayed a Python loop and realized ~7.7h/unit under VM contention (30-43 days
projected; killed at unit 3/108). Light-load microbenchmarks hid it: serial
per-fold was only 0.065s uncontended — the blowup was contention-amplified
tiny-op overhead, exactly the vectorize-rule's overhead class.

**How to apply:** before attesting a fit phase batched, enumerate EVERY serial
loop in it (floors, LOCO chain, grid fits, gate) and name each one's batching
or FLOP-floor justification; also check for duplicate deterministic fits
(#811's fit_cell computed LOCO(Cplus→Vplus) twice). Tombstone at the CALL-SITE
dispatch (e.g. `fitM._ridge_loco_pred(path=...)`) when the shared serial
function has other legitimate callers. Impl: `_ridge_predict_loco_batched` in
`scripts/issue658_fit_predictors.py` (#811 r14, commit d38a262db5).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Batched PRESS-LOCO twin exact + enumerate every loop before attesting "batched"](feedback_batched_press_loco_twin_exact.md) — nested-CV LOCO ridge batches EXACTLY (fold-chunk bmm+eigh, rtol<=1e-7); a round that batched floors left the LOCO chain serial (30-43d realized) (#811 r14)
