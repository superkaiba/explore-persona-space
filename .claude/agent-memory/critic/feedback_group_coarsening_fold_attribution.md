---
name: Group-coarsening fold-change attribution
description: Re-folding at a coarser grouping unit (novel→author) — the R² drop bundles leakage removal with n_train/diversity loss co-localized on the dominant-group fold; subtraction claims need a size-matched random-regrouping null
type: feedback
---

Rule: when an amendment re-folds an existing fit at a COARSER grouping unit (e.g. novel→author, #931 v9) and registers a subtraction claim ("component ≈ R_old − R_new"), the drop is confounded by (i) reduced n_train in the fold holding the dominant group (Austen fold trained on 64% of rows vs ~80%) and (ii) reduced train-group diversity — and these co-localize on the SAME fold where the leakage removal bites, so per-fold diagnostics CANNOT separate them. Existence reads (obs vs same-fold null band) are coarsening-robust (band shares the fold structure); only the attribution/subtraction claim is fatal.

**Why:** the n_train artifact was the same order as the predicted effect (power curve showed strong n-sensitivity at exactly that scale), so a zero-style world could still fire the "component measured" row — a false positive on the amendment headline.

**How to apply:** demand a size-matched random-regrouping null (M random assignments of fine groups into pseudo-groups matching the coarse-group size multiset, same K/seed, obs-only at headline layer — cheap on the cached-eigh path) OR re-register the subtraction row as an explicit UPPER BOUND with attribution-neutral vocabulary ("author-level", not "author-style" — author identity carries era/genre/topic too). Watch for an assumptions bullet folding the n_train reduction into "the construct" (fine for existence, not for subtraction).
