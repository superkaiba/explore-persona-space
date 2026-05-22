---
name: cipher-3gram-pigeonhole
description: Token-novelty gates on cipher held-out sets fail when training footprint saturates the n-gram space; 3-grams over 27-symbol alphabet (a-z + space) have only ~20k cells and saturate fast.
metadata:
  type: feedback
---

When a dataset-design plan registers "N held-out token-novel" against an
n-character ciphertext-substring overlap rule, the gate is only
satisfiable if `(training_size × avg_chars_per_ct) << alphabet^n`. For
issue #192 (08:16:48): 800 training pairs × ~20 chars × on `a-z + space`
(27 symbols, so `27^3 ≈ 20k` 3-gram cells) saturates the cipher 3-gram
space; stage-2 swap loop ran 20,000 attempts finding zero novel 3-grams.

**Why:** held-out plaintexts were drawn from the *same* English-noun +
first-name pool as training plaintexts. Because the affine cipher is
deterministic per-letter and the plaintext pool is narrow, ciphertext
3-gram coverage is nearly bijective with plaintext 3-gram coverage —
which is already saturated by 800 training pairs.

**How to apply:** during preflight or planner review of any dataset-
construction plan that imposes a "≥N token-novel" gate via n-gram
substring overlap on a cipher: compute `train_size × avg_ct_len /
alphabet^n` BEFORE launching. If > 0.3, the gate is at risk of being
mathematically unsatisfiable. Fixes (in increasing impact): widen n
(4-gram = 531k cells, plenty of headroom), reduce N_CIPHER_TRAIN, or
sample held-out plaintexts from a disjoint word pool.

Related: [[feedback_no_eval_shortcuts]] — don't silently relax the
gate; bounce back to implementer with diagnosis + 3 fix options.
