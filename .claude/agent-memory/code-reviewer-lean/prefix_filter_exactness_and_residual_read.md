---
name: prefix-filter-exactness-and-residual-read
description: Certify an "exact" prefix-filtered similarity sweep by full-pool sparse-matmul brute force (integer-arithmetic threshold), and adjudicate a disclosed lexical residual by classifying pairs (duplicate vs template) with a difflib-opcode diff-span read
metadata:
  type: feedback
---

Two probes that settled #2658 group-D round 3 (2026-09-02):

1. **Full-pool brute force beats a subsample, and it is cheap.** A claimed-exact
   rarest-prefix-filter candidate generator (lemma: J(A,B) >= t ⇒ A hits one of
   B's floor((1-t)|B|)+1 rarest shingles) is certified by bypassing the filter
   entirely: binary sparse item×shingle matrix (drop df==1 columns — they can't
   contribute to any intersection; keep ORIGINAL set sizes for the union),
   blocked `X[block] @ X.T` matmul, threshold in INTEGER arithmetic
   (J >= 4/5 ⇔ 9·inter >= 4(|A|+|B|)) to dodge float-0.8 boundary semantics,
   then compare TRANSITIVE-CLOSURE partitions (module fresh-UF vs brute-pair UF)
   — pair-level diffs inside one component are invisible and harmless. 13,204
   stems / 87M pairs ran in ~25 s. Float subtleties worth stating but empirically
   vacuous: `int((1-0.8)*n)+1` is one short of the true floor+1 at n≡0 (mod 5)
   (rescued because both directions failing forces J <= 2/3), and float(0.8) >
   4/5 makes the size-ratio precondition drop a J-exactly-4/5 strict-subset pair
   that the verification would accept.

2. **A "template similarity, not duplication" exclusion argument is testable in
   one pass.** For every flagged cross-boundary pair, difflib SequenceMatcher on
   normalized stems; sort by ratio; print ONLY the non-equal opcode spans.
   Parameter/polarity/subject substitutions ('8'->'5', 'not '->'', 'supply'->
   'demand', formula swaps) = distinct problems sharing a template; punctuation/
   preposition/modal-only spans ('in'->'to' + comma, 'could'->'will') = genuine
   duplicates. In #2658: 2 genuine twins out of 782 pairs/205 flagged test items
   — the exclusion validated, residual ~0.03% of sealed test. Also read the top
   sub-threshold charJ near-misses: shared-stimulus different-question pairs
   (same passage/fact-pattern, different final question) cluster just under the
   near-dup threshold and are their own disclosure class.

**Why:** a false zero from a broken filter looks like closure (worse than a
disclosed residual), and a residual-size argument by narrative was exactly what
the prior round FAILed on — both are settled by measurement, not adjudication.

**How to apply:** any "exact at any pool size" candidate-generation claim
(prefix filter, banding, LSH said to be lossless), and any disclosed
similarity-tier residual whose treatment is exclusion+disclosure. Pairs with
[[superfamily-split-freeze-review-recipe]] and
[[keyed-id-edge-exemption-split-straddle]].
