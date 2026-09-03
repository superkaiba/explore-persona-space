---
name: normalizer-form-vs-acceptance-instances
description: A prescribed normalization form ("fold non-alnum to single space") can fail its own acceptance instances ("the 8 punctuation twins merge") — probe the named instances FIRST; the #2658 "punctuation twins" were whitespace-DELETION twins needing DROP, not FOLD (#2658 D r3)
metadata:
  type: feedback
---

Before implementing a brief-prescribed text-normalization/criterion form,
run the brief's own NAMED acceptance instances through the prescribed form.
On #2658 group-D r3 the brief prescribed "alnum-plus-single-space" folding
for the identity edge with acceptance "the 8 punctuation-only twins merge" —
but the 8 pairs (measured) differed by REMOVED whitespace
('chromosomesare' vs 'chromosomes are'), equal only under
casefold+DROP-all-non-alnum; the fold form merges NONE of them, and 4 of 8
sat below the charJ 0.8 near-dup tier, so no other edge would catch them.

**Why:** the upstream verdict measured "drop ALL non-alphanumerics" but the
brief's wording compressed it to "fold punctuation"; wording drift between a
measurement and its prescription is invisible unless the instances are
re-probed. Adopting DROP was then validated on duplicate-detection grounds:
all 20 drop-vs-fold delta groups in the 13,204-stem pool were
space-placement typo variants — zero content-fusion (e.g. no "2+2"→"22")
false merges.

**How to apply:** (1) reproduce the named instances under the prescribed
form BEFORE building; on divergence, implement what the acceptance
criterion requires and report the deviation with the measurement. (2) When
widening a normalizer, enumerate the full delta-set it newly merges
(pool-wide, not just cross-boundary) and eyeball for false merges — that is
duplicate-detection-grounds validation, distinct from forbidden
tune-after-seeing-cell-counts. Related:
[[refreeze-moves-pilot-membership-downstream-frozen-artifacts]].
