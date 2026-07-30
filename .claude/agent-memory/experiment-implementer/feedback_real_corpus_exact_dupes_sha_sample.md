---
name: real-corpus exact dupes break sha-keyed samples
description: Sha-keyed sampling from a near-dupe-screened real corpus must dedup DURING the draw; measure a frozen pinned split before asserting global uniqueness over it
type: feedback
---

A real-corpus pool screened only NEAR-DUPE-vs-eval-targets still contains
exact-duplicate texts, so any content-keyed (sha) sample must dedup DURING the
draw — taken-set seeded with the pinned rows, top up from the continuing
permutation order (seed-stable). **Why:** #1768 p0 crashed on the real #779
n1M corpus (`AssertionError: duplicate prompt shas in sample`) after every
synthetic smoke passed; and the frozen #779 pinned val/test itself carries 82
internal duplicate-sha rows (1,400 → 1,318 unique; val∩test overlap 13), so a
GLOBAL sha-uniqueness postcondition over a frozen pinned split is
unsatisfiable by construction and re-crashes the relaunch. **How to apply:**
when sampling by content hash from LMSYS/WildChat-class corpora, (1) dedup
in-draw with pinned-row priority, (2) MEASURE the pinned split's own
duplicate content before writing any uniqueness assert that quantifies over
it, (3) scope postconditions to what the fix guarantees (train-unique +
train∩pinned = ∅ + exact counts), never global. (#1768 crash-fix round,
fix 07823360; root cause confirmed.)
