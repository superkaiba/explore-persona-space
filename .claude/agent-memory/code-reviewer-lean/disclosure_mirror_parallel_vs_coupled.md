---
name: disclosure-mirror-parallel-vs-coupled
description: A disclosure that mirrors live predicates (edge filters, gate criteria) is either STRUCTURALLY coupled (shared helper / coupling test) or merely PARALLEL — a parallel mirror plus a guard that only fires on realized-nonzero drift lies silently in the zero case
metadata:
  type: feedback
---

When a diff adds a DISCLOSURE derived by re-implementing another function's
predicates (e.g. #2658 `_edge_domains` mirroring `build_superfamilies`' edge
filters to label overlap `measured` vs `structurally-inert`), classify the
coupling before crediting it:

1. **Structurally coupled** = both sites call ONE shared predicate helper, or
   a test drives the REAL graph across the mirrored boundary asserting the
   mirror's claim (e.g. keyed-vs-free-text exact-duplicate texts do NOT
   merge). Predicate changes then break loudly.
2. **Parallel** = a hand-copied re-implementation. A runtime consistency
   raise (`barred nonempty despite disjoint domains`) guards only ONE
   direction — it fires only when the drifted predicates produce a REALIZED
   nonzero overlap in this dataset. The zero-overlap case (the common one)
   leaves the disclosure silently false after any predicate change.

Check whether the round's own tests would fail under the known queued
predicate change: if the fixture texts share no duplicates, a real-body
"inert" test keeps passing post-change while the label is wrong — that is
the tell of a parallel mirror.

**Why:** #2658 round-2 shipped a predicate-by-predicate-correct mirror whose
only drift guard was the realized-nonzero raise, with a group-D edge-criteria
change already queued — the verdict had to state "parallel, not coupled" so
the D round knew it must update the mirror in lockstep.

**How to apply:** verify the mirror predicate-by-predicate against the live
function FIRST (it can be right today), then answer the coupling question
separately; a correct-today parallel mirror with a queued predicate change
is a CONCERN naming the lockstep file+test, not a PASS-silent. Pairs with
[[keyed-id-edge-exemption-split-straddle]] (the queued change itself) and
[[superfamily-split-freeze-review-recipe]] (probe 1's inert-tell).

**Validated CLOSED shape (#2658 a76241f96f4):** both sites dispatch on ONE
set of shared predicate helpers (edge_problem_id_key / edge_text_identity_key
/ edge_lexical_member), domains become per-EDGE rather than per-population,
and a duplicate-content test drives the REAL graph asserting
merged == bool(spanning-domains). Certify by reading BOTH call paths (a
helper called by one side and duplicated by the other is NOT coupling).
Residual to note as nit: a future edge added INLINE without a helper still
drifts the disclosure — nothing forces the helper pattern.
