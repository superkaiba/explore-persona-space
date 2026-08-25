---
name: shape-complete-scope-sweeps
description: When a scope-of-computation defect class recurs across rounds, the composed sweep must enumerate QUANTITY SHAPES (intervals, single values, sequences, endpoint pairs, ratios, sameness/identity claims), not one syntactic shape — an interval-only sweep PASSed wrongly at #823 r7.
metadata:
  type: feedback
---

When composing a re-gate prompt whose primary ask is a sweep for a
scope-of-computation defect class (a quantity computed on a subset of
cells but written as covering all), enumerate the sweep by QUANTITY
SHAPE, explicitly and exhaustively: intervals/ranges; single values;
sequences/per-cell lists; endpoint pairs ("from X to Y", "rises A → B",
n/d pairs); ratios and fractions; and CLAIMS OF SAMENESS OR IDENTITY
("the same", "identical", "re-fits", "matched", "likewise") — the last
asserts a numeric equality without printing a number, so no numeric
grep or enumeration catches it. Have the twin report shape category per
swept quantity (quoted quantity -> shape -> implied cell set -> artifact
cell set -> verdict).

**Why:** #823 round 7 (2026-08-24) — the Claude critic's third
exhaustive sweep enumerated every quoted numeric INTERVAL, found none
remaining, and PASSed; three instances of the same defect class
survived because they were a realized mask-value list, an n/d endpoint
pair, and a qualitative sameness claim that was flatly false at one
cell (43,987 vs 45,458). The Codex twin caught all three and the
reconciler overturned the PASS. A category-scoped sweep certifies
nothing about out-of-category instances; three consecutive rounds of
the class each escaped through a shape the previous sweep did not name.

**How to apply:** any round whose brief says a sweep "found none
remaining" in a prior round: treat the prior sweep's SHAPE SCOPE as the
first thing to audit, widen the composed enumeration to all six shapes,
and instruct the twin that prior PASS sweeps are inputs to verify, not
conclusions to inherit. Related: [[prior-round-prompt-reuse]],
[[Delta-scoped rounds beyond r3 — compose, don't hard-fail]].
