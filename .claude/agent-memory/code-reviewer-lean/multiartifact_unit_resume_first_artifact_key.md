---
name: multiartifact-unit-resume-first-artifact-key
description: A unit writing N artifacts in a loop whose resume-skip keys on the FIRST-written artifact treats every mid-loop crash as complete; key on the LAST artifact or a post-loop sentinel (#2378 R1 g3)
metadata:
  type: feedback
---

When a per-unit driver writes MULTIPLE output artifacts in a terminal loop
(9 rung JSONs + sidecars per pair) and its resume predicate is "skip if
<artifact> exists with matching regime", check WHICH artifact carries the
key: keyed on the first-written one, any crash inside the write loop (each
iteration here ran a 200-draw bootstrap + two n×n kNN batteries — minutes
at production n) leaves a partial unit that every re-run skips as complete.
Downstream consumers fail loud on the missing later artifacts, but the unit
can never self-heal without manual deletion.

**Why:** #2378 R1 g3 (`issue2378_ladder.py::run_pair_unit`): resume keyed on
`__rung1.json`, written first of 27 artifacts; the sibling fits driver was
correct by construction (single payload JSON written AFTER its sidecars).
Same family as [[count-gate-starved-by-resume-skip]] (resume-skip starves a
gate) and [[start-manifest-stale-artifact-done]] (presence-done + start
manifest) — this is the WRITE-ORDER variant: presence of the first artifact
as the done key.

**How to apply:** for every resume/skip predicate in a reviewed driver, list
the unit's full artifact set and the WRITE ORDER; require the key artifact
to be the LAST write (or an explicit done-sentinel after the loop). A
correct pattern to cite: sidecars first, keyed payload JSON last. Also check
the same unit's tier/branch predicates use ONE statistic convention — #2378
g3's Unmappable branch read the fold-mean margin while the tier CI was
pooled-convention, an inconsistent lattice state that crashed the H3
consumer (which loaded rowstats BEFORE its tier drop-check).
