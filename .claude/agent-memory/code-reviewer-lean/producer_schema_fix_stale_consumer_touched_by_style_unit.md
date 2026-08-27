---
name: producer-schema-fix-stale-consumer-touched-by-style-unit
description: A fix unit changes a producer's output schema; the consumer FILE is touched the same round by a DIFFERENT unit (style fixes only), so "consumer was updated" reads true while its read logic still parses the old shape. Probe by executing the new producer's real output through the HEAD consumer. (#2569 r2 shard1)
metadata:
  type: feedback
---

When a fix unit changes a producer's PERSISTED SCHEMA (nesting a level, renaming
record keys, splitting one arm label into per-side labels), grep every consumer
of that artifact and EXECUTE the new producer's real output through the HEAD
consumer — a `git diff --stat` showing the consumer file was modified this
round is a FALSE comfort signal when a different fix unit touched it for
unrelated reasons.

**Why:** #2569 r2 shard 1: unit F1 fixed `issue2569_dw_fleet.cmd_align` to nest
direction reads under `factors[...]["alignments"]` and to emit per-side intruder
arm names ("write"/"read" instead of always "write"). `issue2569_figures.py` WAS
modified the same round — but only by the marker-edgewidth style unit — so
`build_dw_alignment` still parsed the flat pre-fix shape (raised "carries no
scored directions" on every valid new file) and `build_dw_intruder` hardcoded
`["observed"]["write"]` (KeyError on the 5 V-side modules). Nine in-shard tests
were green because producer and consumer live in different test files.

**How to apply:** For any diff that changes what a phase WRITES: (1) grep the
artifact filename + record keys repo-wide for readers; (2) build a real record
via the NEW producer (reuse the producer's own test fixtures) and call each
consumer function on it — a crash or empty-selection raise is the finding;
(3) treat "the consumer file has a diff this round" as evidence of NOTHING
until the hunks are read — split-fix-unit rounds routinely co-touch files for
disjoint reasons. Sibling memories: [[loader-default-narrows-extended-payload]]
(consumer default-list narrowing), [[banked-parent-dual-schema-equivalence]].
