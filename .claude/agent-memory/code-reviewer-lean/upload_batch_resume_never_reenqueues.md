---
name: upload-batch-resume-never-reenqueues
description: upload-as-you-go per-K-batch loops whose resume marks units done from LOCAL sidecars never re-enqueue done-but-unuploaded units — any crash off a K-boundary makes every re-run fail terminally at the exact-set verify with no repair path
metadata:
  type: feedback
---

In a capture/produce loop that uploads per K-unit batch (`batch_pending` fed
ONLY by newly-produced units) with a resume scan keyed on LOCAL sidecar
presence, a crash between a unit write and its batch upload (window: up to
K−1 units — i.e. almost any mid-run crash) strands local-only units. On
resume they are marked done, never re-enqueued, and the phase-end exact-set
Hub verify fails deterministically on EVERY re-run — a fail-loud wedge on
the load-bearing resume of a multi-hour GPU phase, repairable only by hand
(#1901 g2 R1, `phase_p1_capture`).

**Why:** the exact-set verify is correct; the gap is that resume conflates
"produced" with "persisted". The fix is an entry-time (or pre-verify)
diff of local unit names vs the Hub listing, re-enqueuing the missing ones.

**How to apply:** whenever a diff shows (a) per-unit resume keyed on local
files, (b) uploads batched per K, and (c) a terminal exact-set verify —
trace the crash-between-write-and-upload window and demand the re-enqueue
diff. Related: [[spend-consumer-accepts-partial-shard-set]] (consumer-side
partial acceptance; this is the producer-side wedge),
[[force-flag-not-reaching-resume-state]].
