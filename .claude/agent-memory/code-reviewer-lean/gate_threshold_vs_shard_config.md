---
name: gate-threshold-vs-shard-config
description: A plan-registered in-run gate keyed on N accumulated rows PER SHARD silently never fires when the launcher's shard count makes per-shard rows < N — compute rows/shard for every documented config
metadata:
  type: feedback
---

An in-run validity gate that arms only after a fixed per-process accumulation
threshold (e.g. `len(rows) >= 2000`) is config-dependent: the threshold is
reachable at 8 shards (3125 rows/shard on a 25k split) and unreachable at the
same launcher's own 16-shard example (1563 rows/shard) — the plan-registered
gate then silently never runs, with no end-of-shard fallback and no log line
saying so.

**Why:** #1491 round 3a (M2): plan §7 Decision Gate 1 vs the launch script's
documented 32B 2-pod/16-shard config — the gate was dead exactly on the rung
the headline contrast depends on. Same family as smoke/production GATE
CALIBRATION (gotchas.md #1345) but the inverse direction: production config
makes the gate unreachable rather than unsatisfiable.

**How to apply:** for every accumulation-triggered gate in a sharded driver,
compute rows-per-shard for EACH shard count the launcher/plan documents and
check threshold reachability; demand either threshold = min(K, shard rows) or
a fire-once-at-shard-end fallback, plus a log line when the gate never armed.
Sibling: [[handrolled-pod-sentinel-envelope]].
