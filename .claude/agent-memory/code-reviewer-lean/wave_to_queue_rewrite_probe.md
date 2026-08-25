---
name: wave-to-queue-rewrite-probe
description: Certify a shell wave→work-conserving-queue rewrite by verbatim-adapting the flock queue into a live probe (one slow + one failing fake cell); assert late cells start before the slow one ends, rc/wall read-back, exactly-once dispatch (#2479 R2 g1)
metadata:
  type: feedback
---

When a diff replaces strict wave barriers (`wait` on a pids array per wave)
with a work-conserving per-worker queue (flock'd index file + per-cell rc
files), do NOT certify it by reading alone — verbatim-adapt the queue
functions into a /tmp probe with N cells / 2 workers where one cell is SLOW
and one returns a nonzero rc, then assert from the trace: (a) cells beyond
the first wave START while the slow cell is still running (work
conservation); (b) per-cell rc/wall files read back exactly and the nonzero
rc propagates to the parent rc; (c) each index is dispensed exactly once
(read+increment inside one exclusive flock); (d) missing-rc → fail-loud
sentinel value + rc≠0.

**Why:** #2479 R2 g1 (commit d927b4b970): both the P1 gen loop and the
capture launcher's `run_wave` took this rewrite after a codex
strict-wave-scheduling blocker. The probe took ~1 min and proved the whole
contract; reading alone would have left the flock read-modify-write and the
rc read-back unverified.

**How to apply:** also check (1) per-cell CUDA_VISIBLE_DEVICES stays
command-scoped inside the cell runner and dev values are ENTRIES of the
parent CVD allocation ([[fanout-cvd-ordinal-not-entry]]); (2) worker count
= min(width, pending); (3) fatal-rc accounting drains the whole queue
before exiting (matching old wave semantics) and appends via O_APPEND;
(4) non-numeric rc-map sentinel values ("norc") are consumed only where
pre-existing non-numeric values already flow (telemetry), never in `-ne`
arithmetic; (5) bash dynamic scoping of the enclosing function's locals is
correct but worth one glance. Related: a claimed-fixed-here minor may live
in a SIBLING round commit — `git log --oneline --all -- <file>` before
flagging it missing.
