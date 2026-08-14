---
name: count-gate-starved-by-resume-skip
description: A fresh-round count-keyed gate (grep-count over the round's OWN log dir) breaks when round units dedupe/resume-skip into a prior round's artifacts — the skipped unit emits no fresh log, so n_found < n_expect FATALs spuriously and resume re-FATALs forever (#2225 R1 g5)
metadata:
  type: feedback
---

When a diff adds a per-round completeness gate of the shape `n_found=$(grep -rl <token> <this-round-log-dir> | wc -l); [ n_found -ne n_expect ] && exit`, check every path by which an expected unit can complete WITHOUT writing into this round's log dir — especially resume-skip/dedupe against a prior round (the very dedupe the diff may celebrate elsewhere).

**Why:** #2225 round 1 (commit ecccdf1): the §7 octave-shift re-pilot's `[steer-hook]` recheck counted logs in a fresh `p0_train_repilot/` dir, but every octave grid (×0.5 and ×2 alike) overlaps the original pilot grid at exactly one coefficient, so that cell resume-skips (`run_fan_out` prints `skipped-resume` to stdout, writes NO per-cell log) → 3/4 logs → spurious FATAL exit 7 — and on resume ALL cells skip → 0/4 → a permanent crash loop on the designed remedy path. Sibling of gotchas.md "count-keyed liveness gates".

**How to apply:** whenever a count-gate's numerator is files/lines produced THIS round and its denominator is planned units, trace the unit runner for skip/dedupe/cache-hit branches and ask where their evidence lands (stdout vs the counted dir, prior round's dir). Fix shapes: count launched-only units, have skips write a sentinel line into the counted log, or accept evidence from the prior round's dir per-unit.
