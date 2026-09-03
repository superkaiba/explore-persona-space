---
name: one-sided-unique-join-dup-orphan
description: A claimed "strict 1:1 join" that dedupes only ONE side + asserts count equality still admits a duplicated left row shadowing an orphaned right record — check key uniqueness on BOTH sides (#2658 r1 gC)
metadata:
  type: feedback
---

Rule: when a reviewed loader claims a strict 1:1 join between two artifacts
(manifest rows vs raw records), verify it dedupes keys on BOTH sides. The
common shape checks (a) right-side key uniqueness, (b) len(left) == len(right),
(c) every left row finds a right match — and that trio still PASSES when the
LEFT side carries a duplicated key: both dup left rows match the same right
record, count equality holds because one right record is orphaned (never
matched, never labeled), and any per-pair content sha passes since the dups
are identical. Result: one completion double-counted, one silently unlabeled,
with the denominator unchanged.

**Why:** #2658 round 1 group C, `scripts/issue2658_objective_labels.py::
load_cell_inputs` — recs uniqueness + count parity + per-mrow match + row-hash
verify, no manifest-key dedupe. Flagged Minor (upstream writer is structurally
unique), fix is one `seen` set.

**How to apply:** on any join labeled 1:1/bijective, enumerate the four checks
(left-unique, right-unique, count parity, total match) and demand the missing
one; count parity + one-sided uniqueness + total-match is NOT a bijection
proof. Related: [[silent-get-default-beside-fixed-keyerror]] (audit every
field read at the same boundary), [[consumer-flag-producer-never-writes]].
