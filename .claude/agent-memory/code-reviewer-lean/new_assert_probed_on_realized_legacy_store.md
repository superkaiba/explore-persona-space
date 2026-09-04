---
name: new-assert-probed-on-realized-legacy-store
description: When a fix adds an assert that newly runs on an ALREADY-REALIZED path (e.g. a per-row sha equality outside a split branch), probe the realized artifacts directly instead of accepting the structural argument
metadata:
  type: feedback
---

When a round adds a check that runs on EVERY path (placed outside a branch)
while only ONE branch motivated it, the untouched legacy path newly runs the
check too. The implementer's structural argument ("holds by construction")
is graded by machine-verifying the predicate over the realized legacy
artifacts.

**Why:** #2658 r16 moved capture prompt verification split-aware and added a
gen-manifest sha == re-resolved sha assert on every split. Pilot capture was
already realized. A one-shot probe over all 131 realized pilot gen-manifest
files (6,290 rows) confirmed manifest sha == pin sha with 0 dup keys, so the
new assert cannot brick a pilot resume. Cheap (one python loop over jsonl +
the pin table), and it simultaneously grades the round's dup-key exposure on
real data.

**How to apply:** locate the realized artifact set the new universal check
will read (manifests, row_index, sidecars), recompute the asserted predicate
over ALL rows yourself, and report counts (rows, mismatches, dups). Pair
with [[one-sided-unique-join-dup-orphan]] when the check rides a keyed join,
and with [[fails-pre-fix-probe-parent-commit]] for the fix-engaged side.
