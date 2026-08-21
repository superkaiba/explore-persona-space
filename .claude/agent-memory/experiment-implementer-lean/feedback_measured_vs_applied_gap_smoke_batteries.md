---
name: measured-vs-applied-gap-smoke-batteries
description: In smoke batteries, a green producer leg can be green-but-wrong — verify its artifact against the FIRST consumer's contract; and when a plan defines a quantity as a restriction ("n = the intersection"), grep that the measured set is APPLIED, not just recorded
metadata:
  type: feedback
---

Two coupled patterns from #1336 round-4 Unit C-iv (full all_v3 smoke):

1. **Green producer ≠ correct producer.** `c_pool --smoke` exited 0 in 2.6s
   and wrote a manifest — but the manifest over-covered (32 rows vs the 16
   the fixtures could serve), and only the NEXT leg (g0v3's row-coverage
   assert) exposed it. When a battery leg is suspiciously fast or its
   artifact is cheap to probe, diff the artifact's row/id space against the
   first consumer's contract (here: manifest conv ids vs bundle conv_ids)
   before declaring the leg green.

2. **Measured-vs-applied gap.** The plan defined n_pool AS the 5-way kept
   intersection; the implementation MEASURED the intersection (M1 concern)
   but never RESTRICTED the rows to it — plus the probe read a field
   (`conv_id`) the artifact never carries and skipped the kept-filter, so
   even the measurement was broken and would only crash pod-side. When a
   plan sentence defines a dataset/pool by a computed set, grep the producer
   for where that set FILTERS rows; "recorded in the manifest" is not
   "applied".

**Why:** all three sub-bugs were production-crashing but invisible to every
unit test and to the producer's own smoke; the cross-leg battery caught them
one leg downstream, pre-pod.

**How to apply:** in any multi-leg smoke, after each producer leg, spend one
cheap probe on the consumer join key (ids/keys/counts). When fixing, fix at
the SOURCE (make the manifest serveable) rather than relaxing the consumer
assert under smoke — the production contract must stay binding in both
modes. See [[verify-symbols-at-tip-before-insert]] for the sibling
pre-insert discipline.
