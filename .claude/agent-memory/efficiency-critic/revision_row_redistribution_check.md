---
name: revision-row-redistribution-check
description: When a plan revision's §9 row values rise but the pod/grand total barely moves, check whether the prior version counted that work OUTSIDE the row cells before concluding contingency absorbed a real increase
metadata:
  type: feedback
---

When reviewing a plan REVISION whose §9 per-row GPU-h values increased while the
pod total / grand total stayed ~flat, do NOT immediately conclude the contingency
row silently absorbed a real increase. First reconstruct the PRIOR version's total
from its own arithmetic-check line: the prior version may have counted the same
work (e.g. capture + margin TF forwards) in a side arithmetic line outside the row
cells, and the revision merely redistributed it INTO the rows.

**Why:** #2162 v1→v2 — P2 moved 7.3→9.8 and P3 27.5→29.3 (+4.3 apparent), but v1's
pod total 41.6 already included capture 6.2 (56,160×0.4s) + margin 3.7
(89,856×0.15s) via a separate "arithmetic check" line (its P3 row even contained a
truncated drafting sentence about this). v2's 41.7 was pure row-rounding of the
same quantities split P2/P3. Contingency was untouched (5 in both) and its
adequacy actually improved (gate-3 idle eliminated by overlap instead of absorbed).

**How to apply:** in round-2+ PLAN MODE reviews, diff the two versions'
arithmetic-check lines (grep `Arithmetic check` / `GPU-h` in both), verify the
component multipliers (calls × per-call) sum identically across versions, and only
flag contingency erosion when the reconstructed totals genuinely diverge.
