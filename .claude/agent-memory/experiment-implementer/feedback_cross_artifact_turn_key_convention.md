---
name: cross-artifact turn-key convention mismatch at joins
description: Joining per-turn artifacts from two pipelines on stringified integer keys silently mis-aligns when index conventions differ; translate keys explicitly at the join and check the reference key-space shape first
type: feedback
---

Joining per-turn artifacts from two pipelines on stringified integer keys
silently mis-aligns when the index conventions differ (#825 r11 G-C gate:
refit keyed by 1-based exchange ordinal t; the round-10 reference curve keyed
by 0-based assistant turns-list index = 2t−1). A `.get(str(t), {})`-and-skip
join converts the key mismatch into a plausible-looking parity FAIL (13/15
turns "outside CI") instead of a loud error.

**How to apply:** whenever consuming a prior round's / another script's
per-turn (or per-layer, per-cell) dict, (1) verify the reference's key-space
SHAPE against your own (all-odd keys vs contiguous 1..K was a detectable
signature the join never looked at — assert or log it); (2) translate keys
explicitly at the join with the convention documented in a comment + recorded
in the output payload; (3) never use silent `.get(...)`-and-skip across
artifact boundaries — a missing partner key at a join is evidence of a
convention mismatch, not data to drop. (Incident: #825 epm:failure-lesson v9,
fit_gc false FAIL blocked the round-11 headline.)
