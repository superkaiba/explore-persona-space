---
name: amend-phase-striding-filters
description: A regen/amend phase re-deriving order[w::W] must reproduce EVERY pre-striding filter the generation phase applied (e.g. vLLM cell exclusion) — else per-worker merge-set identity asserts fire after regen spend
metadata:
  type: feedback
---

When a repair/regen/amend phase re-derives per-worker shards by striding (`order[w::W]`) over a canonical
enumeration, diff its enumeration pipeline against the ORIGINAL generation phase's: any filter the generation
applied BEFORE striding (work-conservation exclusions, claimed-cell drops, routing freezes) shifts every
worker's slice if the amend phase omits it. A merge-time set-identity assert (dropped != regenerated) catches
it LOUD — but only after the regeneration compute is burned, and it wedges the registered remedy.

**Why:** #2389 r1 g1 — `phase_capregen_anchors` strided unfiltered `rest_ids` while `phase_anchors` strided
vLLM-cell-filtered `rest_ids`; with item 4 engaged, `_merge_anchor_capregen` raises for every shifted worker.
The path is exactly the designed backstop for a partial cap-recalibration timeout.

**How to apply:** for each amend phase, list the generation phase's pre-striding filters (grep between the
`_anchor_context_order`-style call and the `[w::W]` slice) and require either the same filters or a
shard-derived unit set (`{row keys in the worker's OWN shard} ∩ target`) — the latter is immune by
construction. Prefer flagging the shard-derived form as the fix sketch.
Related: [[wrapper-required-kwonly-kwarg]].
