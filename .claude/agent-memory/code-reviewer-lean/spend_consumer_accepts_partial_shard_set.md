---
name: spend-consumer-accepts-partial-shard-set
description: A local-first staging helper (`if dir nonempty: return`) feeding a paid judge wave / gate reduce silently accepts a PARTIAL sharded producer's output — demand a completeness assert against the deterministic cell enumeration (#2254 R1 g3)
metadata:
  type: feedback
---

When a spend-bearing consumer phase (paid judge wave, gate/operating-point
reduce) stages its inputs via the common local-first pattern — `if
comp_root.exists() and any(glob): return comp_root`, else stage whatever the
HF prefix lists — and the producer is SHARDED (per-shard sentinels, per-shard
uploads at shard END), the consumer run after only some shards finished
silently judges and reduces a partial grid: argmax operating points over
missing cells, under-populated null bands, gates decided, downstream pods
launched at wrong operating points. No error anywhere; the reduce's only
guard is usually "≥1 cell per behavior".

**Why:** #2254 R1 g3 (`issue2254_preimage.py::_stage_phase_completions` +
`_reduce_wave1`): the localize grid is DETERMINISTICALLY enumerable
(`_localize_cells` → 385 ids/behavior) yet nothing asserted staged ⊇
expected; the driver had the right idiom one level down ("empty cell list —
never a silent no-op") but not at the consumer.

**How to apply:** whenever a diff adds a consumer that stages a sharded/
resumable producer's per-cell outputs, ask "can the producer's cell
enumeration be recomputed here?" — if yes (it usually is: the same grid
function is in the same file), demand `staged_ids ⊇ expected_ids` raising
with the missing ids, placed BEFORE the first paid call or before the reduce
writes gate/ops artifacts. Legitimately-absent cells need a recorded absence
set to diff against (e.g. a selection_meta `missing` list), not a weaker
gate. Sibling family: [[count-gate-starved-by-resume-skip]] (the INVERSE:
over-strict fresh-count gate), [[presence-redrive-blesses-stale-mirror]]
(presence-only remote check), [[sentinel-path-outside-drain-glob]].
