---
name: fixed-name-tmp-atomic-write-fanout-race
description: A fixed-name `<path>.tmp` + os.replace "atomic" JSON write races when every fan-out worker re-derives and writes the SAME deterministic artifact at startup (#2546 r1 g3)
metadata:
  type: feedback
---

When a driver fans out N workers that each re-derive a shared deterministic
artifact (rowsets/manifest JSON) and write it through a hand-rolled
`tmp = path.with_name(path.name + ".tmp"); write_text; os.replace(tmp, path)`,
the tmp NAME collides across workers: worker A replaces the tmp while worker B
is mid-write, and B's own replace hits FileNotFoundError → worker rc≠0 →
parent FATAL, on identical-content writes. Identical bytes do NOT make the
pattern safe; the tmp path is the race surface.

**Why:** #2546 c59ea9715a — all 4 shard workers call build_rowsets at startup,
same `arm{K}.json.tmp`; fixed in-round by `atomic_io.write_json_atomic`
(process-unique temp, #2336).

**How to apply:** whenever a reviewed diff both (a) spawns workers and (b) has
any write path reachable from worker startup, grep the atomic-write helper for
a process-unique temp name (pid/uuid suffix, or the repo `atomic_io` helper).
Related: [[start-manifest-stale-artifact-done]] (the resume half of shared
artifacts); also seen same review: unit resume fingerprints keyed on params
only (no store-content key) + a cache marker fingerprint recorded but never
re-compared — the [[pilot-pass-report-fingerprint-unchecked]] shape on a
resume path.
