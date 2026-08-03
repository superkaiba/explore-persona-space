---
name: Registered fallback must be ROUTED into the production path, not just built + probe-timed
description: A plan-registered fallback (MinHash near-dupe sketch) that is only probe-timed + CLI-opt-in never fires; "auto" defaults must resolve to the fallback or carry an executed predicate (#1738 r3)
type: feedback
---

A pre-registered fallback exists in three states: (a) built, (b) measured by a probe, (c) ROUTED — only (c) changes production behavior. #1738 r3: the plan registered a bottom-k MinHash near-dupe fallback "if the stream rate degrades >5x"; the driver BUILT the gate, the probe TIMED it (37 rows/s logged), but `build_manifest` resolved `--near-dupe-impl auto` to `"exact"` (`"exact" if impl == "auto" else impl`) and no predicate ever consumed the probe measurement — the production screen spun 5+h in the primary gate on ~600k rows (py-spy confirmed; /proc write_bytes frozen).

**Why:** an `auto` mode whose resolution is hardcoded to the primary is indistinguishable from a wired fallback in every smoke (tiny N makes both fast); only the marker "the fallback is registered" survives review while the routing line contradicts it.

**How to apply:** when a plan registers a fallback/mitigation, (1) grep the production call site for what `auto`/default actually resolves to — assert the routing in the smoke (e.g. `meta["near_dupe"]["impl"] == "minhash_bottomk"`); (2) prefer unconditional routing to the validated fallback over a threshold predicate extrapolated from a small probe (200-target throughput extrapolates ~linearly-wrong to 11.4k targets); (3) treat throughput of the fallback ITSELF as first-class — #1738's crc32-per-gram Python sketch was still ~4.5h at pool scale; the vectorized numpy rolling-hash sketch + fork-Pool fan-out measured 268 rows/s @8 procs (600k ≈ 37 min). Sibling trap found same round: a hit-count prefilter floored on RAW sketch sizes under-drops boilerplate-heavy near-dupes whose shared grams are df-pruned from the index — scale candidate-prefilter floors by the pair's INDEXED (df-surviving) sketch sizes.
