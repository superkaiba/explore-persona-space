---
name: banked-parent-dual-schema-equivalence
description: Reviewing a dual-schema reader over PARENT stores — certify the claimed slot equivalence against the producer's OWN consumer function AND a probe of the real pinned artifact, never the new code's docstring (#2333 R4)
metadata:
  type: feedback
---

When a fix adds a dual-schema extractor over banked PARENT artifacts (e.g.
"#2094 records have no `v_ce`; context-end = `q_span[-1]`"), the docstring's
equivalence claim is the thing under review, not evidence. Certify it two
ways, both cheap:

1. **Producer's own consumer read** — find the parent driver's OWN function
   that consumes the same slot (`issue2094_run.py::_slot_vectors(rec,"ce")
   == span[-1:]`) and the capture line that defines the span coordinates
   (`captured[layer][j, pe:ctx_len]` ⇒ last row = position ctx_len−1). The
   new extractor must match the producer's read, and the FRESH capture's
   position must match too (`ctx_len − 1` on both sides).
2. **Real pinned-artifact probe** — load the staged real store (implementer
   convention: staged under `/mnt/eps-data/<user>/issue<N>_<slug>/` with a
   re-runnable `verify_fix.py`) and print top-level keys, record keys,
   shapes, and the exact key-name casing (`donor_assignment` SINGULAR in
   bank.json vs `donor_assignments` plural in the .pt — same artifact
   family, two spellings). Re-run the implementer's verify script rather
   than trusting its pasted output.

**Why:** the r4 crash existed because the pre-fix code assumed the sibling
store's schema; a review that checks the fix against the fix's own
docstring repeats the same move one level up. #2333 R4: both probes took
~3 tool calls total and turned every docstring claim into a measured fact
(195/195 PASS, layers 0..27 identical, q_span (21,28,3584), nq==ctx_len−pe).

**How to apply:** any diff whose helper branches on artifact schema
(`if "k" in rec ... elif ...`) or claims cross-store slot equivalence.
Pair with [[fails-pre-fix-probe-parent-commit]] (the parent-commit swap
certifies the crash signature) and check the fail-loud terminal branch
names the observed keys. Also probe id-namespace disjointness before
accepting a `{**map_a, **map_b}` merge of per-set maps.
