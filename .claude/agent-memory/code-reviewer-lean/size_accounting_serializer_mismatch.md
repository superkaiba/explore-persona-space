---
name: size-accounting-serializer-mismatch
description: When a writer sizes output with one serializer config but writes with another (e.g. compact-cost accounting vs indent=1 write), cap invariants silently inflate ~8-13% — probe with a production-path measurement (#2321 R1 g1)
metadata:
  type: feedback
---

Whenever a diff enforces a byte-cap on a serialized artifact (shard, index
part, manifest) by ACCUMULATING per-item serialized costs, diff the
accounting serializer call against the WRITE-site serializer call — every
kwarg: `indent`, `separators`, `ensure_ascii`, `sort_keys`. A mismatch means
the cap is checked against a phantom document.

**Why:** #2321 R1 g1 (`orchestrate/packing.py::_write_index_parts`): cost =
`len(json.dumps({src: e}))` (compact) but the part was written with
`indent=1` — newline + depth-indent per line inflated every entry ~25–35 B.
Measured on the production code path: parts landed at **1.129× the cap**
(1.084–1.115× at realistic path lengths), turning a "≤9 MB non-LFS" contract
into 9.75–10.16 MB files that could cross the 10 MB Hub LFS force-route
inside a deletion-bearing commit. The in-code comment asserted ~1 MB of
headroom; measurement refuted it. Tests never caught it because no fixture
wrote a MULTI-part index (the mid-loop flush path was unexercised at any
scale).

**How to apply:** (1) grep the writer for two `json.dumps` (or equivalent)
calls on the same payload — one in the cost expression, one at the write
site — and compare kwargs; (2) if they differ, run a 2-minute production-path
probe: pack/write with an injected small cap, `stat()` the outputs, compute
worst/cap; (3) check the test suite actually fills a container to its cap
(a split/flush path with zero multi-part fixtures is the tell). Under-count
direction (accounted < real) is the dangerous one; accounted > real is safe.
Related: [[fails_pre_fix_probe_parent_commit]] (probe-don't-trust family).
