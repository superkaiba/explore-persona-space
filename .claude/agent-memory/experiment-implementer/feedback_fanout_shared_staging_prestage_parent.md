---
name: fanout-shared-staging-prestage-parent
description: Concurrent fanout units racing a shared hf-staging dest (shared _hfstage scratch + single-file staleness guard) — pre-stage in the parent; per-invocation mkdtemp scratch; guard on the FULL consumer-required file set
type: feedback
---

Concurrent fanout units must never lazily stage a SHARED input dest. Two coupled failure modes (#1315 r5, epm:failure v4): (1) a SHARED staging scratch dir (`dest/_hfstage`) lets one unit's `os.replace` steal a file a sibling's `hf_hub_download` just returned — sibling crashes `FileNotFoundError` at its own replace; (2) a SINGLE-FILE staleness guard (`raw_pos.jsonl exists`) lets a unit consume a sibling's mid-stage / crash-partial dest — derive crashes on the missing sibling sidecar (also the relaunch hazard after a group-reap).

**Why:** the fanout's whole-group reap on first failure leaves arbitrary partial dests, and a one-cell smoke structurally cannot exercise concurrency on the shared dest.

**How to apply:** (a) pre-stage shared inputs ONCE in the parent before `_fanout_units` (also fails input-sha gates before GPU spend); (b) staging helpers download into a per-invocation `tempfile.mkdtemp(dir=dest)` (same FS — `os.replace` can't cross devices) + atomic per-file publish + per-file `target.exists()` skip (every clean return ⇒ complete dest; partial dests self-heal); (c) staleness guards key on the FULL consumer-required file set (the derive precondition), never one proxy file; (d) do NOT sweep other invocations' scratch dirs unconditionally — a live sibling's dir sweep recreates the steal race (legacy fixed names: unconditional; suffixed dirs: age-gate ~1h); (e) skip flock on MooseFS/FUSE — semantics unverified, and (a)+(b) make it unnecessary. Deterministic CPU repro: 2 threads on the real helper body, Hub boundary faked, barrier-synchronized listing + sleep-widened download-return→replace window (`tests/test_issue1315_margin_pool_staging.py`).
