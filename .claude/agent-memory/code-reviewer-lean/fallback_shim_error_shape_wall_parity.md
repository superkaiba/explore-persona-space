---
name: fallback-shim-error-shape-wall-parity
description: Two reusable probes for read-path fallback/shim diffs — does a dependency 404 propagate typed-as-absence (corruption mislabeled as miss), and does a new staging leg inherit the function's existing hang wall?
metadata:
  type: feedback
---

Two probes that found the only substantive concerns in #2321 r1 g2 (hub.py
packed-tree reader shim, commit f804fb4129):

1. **Error-shape probe:** when a shim resolves X via an index and then fetches
   a backing artifact (shard/part), trace what TYPE a missing-backing-artifact
   failure propagates as. If it surfaces as the ABSENCE class
   (`EntryNotFoundError`/`FileNotFoundError`), every exists-probing caller
   (`except EntryNotFoundError: treat-as-absent`) reads archive corruption as
   clean absence — the silent-wrong-answer sibling of the silent-empty class.
   "Propagates as itself" per plan letter can still be the wrong TYPE.
2. **Wall-parity probe:** when a diff extends a function that carries a hang
   wall / hard-exit timeout (e.g. `stage_hub_prefix`'s
   `EPM_HF_STAGE_TIMEOUT_S` + `as_completed(timeout)`), check every NEW leg
   added after/outside the walled region inherits it. In #2321 the packed leg
   ran after the raw pool, serial and unbounded — and the all-packed prefix
   (the feature's normal end-state) got ZERO wall protection.

**Why:** both gaps sat in disclosed comments and passed all tests; only
end-state tracing (post-delete, shim = only read path) surfaced them.
**How to apply:** on any reader-shim / fallback-wiring diff, walk the failure
taxonomy per caller class (prober vs stager) and diff the new leg's timeout
envelope against the host function's existing one.
