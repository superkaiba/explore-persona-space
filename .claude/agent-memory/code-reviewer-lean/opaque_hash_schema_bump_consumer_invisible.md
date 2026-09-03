---
name: opaque-hash-schema-bump-consumer-invisible
description: A fingerprint schema bump that lives only inside the hash preimage cannot be version-detected by consumers — verify producer-side overwrite safety (leftover-shard set equality) and grade "refuses" claims by realized polarity (#2658 fixB r2)
metadata:
  type: feedback
---

When a fix bumps a store fingerprint SCHEMA (v1 -> v2) to invalidate old
artifacts, the version tag lives inside a sha256 preimage: readers see only an
opaque hex, so a pure-old-schema store is INDISTINGUISHABLE from a new one at
every consumer that reads the realized fingerprint from sidecars. Three checks
(#2658 group-B fix round, `capture_fingerprint` v2 + comparators
`_realized_capture_fingerprint`):

1. **Realized polarity vs claimed polarity.** The implementer said v1 stores
   "resume as stale/foreign and refuse" — the realized behavior was
   `shard_done` returning False, i.e. silent in-place REGENERATION (safe
   polarity per [[dial-added-fingerprint-arms-refuse-on-relaunch]], but not a
   refusal). Grade the claim by the branch, and check partial-overwrite
   safety: leftover old-schema shards past the new write range must be caught
   by a set-equality completeness assert (foreign/duplicate arms), not left
   mixed.
2. **Consumer-side residual.** A consumer that reads the realized fingerprint
   accepts any UNIFORM store, old schema included; only mixed-regime refuses.
   Acceptable when the pre-fix consumer had no check at all (no regression)
   and no old store exists — record the residual, don't FAIL it.
3. **Resume-rewrite byte-parity tests must cross the serialization
   round-trip.** A test pinning "rewrite byte-identical to fresh write" is a
   tautology if both sides derive from the same in-memory object; the honest
   form derives the rewrite from the LOADED (round-tripped) body and the
   reference from the in-memory body — same writer, different provenance
   (#2658 `test_resume_cell_rewrites_missing_or_stale_manifest` did this
   correctly). Related: [[twin-transcription-parity-tautology]].

**How to apply:** any fix-round diff pairing a fingerprint/schema bump with a
"stores are invalidated" claim — trace the mismatch branch (refuse vs
regenerate vs accept), sweep every consumer of the fingerprint for
version-blindness, and check the parity test's two sides have distinct
provenance.
