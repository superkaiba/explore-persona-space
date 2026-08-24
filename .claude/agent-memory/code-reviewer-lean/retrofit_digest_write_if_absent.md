---
name: retrofit-digest-write-if-absent
description: A retrofitted provenance/regime digest gated by write-if-absent silently grandfathers artifacts produced BEFORE the digest existed — check the producer branch for an artifacts-already-present refusal (#2502 R2 g2)
metadata:
  type: feedback
---

When a revision ADDS a remote provenance digest (regime.json, fingerprint, manifest) to close a presence-only resume hole, the producer path `if digest absent: publish mine and proceed` re-opens a narrower version of the same hole: a prefix populated by the PRE-fix writer has artifacts but no digest, so the new code blesses unknown-provenance chunks with a fresh digest and presence-skips them.

**Why:** post-fix code publishes the digest before any artifact (so absent-digest+present-artifacts is unreachable in a post-fix-only world), which makes the seam easy to size — it is exactly the set of prefixes the superseded writer touched (usually smoke leftovers). That sizing is what splits Minor from Major: pre-launch, production prefixes fresh ⇒ Minor residual; if the old writer ran in production ⇒ Major.

**How to apply:** on any diff introducing a write-if-absent digest gate, (a) ask "can artifacts exist under this prefix without the digest?" and enumerate who could have produced them; (b) prefer/recommend the tightening `refuse to publish when the prefix already holds artifacts` (one extra scoped listing); (c) verify the digest dict's values are JSON-round-trip-stable scalars (tuples→lists breaks `have != want`); (d) verify concurrent-producer races publish IDENTICAL bytes (e.g. shard fields stripped). Related: [[presence-redrive-blesses-stale-mirror]], [[manifest-first-reader-stale-shard-regen]], [[ported-pin-application-semantics]].
