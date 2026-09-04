---
name: manifest-internal-determinism-replay
description: Certify a committed sampler manifest by replaying its seeded sub-draws from its OWN committed id lists — no network, no input restage (#2479 R1 g4)
metadata:
  type: feedback
---

When reviewing a committed sampling manifest (sample ids + seeded reservation/subset + tier fill + pinned input revisions), the full rerun needs the pinned HF inputs — but two cheaper probes certify most of it locally:

1. **Seeded sub-draw replay:** if the sampler draws a subset FROM ITS OWN recorded list with a recorded seed over a canonical ordering (e.g. `random.Random(0).sample(sorted(sample), k)`), recompute it from the committed manifest alone and require exact equality. Catches hand-edits, seed drift, and post-hoc substitution of either list.
2. **Static-definition byte-compare:** when a committed artifact (panel.json) is the emit of a pure in-repo definition module, run that module's emit at the manifest's `git_commit` provenance SHA and `cmp` bytes ([[tmp-rerun-zero-diff-analysis-artifact]]'s zero-network sub-case). Confirm the SHA is an ANCESTOR of the artifact commit and the module is byte-unchanged between them first — otherwise the replay certifies the wrong code.

**Why:** #2479 R1 g4 — both probes ran in seconds and retired the hand-edit/provenance questions that a full HF restage (tens of MB of real-corpus JSONL, content-hygiene constraints) would have cost far more to answer.

**How to apply:** any panel_manifest/sample-manifest review where the brief asks for internal consistency + provenance. Also check tier availabilities against the intersection table arithmetically (avail_k = source_k − taken_above) — a mismatch there is the cheap tell of a substituted pool. Realized-vs-plan count surpluses that the plan routes to "re-measure at materialization" are observations, not blockers.

3. **Same-named hash, two artifacts, different values (i2658 R15 g3):** when a freeze artifact and a runtime order manifest both carry a per-split `requests_sha256` over the SAME request set and the values differ, do not flag drift on inequality alone. Read each producer's preimage (both hashed `json.dumps([(id, k, seed(id,k)), ...])` with a shared seed helper) and replay BOTH in their own documented walk order (selection: sorted(frame) x sorted(cell). generator: cell_order walk). Exact match on both plus sorted-set equality between the two expansions certifies the manifest is fully derived from the frozen selection. Also grep for any consumer that cross-compares the two keys: none existing = observation only, one existing = real blocker.
