---
name: manifest-first-reader-stale-shard-regen
description: A manifest-first JSONL reader/lister beside an in-place regenerating writer silently serves the PRIOR regime's shards — writers must unlink stale manifest+shard siblings (#823 ext r1 g3)
metadata:
  type: feedback
---

When a driver pairs (a) a manifest-first reader/lister (`if <stem>.manifest.json exists: read shards` else plain `<stem>.jsonl`) with (b) a regeneration path that rewrites ONLY the plain `<stem>.jsonl`, a prior run's manifest + `shardNN` files (written by a >9.5 MB upload-sharding pass) take PRECEDENCE over the fresh rewrite — and they sha-verify against each other, so integrity checks pass on stale data. Two sub-shapes found in one commit (#823 `origin-ladder-more-contexts` r1, `issue823_ladder_ext_capture.py`): (1) a read-back loop consumed the previous regime's rollout text for own_len after a fingerprint-invalidated chunk regen; (2) an `expected_store_files` lister took the stale-manifest branch when the rewritten jsonl came in under the shard threshold, so the store NAME-SET verify PASSed on the stale set while the fresh rows never uploaded.

**Why:** the shard sha map pins shards to the MANIFEST, not to the current jsonl — internal consistency of a stale pair defeats every sha check; only trigger preconditions (same out-root rerun after an upstream regime change + a previously-oversized file) keep it latent, which is exactly the crash-fix reround shape.

**How to apply:** whenever a diff adds a manifest-first reader or a shard-for-upload helper, trace every writer of the underlying `<stem>.jsonl`: regeneration/rewrite sites must unlink `<stem>.manifest.json` + `<stem>.shard*.jsonl` in the same step (or the shard helper must remove stale siblings on its no-shard path). Fingerprint-equality resume on the jsonl does NOT protect the manifest-preferring read. Related: [[start-manifest-stale-artifact-done]], [[presence-redrive-blesses-stale-mirror]].
