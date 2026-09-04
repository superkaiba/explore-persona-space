---
name: frozen-store-over-500kb-hub-plus-index
description: A frozen JSON artifact over the 500 KB pre-commit cap ships as Hub bytes + a small committed index (file sha256, content sha, hub path, regen/verify commands), not line-split shards
metadata:
  type: feedback
---

When a write-once frozen artifact exceeds the repo's 500 KB `check-added-large-files` cap (#2658 r18: two ~1.5 MB evidence-packet stores), prefer persisting the bytes on the HF data repo under the issue prefix and committing a SMALL index JSON pinning per-file sha256, addressable content_sha256, hub path, and the exact regen + verify commands, over line-splitting into committed shards.

**Why:** shard-splitting a single canonical JSON document forces every consumer's read path to reassemble; the Hub + index route keeps the one-file read path, and immutability stays checkable because the freeze rebuilds deterministically from committed sources and refuses drift (rebuild-match-or-raise), with the index sha as the durable pin. Verified post-upload with a scoped list_repo_tree byte-size match.

**How to apply:** [[worktree-commit-git-dash-c-form]] for the commit itself; make the consumer's store-absent error print the regen command so a fresh checkout self-serves.
