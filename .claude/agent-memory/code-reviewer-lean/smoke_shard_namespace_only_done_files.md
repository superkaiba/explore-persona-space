---
name: smoke-shard-namespace-only-done-files
description: Smoke/production runs sharing an out_root need EVERY per-block artifact path namespaced, not just done-files — else uploads sweep smoke shards under the production prefix (#2333 R1 g1)
metadata:
  type: feedback
---

A driver that namespaces its resume DONE-files by smoke vs production
(`smoke_blocks/` vs `blocks/`) but writes rollout/tensor shards at
`<dir>/<block.slug>.<ext>` where the slug excludes the pair set collides
smoke and production artifacts in the shared default out_root. The
production run's incremental `upload_dir_hf(<shared dir>, <PROD prefix>,
glob)` then uploads stale smoke shards (K=1, single pair) under the
PRODUCTION HF prefix in the DEFAULT smoke-then-grid sequence; remote state
converges only at full completion, and an interrupted/partial grid leaves
them there as silently mislabeled production data (#2333 R1 g1, M2).

**Why:** the resume predicate and the artifact store are separate
namespaces; fixing one does not fix the other, and dir-glob uploads bind
to the DIRECTORY, not to which run produced each file.

**How to apply:** when a diff adds a `--smoke` mode sharing an out_root,
enumerate EVERY per-block write path (shards, tensor stores, jsonl) and
check each carries the same namespace split as the done-files; then check
every upload glob against the mixed dir. Pair this with the spend-consumer
check: the downstream judge/analysis loader needs a staged ⊇
enumerable-grid assert, or the contaminated/partial set is consumed
silently ([[spend-consumer-accepts-partial-shard-set]]).
