---
name: HF Hub Commit Rate Limit Strategy
description: When backing up many dirs to HF Hub, batch into few commits via create_commit to avoid 128/hr shared limit
type: feedback
---

HF Hub has a hard rate limit of **128 commits/hour per repo** (shared across all concurrent writers — pods, pipelines, people).

**Why:** In April 2026, parallel pod backups (pod2 + pod4) saturated the quota within 30 min. Each `upload_folder` call = 1 commit. 68 dirs = 68 commits if done naively. With pod2 also uploading, I hit 429 on commit 35 on pod4 and had to wait 65 min between retries. Total: ~8 hours of wall time for 170 GB.

**How to apply:**
- Never loop `upload_folder` for many dirs. Instead use `HfApi.create_commit(operations=[CommitOperationAdd(...), ...])` to batch many files into ONE commit.
- Group by ~60 GB / ~200 files per commit (HF LFS handles big payloads but the commit itself has overhead).
- When a 429 IS hit, sleep at least 65 min — HF's window is rolling-hour, not fixed.
- Sort batches by size DESCENDING so big work pace the early commits; small ones don't waste a commit slot.
- Example: 16 missing dirs / 170 GB → 3 commits (60+60+48 GB) via `create_commit`, vs 16 commits via `upload_folder`.
- Always verify via `list_repo_tree` after commit; tolerate ±5% size mismatch for LFS metadata.
