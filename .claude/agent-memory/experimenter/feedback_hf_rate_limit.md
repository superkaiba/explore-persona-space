---
name: HF Hub bulk-upload mechanics — 128 commits/hr, create_commit batching, upload_large_folder ban
description: 128 commits/hour per repo shared across all writers. Batch many CommitOperationAdd ops into few create_commit calls; never upload_large_folder (silent 0-file bug); parallelize op construction (eager sha256).
type: feedback
---

HF Hub enforces **128 commits/hour per repo**, shared across ALL concurrent writers. Naive `upload_folder`-per-dir loops (1 commit each) saturate it — April 2026 parallel pod backups hit 429 at commit 35 and stretched 170GB to ~8h wall time.

**How to apply:**
- Batch many `CommitOperationAdd` ops into ONE `HfApi.create_commit` (e.g. 16 dirs / 170GB → 3 commits). Group ~60GB / ~200-500 files per commit; keep total commits well under the limit for headroom; sort batches by size descending.
- **Never `HfApi.upload_large_folder`** — documented silent 0-file success bug on dirs containing symlinks. Use `upload_folder` (simple case) or `preupload_lfs_files` + `create_commit` (rate-limit avoidance).
- `CommitOperationAdd(path_or_fileobj=str(path))` does **eager sha256** in `__post_init__` — constructing thousands single-threaded takes hours at TB scale; parallelize op construction with a ThreadPoolExecutor (~16 workers) and preupload LFS in parallel (~8 workers).
- On a 429, sleep ≥65 min (rolling-hour window) and parse any `Retry after N seconds` hint; cap retries.
- Verify each commit via `list_repo_tree(path_in_repo=..., recursive=True)`, tolerating ~±5% size mismatch for LFS metadata. Keep an append-only JSONL manifest for resumability.
