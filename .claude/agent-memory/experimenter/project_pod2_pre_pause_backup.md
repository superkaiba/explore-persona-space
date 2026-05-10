---
name: Pod2 Pre-Pause Backup
description: Strategy for safely backing up a container-disk pod before pause when HF Hub rate limit of 128 commits/hour is blocking. Applies when you have hundreds of model dirs to upload.
type: project
---

**Fact:** Pod2 (thomas-rebuttals-2) uses container disk (`/dev/md127` at `/workspace`), not a network volume. If RunPod terminates, all data is LOST. Pre-pause backup is critical.

**Why:** We had 318 weight dirs (~2 TB) on pod2, needed to back them up before pause. HF Hub enforces 128 commits/hour per repo, so naive one-commit-per-dir backup hits 429s after ~60 commits.

**How to apply:**

1. **Enumerate weight dirs once:** walk `/workspace` but skip `/cache/huggingface`, `/.cache/huggingface`, `/huggingface/hub/models--`, `/snapshots/` paths to avoid crawling HF cache content (which would double-count files).

2. **Slug scheme:** use `slug_for(path)` = full path with `/workspace/` stripped and remaining `/` replaced by `_`. This prevents name collisions across different experiment dirs with the same basename.

3. **Never use `HfApi.upload_large_folder`** — has a documented 0-file silent success bug on dirs with symlinks. Use `HfApi.upload_folder` (simple case) or `preupload_lfs_files` + `create_commit` (for rate-limit avoidance).

4. **For 1+ TB backups at rate-limit risk:** batch many `CommitOperationAdd` ops into one `create_commit` call.
   - Use `OPS_PER_COMMIT = 500`, `MAX_BYTES_PER_COMMIT = 150 GB`
   - Pattern: enumerate files → construct ops in parallel (sha256 is CPU-bound, use 16 workers) → preupload LFS files in parallel (8 workers) → single `create_commit` with 500 ops
   - Keep total commits under ~50 to leave headroom under 128/hr limit

5. **CRITICAL CAVEAT:** `CommitOperationAdd(path_or_fileobj=str(path))` calls `UploadInfo.from_path()` which does **eager sha256** in `__post_init__`. Constructing thousands of these on one thread takes hours for TB-scale data. MUST parallelize op construction with a ThreadPoolExecutor (16 workers gives ~18 GB/s sha256 throughput).

6. **Rate-limit recovery:** if already rate-limited before starting, the 1-hour window resets gradually. Expect to wait 30-60 min between hitting 128/hr and being able to commit again. Script should retry with `Retry after N seconds` parsing and cap retries at ~20.

7. **Verification:** after each commit, use `api.list_repo_tree(path_in_repo=hub_path, recursive=True)` and filter `RepoFile`s. Compare `count` and `size` (within 1% tolerance for LFS dedup).

8. **Manifest:** append-only JSONL at `/workspace/pod2_backup_manifest.jsonl` with `{local_path, hub_path, status, local_file_count, local_size_bytes, verify_msg, ts}`. Use a thread lock for appends.

Script location: `/tmp/pod2_backup_batched.py` locally, `/workspace/pod2_backup.py` on pod2.
