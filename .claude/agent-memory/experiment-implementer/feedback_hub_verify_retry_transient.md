---
name: hub-verify-path transport retry + prefix-batched listing
description: Upload/verify-path Hub calls must ride retry_transient and one prefix-scoped listing — an unretried per-file file_exists HEAD probe let a single HF 429 kill a run post-upload (#1335)
type: feedback
---

Wrap every fresh Hub call on an upload/verify path in `hub.retry_transient` (Retry-After-aware, budgeted) and verify sharded uploads with ONE prefix-scoped listing per destination repo — never per-file `api.file_exists` loops.

**Why:** #1335 att-20260715-134136 — `list_hf_files_under_path`'s exact-file fallback issued an un-retried `file_exists` HEAD per shard; one transient 429 ('maximum queue size reached') crashed a healthy run 2.8h in, AFTER its uploads had succeeded. A transport error is retried, never fatal (the llm-judging rule-24 analogue for the upload path).

**How to apply:** when writing or reusing shard-upload / upload-verify code, grep the path for bare `file_exists` / `get_hf_file_metadata` / per-file existence loops; route through `_batched_verify`-style prefix listings + `retry_transient`. Pin with a 429-then-success test and a ≤2-listings batching test (see tests/test_upload_sharded.py pins from #1335 round 5).
