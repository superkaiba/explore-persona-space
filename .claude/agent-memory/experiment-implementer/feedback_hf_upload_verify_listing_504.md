---
name: HF post-upload verify listing 504 escapes the upload retry wrapper
description: list_repo_files paginates /tree/main and pagination retries ONLY 429 on follow-up pages; a 504 on a cursor page propagates straight through a bare verify call even when upload_folder is retry-wrapped. Wrap the verify call too.
type: feedback
---

A retry wrapper around `HfApi.upload_folder` does NOT protect the
`list_repo_files` / `list_repo_tree(recursive=True)` post-upload VERIFY
call — they are separate calls, and the 504 on a large repo lands on the
verify, AFTER the upload has already committed every file.

**Why:** `huggingface_hub.utils._pagination.paginate` retries only **429**
on follow-up cursor pages (`http_backoff(..., retry_on_status_codes=429)`),
NOT 5xx. So when `list_repo_files` paginates
`/api/datasets/.../tree/main?...&recursive=true&limit=1000&cursor=...` over a
big repo, a 504 Gateway Time-out on any follow-up page raises
`HfHubHTTPError(504)` straight through an unwrapped caller. `upload_folder`
itself does NOT paginate `tree` unless `delete_patterns` is set, so the
crash is the verify listing, not the upload. (#658 r3, 2026-06-30: two
consecutive runs uploaded all 12081 files / 38.6 GB, then crashed rc=1 on
the verify `list_repo_files` 504. The diagnosis hypothesis "the upload
wrapper is URL-filtered" was wrong — the upload wrapper was fine; the
escape was a completely separate, unwrapped verify call.)

**How to apply:** any HF-upload-heavy script that verifies via
`list_repo_files` (or any recursive tree listing) on a large repo MUST wrap
the verify call in the SAME transient-5xx retry as the upload — generalize
the retry helper to `_retry_on_transient_hf(fn, *args, ...)` and wrap BOTH.
A 504 `HfHubHTTPError` carries `response.status_code == 504` AND "Gateway
Time-out" in `str()`, so a status-code-OR-message transient classifier
catches it; preserve the storage-quota-403 immediate-reraise so an overflow
fallback still fires. `upload_large_folder` is NOT a drop-in fix here — it
has no `path_in_repo` (uploads to repo root) and does not touch the script's
own verify call anyway.
