---
name: Hub queue-full 429 storms + LocalEntryNotFoundError masking
description: HF Hub "maximum queue size reached" 429s hit plain REST (tree/resolve/commit) before any xet/hf_transfer byte transfer; a 429 on hf_hub_download's HEAD surfaces as LocalEntryNotFoundError (an EntryNotFoundError subclass) — 404-shaped but actually transport
type: feedback
---

Wrap EVERY Hub call on upload / verify / staging legs in `hub.retry_transient`
(= `_retry_upload`: Retry-After-aware, `EPM_HF_RETRY_BUDGET_S` wall budget
default 1800 s, 6-attempt floor, quota-403 immediate re-raise, exhaustion
fail-loud naming `what` + attempts). Three coupled facts (#1345 crash-fix r5,
att-20260715-175238):

1. **"maximum queue size reached" is the Hub SERVER's queue-full 429 body** on
   plain REST endpoints — observed on `/api/.../tree` (list_repo_tree),
   `/resolve/...` HEAD (file_exists / hf_hub_download metadata), and
   `/api/.../commit/main` (upload_file). It fires BEFORE any accelerated byte
   transfer, so `HF_XET_HIGH_PERFORMANCE` / `HF_HUB_ENABLE_HF_TRANSFER` are not
   implicated. Storms are intermittent — a retry usually clears in 1-2 attempts.
2. **A 429 on the HEAD inside `hf_hub_download` surfaces as
   `LocalEntryNotFoundError`** — an `EntryNotFoundError`/`FileNotFoundError`
   subclass whose message reads like a genuine not-found ("cannot find the
   requested files ... check your connection"). Do NOT conclude absence from
   it; `retry_transient` classifies it transient via the "connection" text, so
   a wrapped call self-heals.
3. **The tree endpoint 404s on exact FILE paths by design** (hub 0.36.2,
   #939), so per-file verify probes ALWAYS route through the `file_exists`
   fallback — any bare (un-retried) call on that fallback path converts a lone
   429 into a fatal AFTER the upload already landed (the #1345 crash: the
   shard was on the Hub; only the verify died).

**Why:** the fix placed retry at the SOURCE modules (`hub.list_hf_files_under_path`
fallback, `upload_sharded` upload_file sites) per artifact-reuse's
"fix the source module" law — never caller-side re-wraps.

**How to apply:** any new script doing Hub IO (upload_folder, upload_file,
hf_hub_download, file_exists, list_repo_tree) wraps each call in
`retry_transient(lambda: ..., what=f"<op>(<repo>:<path>)")`; in loops, bind
loop vars as lambda defaults (ruff B023). In tests, no-op `time.sleep` +
`EPM_HF_RETRY_BUDGET_S=0` (attempt-bound, 6 calls) or transient-looking fake
errors ("500 ..." messages) will sleep for real. Regression pin:
`tests/test_upload_sharded.py::test_verify_fallback_retries_429_then_success`.
