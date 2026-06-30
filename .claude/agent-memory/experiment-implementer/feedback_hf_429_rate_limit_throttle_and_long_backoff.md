---
name: HF Hub 429 rate-limit — throttle the burst AND back off longer than 5xx
description: snapshot_download of many files bursts 1 req/file (xet-read-token), trips HF's 2500-req/5min per-org quota → fatal 429; cap max_workers + give 429 a 60-300s backoff (NOT the short 5xx 10-180s)
type: feedback
---

A bulk `huggingface_hub.snapshot_download` of many files (#658 PV store: ~12000
`rollout_acts/**/*.pt`) issues ONE request per file (xet-read-token) and, at the
default parallelism (8-16 concurrent), bursts past HF Hub's per-org rate limit
(**2500 api requests per 5-minute window**), dying with a fatal
`HfHubHTTPError: 429 ... We had to rate limit you, you hit the quota of 2500 api
requests per 5 minutes period`. It ran cleanly ~38 min then died on a final
batch (#658 failure v10, 2026-06-30).

**Why:** the 429 is a DOWNLOAD rate limit (xet-read-token, one per file), a
DIFFERENT quota from the upload-side throttles in `upload-policy.md` (the
256-commits/hr repo-commit limit, and the account-wide LFS storage 403). Don't
conflate them.

**Two-sided fix — do BOTH:**
1. **Throttle the burst at the source:** pass `max_workers=4` to
   `snapshot_download` (4 workers × ~1 req/s ≈ 240 req/min = 1200/5min, under the
   2500 quota with margin). `max_workers=2` for extra safety. This *prevents*
   tripping the limit.
2. **Make the transient-retry wrapper back off LONGER on 429 than on 5xx.** A 429
   clears only over the 5-minute window, so a short 5xx-style backoff (10, 20,
   40... capped 180s) just re-trips it. For 429: honor the `Retry-After` response
   header when present (clamped to a `[60, 300]`s window), else exponential from a
   **60s base capped at 300s** (the 5-min window). Keep 5xx/timeout on the short
   `min(180, 10·2^(attempt-1))` backoff. `Retry-After` round-trips through
   `HfHubHTTPError.response.headers.get("Retry-After")` (CaseInsensitiveDict).
   429 detection: `getattr(err.response, "status_code", None) == 429` OR a "429 /
   too many requests / rate limit" message substring.

The 5xx and 429 transient classes need DIFFERENT backoffs even though both are
"retry-worthy" — a single short backoff covers neither well.

**Why this generalizes:** any per-issue script that `snapshot_download`s a
multi-thousand-file store (per-rollout activations, per-cell tensors, large
sharded datasets) hits the same wall. Bake `max_workers` + the 429-aware backoff
into the shared transient-HF retry helper so siblings inherit it. Impl of record:
`scripts/issue658_extract_rb_personavectors.py` (`_backoff_seconds` /
`_is_rate_limit_429` / `_retry_after_seconds` / `_retry_on_transient_hf`) +
`scripts/issue658_rb_pv_fit.py::_resolve_pv_store`. #658 r4.
