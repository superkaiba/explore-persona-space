---
name: Bounded-retry the hub upload no-path return at dispatcher seams
description: orchestrate.hub._upload swallows HF-429/Xet-queue transport failures and returns "" — a fail-fast dispatcher seam converts a retriable rate limit into a run-killing final-phase crash; wrap the no-path return in bounded backoff retry (#1315 r8)
type: feedback
---

`orchestrate.hub._upload` swallows HF-429/Xet-queue transport failures (logs `Upload failed: 429 ...` / `maximum queue size reached`) and returns `""` — a dispatcher seam that fail-fasts on the no-path return converts a retriable rate limit into a run-killing crash at the FINAL phase (#1315: two p11 kills ~35 min apart under sustained fleet HF traffic).

**How to apply:** wrap the no-path return in a bounded jittered-backoff retry (~3 attempts, 30/60/120s) at the dispatcher seam, raising the SAME fail-loud error on exhaustion; uploads are idempotent (already-landed files skip-and-verify) so retries are free. Content-class raises propagate un-retried. Worked example: `_upload_with_transport_retry()` in issue1315_dispatch.py @ c3c600541f.

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Upload loops need 5xx retry + skip-set pairing](feedback_upload_loop_retry_plus_skip_set.md) — bounded 5xx retry (4xx loud) + pre-fetched skip set; verify on a FRESH listing. #542.
- [Bounded-retry the hub upload no-path return](feedback_hub_upload_no_path_transport_retry.md) — hub._upload swallows 429/Xet-queue and returns empty; dispatcher seams retry with backoff, fail-loud on exhaustion (#1315 r8)
