---
name: Bounded-retry the hub upload no-path return at dispatcher seams
description: orchestrate.hub._upload swallows HF-429/Xet-queue transport failures and returns "" — a fail-fast dispatcher seam converts a retriable rate limit into a run-killing final-phase crash; wrap the no-path return in bounded backoff retry (#1315 r8)
type: feedback
---

`orchestrate.hub._upload` swallows HF-429/Xet-queue transport failures (logs `Upload failed: 429 ...` / `maximum queue size reached`) and returns `""` — a dispatcher seam that fail-fasts on the no-path return converts a retriable rate limit into a run-killing crash at the FINAL phase (#1315: two p11 kills ~35 min apart under sustained fleet HF traffic).

**How to apply:** wrap the no-path return in a bounded jittered-backoff retry (~3 attempts, 30/60/120s) at the dispatcher seam, raising the SAME fail-loud error on exhaustion; uploads are idempotent (already-landed files skip-and-verify) so retries are free. Content-class raises propagate un-retried. Worked example: `_upload_with_transport_retry()` in issue1315_dispatch.py @ c3c600541f.
