---
name: FileLock not re-entrant across instances
description: filelock.FileLock is re-entrant on the SAME instance only; two instances on one path deadlock at OS-level flock, wait the full timeout=, then raise Timeout.
type: feedback
---

`filelock.FileLock` re-enters only via the same instance's internal `_lock_counter`; two DIFFERENT instances on the same path contend for the OS `flock(2)` and the inner one blocks its full `timeout=` then raises `filelock.Timeout`.

**Why:** #228 R4 — an outer "belt and braces" lock in `pregenerate_one_source` wrapped the existing inner lock in `generate_and_cache_onpolicy_data`; review missed it because the locks were in separate files. Symptom: worker logs the "generating on-policy cache" line, sits at 10-15% CPU for 600s with GPU at 0 MiB (vLLM never imports — the lock blocks first), then Timeout.

**How to apply:** when adding locks around code that may already lock the same path: remove one (keep the outermost layer covering the read-then-write atomicity you need), or pass the SAME lock instance through and re-`with` it. Trace the call chain across files before approving nested locking.
