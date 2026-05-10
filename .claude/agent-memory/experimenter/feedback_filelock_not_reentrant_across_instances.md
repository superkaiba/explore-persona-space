---
name: FileLock not re-entrant across instances
description: Python `filelock.FileLock` is re-entrant on the SAME instance only — different instances on the same path deadlock at OS-level flock
type: feedback
---

`filelock.FileLock` (3.x) is re-entrant on the **same instance** only. Two
*different* `FileLock(path, ...)` instances pointing at the **same path** will
deadlock if the second one is acquired while the first still holds the lock.
The inner waits its full `timeout=` seconds, then raises `filelock.Timeout`.

**Why:** Both instances try to acquire the OS-level `flock(2)` on the same
inode. The kernel sees them as different file descriptors and refuses the
second exclusive lock. Re-entrancy works on the same instance because the
class tracks `_lock_counter` internally and skips the OS call.

**How to apply:** When refactoring code to add "belt and braces" filelocks
(e.g., #228 R4 added an outer lock in `pregenerate_one_source` while keeping
the inner lock in `generate_and_cache_onpolicy_data`), either:
1. Remove one of the two locks (recommended — pick the outermost layer that
   covers the read-then-write atomicity you actually need).
2. Pass the lock object through to inner functions and `with lock:` again
   on the same instance (re-entrant — works).
3. Use `threading.RLock()` if it's a single-process re-entrancy case (but
   filelocks exist precisely because you need cross-process safety).

**Symptom in practice:** Worker process spawns, logs "[source] generating
canonical 10-persona on-policy cache", then sits at ~10-15% CPU for 600s
before raising Timeout. GPU memory stays at 0 MiB the whole time — vLLM
never even imports because the inner FileLock acquisition blocks first.

**Caught in #228 R4 second-launch attempt** — code-review missed it because
the outer + inner locks were in separate files and the reviewer didn't trace
the call chain across `pregenerate_onpolicy_cache_228.py` → `run_leakage_v3_onpolicy.py`.
