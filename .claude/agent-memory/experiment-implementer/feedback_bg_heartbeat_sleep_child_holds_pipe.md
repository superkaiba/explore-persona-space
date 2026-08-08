---
name: bg heartbeat sleep child holds stdout pipe
description: Killing a background heartbeat subshell does NOT reap its in-flight `sleep` child; the orphan keeps the inherited stdout fd open and any pipe-capturing parent (pytest subprocess.run, CI) blocks until the sleep expires.
type: feedback
---

In a bash launcher with a background heartbeat subshell (`( while true; do echo ...; sleep $N; done ) & HB_PID=$!`), a bare `kill "$HB_PID"` in the EXIT trap kills only the subshell — the in-flight `sleep` child is reparented to init and keeps the inherited stdout/stderr fds open. Harmless when output goes to a file, but any parent capturing the script's output through a PIPE (pytest `subprocess.run(capture_output=True)`, CI wrappers) blocks on EOF until the sleep expires (#601 round 7: a 60-s heartbeat hung a 30-s test timeout).

**Why:** bash delivers default-disposition TERM to the subshell immediately, but children are never signaled; pipe EOF requires ALL fd holders dead.

**How to apply:** deterministic teardown in the trap — `kill -STOP "$HB_PID"` (freeze so no new sleep forks) → `pkill -P "$HB_PID"` (reap the in-flight sleep while parentage is intact; after the subshell dies, `-P` no longer matches the reparented child) → `kill -CONT "$HB_PID"` → `kill "$HB_PID"`. Alternative: run the heartbeat under its own `setsid` and `kill -- -PGID` (but then `$$` inside is no longer the driver pid — pass it explicitly).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [bg heartbeat sleep child holds pipe](feedback_bg_heartbeat_sleep_child_holds_pipe.md) — killed subshell's in-flight sleep holds stdout; STOP→pkill-child→CONT→TERM in the trap. #601.
