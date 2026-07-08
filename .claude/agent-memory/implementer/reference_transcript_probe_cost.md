---
name: Transcript-tail probe cost is resolution-dominated
description: watcher transcript probes cost ~165ms/pid warm, ~160ms of it happy-log resolution; _transcript_tail_rows re-resolves internally (timing it after a separate resolve double-counts)
type: reference
---

Measured on the live VM (task #1127, 2026-07-08): one watcher transcript-tail
probe (`autonomous_session_watch._transcript_tail_rows(pid, 262144)`) costs
~165 ms warm per pid — ~160 ms of it is
`session_resolver._resolve_transcript_via_happy_log(pid)` (scans
`~/.happy/logs`); the 256 KB read+JSON-parse itself is ~5 ms.

Two consequences for future watcher perf/timing work:

- `_transcript_tail_rows` calls the resolver INTERNALLY — timing
  "resolve + _transcript_tail_rows" double-counts the ~160 ms resolution.
- Any "parse is too slow, add a byte pre-scan" optimization targets the
  wrong term; the win, if ever needed, is memoizing pid -> transcript
  resolution per tick across the watcher passes that each resolve
  fleet-wide (stale-registration, idle-unmapped, wedge probes).
