---
name: stall-anchor-log-mtime-zero
description: Stall detectors anchored on `now - log_mtime` false-positive at startup because stat returns 0 on missing files.
metadata:
  type: feedback
---

Watchdog / stall-detector loops that compute freshness as
`gap = $(now) - $(stat -c %Y "$LOG_FILE")` fire an immediate
false-positive when the log file doesn't exist yet: stat returns 0,
gap becomes `now - 0 ≈ 1.7e9 seconds`, and the watchdog kills the
dispatcher within one poll interval.

**Why:** `stat -c %Y missing_file` exits non-zero with empty stdout
under `set -e`; the `|| echo 0` fallback yields 0, which the arithmetic
treats as the epoch. The dispatcher hasn't had time to touch the file
on cycle startup — especially with Python startup latency.

**How to apply:** Anchor the freshness window to
`max(cycle_start_timestamp, log_mtime)`, not just log_mtime. The
stall freshness window only opens after the cycle begins, so the
first STALL_GAP_SECONDS are always granted regardless of log mtime.

Same principle applies to any "is the producer making progress" check
that uses file mtime: snapshot a baseline timestamp when the producer
launches, and use that as the lower bound on the freshness window.

Seen in task #365 round-7 watchdog tests — round-6 watchdog would have
killed a `sleep 3` dispatcher within 6 seconds of cycle start, never
respawning cleanly.
