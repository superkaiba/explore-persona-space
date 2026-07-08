---
name: append-mode unit logs carry stale tracebacks across relaunches
description: After a crash-fix relaunch, the prior attempt's traceback sits at the HEAD of the same append-mode unit log — only errors AFTER the fix-engaged line count against the new run
type: feedback
---

Per-unit fan-out logs open in APPEND mode, so a crash-fix relaunch appends to
a log whose HEAD still carries the prior attempt's traceback. A bare
`grep Traceback/FileNotFoundError` false-positives the healthy new run.

**How to apply:** when verifying a relaunch against unit logs, gate on
ORDERING — only errors timestamped after the fix-engaged line (or after the
new launch's first line) count. (#1112 attempt 6, 2026-07-08.)
