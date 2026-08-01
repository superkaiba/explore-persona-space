---
name: Fixed-filename disk probes race concurrent workers (EBADF)
description: A create/fallocate/unlink probe at a FIXED path races itself across concurrent workers on shared filesystems — uniquify the filename per invocation at the source helper
type: feedback
---

A fixed-filename create/fallocate/unlink disk probe (e.g. `.preflight_disk_probe.tmp` under a shared out-root) races itself when N concurrent worker processes run it: a sibling's unlink/recreate invalidates an open fd mid-`posix_fallocate`, surfacing `OSError EBADF` — an errno OUTSIDE the probe's handled sets (ENOSPC/EDQUOT/EOPNOTSUPP), so the worker crashes at startup.

**Why:** #1979 fellows job 16686 (2026-08-01) — 8 parallel per-unit workers each ran `assert_out_root_headroom(out_root)` at startup; 5 lost the race and died rc=1 pre-work. Fixed at source in `orchestrate/preflight.py::_probe_writable_bytes` (commit 11a6c405cd): probe path uniquified to `.preflight_disk_probe.<pid>.<8-hex-uuid>.tmp`, cleanup self-scoped + missing-file tolerant.

**How to apply:** when writing ANY per-worker startup probe (disk headroom, writability, lock canaries) that creates temp files on a shared path, uniquify the filename per invocation (pid + uuid). Never serialize at the caller and never swallow EBADF (it can mask real fd bugs). Known un-fixed siblings of the pattern at fix time: `scripts/issue1481_marker.py:1482`, `scripts/pod_disk_guard.py:84` (wf-fix filed).
