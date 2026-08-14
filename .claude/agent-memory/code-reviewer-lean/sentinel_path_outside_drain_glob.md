---
name: sentinel-path-outside-drain-glob
description: A CONFORMING sentinel envelope written into a /workspace/logs/issue-<N>/ SUBDIRECTORY is invisible to poll_pipeline's non-recursive top-level drain glob — check the PATH, not just the keys (#2225 R1 g4)
metadata:
  type: feedback
---

When a pod dispatcher writes sentinels via a correct helper
(`issue778_lib.write_results_sentinel` — all `_SENTINEL_REQUIRED_KEYS`
present), the file can STILL never reach the VM: the drain shell globs
`/workspace/logs/issue-<N>-*.json` (poll_pipeline.py:2321-2333, path-terminal,
non-recursive; the `epm_results` presence probe at :2044 is equally
top-level), so a `LOG_ROOT=/workspace/logs/issue-<N>` DIRECTORY passed as
`logs_dir` parks every sentinel one level down where `*` cannot reach.

**Why:** #2225 R1 g4 (commit 8b2c549c65): `issue2225_dispatch.sh` set
`LOG_ROOT="${EPM_I2225_LOG_ROOT:-/workspace/logs/issue-2225}"` and threaded it
into `write_results_sentinel(logs_dir=...)` — envelope perfect, path dead; the
plan §9 even pinned the top-level paths (`/workspace/logs/issue-2225-*.json`)
and justified `backend: runpod` on that contract. P3's `epm:results` and the
P0 designed-halt octave note would never drain.

**How to apply:** in any diff touching pod-side sentinel emission, check TWO
things: (1) envelope keys ([[handrolled-pod-sentinel-envelope]]), and
(2) the RESOLVED parent dir of the sentinel path == `/workspace/logs` (or the
lane's drained root) — a per-issue log SUBDIR default (`logs/issue-<N>/`) is
the trap shape; per-phase LOGS may live in subdirs, SENTINELS may not. Also
costs stall detection: `/workspace/logs/issue-<N>-*.log` globs miss subdir
logs.
