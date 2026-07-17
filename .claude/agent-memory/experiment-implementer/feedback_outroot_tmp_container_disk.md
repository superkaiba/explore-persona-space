---
name: outroot-tmp-container-disk
description: Dispatcher out-roots must never default to /tmp — on RunPod /tmp is the ~50 GB container overlay, not the /workspace volume; probe headroom per phase
type: feedback
---

Never default a dispatcher/driver out-root (or staging dir) to /tmp: on RunPod, /tmp rides the ~50 GB container overlay disk and a single consolidated 7B FT checkpoint (~15-25 GB) plus staged inputs fills it mid-run (SafetensorError ENOSPC at save; #1333 attempt 3, 2026-07-15 — GCP masked the bug because its /tmp rides the 300 GB boot disk). 
**Why:** lane-portability — the same workload runs on GCP boot disks AND RunPod container disks; only /workspace is big on both.
**How to apply:** self-resolve out-roots to a repo-root-under-/workspace anchor (e.g. `<checkout>/data/issue_<N>/{smoke,run}` — also inside the GCE crash-persist glob), keep /tmp only as a local-CPU-test fallback, gate each big-write phase with a statvfs floor + 1 GB posix_fallocate canary (fail loud with numbers BEFORE the write), and on any relaunch wipe the stale /tmp tree first. Worked fix: issue1333_dispatch.py `_default_out_root` + `_assert_out_root_headroom` (commit 5bb36f2ffe).
