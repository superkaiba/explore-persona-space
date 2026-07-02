---
name: Disk-full corrupts post-marker append (interleaved write)
description: A full VM root disk makes post-marker write a truncated, newline-less line; a retry concatenates onto it, producing one unparseable interleaved events.jsonl line that view silently skips.
type: feedback
---

When the VM root disk `/` is at 100%, `task.py post-marker` writes a
PARTIAL line with NO trailing newline (the write is cut off mid-record),
exits non-zero, and a retry APPENDS its full record onto the same
physical line — producing one corrupted interleaved line that
`task.py view --json` silently skips ("skipping malformed line N").
The marker then looks ABSENT even though the valid record is embedded.

**Why:** `post-marker` writes events.jsonl + git-commits; both fail on a
full disk. The shell itself fails too — bare `df`/`stat`/`cd` exit 1
because the harness's own `/tmp/claude-*` output write fails.

**How to apply:**
1. First symptom of "every command exits 1 with no output" = `/` is
   full. Free space WITHOUT writing: `rm -f /tmp/claude-1001/*/*/tasks/*.output`
   (deletes the harness's own stale bg-task logs). Then `df -h /` works.
2. After a post-marker that returned the record but `view` shows it
   missing: `tail -3 events.jsonl`, parse each line; a line that starts
   `...{"ts": ...}{"ts": ...}` is two interleaved writes. The LAST
   complete `{"ts": ...}` object is the valid record.
3. Fix: read all lines, find the embedded-record start (`bad.find('{"ts": "<retry-ts>"')`),
   `json.loads` the tail to validate, rewrite events.jsonl = lines[:N-1] +
   [the clean record]; validate EVERY line parses before `os.replace`.
   Back up to `events.jsonl.bak-*` first.
4. The data disk `/mnt/eps-data` (worktree caches) and `/` both matter;
   the vm_disk_guard cron should fire but acting yourself (temp cleanup)
   is faster mid-session. Loading 64 × 155MB `.pt` files is read-only
   (no write), so that is NOT the cause — the cause is fleet-wide fill.

Incident: #697 round-2 analyzer (2026-06-30) — v2 epm:interpretation
post corrupted line 213 into an interleaved write; recovered by splitting
the line and re-validating.
