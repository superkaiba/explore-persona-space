---
name: stale-index-lock-pre-launch-probe
description: Probe + clear a confirmed-stale .git/index.lock pre-launch on pods whose workload tail commits pod-side — a pre-existing 0B lock kills the result commit at the very end
type: feedback
---

Before launching any workload whose TAIL commits pod-side (result JSONs to the
issue branch), probe `/workspace/<repo>/.git/index.lock`: a PRE-EXISTING stale
lock (0 bytes, mtime older than the launch, `pgrep -a git` empty) survives the
whole multi-hour run and then kills the workload's own `git commit` at the last
step — loud `File exists` crash, results sentinel never written (incident #1336,
2026-07-15: G1-halt upload tail died there; re-drive took seconds once cleared).

**Why:** bootstrap/provision interruptions leave 0-byte locks; git ops during
the run don't touch them; only the tail's commit collides.

**How to apply:** add to the pre-launch checks: `ls -la .git/index.lock` — if
present AND 0B AND old mtime AND no live git process, `rm` it (never remove a
lock a live process holds). The write-side sibling of the removed-dir/silent-pull
git traps.
