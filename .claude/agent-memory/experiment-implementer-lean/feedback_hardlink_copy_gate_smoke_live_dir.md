---
name: hardlink-copy-gate-smoke-live-dir
description: "Smoke a read+write phase against a live-run-owned artifact dir via a SAME-FILESYSTEM `cp -al` hardlink copy — safe iff every writer uses atomic os.replace; /tmp is a different fs (cross-device link fails)"
metadata:
  type: feedback
---

When a round must smoke a phase that READS a real artifact tree a live
detached run OWNS (the #2658 r13 gate smoke while `--phase all` wrote
`power/*`), copy the tree with `cp -al` (hardlinks, instant, zero extra
data) into a scratch root ON THE SAME FILESYSTEM and run the phase there.

**Why it is safe:** this repo's writers go through
`atomic_io.write_json_atomic` (temp file + `os.replace`), and a rename onto
a hardlinked dentry BREAKS the link instead of writing through it, so the
smoke's writes never touch the real files and the live run's writes never
corrupt the snapshot (the copy keeps the old inode, a consistent read).
Verify after: real file `nlink=1` + live-run mtime. Append-mode files
(`open("a")` ledgers) DO write through hardlinks — safe only when the
smoked phase never appends to them (gate reads, never appends).

**Two traps:** (1) `/tmp` is usually a DIFFERENT filesystem from
`/mnt/eps-data` worktrees, so `cp -al` to /tmp fails with cross-device
link — put the scratch under `/mnt/eps-data/$USER/...`; (2) remove the
scratch in the same turn (rm -rf only unlinks the extra links).

**How to apply:** any lean round whose brief forbids running a phase on
the real dir while a live pid owns it, but the phase itself is
read-mostly + atomic-rename-writing. Related:
[[lean-session-waits-and-tmp-collisions]].
