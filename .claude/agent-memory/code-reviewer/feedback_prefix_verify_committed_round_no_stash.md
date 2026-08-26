---
name: prefix-verify-committed-round-no-stash
description: Re-verifying pre-fix test failure on a COMMITTED round — git stash is a no-op on a clean tree and a pop applies a FOREIGN autostash; use checkout-base-blob instead
metadata:
  type: feedback
---

When a brief says "verify the tests fail pre-fix via in-place stash round-trip",
that recipe is only valid while the fix is UNCOMMITTED (the implementer's state).
At review time the fix is committed and the tree is clean, so:

1. `git stash push -- <file>` stashes NOTHING (silently, rc=0 under `-q`) — the
   tests then run against the post-fix HEAD blob and "pass", which looks like a
   fabricated pre-fix claim when it is actually your own no-op stash.
2. The paired `git stash pop` then pops a FOREIGN pre-existing entry — issue
   worktrees carry rescued autostashes from repo-root sync recoveries
   (#2546 r16: stash@{0..4} incl. "rescued autostash from stale rebase-merge
   husk") — and can CONFLICT (UU rows), dirtying the worktree with another
   session's content while the entry is "kept".

**Why:** hit verbatim on #2546 r16 (2026-08-26): the pop applied a foreign
autostash (UU tasks/REGISTRY.json + 2 staged files); cleanup = `git -C WT reset`
+ `git -C WT checkout HEAD -- <paths>` + rm the popped-in untracked file, leaving
the foreign stash entry untouched.

**How to apply:** on a committed round, verify pre-fix failure with the base
blob instead: `git -C <WT> checkout <base-sha> -- <changed file>` → run the
tests → `git -C <WT> checkout HEAD -- <file>` → confirm `status --porcelain`
clean. Never `git stash pop` unless YOUR push verifiably created an entry
(check `git stash list` before AND after the push). Related: [[plan-first]].
