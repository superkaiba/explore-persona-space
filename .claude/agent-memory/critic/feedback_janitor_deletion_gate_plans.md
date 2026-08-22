---
name: janitor-deletion-gate-plans
description: Review checklist for evidence-gated /tmp-scratch deletion plans (#2127 v2) — live-process /proc probe, fallback-rmtree-on-git-refusal, pytest-timeout-less hang fixtures, hardlink/marginal-byte verification recipes
metadata:
  type: feedback
---

Reviewing janitor plans that DELETE filesystem trees under an evidence gate (#2127 v1→v2):

**Probes that reproduced the planner's claims exactly (reuse these):**
- Marginal-vs-apparent bytes: walk `lstat`s, count per-inode occurrences, `marginal = Σ size where in-tree occurrence count == st_nlink`. #2127's four figures (8.84/0.08/9.88/10.14 GiB) reproduced exactly; `mkstest-main` was 0.08 marginal vs 9.67 apparent (uv-hardlinked venv, the #596 class).
- Shape-glob sim: `fnmatch` over live `/tmp` top level, uid-owned dirs only — 383 matches reproduced; assert denylist targets (`claude-1001`, `pytest-of-*`) are BOTH shape-non-match AND denylisted.
- Worktree class: `.git`-FILE (`gitdir: .../.git/worktrees/<name>`) vs `.git`-DIR discriminates linked worktrees from clones; stash is SHARED (refs/stash, survives worktree removal — dropping the stash probe for worktrees is correct); `git status --porcelain` in a worktree reads its OWN index (staged-never-committed shows up); tracked-checkout content blob-verifies against the shared odb by construction; check `git ls-files -o -i --exclude-standard` to see gitignored content status can't show.

**Holes to check in v2-style fixes (why: each was found or nearly-missed here):**
1. **Live-silent-process residual.** Scoping the atime keep-signal to `st_nlink==1` removes the accidental protection hardlink-shared venv atimes gave live processes. A >48h-running process that wrote nothing in-tree and re-read no nlink-1 file in 48h is invisible to BOTH recency keys → reap-eligible while live. Cheap closing gate: scan `/proc/[pid]/{cwd,exe,fd/*}` readlinks for a candidate-prefix hit (measured ~2 s for ~2,000 procs) → keep with reason. Demand it be named as a residual or added as a gate.
2. **Fallback-rmtree converts git's protective refusal into deletion.** `git worktree remove --force` fails on a lock acquired post-gate (locked removal needs `--force` twice) — a blind `shutil.rmtree` fallback then deletes the just-locked tree. Fix: on remove-failure, re-check the `locked` file / keep-with-reason, never blind-rmtree.
3. **`git worktree prune` is global.** The fallback prune sweeps ALL ~95 registrations; a peer worktree on a temporarily-missing mount (`/mnt/eps-data/tmp/step9c-scratch-*`) gets unregistered (repairable via `git worktree repair`, no data loss — Concern not blocker).
4. **Hang-as-failure fixtures need an explicit timeout mechanism.** `pytest-timeout` is NOT in this repo's deps; a FIFO fixture whose deleted-guard failure mode is "hangs into its pytest timeout" would hang the Step-9c suite instead of FAILing. Demand thread-join/`signal.alarm`/subprocess-timeout in the test, or the dep added.
5. **Empty `.gitmodules` + committed gitlink** (`external/open-instruct` here): `git submodule status` errors; empty gitlink dirs are invisible to status and hold no files — populated foreign-repo content fails blob-verify (safe direction). `external/*` other dirs are TRACKED regular files, not submodules — verify with `git ls-files -s`.

**How to apply:** any plan reaping under recency + evidence gates gets: the /proc live-owner probe question, the git-refusal-fallback question, the fixture-timeout-mechanism question, and independent reproduction of every measured byte/count figure (they reproduce in minutes and ground the whole verdict).
