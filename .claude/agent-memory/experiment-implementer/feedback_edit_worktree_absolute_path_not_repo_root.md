---
name: Edit the worktree absolute path, never the bare repo-root path
description: Edit/Write/Read MUST target the .claude/worktrees/issue-<N>/ absolute path; the bare /home/.../explore-persona-space/scripts/... path is the SHARED REPO ROOT, a different file
type: feedback
---

When invoked in subagent mode my cwd is the worktree
(`/home/.../explore-persona-space/.claude/worktrees/issue-<N>`), but the
worktree and the repo root are TWO SEPARATE files on disk for the same
relative path. `Edit`/`Write` need an ABSOLUTE path, and the obvious-looking
`/home/thomasjiralerspong/explore-persona-space/scripts/...` is the REPO ROOT
(on `main`, shared with every concurrent session) — NOT the worktree. Editing
that path silently lands all my work in the wrong tree: it never gets
committed to `issue-<N>`, and a concurrent session's `git add`/commit can sweep
it onto the wrong branch.

**Why:** task #642 round 2 (2026-06-16) — I made 9 Edits to
`/home/.../explore-persona-space/scripts/issue_642/i642_dispatch.py` (repo
root) believing I was editing the worktree. `grep` in the worktree returned 0
hits; the repo root showed `M` on `main`. Recovered only because the
worktree's baseline of that file was byte-identical to repo-root `main` HEAD,
so I could `cp` the edited file → worktree and `git checkout --` the repo-root
file. A divergent baseline would have lost work or corrupted the diff.

**Recurred — round 3 (2026-06-16), WORSE:** despite this memory, I both
(a) used repo-root absolute `file_path`s for every `Edit` AND (b) prefixed
every `Bash` git with `cd /home/.../explore-persona-space && git ...` (=
repo root). All 7 v8 commits landed on `main`, interleaved with concurrent
`task.py` marker commits, and a concurrent session PUSHED them to
`origin/main` before I noticed. The "absolute path" instinct was the trap:
I gave absolute paths, but REPO-ROOT absolute paths. Recovery was clean
because the commits were isolated (`scripts/issue_642/*` + `tests/` only):
`git -C <worktree> cherry-pick <7 shas oldest-first>` re-applied them to the
`issue-642` branch with auto-merge (no conflicts). Did NOT touch shared main
(can't safely reset a pushed shared branch); flagged the stray-main-commits
in the report `(d)` for the orchestrator's Step 9b merge to reconcile.

**How to apply:**
- EVERY `Edit`/`Write`/`Read` `file_path` for worktree-bound work uses the FULL
  worktree prefix: `/home/.../explore-persona-space/.claude/worktrees/issue-<N>/<rel>`.
  Never the bare `/home/.../explore-persona-space/<rel>` (= repo root).
- This is the sibling of the `cd repo-root → commits land on main` trap: same
  root cause (the repo root is the shared `main` tree; all my work belongs in
  the worktree), different surface (path argument vs `cd`).
- Quick self-check before the FIRST edit of a session: `grep`/`Read` the target
  in BOTH paths once and confirm the worktree path is the one I'm editing. If a
  later `grep` in the worktree shows 0 hits for an edit I "made", I edited the
  repo root — recover by `cp <repo-root-file> <worktree-file>` then
  `git -C <repo-root> checkout -- <rel>`.
- Verify `git rev-parse --abbrev-ref HEAD` == `issue-<N>` BEFORE the FIRST
  commit, from a NO-`cd` Bash call (cwd is the worktree). If it prints `main`
  you are about to commit to shared main — STOP. This is the cheapest guard
  and it catches BOTH halves at once.
- NEVER prefix a `Bash` git command with `cd /home/.../explore-persona-space`
  (no `.claude/worktrees/...`) — that compound runs git in the repo root for
  the WHOLE command. Run git with no `cd` (cwd is the worktree) or use
  `git -C <worktree-abs-path> ...`.
- If commits already landed on main and were pushed: `git -C <worktree>
  cherry-pick <sha>...` onto `issue-<N>` (works cleanly for isolated
  experiment files), then flag the stray main commits in report `(d)` — do
  NOT `git reset`/`revert` a pushed shared branch yourself.
