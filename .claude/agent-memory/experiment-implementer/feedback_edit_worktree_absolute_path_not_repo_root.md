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
- Verify `git rev-parse --abbrev-ref HEAD` == `issue-<N>` before every commit
  (commits run from the worktree cwd, so this catches the branch half).
