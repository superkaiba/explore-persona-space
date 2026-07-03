---
name: Worktree edits need ABSOLUTE worktree paths
description: In subagent mode with a worktree brief, bare src/... paths in Edit/Read/Grep resolve to the repo root (main), NOT the worktree — always use the absolute worktree path
type: feedback
---

When a subagent brief gives a worktree (`.claude/worktrees/issue-<N>`), the
Edit / Read / Grep / Glob tools resolve a BARE relative-looking path like
`src/explore_persona_space/backends/router.py` (and even an absolute
`/home/.../explore-persona-space/src/...`) to the **repo root**, which is
pinned to `main` — NOT the worktree. Editing there strands the work on `main`
and a `uv run` from the worktree imports the unmodified file.

**Rule:** in worktree-bound subagent mode, prefix EVERY Edit/Read/Grep
`file_path`/`path` with the full worktree root, e.g.
`/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-656/src/...`.
Verify once at the start: after the first edit, run
`git -C <repo-root> status --short <file>` — if the repo root shows the
modification, the edit landed in the WRONG tree.

**Why:** CLAUDE.md "NEVER git checkout a branch in the repo-root tree — keep
it on main; do all branch work in a worktree." Editing the repo-root src
files directly is the same hazard by another route: a concurrent
`git add && git commit` in the shared root could sweep the stray edits onto
the wrong branch.

**How to apply:** Recovery when it happens (incident: task #656, 2026-06-17 —
all of gcp.py/router.py/__init__.py landed in the repo root): capture the
repo-root diff (`git diff -- <files> > /tmp/x.patch`), `git apply` it inside
the worktree (the worktree files are still pristine `main`, so it applies
clean), then `git checkout -- <files>` in the repo root to restore it. Then
redo all subsequent edits against the worktree path.
