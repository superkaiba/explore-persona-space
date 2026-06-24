---
name: Edit/Write land at repo-root mirror, not the issue-N worktree
description: In a worktree session the Edit/Write tools can materialize files at the repo-root tree (on main) instead of the worktree path; verify on-disk with grep before committing, copy into the worktree, restore repo root to main.
type: feedback
---

In an `issue-<N>` worktree session, the Edit/Write tools can resolve a
worktree absolute path to the **repo-root mirror** (`/home/.../explore-persona-space`,
which is on `main`) instead of the worktree
(`.claude/worktrees/issue-<N>`). The tool reports success and the
file-state-current confirmations look right, so all in-session lints /
tests / smokes PASS — but they ran against the repo-root copy, and the
WORKTREE (what gets merged for the issue) still has the OLD code. Worse:
the edits sit uncommitted on the shared `main` tree.

**Why:** the harness session cwd / file-tracking maps the edits to the
repo-root tree. Same family as the prior memory "Write tool lands in
session cwd" — but here it's silent across BOTH Edit and Write and the
landing target is the dangerous shared `main` root.

**How to apply:**
- Before `git add`/commit, ALWAYS verify the edits are physically in the
  worktree, not the repo root:
  ```bash
  grep -c "<a-symbol-i-just-added>" <worktree>/path/to/file.py   # must be > 0
  grep -c "<a-symbol-i-just-added>" <repo-root>/path/to/file.py  # if > 0, edits leaked to main
  ```
- If they leaked to repo root: `cp` each modified/new file into the
  worktree (`cp <repo-root>/<f> <worktree>/<f>`), then restore ONLY your
  files at the repo root to main (`git -C <repo-root> checkout HEAD -- <tracked files>`;
  `rm` your new untracked files from the repo root). NEVER touch
  concurrent sessions' dirty files (`pods_ephemeral.json`, agent-memory
  edits) at the repo root.
- The editable install resolves `explore_persona_space` to the WORKTREE
  `src/` when cwd is the worktree, so re-running pytest + the dispatcher
  `--verify-imports` + the CPU smoke FROM THE WORKTREE cwd is the
  authoritative re-validation after the copy.
- Incident: #653 v8 round (2026-06-24) — all 5 files (4 modified + 1 new
  test) landed at the repo root on `main`; caught only because
  `git status` in the worktree showed only the pre-existing dirty file,
  not my edits. ~10 min recovery (copy-into-worktree + restore-repo-root).
