---
name: Write/Edit/Bash resolve against session cwd (repo root), not the worktree
description: New files written with a worktree absolute path can still land in the repo-root mirror path; verify on disk + commit ONLY from the worktree
type: feedback
---

When implementing inside `.claude/worktrees/issue-<N>`, the session CWD is the
REPO ROOT (`/home/thomasjiralerspong/explore-persona-space`), and a `Write` to a
new file under the worktree absolute path can still materialize at the
**repo-root mirror path** instead of the worktree (observed #653 round 1: a
`Write` to `.../worktrees/issue-653/src/.../issue_653/__init__.py` created the
file at `<repo-root>/src/.../issue_653/__init__.py` — the worktree dir stayed
empty). The repo root is on `main`, shared with concurrent committers, so a
stranded new file is at risk of being swept onto the wrong branch.

**Why:** the worktree dir often does not exist yet when the new subpackage is
first written; the tool's path resolution falls back to the session CWD's mirror
of the same relative path. (`mkdir -p <worktree>/...` in a prior Bash call did
NOT prevent it.)

**How to apply:**
- After EVERY `Write` of a NEW file under a worktree, immediately
  `ls -la <worktree-abs-path>` to confirm it landed there; if it landed in the
  repo root, `mv` it into the worktree and `rmdir` the stray root dir BEFORE any
  `git add` (the stray is untracked, so `main` is safe as long as you never
  `git add -A`).
- Run ALL `git add`/`git commit`/`ruff`/smoke from inside the worktree
  (`cd <worktree> && ...`), never from the repo root, so the explicit-path stage
  targets the worktree's index. `git rev-parse --abbrev-ref HEAD` must read
  `issue-<N>` before committing.
- `Edit`/`Read` on an EXISTING worktree file resolve correctly ONLY when you pass
  the FULL WORKTREE absolute path (`.../.claude/worktrees/issue-<N>/scripts/x.py`).
  Passing the repo-root absolute path (`<repo-root>/scripts/x.py`) silently edits
  the REPO-ROOT MIRROR instead — and the harness's file-state tracker + Read tool
  will happily show you that repo-root copy, so the edit "succeeds" while the
  worktree file is untouched (#778 round 1: 8 Edits + 5 Writes all landed in
  repo-root because I used repo-root abs paths; the worktree files stayed at the
  stale prior-round version, and `git status` in the worktree showed no change).
  ALWAYS spell out the `.claude/worktrees/issue-<N>/` prefix in every Edit/Write
  `file_path`, and periodically `grep -c <my-marker> <WT>/<file>` vs the repo-root
  copy to confirm which mirror has the change.
- Sibling of `feedback_cd_repo_root_commits_land_on_main.md` — that one is about
  `cd <repo-root> && git commit` committing to main; this one is about the Write
  tool placing the source file there in the first place.
