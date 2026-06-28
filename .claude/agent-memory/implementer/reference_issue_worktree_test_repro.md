---
name: Issue-worktree cwd-failure repro recipe
description: How to reproduce/verify test failures that only occur when pytest runs inside a .claude/worktrees/issue-<N> path (the /issue Step 9c gate environment)
type: reference
---

To reproduce a "tests fail only inside an issue worktree" report (the watcher
passes infer `issue=<N>` from any session path matching
`spawn_session._WORKTREE_ISSUE_RE = r"/\.claude/worktrees/issue-(\d+)/?$"`),
build a scratch detached worktree whose PATH matches the regex — no real
issue branch needed:

```bash
mkdir -p /tmp/eps-repro/.claude/worktrees
git worktree add --detach /tmp/eps-repro/.claude/worktrees/issue-9999 <sha>
cd /tmp/eps-repro/.claude/worktrees/issue-9999
/home/thomasjiralerspong/explore-persona-space/.venv/bin/python -m pytest tests/<file> -q
# cleanup: git worktree remove --force <path>; rm -rf /tmp/eps-repro
```

Notes: checkout is ~5GB (sparse-cone `!` exclusions don't apply on plain
worktree add — acceptable for a scratch repro); use the MAIN checkout's
`.venv/bin/python` (no worktree venv build); pre-fix run reproduces, re-run
at the fix commit verifies. Used for the #580 fix (2026-06-12): `_Z_ROOT`
derived from the real `spawn_session.PROJECT_ROOT` collided with the
issue-inference regex — fixed by pinning `asw.PROJECT_ROOT` to a synthetic
issue-pattern-free root in the pass-level patch helpers.
