---
name: commit-state-isolated-test-run
description: Split-review recipe — run a scoped commit's own tests against its own file state when later commits extend the same files (temp-dir extraction, dep-diff precheck, ruff path caveat)
metadata:
  type: feedback
---

When a split-review commit's files are EXTENDED by a later in-round commit
(HEAD ≠ commit state), running the suite at HEAD certifies the extension, not
the scoped commit. Recipe (#2587 r1 g3): (1) verify the commit's dependency
modules are unchanged `git diff <sha>..HEAD --stat -- <dep paths>` (empty ⇒
HEAD venv/src is faithful); (2) `git show <sha>:<path>` the scoped script +
test into `/tmp/<slug>/{scripts,tests}/` PLUS every transitive scripts-dir
import (chase `ModuleNotFoundError` iteratively — issue-common files import
each other); (3) `uv run pytest /tmp/.../tests/...` from the worktree.

**Why:** the only honest test of "the commit as it leaves the file" when the
same file gains 1,000+ lines in a sibling group's commit.

**How to apply:** any SPLIT-REVIEW SUB-SCOPE brief noting "same file extended
by commit X". Caveat: `ruff check` on the /tmp copies loses pyproject
`per-file-ignores` (path patterns don't match /tmp) — spurious style hits;
re-run ruff on the real repo paths before reporting a lint finding. Clean up
the /tmp tree after.
