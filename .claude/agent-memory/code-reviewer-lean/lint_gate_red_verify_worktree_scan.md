---
name: lint-gate-red-verify-worktree-scan
description: Before flagging a committed file as workflow_lint gate-red, run that check with its scan root pinned to the WORKTREE; checker docstrings can overstate match shapes (#2388 R1 g5)
metadata:
  type: feedback
---

Before flagging a round-committed file as red on a bundled `workflow_lint.py`
check, settle it EMPIRICALLY with the scan root pinned to the worktree — e.g.
`import workflow_lint as wl; wl.check_upload_file_in_loop(scripts_dir=Path("scripts"))`
from the worktree cwd — never from the checker's docstring alone.

**Why:** two traps stacked in #2388 R1 g5. (1) The flag-form CLI run
(`uv run python scripts/workflow_lint.py --check-upload-file-in-loop`) can
resolve its scan root to the MAIN repo checkout, silently scanning a tree that
does not contain the branch's new file — a PASS there proves nothing about the
worktree file. (2) The checker's docstring promised shape-B matching of
`_upload(..., upload_as_file=True)` in a loop, but the attribute-form
`hub._upload(tar, ..., upload_as_file=True)` inside `for benchmark in ...`
scanned CLEAN even with the scan root pinned to the worktree (0 findings) — a
docstring-derived "gate red" flag would have been a false positive costing a
re-roll round.

**How to apply:** any time a verdict wants to claim "this commit trips lint
check X": run X directly with an explicit scan-root kwarg (or cwd-relative
path) inside the worktree, read the findings list, and only then write the
finding. Companion probe for ruff: certify lint-INTRODUCTION with
`uv run ruff check --stdin-filename <repo-rel path> < <(git show <sha>^:<path>)`
— parent blob clean + HEAD red = commit-introduced (the C901 complexity class
fires exactly this way when a commit adds a branch chain to an already-large
function). Related: [[untracked-twin-add-certification]].
