---
name: cert-blob-recovery-after-clobber
description: Uncommitted shared-root edits clobbered by a concurrent session are recoverable from the inline-lint-gate cert blob SHAs via git cat-file
metadata:
  type: reference
---

The inline payload lint gate (`scripts/inline_lint_gate.py`) certifies each
payload file's exact content by GIT BLOB SHA (printed as
`certified <path> (<sha>)` and recorded in /tmp/eps-inline-lint-cert-v1.txt),
and those blobs land in the object DB. If a concurrent session's destructive
working-tree op wipes your UNCOMMITTED edits on the shared repo root (it
happened mid-round on 2026-08-05: all four tracked files of a gated payload
reverted to HEAD; `git stash list` + root-sync-rescue had nothing), recover
with `git cat-file blob <cert-sha> > <path>` — after first checking
`git diff --stat HEAD:<path> <cert-sha>` shows exactly your hunks (if HEAD
moved on that file since certification, 3-way-apply your hunks instead of
blob-restoring).

**How to apply:** before hand-reconstructing lost work, probe (1)
`git stash list` autostashes, (2) `~/.task-workflow/root-sync-rescue/`,
(3) the gate cert SHAs via `git cat-file -e <sha>`. Commit as soon as the
gate passes — the uncommitted window on the shared root is the exposure.
