---
name: Never git stash to compare against pristine HEAD in a shared-.git worktree
description: git stash / stash pop on the shared MooseFS .git (concurrent committers) can lose uncommitted work; prove pre-existing failures another way
type: feedback
---

Do NOT use `git stash` + `git stash pop` to temporarily revert your uncommitted
edits (e.g. "is this lint/test failure pre-existing on pristine HEAD?") in an
issue-<N> worktree. The `.git` stash stack is SHARED across every worktree +
every concurrent VM session, and a concurrent committer's stash push/pop can
race yours: the pop can restore the WRONG stash (or a stash that captured only
a concurrent session's file), silently DROPPING your own uncommitted hub.py /
test.py edits.

**Why:** Incident 2026-07-01, task #794 (this file's origin). I ran `git stash`
to re-run `workflow_lint.py` on pristine branch HEAD, then `git stash pop`. The
pop reported `Dropped refs/stash@{0} (2062a816...)` and restored ONLY a
concurrent session's `.claude/agents/experiment-implementer.md` change — my two
files' edits were GONE (grep for `def _list()` returned 0). Had to re-apply both
Edits from scratch. Ties to the existing memory
`feedback_mutation_restore_wipes_uncommitted` (never `git checkout --` to undo a
mutation in a file with uncommitted edits).

**How to apply:** To prove a lint/test failure is pre-existing (not introduced
by your diff), do it WITHOUT reverting your working tree:
- If the failing file is NOT in your diff (`git diff --name-only HEAD`), that
  alone proves your change didn't introduce it — state that + point at the
  failing path. (#794: the offender was `scripts/issue744_dump_and_stream.py`,
  never in my diff.)
- If you must see HEAD's version of a specific file, use `git show HEAD:<path>`
  (read-only, touches nothing) or check it out to a temp path
  (`git show HEAD:<path> > /tmp/pristine_x`), never `git stash`.
- To prove a test's pre/post-fix distinction, reason about it structurally
  (e.g. "the exhaustion test asserts call_count==6, impossible without the
  retry wrap") rather than reverting the fix.
