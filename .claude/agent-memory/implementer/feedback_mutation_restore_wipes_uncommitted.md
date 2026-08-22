---
name: mutation-restore-wipes-uncommitted
description: Never restore a sed/mutation-tested file with `git checkout --` while OTHER uncommitted edits live in the same file — it resets to HEAD and wipes them; revert by inverse sed or commit first
metadata:
  type: feedback
---

When mutation-testing a file that ALSO carries your uncommitted edit (e.g. sed-flip
`reverse=True` → `reverse=False` in a script you just added a guard to), do NOT restore
with `git checkout -- <file>`: it resets to HEAD and silently wipes the uncommitted edit
along with the mutation.

**Why:** 2026-06-11 pm_queue_report run — the parser.error guard (uncommitted) was wiped
by the post-mutation `git checkout --` and had to be re-applied; the empty `git diff --stat`
was the only tell. The round-2 [[no-agent-tool-in-spawn]] reviewer brief now warns its
reviewer of the same trap.

**How to apply:** Either (a) revert the mutation by the inverse sed (the restore is then
exact and edit-preserving), or (b) commit your verified edits BEFORE any mutation pass,
then `git checkout --` is safe. Verify restoration with `git diff --stat` showing the
EXPECTED pending diff, not an empty one.

**Pre-mutation gate (added 2026-08-12 — tripped AGAIN on #2127 r2 despite this memory):**
knowing the rule post-hoc does not fire it in the moment. Make it mechanical: BEFORE any
guard-deletion/mutation round, run `git -C "$WT" status --porcelain -- <target file>`; if
the target is dirty, COMMIT FIRST (option b) — always, no judgment call. The #2127 tell was
the test suite regressing to pre-fix behavior right after a "clean" restore: if tests that
passed pre-mutation fail post-restore, suspect this wipe before debugging the code.
