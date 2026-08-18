---
name: never-edit-body-in-place-stash-race
description: Editing tasks/<status>/<N>/body.md in place at the shared root can be silently reverted by the pre-commit stash race before set-body reads it; build the body off-root and verify landing by commit SHA.
metadata:
  type: feedback
---

Never Edit/patch a task's `body.md` in place at the shared repo root. Build the
edited body in a FRESH off-root file (/tmp or .claude/cache), `diff` it against
the live body to confirm exactly the intended changes, hand THAT file to
`task.py set-body`, then verify landing by git evidence — `git log -1
--format=%H -- tasks/<status>/<N>/body.md` shows a NEW commit and `git show
<sha>:<path> | grep <new sentence>` finds the edit. A clean `verify_task_body`
PASS + "ok" from set-body is NOT landing evidence.

**Why:** #2333 micro-round 4 (2026-08-18) — two Edit-tool changes to the live
body.md sat tracked-modified-uncommitted for ~2 min; a concurrent fleet
commit's pre-commit stash cycle (`git checkout -- .`, the #2015 race) reverted
them; `set-body` then read the already-reverted file and no-op'd (identical
bytes ⇒ no new commit). The posted v4 marker asserted edits + PASS for a body
state that never existed; the critic found the body byte-identical to the
prior commit and a full correction round (v5) was needed.

**How to apply:** every analyzer body write — first drafts, revision rounds,
re-folds. The off-root-draft + set-body flow was already the Step 4/6 shape
for full bodies; this extends it to SMALL targeted edits, where in-place Edit
is tempting. After set-body, also check stderr for the deferred-commit ERROR
(rc=0 with a deferred commit is possible; never claim landed on it). Related:
[[commit-figures-before-post]].
