---
name: shared-worktree-partial-stage-commit
description: In a multi-round shared worktree, `git commit -- <path>` commits WORKING-TREE content and sweeps a live sibling's uncommitted edits; stage only your hunks via `git apply --cached` then bare-commit the verified index
metadata:
  type: feedback
---

When a shared file in a concurrent-fix-round worktree carries BOTH your edits
and a live sibling's uncommitted edits, the pathspec commit form
(`git commit -m ... -- <path>`) does NOT commit the index for that path — it
commits the CURRENT WORKING-TREE content, sweeping the sibling's in-progress
work into your commit.

**Why:** git's documented pathspec-commit semantics; hit on #2658 group-J
(2026-09-02) where tests/test_issue2658_unit12.py held my fixture updates plus
a live figure-round's two unfinished (lint-red) tests.

**How to apply:** (1) build a patch of only your hunks (`git diff -- <path>`,
cut at `^@@` boundaries), `git apply --cached <patch>`; (2) stage your
wholly-owned files normally; (3) verify the index holds ONLY your entries
(`git status --porcelain`, column-1 letters); (4) bare `git commit -F <msg>`
— the CLAUDE.md pathspec rule exists to avoid sweeping others' STAGED files,
which the verification step discharges; the pathspec form would violate the
rule's intent here. Post-commit, confirm the sibling's edits survive unstaged
(`git status --porcelain -- <path>` still shows ` M`) and the committed blob
excludes them (`git show HEAD:<path> | grep -c <their marker>`). Related:
[[worktree-commit-and-selector-vintage]].
