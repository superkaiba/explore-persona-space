---
name: plan-disclosure-edit-needs-new-version
description: A brief-sanctioned "edit the plan" disclosure update must land as a NEW plan version (task.py new-plan-version), never an in-place plans/vK.md edit — #2123 immutability Arm W FAILs the no-flags gate on any M to a persisted plan version, probing the REPO ROOT even from worktrees
metadata:
  type: feedback
---

A brief saying "the plan edit is sanctioned as a disclosure update" does NOT license an
in-place edit of `tasks/**/plans/v<K>.md`. **Why:** `workflow_lint.py
--check-plan-version-immutability` Arm W (#2123, bundled into the no-flags default run = the
Step 9c gate) FAILs on `M`/`D` in either porcelain column for the plans pathspec, and it
probes the REPO ROOT even when invoked from a worktree — so an in-flight in-place edit can
fail a CONCURRENT session's lint leg, and a committed one violates the Arm H history
contract permanently. **How to apply:** compose the full v(K+1) = vK carried forward
verbatim + the disclosure insertion, land it via `task.py new-plan-version <N> --file ...`
(self-contained, so the #2255 thin-amendment refusal passes; pure-add = Arm W exemption (i),
safe beside a live lint leg; plan.md symlink re-points automatically). Read "no
plan-amendment note needed" as "no separate justification marker", not "edit in place".
Worked example: #2336 round 3 (plan v4, commit f847c6b225). Related: [[worktree-commit-use-git-dash-c]].
