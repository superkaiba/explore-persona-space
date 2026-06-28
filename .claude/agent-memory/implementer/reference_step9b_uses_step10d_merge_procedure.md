---
name: Step 9b experiment-merge IS the Step 10d procedure
description: Step 9b auto-merge for experiments reuses the Step 10d procedure (gh pr merge --rebase + guards + artifact-confirmed defer); git merge --no-ff is NOT used there
type: reference
---

The `/issue` experiment terminal-point auto-merge ("Step 9b") does NOT use
`git merge --no-ff issue-<N>` — that string appears NOWHERE in SKILL.md.
Step 9b at `awaiting_promotion` literally says "run the **Step 10d
auto-merge procedure**" (SKILL.md ~line 4981). Both the experiment path
(Step 9b) and the code path (Step 10d) share ONE merge procedure at
SKILL.md ~6033+.

That procedure already has a full safe/unsafe decision tree:
- **Guards 1-3** (foreign-tasks / status / branch-content+non-mainline).
  Guard 3 trips when `ON_MAINLINE=no` OR own-commits (`origin/main...HEAD`
  three-dot) touch foreign-tasks/out-of-scope paths. `BEHIND` alone is
  never the trip (every task.py marker is a commit; same-day branches read
  BEHIND in the thousands — #537 read BEHIND=17057/8019).
- **Safe case** (guard 3 clean): `gh pr merge --rebase`; on real conflict
  → **merge-conflict recovery** (resolve IN THE WORKTREE, never the shared
  root; re-run tests; re-check mergeability; retry) → else `epm:merge-failed`.
- **Unsafe case** (guard 3 tripped): **artifact-confirmed merge** = the
  canonical defer-with-disclosure path. Posts `epm:merged {artifact_confirmed:
  true, full_rebase_deferred: true, reason, verified_paths}` (or `surgical_checkout:
  true, files: [...]`). This IS "the body is durable, defer the rebase."
- **New-shared-`src/`-infra guard** (added 2026-06-13, commit 2907b5cd12):
  runs FIRST in the unsafe case; if the branch ADDED new `src/explore_persona_space/`
  modules, it REFUSES the artifact-confirmed/surgical degrade (which can't
  carry shared src/) and routes to full-rebase or `epm:merge-failed`. This
  exists to PREVENT silently stranding new shared src/ (incident #595).

**Why this matters for candidates:** a "Step 9b has no conflict/defer
fallback" candidate is almost always already-handled. Worse, a proposed
"defer if body links resolve on main" (`body_durable: true`) fix is the
EXACT anti-pattern the new-src guard forbids — deferring strands any new
shared src/ the branch added. The `epm:merged {full_rebase_deferred: true}`
marker already serves the disclosure role; a new `epm:worktree-merge-deferred`
marker would be a redundant duplicate.

`git merge --no-ff` DOES appear in `.claude/rules/workflow-fix-on-bug.md`
— but that's the ORCHESTRATOR's own workflow-fix merge (a different merge
site), not Step 9b. Don't conflate the two.
