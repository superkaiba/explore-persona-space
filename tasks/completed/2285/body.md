---
title: 'workflow-fix: Step 10d Guard 2 asserts the pre-#1723 status ordering (says
  code paths are at completed; #1723 keeps them at running)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-14T05:04:51Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2281 Step 10d Guard 0-5 run: Guard 2 asserts the
  status is ''completed for code paths, flipped in Step 10 step 6 BEFORE this step'',
  which is the ordering #1723 removed — the live task read status: running, kind:
  infra.'
workflow: v1
---
# Step 10d Guard 2 asserts the pre-#1723 status ordering ("`completed` for code paths, flipped in Step 10 step 6 BEFORE this step")

## The defect

`.claude/skills/issue/SKILL.md` Step 10d, **Guard 2** (currently ~line 11506) reads:

> 2. **Status already off `running`.** By both trigger points the status is well
>    past `running` (`awaiting_promotion` for experiments; `completed` for code
>    paths, flipped in Step 10 step 6 BEFORE this step). A crash mid-merge
>    therefore cannot strand a terminated-pod task at `running`. On a later
>    `/issue <N>` resume: if the PR is already merged AND status is still
>    `running` for any reason, auto-advance rather than re-dispatching.

The parenthetical describes the ordering **#1723 removed**. Step 10 step 6 now says
the opposite for exactly this population:

> **If `epm:merged` is NOT yet present** (code-change path arriving via Step 9c
> PASS — `kind: infra | batch | analysis | survey`): do NOT tear down the cron
> here and do NOT apply the terminal status / `epm:done` here. Advance directly
> to Step 10d — its success path posts `epm:merged` and THEN, in its
> `#### Terminal teardown (code-change path only)` sub-section, fires
> CRON-TEARDOWN + `set-status completed` + `epm:done` in that order.

with the rationale stated in the same step:

> the previous ordering (teardown → `set-status completed` → `epm:done` → Step
> 10d) left the entire Step 10d merge window (up to ~33 min under fleet churn)
> with NO `/issue-tick` re-drive coverage AND with the durable record reading
> `completed`+`epm:done` on an unmerged branch — the `completed_unmerged_pass`
> (#1540, #1653) flag class.

So a `kind: infra | batch | analysis | survey` task is at **`running`** for the whole
of Step 10d, deliberately. Guard 2 asserts it is at `completed`, and attributes that
to a Step-10-step-6 action that step explicitly no longer performs.

## Why it matters (the misread this invites)

Guard 2 is a numbered GUARD in a block an orchestrator executes in order. An
orchestrator that reads it as a PRECONDITION rather than stale narration has two
bad moves available, and both re-open the gap #1723 closed:

1. Flip the status to `completed` before merging "because Guard 2 says it should
   already be there" — reproducing the `completed`+`epm:done`-on-an-unmerged-branch
   record and dropping `/issue-tick` coverage over the ~30-40 min merge window.
2. Treat `status: running` at Step 10d entry as a guard FAILURE and bounce /
   block, stalling a task whose state is correct.

Observed on **#2281** (this task's filer): the Step 10d run read `status: running`,
`kind: infra` at Guard 2 and had to reconcile the guard against #1723 inline before
proceeding. Nothing was blocked — the drift cost a reconciliation, not a round — but
the next reader may resolve it the other way.

Secondary, smaller inaccuracy in the same bullet: "cannot strand a **terminated-pod**
task at `running`" is experiment-path framing. A code-change path never provisions a
pod, so pod-stranding was never its risk; its actual crash-safety story under #1723 is
the armed `/issue-tick <N>` cron + Step 10d's own idempotent re-entry
(`epm:merged`-keyed) + the `completed_unmerged_pass` watcher flag.

## Suggested fix

Re-derive the bullet rather than swapping one word. Sketch:

- **Experiments:** unchanged — `awaiting_promotion` at the Step 9b trigger point.
- **Code paths (`infra` / `batch` / `analysis` / `survey`):** state that the status
  STAYS at `running` through Step 10d BY DESIGN (#1723), that the terminal flip fires
  from this step's `#### Terminal teardown (code-change path only)` sub-section AFTER
  `epm:merged`, and that this is what preserves `/issue-tick` re-drive coverage over
  the merge window.
- Re-base the crash-safety sentence per path: pod-stranding for the experiment path;
  armed cron + idempotent `epm:merged`-keyed re-entry + `completed_unmerged_pass` for
  the code path.
- Keep the final resume sentence as-is (still correct).

Worth a scan for the same stale parenthetical elsewhere in Step 10d / Step 9b prose
while in there, and worth considering whether the `--check-references`-style lint has
any purchase on this class (a guard bullet asserting a superseded ordering is not
currently mechanically detectable, so the honest answer may be "no — prose only").

## Provenance

Surfaced by the #2281 Step 10d run (`kind: infra`, worktree
`.claude/worktrees/issue-2281`) while executing the Guard 0-5 block: Guard 2's
asserted precondition was false against the live task state, and the contradiction
resolves in #1723's favour. No existing task covers it — the dedup scan over
`tasks/*/*/body.md` found 20 Step-10d-titled tasks (nearest neighbours: #2246 on the
mid-gate worktree reap, plus the Guard-1 three-dot and lint-vintage fixes) and none
touching the Guard-2 status assertion.

workflow_fix_target: .claude/skills/issue/SKILL.md
