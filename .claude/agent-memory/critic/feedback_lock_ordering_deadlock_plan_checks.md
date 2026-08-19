---
name: lock-ordering-deadlock-plan-checks
description: Reviewing a lock-ordering/deadlock infra fix plan — three verifications: hoist site is the ONLY unbounded acquisition of the contended lock; the sanctioned recovery tool has no self-wait circularity through the blocked resolver; a "knob already exists" option-rejection is read from the env-getter + its caller, not the body's claim
metadata:
  type: feedback
---

For an infra plan fixing a "writers hold lock L while waiting on condition C;
the only clearer of C needs L" deadlock (the #2295 shape — task.py held
`~/.task-workflow/lock` through the 120 s rebase-husk wait while
`sync_repo_root.py` needed it to `rebase --abort`), three checks decide the
review:

1. **Hoist-site completeness.** The chosen fix (resolve/wait BEFORE acquiring
   L) closes the CLASS only if the hoist site is the single unbounded
   acquisition of L. Grep every `fcntl.flock`/`LOCK_EX` on the lock file:
   in #2295 `_locked()` (task_workflow.py:806) was the only unbounded
   LOCK_EX; the helper's own `acquire_task_workflow_lock` is LOCK_NB +
   deadline (fails safe), so hoisting at `_locked()` covered every holder.
2. **Recovery-tool self-wait circularity.** When the plan points blocked
   writers at a recovery tool (timeout-message edit), check the tool's own
   default-argument resolution path does NOT route through the same waiting
   resolver — `sync_repo_root.py` defaults `--repo` to
   `task_workflow.primary_checkout_root()`, which deliberately has NO branch
   guard / husk wait (task_workflow.py:743-751); had it called `repo_root()`
   the recommended recovery would self-wait/raise before reaching preflight.
3. **"Knob already exists" rejections are read from code.** #2295's body
   asserted "no CLI flag and no environment variable" for the helper's
   task-lock wait; the knob existed (`EPM_ROOT_SYNC_LOCK2_WAIT_S`,
   sync_repo_root.py:246-248 → `_run_locked` L1952) with a test. Verify the
   env-getter AND its caller before crediting either the body or the plan.

**Why:** all three are cheap greps that decide whether the fix closes the
class vs one instance, and whether the body's option framing is factually
sound — in #2295 the plan was right and the BODY was wrong on option 3.

**How to apply:** any wf-fix plan touching `task_workflow.py` locking,
`sync_repo_root.py`, or a hold-while-wait deadlock; pairs with
[[infra-plan-review-checklist]].
