---
name: hydra-per-key-additive-prefix
description: Hydra +foo=X vs foo=X rules depend per-key on whether key is in the default struct; never bulk-add or bulk-remove + prefix across a related key group
metadata:
  type: feedback
---

When a Hydra invocation fails with `Could not override 'training.X'. ... use +training.X=...  Key 'X' is not in struct` and the natural fix is "add the `+` prefix" — that fix applies ONLY to keys explicitly NOT in the resolved config struct. Sibling keys in the same dotted-path group (e.g. `training.learning_rate`, `training.max_steps`, `training.epochs`) may have DIFFERENT in-struct vs not-in-struct status depending on the config defaults.

**Why:** Task #416 launch attempt 1 (2026-05-28) failed at smoke_train Hydra parse with `Could not override 'training.learning_rate'`. The implementer's hotfix b9f68e80 bulk-REMOVED `+` from learning_rate AND max_steps AND epochs. Re-launch attempt 2 died at the same phase with `Could not override 'training.max_steps'` — max_steps was NOT in the struct and DID need the `+`. The bulk-fix moved the right direction for one key and the wrong direction for another, costing a full subagent turn.

**How to apply:**
- When you see this error in a launch tail, do NOT just bounce the implementer with "add +" or "remove +". Recommend they:
  1. `cat configs/training/*.yaml` (or whichever Hydra group is failing) to list keys actually in the default struct.
  2. Per-key decide: in-struct → plain `group.key=X` (override); not-in-struct → `+group.key=X` (append).
  3. Add a Hydra dry-run smoke test (`uv run python scripts/train.py --cfg job <overrides>` or equivalent) BEFORE nohup launch to catch this in <2s instead of via a failed nohup that has to bounce through experimenter.
- Tag with [[trl-conversational-format-in-format-dataset]] and [[epochs-negative-one-zero-steps]] as the third member of the "Hydra/config edge cases that surface only at training start and survive a smoke gate that doesn't actually compose the training config" family.
