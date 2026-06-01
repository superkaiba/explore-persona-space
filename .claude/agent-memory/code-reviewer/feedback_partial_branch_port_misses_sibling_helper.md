---
name: partial-branch-port-misses-sibling-helper
description: FAIL re-run diffs that port a recipe/callback from a branch but miss a load-bearing SIBLING helper in the same file (e.g. _resolve_duration_kwargs alongside SaveAtSpecificSteps); max_steps silently ignored → zero-step training.
metadata:
  type: feedback
---

When an experiment re-runs a parent recipe whose scripts live ONLY on the
parent branch (e.g. `issue-432@6c562eb`, never merged to `main`), the
implementer ports the obvious files but can miss a load-bearing SIBLING
helper in the SAME source file at the SAME commit.

**Why:** task #456 (round 1) ported `SaveAtSpecificSteps` (checkpoint-saver)
from `train/trainer.py@6c562eb` but NOT its sibling `_resolve_duration_kwargs`
(same file, ~30 lines above). `_resolve_duration_kwargs` threads
`cfg.training.max_steps` into `SFTConfig`; without it, `++training.max_steps=1600`
is silently ignored, `num_train_epochs=-1` reaches HF, and the model trains
ZERO steps (no 22 checkpoints → entire experiment broken). The helper even
RAISED on the `epochs<=0, max_steps<=0` combo, so the cheap fail-fast guard
was also lost; failure would only surface after a pod + 7B model load.

**How to apply:** on any re-run/port diff, for EACH ported symbol, `git show
<parent_commit>:<file>` and check what ELSE in that file the ported symbol's
call site depends on. Specifically:
- If the pipeline passes `++training.max_steps=K` and/or `++training.epochs=-1`,
  GREP the on-HEAD `SFTConfig(...)`/`DPOConfig(...)` construction for `max_steps`.
  If `max_steps` is NOT passed (only `num_train_epochs=training.epochs`), it's a
  silent no-op → Critical. Confirm via `git show main:...trainer.py | grep -c
  _resolve_duration_kwargs` (0 = never on main = port required).
- Verify ported config inheritance with `git diff <commit>:<config> HEAD:<config>`
  (empty diff = truly inherited). #456's `configs/lora/default.yaml` WAS clean.
- General rule: a partial port is the #1 correctness risk on re-run tasks whose
  recipe is branch-only. The diff "looks complete" (all named files present) but
  a helper one function away is missing.

Related: [[feedback_eval_rig_per_phase_checkpoint]] (another #432/#385-family
trainer footgun).
