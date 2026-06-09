---
name: Unmerged parent branch drift (TrainLoraConfig field churn)
description: When cherry-picking experiment code from an unmerged parent branch, the parent's TrainLoraConfig schema is OFTEN out of sync with main; diff sft.py up front + run a signature smoke before commit.
type: feedback
---

When porting a recipe from an unmerged parent branch (e.g. #464 → #529 cherry-pick from `origin/issue-464` SHA `0905fc70`), the parent's `TrainLoraConfig` schema can be SIGNIFICANTLY ahead of `main`. Concretely observed on the 2026-06-08 #529 implementation:

- `marker_logprob_trajectory: dict | None` — REMOVED on main after #464 branch was authored
- `marker_text: str | list[str]` (parent multi-marker support) → `marker_text: str` (single string on main)
- `marker_tail_tokens: int = 32` (parent default) → `marker_tail_tokens: int = 0` (main default)
- `MarkerOnlyDataCollator` constructor signature changed (`list[int]` only, no nested `list[list[int]]`)

Each one would silently crash at the first GPU launch with `TypeError: TrainLoraConfig.__init__() got an unexpected keyword argument 'marker_logprob_trajectory'` or similar.

**Why:** Per CLAUDE.md "Porting a recipe from an unmerged parent branch", the entire train+eval code path needs to be diff'd against the parent SHA. The cherry-pick brings scripts that import library functions; those functions can have been refactored on main.

**How to apply:** Before committing the cherry-picked rig + your extensions, run:

```bash
git diff <parent-sha> origin/main -- src/explore_persona_space/train/sft.py \
    src/explore_persona_space/train/callbacks.py \
    src/explore_persona_space/experiments/{train,eval}/
```

Then a one-line signature smoke per kwarg the dispatcher passes:

```python
from dataclasses import fields
from explore_persona_space.train.sft import TrainLoraConfig
cfg_fields = {f.name for f in fields(TrainLoraConfig)}
caller_passes = {'gpu_id', 'epochs', 'marker_logprob_trajectory', ...}
assert not (caller_passes - cfg_fields), f"missing: {caller_passes - cfg_fields}"
```

This catches the partial-port crash class at experiment-implementer time, before code-review or pod launch. Incident #529 (2026-06-08): the `marker_logprob_trajectory` field was retired on main; the unguarded `cfg = TrainLoraConfig(marker_logprob_trajectory=traj_cfg, ...)` call would have crashed at the first cell instantiation. Fix was to make the field conditional via `dataclasses.fields()` introspection and reject the no-longer-supported multi-marker path with a fail-loud message.
