---
name: Scope — Infrastructure Only
description: Hard line between what implementer owns vs what experimenter owns
type: project
---

Implementer owns: refactors, bug fixes, new utilities (src/explore_persona_space/**), config reorganizations, build / sync / pod-management scripts (scripts/pod.py, scripts/sync_*.py), CLAUDE.md-adjacent rules, preflight + orchestration code.

Experimenter owns: new training scripts for a specific research condition, data-generation scripts for a specific run, Hydra configs in configs/condition/, eval pipelines tied to a particular eval.

**Why:** Keeps review loads separate. A shared-infra bug affects every future experiment; experiment-specific code affects one run. Different risk profiles need different review strictness.

**How to apply:**
- If the task touches files used by >1 experiment, it's implementer territory. Run full code-reviewer loop.
- If the task is scoped to one condition or one eval, defer to experimenter (in subagent mode, flag back to research-pm and decline; in main agent mode, tell the user to use experimenter).
- Edge case: a change to `src/explore_persona_space/train/trainer.py` is always implementer, even if motivated by one experiment, because the monkey-patched Trainer is load-bearing for all training.
