---
name: epochs-negative-one-zero-steps
description: training.epochs=-1 + max_steps>0 silently runs ZERO steps — SFTConfig(num_train_epochs=-1) means 0 epochs and max_steps does not rescue the negative path. Use epochs=1 with max_steps.
metadata:
  type: feedback
---

`training.epochs=-1` paired with `+training.max_steps=N` (intending "exactly N steps") silently runs ZERO iterations: Hydra passes `-1` into `SFTConfig(num_train_epochs=-1)`, HF Trainer reads it as 0 epochs, and `max_steps` only overrides POSITIVE epoch counts.

**Signature (instant recognition):** `0it [00:00, ?it/s]`, NEGATIVE `train_samples_per_second`, ~17ms train_runtime, zero loss, skip straight to "Merging adapter", `*_step_checkpoints/` created but empty.

**Why:** #385 round-4 (2026-05-25) — smoke passed the format criterion but failed checkpoint criteria because of this; cost ~3 min pod time + a bounce round.

**How to apply:** on that signature, diagnose immediately (don't re-read format_dataset traces). Bounce `failure_class: code` with the fix: when `max_steps > 0`, set `num_train_epochs` to a huge positive (or assert against `epochs <= 0 and max_steps <= 0`). When YOU author launch commands, use `training.epochs=1` with max_steps, never `-1`. Related: [[trl-conversational-format-in-format-dataset]], [[feedback_hydra_per_key_additive_prefix]] — same "composes only at training start, survives smoke" family.
