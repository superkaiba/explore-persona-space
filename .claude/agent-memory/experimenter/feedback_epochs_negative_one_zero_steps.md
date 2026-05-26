---
name: epochs-negative-one-zero-steps
description: training.epochs=-1 + max_steps>0 yields ZERO training steps via SFTConfig(num_train_epochs=-1); HF Trainer's max_steps override doesn't rescue a negative epochs count.
metadata:
  type: feedback
---

When a plan or launch brief specifies `training.epochs=-1` paired with `+training.max_steps=N` (intending "run exactly N steps regardless of epochs"), the TRL/HF Trainer pathway in `src/explore_persona_space/train/trainer.py` silently runs ZERO steps.

**Signature in log (instant recognition):**
- `0it [00:00, ?it/s]` (zero iteration progress bar)
- `{'train_runtime': 0.0176, 'train_samples_per_second': -34120.375, 'train_steps_per_second': -2160.957, 'train_loss': 0.0, 'epoch': 0}` (NEGATIVE throughputs, sub-second runtime, zero loss)
- Trainer skips straight to "Merging adapter into base model"
- `*_step_checkpoints/` dir is created but empty (no `checkpoint-N/` subdirs)

**Why:** Hydra's `cfg.training.epochs: -1` flows verbatim into `SFTConfig(num_train_epochs=-1)`. HF Trainer interprets a negative epoch count as "0 epochs", running zero iterations. `max_steps` does NOT override this on the negative-epoch path; it only overrides positive epoch counts. The negative-throughput numbers in the train_runtime dict are the giveaway (division by ~0 runtime over a 0-sample run).

**Why:** Burned in task #385 round-4 launch (2026-05-25). Round-3 had just shipped the `format_dataset` conversational fix; smoke passed criterion-4 (format) but FAILED criteria 1+2 (no save callback fire, empty checkpoint dir) because of this. Cost ~3 minutes of pod time + a round-5 bounce.

**How to apply:**
- During smoke launch, if the log shows `0it [00:00, ?it/s]` and negative `train_samples_per_second`, immediately diagnose as "epochs<=0 + max_steps>0" — don't waste cycles re-reading the format_dataset traceback.
- Bounce to implementer (`failure_class: code`) with the clear fix path: in `train/trainer.py` when `max_steps > 0`, set `num_train_epochs = sys.maxsize` (or 999_999) so the trainer's max_steps guard fires; OR add an assert that rejects `epochs <= 0 AND max_steps <= 0`.
- When YOU author a plan-followup launch command with this pattern, prefer `training.epochs=1` over `-1` (HF Trainer will still cap at max_steps when both are set positive).

Related: [[trl-conversational-format-in-format-dataset]] (the round-3 bug; both bugs hide in the same code path and surface in sequence).
