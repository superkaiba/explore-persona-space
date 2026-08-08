---
name: on_log never sees direct wandb.log metrics
description: TrainerCallback.on_log only receives trainer-routed logs; keys another callback emits via direct wandb.log() are invisible — read the producer's artifact file instead
type: feedback
---

`TrainerCallback.on_log` only observes logs routed through the HF trainer's log
pipeline (`trainer.log` / `state.log_history`). Metrics a callback emits via
DIRECT `wandb.log(...)` (e.g. `MarkerBandStopCallback`'s
`marker/band_stop_step` + `band_stop_delta_nats`, `eval/callbacks.py`
~:1008-1015) never reach `on_log` — an on_log subscription to another
callback's wandb keys silently reads nothing.

**Why:** task #621 (2026-06-12): a `_Recorder` on_log subscription to the
band-stop keys always wrote `fired: false`, producing a FALSE "band miss" at
cap 16 AND cap 32 on smoke runs that had actually banded in [5,12] (WandB
ground truth 5.08/5.36 nat); the pipeline failed "r=1 cannot reach the band"
and the GCP EXIT trap powered the VM off — a full launch cycle lost to a
verdict-read bug the CPU smoke structurally could not catch (the real callback
only runs on GPU).

**How to apply:** derive cross-callback outcomes from the PRODUCING callback's
artifact file (`band_trajectory.json`, atomically rewritten per probe) plus
`state.global_step` / `state.max_steps` captured at `on_train_end` — never
from log-key sniffing. When a verdict path can only execute on GPU, add a
synthetic-artifact CPU test (in-band / overshoot / full-cap-miss cases) so the
classification logic is covered pre-launch.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [on_log never sees direct wandb.log](feedback_on_log_never_sees_direct_wandb_log.md) — derive cross-callback outcomes from the producer's artifact file, never on_log key-sniffing (#621 false band-miss)
