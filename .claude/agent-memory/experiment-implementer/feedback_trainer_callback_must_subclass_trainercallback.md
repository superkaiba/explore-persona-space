---
name: trainer-callback-must-subclass-TrainerCallback
description: Hand-rolled HF Trainer callbacks must subclass transformers.TrainerCallback; smoke the REAL SFTTrainer path (tiny same-arch model, CPU) for GPU-bound phases — dry-run/import-check substitutes never traverse Trainer.__init__.
type: feedback
---

A hand-rolled callback passed to `train_lora(callbacks=[...])` / any HF Trainer MUST
subclass `transformers.TrainerCallback`. `CallbackHandler.call_event` fires EVERY
lifecycle event on every callback — `on_init_end` first, inside `Trainer.__init__` —
and only the subclass inherits the no-op defaults. A docstring *claiming* the base
class is not the base class (#816 round-2 production crash: `AttributeError:
'PreventativeSteeringCallback' object has no attribute 'on_init_end'`; all
preventative cells died at SFTTrainer init after 53/53 steering cells had burned
GPU-hours).

**Why:** the round-1 smoke "passed" because the phase is GPU-bound (`device_map={"": 0}`)
and the CPU smoke substitute (dry-run / signature / import-check) never constructed a
real SFTTrainer — `on_init_end` fires only inside `SFTTrainer.__init__`.

**How to apply:** when a phase attaches a custom Trainer callback, the smoke MUST
traverse the real `SFTTrainer.__init__ → on_init_end → on_train_begin → step →
on_train_end` lifecycle — a tiny same-arch model (e.g. Qwen2 0.5B) on CPU with
max_steps=1-2 does it in minutes. Assert the callback subclasses TrainerCallback in
code review whenever `callbacks=` appears in a diff.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [TrainerCallback subclass + real-path smoke](feedback_trainer_callback_must_subclass_trainercallback.md) — hand-rolled HF callbacks must subclass TrainerCallback; GPU-bound-phase smokes must traverse real SFTTrainer.__init__ (#816)
