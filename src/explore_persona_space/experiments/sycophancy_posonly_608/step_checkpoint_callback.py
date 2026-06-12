"""Task #608 follow-up ``sub-ceiling-install`` — step-list checkpointing.

``StepListCheckpointCallback`` saves an adapter checkpoint at an EXPLICIT list
of optimizer steps (plan v5 §4 diff 2) by setting ``control.should_save = True``
in ``on_step_end``. Pair it with ``TrainLoraConfig(save_strategy="no",
save_only_model=True)``: the Trainer's ``DefaultFlowCallback`` (registered
first) never touches ``should_save`` under ``save_strategy="no"``, this
callback (appended after) sets it at exactly the named steps, and
``Trainer._maybe_log_save_evaluate`` then writes ``checkpoint-<global_step>``
(adapter-only, ~330 MB, no optimizer/scheduler state).

``state.global_step`` counts OPTIMIZER steps (post gradient-accumulation) and
is incremented BEFORE ``on_step_end`` fires, so membership in ``steps`` reads
as "this many optimizer steps completed". No edit to ``train/sft.py`` — the
callback rides the existing ``train_lora(callbacks=[...])`` parameter.
"""

from __future__ import annotations

from transformers import TrainerCallback


class StepListCheckpointCallback(TrainerCallback):
    """Set ``control.should_save`` at exactly the named global steps."""

    def __init__(self, steps: tuple[int, ...] | list[int]):
        self.steps = frozenset(int(s) for s in steps)
        if not self.steps:
            raise ValueError("StepListCheckpointCallback needs a non-empty step list")
        if any(s <= 0 for s in self.steps):
            raise ValueError(f"Checkpoint steps must be positive, got {sorted(self.steps)}")

    def on_step_end(self, args, state, control, **kwargs):
        """Trigger a save when the just-completed optimizer step is in the list."""
        if state.global_step in self.steps:
            control.should_save = True
        return control
