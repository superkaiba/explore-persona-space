"""TrainerCallbacks specific to checkpointing schedules outside HF/TRL's built-ins.

Currently houses ``SaveAtSpecificSteps``, used by issue #385 to capture LoRA
adapter snapshots at an arbitrary, irregular list of optimizer steps (e.g.
``[5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600]``).
TRL's native ``save_strategy='steps'`` only supports a constant cadence.

This module is intentionally separate from
``explore_persona_space.eval.callbacks`` (which holds the periodic-eval
callbacks gated by ``cfg.periodic_eval.enabled``). Step-list checkpointing is
NOT a periodic eval — it's a pure checkpoint-snapshot mechanism and lives in
``train/`` so it can be wired into ``train_phase`` independent of the
periodic-eval gate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class SaveAtSpecificSteps(TrainerCallback):
    """Save a LoRA adapter checkpoint at each step in ``steps_list``.

    Fires from ``on_step_end``. Writes ``{output_dir}/checkpoint-{step}/`` and
    calls ``model.save_pretrained`` (PEFT adapter-only save: ~150 MB per
    checkpoint at LoRA r=32 on Qwen-2.5-7B). The tokenizer is also saved when
    available so each checkpoint dir is a self-contained adapter that vLLM can
    load via ``LoRARequest(lora_path=...)``.

    Each requested step fires at most once per training run (idempotent via
    ``_fired`` set). Saves are restricted to the local rank-0 process so DDP /
    multi-GPU training does not race on the same directory.

    Args:
        steps_list: Integer global_step values at which to save. Order does
            not matter; duplicates are collapsed.
        output_dir: Base directory under which ``checkpoint-{step}`` dirs are
            written.
    """

    def __init__(self, steps_list: Iterable[int], output_dir: str | Path):
        self.steps_set: set[int] = {int(s) for s in steps_list}
        if not self.steps_set:
            raise ValueError("SaveAtSpecificSteps requires at least one step")
        self.output_dir = Path(output_dir)
        self._fired: set[int] = set()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step not in self.steps_set or step in self._fired:
            return
        # DDP / multi-GPU safety: only rank 0 writes.
        if getattr(args, "local_process_index", 0) != 0:
            return
        if model is None:
            logger.warning("SaveAtSpecificSteps: model is None at step %d; skipping save", step)
            return
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # PEFT adapter-only save: writes adapter_config.json + adapter_model.safetensors.
        # On a non-PEFT model this still works (writes full weights) but the use case
        # here is LoRA only.
        model.save_pretrained(str(ckpt_dir))
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if tokenizer is not None:
            tokenizer.save_pretrained(str(ckpt_dir))
        self._fired.add(step)
        logger.info("SaveAtSpecificSteps: saved checkpoint-%d to %s", step, ckpt_dir)
