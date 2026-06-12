# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Checkpoint-grid pruning callback for the #597 Arm B fine early grid.

Arm B saves every 4 optimizer steps (``save_steps=4``) so the early install
window gets 4-step resolution, but keeping all 132 checkpoints would cost
~2.3× the panel-probe compute and ~40 GB of MooseFS quota per source. The
``CheckpointGridPruneCallback`` prunes, immediately after every save, any
``checkpoint-<step>`` dir whose step is NOT in the registered keep grid
(``B_GRID`` = {4..60:4} ∪ {80..520:20} ∪ {528}).

NEVER combine with ``save_total_limit`` — HF's rotation would delete keeper
checkpoints from the FRONT of the ladder; this callback is the only pruning
mechanism Arm B uses.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


class CheckpointGridPruneCallback(TrainerCallback):
    """Prune ``checkpoint-<step>`` dirs whose step is not in ``keep_steps``.

    Runs on every ``on_save`` event: lists ``args.output_dir`` for
    ``checkpoint-*`` subdirs and ``shutil.rmtree``s any whose parsed step is
    not in the keep set. Steps that don't parse as integers are left alone
    (fail-safe: never delete a dir we can't positively identify).

    Args:
        keep_steps: iterable of optimizer steps whose checkpoints survive
            (e.g. ``B_GRID``). Stored as a frozenset.

    Attributes:
        pruned_steps: list of steps pruned so far (for logging/tests).
    """

    def __init__(self, keep_steps: Iterable[int]):
        self.keep_steps = frozenset(int(s) for s in keep_steps)
        if not self.keep_steps:
            raise ValueError("CheckpointGridPruneCallback requires a non-empty keep_steps grid")
        self.pruned_steps: list[int] = []

    def prune_dir(self, output_dir: Path | str) -> list[int]:
        """Prune non-grid checkpoint dirs under ``output_dir``; return pruned steps."""
        out = Path(output_dir)
        pruned_now: list[int] = []
        for d in sorted(out.glob("checkpoint-*")):
            if not d.is_dir():
                continue
            m = _CKPT_RE.match(d.name)
            if m is None:
                continue
            step = int(m.group(1))
            if step in self.keep_steps:
                continue
            shutil.rmtree(d)
            pruned_now.append(step)
        if pruned_now:
            self.pruned_steps.extend(pruned_now)
            logger.info(
                "CheckpointGridPruneCallback: pruned %d off-grid checkpoint(s) %s under %s "
                "(%d pruned total)",
                len(pruned_now),
                pruned_now,
                out,
                len(self.pruned_steps),
            )
        return pruned_now

    def on_save(self, args, state, control, **kwargs):
        """HF Trainer hook: prune right after each checkpoint save."""
        self.prune_dir(args.output_dir)
        return control


class HaltAfterStepCallback(TrainerCallback):
    """Stop training right after the checkpoint at ``halt_step`` is written.

    Save-driven halt (#597 follow-up `dense-early-contrastive-grid`, plan v3
    §3): ``max_steps`` stays 528 so the cosine + warmup LR schedule is
    untouched for steps 1–``halt_step`` (``max_steps=60`` would change both
    denominators). HF Trainer writes the checkpoint BEFORE firing ``on_save``
    and checks ``control.should_training_stop`` at the end of the step loop,
    so the ``checkpoint-<halt_step>`` dir is on disk when the halt fires.

    Args:
        halt_step: optimizer step whose save triggers the stop (e.g. 60).
        save_steps: the Trainer's ``save_steps`` — asserted to divide
            ``halt_step`` so the halt fires ON a save event (a non-multiple
            would silently never halt and train all 528 steps).
    """

    def __init__(self, halt_step: int, save_steps: int):
        if halt_step % save_steps != 0:
            raise ValueError((halt_step, save_steps))
        self.halt_step = halt_step
        self.save_steps = save_steps

    def on_save(self, args, state, control, **kwargs):
        """HF Trainer hook: request a clean stop once ``halt_step`` is saved."""
        if state.global_step >= self.halt_step:
            logger.info(
                "HaltAfterStepCallback: checkpoint at step %d saved (halt_step=%d) — "
                "stopping training (max_steps=%d untouched; schedule identity preserved)",
                state.global_step,
                self.halt_step,
                state.max_steps,
            )
            control.should_training_stop = True
        return control
