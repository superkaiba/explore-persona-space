"""Regression test for the round-4 smoke crash on task #405.

The bug: ``ProbePanelLogprobCallback`` (in ``scripts/issue405_run_cell.py``) was
a plain class — it did NOT subclass ``transformers.TrainerCallback``. HF's
``CallbackHandler.call_event`` calls ``getattr(callback, event)(...)`` for EVERY
trainer lifecycle event (``on_init_end``, ``on_train_begin``, ``on_epoch_begin``,
``on_step_end``, ``on_log``, ``on_save``, ...). The callback only defined
``on_train_begin`` + ``on_step_end``, so the FIRST event the handler fires —
``on_init_end``, dispatched from ``SFTTrainer.__init__`` — raised::

    AttributeError: 'ProbePanelLogprobCallback' object has no attribute 'on_init_end'.
    Did you mean: 'on_step_end'?

Trainer construction crashed before a single training step ran; all 4 worktree
review rounds passed code-review while the pod-side smoke crashed in ~5s.

The fix is to subclass ``TrainerCallback`` so the parent provides no-op defaults
for every lifecycle event we don't implement.

These tests assert:
  (1) the subclass relationship holds (the structural fix), AND
  (2) driving the callback through HF's ``CallbackHandler.call_event`` for
      every lifecycle event raises NO AttributeError (the behavioural fix —
      reproduces the exact code path the trainer takes at construction).

The test uses a minimal CPU-only fake (no model, no tokenizer needed for the
no-op events). ``on_train_begin`` is exercised with ``tokenizer=None`` which
hits the callback's documented "no tokenizer → disable" branch (no GPU/HF
heavy machinery required).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _import_callback():
    from issue405_run_cell import ProbePanelLogprobCallback

    return ProbePanelLogprobCallback


def _make_callback():
    """Cheap construction — no GPU/HF state needed for the lifecycle-event sweep."""
    ProbePanelLogprobCallback = _import_callback()
    return ProbePanelLogprobCallback(
        probe_personas=[("paramedic", "You are a paramedic.")],
        sample_R="Sure, here is a response.",
        sample_question="What is 2+2?",
        marker_text=" ※",
        marker_token_id=83399,
        log_every_steps=5,
    )


def test_subclasses_trainer_callback():
    """Structural fix — without this, every other event raises AttributeError.

    `transformers.TrainerCallback` provides no-op defaults for all 14 lifecycle
    events; subclassing is how HF's CallbackHandler can call ANY of them without
    AttributeError. The plain class declaration `class Foo:` is the bug.
    """
    from transformers import TrainerCallback

    ProbePanelLogprobCallback = _import_callback()
    assert issubclass(ProbePanelLogprobCallback, TrainerCallback), (
        "ProbePanelLogprobCallback MUST subclass transformers.TrainerCallback so "
        "CallbackHandler.call_event finds no-op defaults for events the callback "
        "doesn't implement. Without the subclass, on_init_end (the FIRST event "
        "fired, from SFTTrainer.__init__) raises AttributeError and trainer "
        "construction dies before any training step runs."
    )


def test_callback_handler_fires_all_lifecycle_events_without_error():
    """Behavioural fix — drives the callback through HF's exact code path.

    `CallbackHandler.call_event(event, args, state, control, **kwargs)` is what
    SFTTrainer.__init__ invokes for `on_init_end`, what `Trainer.train` invokes
    for `on_train_begin` / `on_step_end` / `on_log` / `on_epoch_begin` / etc.,
    and what `Trainer.save_model` invokes for `on_save`. If ANY of these raise
    AttributeError, the trainer crashes mid-flight.

    Reproduces the round-4 smoke crash: pre-fix, `on_init_end` raises
    AttributeError immediately. Post-fix (with TrainerCallback parent), every
    event is a safe no-op except the two the callback implements.
    """
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from transformers.trainer_callback import CallbackHandler

    cb = _make_callback()

    args = TrainingArguments(output_dir="/tmp/test_issue405_probe_cb")
    state = TrainerState()
    control = TrainerControl()

    # CallbackHandler is what SFTTrainer uses internally. Construct it the same
    # way Trainer does: callbacks list, model=None (no_op for these events),
    # processing_class=None, optimizer/lr_scheduler=None.
    handler = CallbackHandler(
        callbacks=[cb],
        model=None,
        processing_class=None,
        optimizer=None,
        lr_scheduler=None,
    )

    # Every lifecycle event SFTTrainer's training path fires. on_init_end is
    # the one that crashed at SFTTrainer.__init__ in round 4. on_train_begin
    # is exercised with tokenizer=None which trips the callback's documented
    # "disable" path (no GPU/HF model needed). on_step_end with model=None is
    # also a no-op per the callback's own guard.
    #
    # on_predict is EXCLUDED — it requires a `metrics` kwarg and is only
    # fired by Trainer.predict(), never by Trainer.train() / SFTTrainer
    # construction. The round-4 crash was on the training path, not predict.
    events = [
        "on_init_end",
        "on_train_begin",
        "on_epoch_begin",
        "on_step_begin",
        "on_substep_end",
        "on_pre_optimizer_step",
        "on_optimizer_step",
        "on_step_end",
        "on_log",
        "on_evaluate",
        "on_save",
        "on_prediction_step",
        "on_epoch_end",
        "on_train_end",
    ]
    for event in events:
        try:
            handler.call_event(event, args, state, control)
        except AttributeError as e:
            pytest.fail(
                f"CallbackHandler.call_event({event!r}) raised AttributeError: {e}. "
                "This is the round-4 smoke-crash class: ProbePanelLogprobCallback "
                "no longer subclasses TrainerCallback, so events it doesn't "
                "implement are missing and getattr() blows up. Re-add the "
                "`class ProbePanelLogprobCallback(TrainerCallback)` subclass."
            )


def test_on_step_end_is_a_no_op_when_model_is_none():
    """Sanity: the callback's own implemented on_step_end early-exits on model=None.

    Documents the early-exit path the lifecycle-events test relies on (so a
    future refactor that breaks the guard doesn't quietly invent a new crash).
    """
    from transformers import TrainerControl, TrainerState, TrainingArguments

    cb = _make_callback()
    args = TrainingArguments(output_dir="/tmp/test_issue405_probe_cb")
    state = TrainerState()
    control = TrainerControl()

    # No exception expected — early exit on model is None.
    cb.on_step_end(args, state, control, model=None)
