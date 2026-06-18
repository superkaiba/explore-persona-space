"""CPU-only tests for train_lora's per-call WandB run lifecycle (#577).

Pins the fix for the #527 sweep-dispatcher regression: in-process sweep
dispatchers call ``train_lora()`` once per cell with ``report_to="wandb"`` and
a per-cell ``run_name``, but HF's ``WandbCallback`` only calls ``wandb.init``
when ``wandb.run is None``. The first cell's run was never finished, so cells
2..N logged into the STALE first run with global_steps rewound to 0, and WandB
silently dropped the out-of-order writes — 17/18 #527 cells lost all training
telemetry. train_lora now finishes any run IT created before returning (and on
the exception path), so the next cell re-inits a fresh run; caller-owned runs
that existed before the call are never touched.

Tested via a fake ``wandb`` module injected into ``sys.modules`` — no network,
no GPU, runs in <1s.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from explore_persona_space.train.sft import (
    _finish_wandb_run_if_owned,
    _wandb_run_active,
    train_lora,
)


@pytest.fixture
def fake_wandb(monkeypatch):
    """Inject a minimal fake wandb module with a controllable .run."""
    mod = types.ModuleType("wandb")
    mod.run = None
    mod.finish_calls = []

    def finish(exit_code=0):
        mod.finish_calls.append(exit_code)
        mod.run = None

    mod.finish = finish
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


# ---------------------------------------------------------------------------
# _wandb_run_active
# ---------------------------------------------------------------------------


def test_run_active_false_when_wandb_never_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    assert _wandb_run_active() is False


def test_run_active_false_when_no_run(fake_wandb):
    assert _wandb_run_active() is False


def test_run_active_true_when_run_exists(fake_wandb):
    fake_wandb.run = object()
    assert _wandb_run_active() is True


# ---------------------------------------------------------------------------
# _finish_wandb_run_if_owned
# ---------------------------------------------------------------------------


def test_preexisting_run_is_never_finished(fake_wandb):
    """Caller-owned runs (e.g. orchestrate/runner.py init_wandb) stay open."""
    fake_wandb.run = object()
    _finish_wandb_run_if_owned(run_preexisting=True, exit_code=0)
    assert fake_wandb.finish_calls == []
    assert fake_wandb.run is not None


def test_owned_run_is_finished(fake_wandb):
    """A run created during the call is finished → next cell re-inits (#527)."""
    fake_wandb.run = object()
    _finish_wandb_run_if_owned(run_preexisting=False, exit_code=0)
    assert fake_wandb.finish_calls == [0]
    assert fake_wandb.run is None


def test_owned_run_finished_with_crash_exit_code(fake_wandb):
    fake_wandb.run = object()
    _finish_wandb_run_if_owned(run_preexisting=False, exit_code=1)
    assert fake_wandb.finish_calls == [1]


def test_noop_when_no_run_exists(fake_wandb):
    _finish_wandb_run_if_owned(run_preexisting=False, exit_code=0)
    assert fake_wandb.finish_calls == []


def test_noop_when_wandb_never_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    _finish_wandb_run_if_owned(run_preexisting=False, exit_code=0)  # must not raise


def test_finish_failure_is_swallowed(fake_wandb):
    """Teardown must never mask the training result / original exception."""
    fake_wandb.run = object()

    def exploding_finish(exit_code=0):
        raise RuntimeError("wandb backend gone")

    fake_wandb.finish = exploding_finish
    _finish_wandb_run_if_owned(run_preexisting=False, exit_code=0)  # must not raise


# ---------------------------------------------------------------------------
# train_lora wiring (source-level invariant; loading a real model on CPU is
# out of scope for a unit test)
# ---------------------------------------------------------------------------


def test_train_lora_wires_the_per_call_lifecycle():
    src = inspect.getsource(train_lora)
    assert "wandb_run_preexisting = _wandb_run_active()" in src, (
        "train_lora must capture run ownership at entry (#527 per-cell wandb.init fix)"
    )
    assert "_finish_wandb_run_if_owned(" in src, (
        "train_lora must finish the run it created before returning (#527)"
    )
    # The finish must sit in a finally block so a crashed cell cannot leak its
    # run into the next cell of an in-process sweep.
    assert "finally:" in src.split("_finish_wandb_run_if_owned(")[0].rsplit("try:", 1)[-1], (
        "_finish_wandb_run_if_owned must run from the finally block of the train/upload section"
    )


def test_ownership_capture_precedes_training():
    src = inspect.getsource(train_lora)
    assert src.index("wandb_run_preexisting = _wandb_run_active()") < src.index(
        "trainer.train()"
    ), "ownership must be read BEFORE the trainer can init a run"
