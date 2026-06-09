"""Pin issue #527 Critical-1 fix: ``MarkerBandStopCallback`` surfaces its
source-delta + band-stop scalars through the trainer's ``on_log`` lifecycle
so sibling ``TrainerCallback.on_log`` subscribers (e.g. the per-cell
``BandStopRecorder`` that writes ``marker_band_stop_result.json`` to disk)
actually receive them.

Round-1 review surfaced that ``on_step_end`` called ``wandb.log(...)``
directly, never invoking ``Trainer.log(...)`` / appending to
``state.log_history`` / firing the lifecycle hook. The recorder-sibling
subscribed to ``on_log`` and saw ONLY the trainer's built-in loss/lr/
grad_norm scalars, so its ``fired`` stayed ``False`` for every cell and
the smoke gate verdict was a deterministic ``FAIL``.

This test pins the round-2 fix: extending ``MarkerBandStopCallback`` with
an ``on_log`` hook that merges its trajectory + firing-event scalars into
the trainer's ``logs`` dict.

Run with: ``uv run pytest tests/test_issue_527_band_stop_wiring.py -x``
"""

# math/scientific notation in docstrings

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def callback_factory():
    """Make a MarkerBandStopCallback with a tiny synthetic probe batch."""
    from explore_persona_space.eval.callbacks import MarkerBandStopCallback

    def _make(*, low_nats: float = 5.0, high_nats: float = 12.0, min_steps: int = 0):
        # 2-row probe batch; the actual values don't matter for the wiring
        # test — we stub _read_logp_trained / _read_logp_with_base below.
        input_ids = torch.zeros((2, 8), dtype=torch.long)
        marker_positions = torch.tensor([5, 5], dtype=torch.long)
        attention_mask = torch.ones((2, 8), dtype=torch.long)
        cb = MarkerBandStopCallback(
            marker_token_ids=[83399],
            probe_input_ids=input_ids,
            probe_marker_positions=marker_positions,
            probe_attention_mask=attention_mask,
            low_nats=low_nats,
            high_nats=high_nats,
            eval_every_steps=1,
            min_steps=min_steps,
            log_prefix="marker",
        )
        return cb

    return _make


def _fake_state(global_step: int = 10, max_steps: int = 100):
    return SimpleNamespace(global_step=global_step, max_steps=max_steps, log_history=[])


def _fake_control():
    return SimpleNamespace(should_training_stop=False, should_save=False)


def test_on_log_publishes_source_delta_after_on_step_end(callback_factory):
    """After on_step_end caches a reading, on_log must merge it into logs."""
    cb = callback_factory()
    state = _fake_state(global_step=10)
    control = _fake_control()
    # Stub the model forward reads so we don't need a real torch model.
    with (
        patch.object(cb, "_read_logp_with_base", return_value=torch.tensor([-20.0, -20.0])),
        patch.object(cb, "_read_logp_trained", return_value=torch.tensor([-17.0, -17.0])),
    ):
        # Run the actual on_step_end (it computes delta_mean = 3.0 — OUTSIDE
        # the [5, 12] band, so should_stop is False and we exercise the
        # pure-trajectory path).
        cb.on_step_end(args=None, state=state, control=control, model=object())

    # The trajectory scratch must be cached.
    assert cb._last_delta_mean is not None, "on_step_end did not cache _last_delta_mean"
    assert cb._fired_step is None, "band-stop should NOT have fired (delta=3 < low=5)"

    # Now the trainer fires on_log with an empty logs dict — the canonical
    # `TrainerCallback.on_log` lifecycle hook. Round-1 bug: this dict
    # would be returned unmodified. Round-2 fix: merge the callback's
    # source-delta scalars into it.
    logs: dict = {}
    cb.on_log(args=None, state=state, control=control, logs=logs)
    assert "marker/source_delta_nats" in logs, (
        f"on_log did NOT merge source_delta_nats; logs keys = {sorted(logs.keys())}"
    )
    assert logs["marker/source_delta_nats"] == pytest.approx(3.0, abs=1e-6)
    assert "marker/source_logp_mean" in logs
    # band-stop event keys must NOT appear when the callback hasn't fired.
    assert "marker/band_stop_step" not in logs


def test_on_log_publishes_band_stop_event_when_callback_fires(callback_factory):
    """When on_step_end triggers the band-stop, on_log must publish the firing event."""
    cb = callback_factory(low_nats=2.0, high_nats=12.0, min_steps=0)
    state = _fake_state(global_step=10)
    control = _fake_control()
    # Tune the stubs so delta_mean lands inside the [2, 12] band.
    with (
        patch.object(cb, "_read_logp_with_base", return_value=torch.tensor([-20.0, -20.0])),
        patch.object(cb, "_read_logp_trained", return_value=torch.tensor([-13.0, -13.0])),
    ):
        cb.on_step_end(args=None, state=state, control=control, model=object())

    # band-stop should have fired.
    assert cb._stopped is True
    assert cb._fired_step == 10
    assert control.should_training_stop is True
    assert control.should_save is True

    # on_log must surface the firing event.
    logs: dict = {}
    cb.on_log(args=None, state=state, control=control, logs=logs)
    assert "marker/source_delta_nats" in logs
    assert "marker/band_stop_step" in logs, (
        f"on_log did NOT publish band_stop_step; logs keys = {sorted(logs.keys())}"
    )
    assert logs["marker/band_stop_step"] == 10
    assert logs["marker/band_stop_delta_nats"] == pytest.approx(7.0, abs=1e-6)


def test_on_log_noop_before_on_step_end(callback_factory):
    """A no-op on_log fired before the first on_step_end leaves logs alone."""
    cb = callback_factory()
    logs: dict = {"train/loss": 0.5}  # the trainer's built-in scalars
    cb.on_log(args=None, state=_fake_state(global_step=0), control=_fake_control(), logs=logs)
    assert sorted(logs.keys()) == ["train/loss"], (
        f"on_log mutated logs before any cached reading; logs = {logs}"
    )


def test_recorder_sees_callback_keys_through_on_log():
    """The #527 BandStopRecorder must receive the source-delta keys on on_log."""
    import tempfile
    from pathlib import Path

    from scripts.run_issue527_train import _make_band_stop_recorder

    with tempfile.TemporaryDirectory() as tmp:
        recorder = _make_band_stop_recorder(output_dir=Path(tmp), low_nats=5.0, high_nats=12.0)
        # Simulate the orchestrated handoff: callback computes a reading,
        # publishes through on_log, recorder receives it.
        state = SimpleNamespace(global_step=20)
        recorder.on_log(
            args=None,
            state=state,
            control=_fake_control(),
            logs={
                "marker/source_delta_nats": 8.5,
                "marker/source_logp_mean": -11.5,
            },
        )
        assert recorder._last_delta == pytest.approx(8.5, abs=1e-6)
        assert recorder._last_step == 20
        # Firing event delivered on a later log boundary.
        state2 = SimpleNamespace(global_step=22)
        recorder.on_log(
            args=None,
            state=state2,
            control=_fake_control(),
            logs={
                "marker/source_delta_nats": 8.5,
                "marker/band_stop_step": 22,
                "marker/band_stop_delta_nats": 8.5,
            },
        )
        assert recorder.fired is True
        assert recorder.fired_step == 22
        assert recorder.final_delta_nats == pytest.approx(8.5, abs=1e-6)


def test_recorder_accepts_legacy_marker_band_stop_prefix():
    """The recorder also accepts the legacy log_prefix="marker_band_stop" namespace."""
    import tempfile
    from pathlib import Path

    from scripts.run_issue527_train import _make_band_stop_recorder

    with tempfile.TemporaryDirectory() as tmp:
        recorder = _make_band_stop_recorder(output_dir=Path(tmp), low_nats=5.0, high_nats=12.0)
        recorder.on_log(
            args=None,
            state=SimpleNamespace(global_step=15),
            control=_fake_control(),
            logs={
                "marker_band_stop/source_delta_nats": 7.0,
                "marker_band_stop/band_stop_step": 15,
                "marker_band_stop/band_stop_delta_nats": 7.0,
            },
        )
        assert recorder.fired is True
        assert recorder.fired_step == 15
        assert recorder.final_delta_nats == pytest.approx(7.0, abs=1e-6)
