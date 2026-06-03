"""Task #479: CheckpointAtStepsCallback fires at absolute optimizer steps.

End-to-end unit test on a mock Trainer state — no model weights, no GPU. The
callback's contract is: on each `on_step_end`, save the adapter to
`<ckpt_root>/step_<S>/` the FIRST time `state.global_step` reaches `S`. The
endpoint step is recorded but its directory is the caller's `output_dir`
(filled in by the runner after train_lora returns).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
    CheckpointAtStepsCallback,
)


class _RecordingModel:
    """Minimal stand-in for a PEFT model with save_pretrained."""

    def __init__(self):
        self.saves: list[str] = []

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.bin").write_bytes(b"")
        self.saves.append(path)


def _drive(callback: CheckpointAtStepsCallback, model, max_steps: int) -> None:
    """Simulate Trainer iterating from step 1 → max_steps, calling on_step_end each step."""
    for step in range(1, max_steps + 1):
        state = SimpleNamespace(global_step=step, max_steps=max_steps)
        callback.on_step_end(args=None, state=state, control=None, model=model)
    end_state = SimpleNamespace(global_step=max_steps, max_steps=max_steps)
    callback.on_train_end(args=None, state=end_state, control=None, model=model)


def test_saves_at_each_target_step(tmp_path):
    model = _RecordingModel()
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=(5, 10, 20))
    _drive(cb, model, max_steps=30)
    # Three mid-run saves (5, 10, 20); 30 is past the largest non-endpoint step
    # but is NOT in the target steps so no endpoint save fires.
    assert len(model.saves) == 3
    assert (tmp_path / "step_0005").exists()
    assert (tmp_path / "step_0010").exists()
    assert (tmp_path / "step_0020").exists()
    idx = cb.index()
    assert set(idx) == {"0005", "0010", "0020"}
    assert idx["0005"]["step"] == 5
    assert idx["0010"]["step"] == 10
    assert idx["0020"]["step"] == 20


def test_endpoint_step_defers_to_caller(tmp_path):
    """The endpoint step is recorded with path=None (the caller fills it)."""
    model = _RecordingModel()
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=(5, 10, 20))
    _drive(cb, model, max_steps=20)
    # Mid-run: only step 5 and 10 saved to disk (step 20 == max_steps deferred).
    assert len(model.saves) == 2
    idx = cb.index()
    assert "0020" in idx
    assert idx["0020"]["path"] is None  # caller fills with the final adapter dir.
    assert idx["0005"]["path"] is not None
    assert idx["0010"]["path"] is not None


def test_does_not_save_twice_at_same_step(tmp_path):
    """Repeated on_step_end calls with the same global_step trigger one save."""
    model = _RecordingModel()
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=(5,))
    state = SimpleNamespace(global_step=5, max_steps=10)
    cb.on_step_end(args=None, state=state, control=None, model=model)
    cb.on_step_end(args=None, state=state, control=None, model=model)
    cb.on_step_end(args=None, state=state, control=None, model=model)
    assert len(model.saves) == 1
    assert (tmp_path / "step_0005").exists()


def test_step_threshold_first_crossing(tmp_path):
    """If global_step JUMPS past a target (gradient accumulation), still save once."""
    model = _RecordingModel()
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=(5, 10))
    # Simulate global_step skipping from 3 to 7 (e.g. gradient_accumulation_steps>1
    # where on_step_end fires per OPTIMIZER step, not per micro-batch).
    for global_step in [1, 3, 7, 8, 9, 10]:
        state = SimpleNamespace(global_step=global_step, max_steps=10)
        cb.on_step_end(args=None, state=state, control=None, model=model)
    # max_steps == 10, step 10 IS the endpoint → deferred. Only step 5 saves
    # (at the global_step=7 crossing, since that's the first step ≥ 5).
    assert len(model.saves) == 1
    assert (tmp_path / "step_0005").exists()
    saved_step = cb.index()["0005"]["step"]
    assert saved_step == 7, f"expected first-crossing step 7, got {saved_step}"


def test_no_target_steps(tmp_path):
    """Empty steps tuple is a no-op — no saves, empty index."""
    model = _RecordingModel()
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=())
    _drive(cb, model, max_steps=10)
    assert model.saves == []
    assert cb.index() == {}


def test_steps_sorted_uniquified(tmp_path):
    """Constructor sorts and de-duplicates the input steps tuple."""
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=(20, 5, 10, 5, 20))
    assert cb.steps == [5, 10, 20]


def test_model_none_is_noop(tmp_path):
    """If Trainer doesn't pass model, the callback no-ops instead of crashing."""
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=(5,))
    state = SimpleNamespace(global_step=5, max_steps=10)
    cb.on_step_end(args=None, state=state, control=None, model=None)
    assert cb.index() == {}


def test_max_steps_zero_is_noop(tmp_path):
    """If max_steps == 0 (TRL pre-init), no saves until it's set."""
    model = _RecordingModel()
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=(5,))
    state = SimpleNamespace(global_step=5, max_steps=0)
    cb.on_step_end(args=None, state=state, control=None, model=model)
    assert model.saves == []


@pytest.mark.parametrize(
    "steps,max_steps,expected_saved",
    [
        # 11-step #479 schedule with max_steps == endpoint: 10 mid-run saves.
        ((5, 10, 20, 35, 50, 75, 100, 125, 150, 200, 250), 250, 10),
        # Smoke 2-step list: 1 mid-run save (step 1), endpoint 2 deferred.
        ((1, 2), 2, 1),
    ],
)
def test_matches_479_schedules(tmp_path, steps, max_steps, expected_saved):
    model = _RecordingModel()
    cb = CheckpointAtStepsCallback(ckpt_root=tmp_path, steps=steps)
    _drive(cb, model, max_steps=max_steps)
    assert len(model.saves) == expected_saved, (
        f"steps={steps}, max_steps={max_steps}: expected {expected_saved} saves, "
        f"got {len(model.saves)} ({model.saves})"
    )
