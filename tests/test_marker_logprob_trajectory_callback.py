"""CPU-only regression test for MF-C trajectory callback's WandB logging path.

Round-1 review blocker #1 (issue #464): the callback's WandB logging used
``trainer = kwargs.get("trainer")`` which is always None (HF Trainer's
CallbackHandler doesn't pass `trainer` in kwargs). So every callback
firing silently fell through to a logger.info fallback and no
``marker_logp/*`` metric ever reached WandB. The round-2 fix uses
``wandb.log(..., step=state.global_step)`` directly (mirrors the working
pattern in eval/callbacks.py:163-166).

This test exercises the full callback path without HF Trainer or vLLM by:
  * monkeypatching the subprocess call so the eval script is never run
  * writing a fake per-key-logp JSON to the expected ``out_file`` path
  * monkeypatching ``wandb.log`` and ``wandb.run`` to capture the call
  * invoking ``on_step_end`` directly with a fake state

The single behavioral assertion: ``wandb.log`` is called with
``step=state.global_step`` (the bug was step never reached WandB).
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from explore_persona_space.train.callbacks import MarkerLogprobTrajectoryCallback


class _FakeModel:
    """Minimal stand-in for a PEFT-wrapped model that supports save_pretrained."""

    def __init__(self):
        self.save_pretrained_calls: list[str] = []

    def save_pretrained(self, path):
        self.save_pretrained_calls.append(str(path))
        # Touch a dummy file so callback's downstream check is plausible.
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.safetensors").write_bytes(b"")


def _make_state(global_step: int):
    """Build a fake TrainerState-like object exposing only ``.global_step``."""
    return types.SimpleNamespace(global_step=global_step)


def _make_probe_file(tmp_path: Path) -> Path:
    """Write a minimal probe-file JSON the callback's CLI would accept."""
    payload = {
        "schema_version": "i464_marker_traj_v1",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "probes": [
            {
                "key": "test_arm/pirate/system_pirate",
                "full_ids": [1, 2, 3],
                "marker_id": 3,
                "slot": 2,
            }
        ],
    }
    p = tmp_path / "probes.json"
    p.write_text(json.dumps(payload))
    return p


def test_wandb_log_called_with_step_keyed_metric(monkeypatch, tmp_path):
    """The callback MUST log per-key metrics via wandb.log with step=global_step."""
    # 1. Build probe file + adapter dump dir.
    probe_file = _make_probe_file(tmp_path)
    adapter_dump_dir = tmp_path / "adapter"

    # 2. Patch subprocess.run so the eval subprocess is never actually invoked.
    #    Instead, write a fake out_file payload the callback expects.
    def fake_run(cmd, env=None, check=False, capture_output=True, text=True, timeout=None):
        # Find the --out-file argument; write a deterministic per-key payload.
        out_path = Path(cmd[cmd.index("--out-file") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": "i464_marker_traj_v1_out",
                    "n_probes": 1,
                    "per_key_logp": {"test_arm/pirate/system_pirate": -3.25},
                }
            )
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("explore_persona_space.train.callbacks.subprocess.run", fake_run)

    # 3. Build a fake `wandb` module exposing `run` (truthy) and `log` (captured).
    captured: dict = {}

    def fake_log(metrics, step=None):
        captured["metrics"] = dict(metrics)
        captured["step"] = step

    fake_wandb = types.SimpleNamespace(
        run=MagicMock(),  # truthy = a run is active
        log=fake_log,
    )
    # The callback does `import wandb` inside on_step_end; substitute the module.
    import sys

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    # 4. Build the callback + invoke on_step_end.
    cb = MarkerLogprobTrajectoryCallback(
        probe_file=str(probe_file),
        step_every=10,
        adapter_dump_dir=str(adapter_dump_dir),
        log_prefix="marker_logp",
    )
    model = _FakeModel()
    state = _make_state(global_step=10)
    cb.on_step_end(args=None, state=state, control=None, model=model)

    # 5. Behavioral assertion: wandb.log was called with the step-keyed metric.
    assert captured, "wandb.log was never called — round-1 bug regression"
    assert captured["step"] == 10, f"wandb.log step={captured['step']}, expected 10"
    assert captured["metrics"] == {"marker_logp/test_arm/pirate/system_pirate": -3.25}, (
        f"wandb.log metrics={captured['metrics']}, expected per-key-prefixed dict"
    )


def test_no_wandb_log_when_no_active_run(monkeypatch, tmp_path):
    """When wandb.run is None (no active training run), no log call should happen."""
    probe_file = _make_probe_file(tmp_path)
    adapter_dump_dir = tmp_path / "adapter"

    def fake_run(cmd, **kw):
        out_path = Path(cmd[cmd.index("--out-file") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": "i464_marker_traj_v1_out",
                    "n_probes": 1,
                    "per_key_logp": {"test_arm/pirate/system_pirate": -3.25},
                }
            )
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("explore_persona_space.train.callbacks.subprocess.run", fake_run)

    log_calls: list = []

    def fake_log(metrics, step=None):
        log_calls.append((metrics, step))

    fake_wandb = types.SimpleNamespace(run=None, log=fake_log)
    import sys

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cb = MarkerLogprobTrajectoryCallback(
        probe_file=str(probe_file),
        step_every=10,
        adapter_dump_dir=str(adapter_dump_dir),
    )
    cb.on_step_end(args=None, state=_make_state(10), control=None, model=_FakeModel())

    assert not log_calls, f"wandb.log called with {log_calls} despite wandb.run=None"


def test_skipped_when_step_not_multiple(monkeypatch, tmp_path):
    """Callback must NOT run subprocess when step is not a multiple of step_every."""
    probe_file = _make_probe_file(tmp_path)
    adapter_dump_dir = tmp_path / "adapter"

    sub_calls: list = []

    def fake_run(cmd, **kw):
        sub_calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("explore_persona_space.train.callbacks.subprocess.run", fake_run)

    cb = MarkerLogprobTrajectoryCallback(
        probe_file=str(probe_file),
        step_every=10,
        adapter_dump_dir=str(adapter_dump_dir),
    )
    # Step 5 is NOT a multiple of step_every=10 → callback should be a no-op.
    cb.on_step_end(args=None, state=_make_state(5), control=None, model=_FakeModel())

    assert not sub_calls, "subprocess invoked despite step % step_every != 0"


def test_skipped_at_step_zero(monkeypatch, tmp_path):
    """Step 0 is always skipped (avoids logging the un-trained adapter)."""
    probe_file = _make_probe_file(tmp_path)
    adapter_dump_dir = tmp_path / "adapter"

    sub_calls: list = []

    def fake_run(cmd, **kw):
        sub_calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("explore_persona_space.train.callbacks.subprocess.run", fake_run)

    cb = MarkerLogprobTrajectoryCallback(
        probe_file=str(probe_file),
        step_every=10,
        adapter_dump_dir=str(adapter_dump_dir),
    )
    cb.on_step_end(args=None, state=_make_state(0), control=None, model=_FakeModel())
    assert not sub_calls, "subprocess invoked at step=0"


def test_invalid_step_every_raises():
    """``step_every <= 0`` must fail at construction."""
    with pytest.raises(ValueError, match="step_every"):
        MarkerLogprobTrajectoryCallback(probe_file="x", step_every=0, adapter_dump_dir="/tmp/x")
