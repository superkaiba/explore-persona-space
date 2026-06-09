# em-dash + Qwen marker " ※" + Greek ΔG intentional
"""Tests for the #534 sub-stop checkpointing path.

Covers the two NEW pure pieces the #534 pipeline rides on:

  1. ``scripts/i534_select_fractions.py::select_fractions`` — the post-hoc
     fraction → snapshot mapping (happy path; not-stopped flag; S<4 dedup;
     sparse-snapshot nearest with `exact` flags; fail-loud on missing meta /
     missing snapshots) and ``check_logit_readout_valid`` (gauge trip on an
     lm_head-bearing adapter).
  2. ``MarkerBandStopCallback`` snapshot gating — per-step snapshots fire
     BEFORE the stop predicate (the stop-step snapshot exists), respect the
     cap, are a no-op for legacy constructions, and ``on_train_end`` writes
     the ``band_stop_meta.json`` sidecar with the realized stop step.

CPU-only; no model downloads; sub-second.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "i534_select_fractions.py"


@pytest.fixture(scope="module")
def selector_mod():
    """Import `scripts/i534_select_fractions.py` as a module."""
    spec = importlib.util.spec_from_file_location("i534_select_fractions", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_snapshots(tmp_path: Path, steps: list[int], *, meta: dict) -> Path:
    snap = tmp_path / "snapshots"
    snap.mkdir()
    for s in steps:
        d = snap / f"step_{s:04d}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(b"fake")
    (snap / "band_stop_meta.json").write_text(json.dumps(meta))
    return snap


def test_select_exact_hits_at_k1_s20(selector_mod, tmp_path):
    """k=1, S=20 → targets {5, 10, 15, 20}, all exact, 4 distinct steps."""
    snap = _make_snapshots(
        tmp_path,
        list(range(1, 21)),
        meta={"stopped": True, "stop_step": 20, "stop_reason": "band"},
    )
    sel = selector_mod.select_fractions(snap)
    assert {k: v["step"] for k, v in sel["index"].items()} == {
        "0.25": 5,
        "0.50": 10,
        "0.75": 15,
        "1.00": 20,
    }
    assert all(m["exact"] for m in sel["manifest"])
    assert sel["distinct_steps"] == 4
    # Paths point at the actual snapshot dirs.
    for v in sel["index"].values():
        assert Path(v["path"]).is_dir()


def test_select_not_stopped_uses_last_step_and_flags(selector_mod, tmp_path, caplog):
    """stop_reason=epoch_ceiling → selection proceeds off the last step, loudly."""
    snap = _make_snapshots(
        tmp_path,
        list(range(1, 13)),
        meta={"stopped": False, "stop_step": 12, "stop_reason": "epoch_ceiling"},
    )
    with caplog.at_level("WARNING"):
        sel = selector_mod.select_fractions(snap)
    assert sel["stop_meta"]["stopped"] is False
    assert sel["index"]["1.00"]["step"] == 12
    assert any("never band-stopped" in r.message for r in caplog.records)


def test_select_s_below_4_keeps_all_frac_keys(selector_mod, tmp_path):
    """S=2 → duplicate selected steps; all 4 frac keys kept; distinct < 4."""
    snap = _make_snapshots(
        tmp_path, [1, 2], meta={"stopped": True, "stop_step": 2, "stop_reason": "band"}
    )
    sel = selector_mod.select_fractions(snap)
    assert set(sel["index"]) == {"0.25", "0.50", "0.75", "1.00"}
    assert sel["distinct_steps"] == 2
    assert sel["index"]["0.25"]["step"] == 1  # max(1, round(0.5)) clamps to 1
    assert sel["index"]["1.00"]["step"] == 2


def test_select_sparse_snapshots_nearest_with_exact_flags(selector_mod, tmp_path):
    """k=2 snapshots (even steps only) → odd targets pick nearest, exact=False."""
    snap = _make_snapshots(
        tmp_path,
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        meta={"stopped": True, "stop_step": 20, "stop_reason": "band"},
    )
    sel = selector_mod.select_fractions(snap)
    by_frac = {m["frac"]: m for m in sel["manifest"]}
    # target 5 → tie between 4 and 6 → earlier (4) wins.
    assert by_frac[0.25]["target_step"] == 5
    assert by_frac[0.25]["selected_step"] == 4
    assert by_frac[0.25]["exact"] is False
    # target 15 → tie between 14 and 16 → earlier (14).
    assert by_frac[0.75]["selected_step"] == 14
    assert by_frac[1.0]["selected_step"] == 20
    assert by_frac[1.0]["exact"] is True


def test_select_missing_meta_fails_loud(selector_mod, tmp_path):
    snap = tmp_path / "snapshots"
    snap.mkdir()
    (snap / "step_0001").mkdir()
    with pytest.raises(FileNotFoundError, match="band_stop_meta"):
        selector_mod.select_fractions(snap)


def test_select_no_snapshots_fails_loud(selector_mod, tmp_path):
    snap = tmp_path / "snapshots"
    snap.mkdir()
    (snap / "band_stop_meta.json").write_text(
        json.dumps({"stopped": True, "stop_step": 20, "stop_reason": "band"})
    )
    with pytest.raises(RuntimeError, match="no step_"):
        selector_mod.select_fractions(snap)


def test_gauge_check_trips_on_lm_head_key(selector_mod, tmp_path):
    """An adapter whose safetensors carries an lm_head key → valid=False."""
    import torch
    from safetensors.torch import save_file

    d = tmp_path / "adapter"
    d.mkdir()
    save_file(
        {
            "base_model.model.lm_head.lora_A.weight": torch.zeros(2, 2),
            "base_model.model.q_proj.lora_A.weight": torch.zeros(2, 2),
        },
        str(d / "adapter_model.safetensors"),
    )
    (d / "adapter_config.json").write_text(json.dumps({"target_modules": ["q_proj"]}))
    verdict = selector_mod.check_logit_readout_valid(d)
    assert verdict["valid"] is False
    assert any("lm_head" in p for p in verdict["problems"])


def test_gauge_check_passes_clean_adapter(selector_mod, tmp_path):
    import torch
    from safetensors.torch import save_file

    d = tmp_path / "adapter"
    d.mkdir()
    save_file(
        {"base_model.model.q_proj.lora_A.weight": torch.zeros(2, 2)},
        str(d / "adapter_model.safetensors"),
    )
    (d / "adapter_config.json").write_text(
        json.dumps({"target_modules": ["q_proj"], "modules_to_save": None})
    )
    assert selector_mod.check_logit_readout_valid(d)["valid"] is True


# ── MarkerBandStopCallback snapshot gating ──────────────────────────────────


class _DummyAdapterModel:
    """Stand-in PEFT model: save_pretrained writes an adapter dir."""

    def __init__(self):
        self.n_saves = 0

    def save_pretrained(self, d: str) -> None:
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "adapter_model.safetensors").write_bytes(b"fake")
        self.n_saves += 1


def _make_callback(tmp_path: Path, **overrides):
    import torch

    from explore_persona_space.eval.callbacks import MarkerBandStopCallback

    kwargs = dict(
        marker_token_ids=[83399],
        probe_input_ids=torch.tensor([[1, 2, 3]]),
        probe_marker_positions=torch.tensor([1]),
        probe_attention_mask=torch.tensor([[1, 1, 1]]),
        snapshot_every_steps=1,
        snapshot_dir=tmp_path / "snapshots",
        snapshot_max_count=64,
    )
    kwargs.update(overrides)
    return MarkerBandStopCallback(**kwargs)


def _state(step: int, max_steps: int = 300):
    from transformers import TrainerState

    st = TrainerState()
    st.global_step = step
    st.max_steps = max_steps
    return st


def test_snapshot_fires_before_stop_predicate(tmp_path, monkeypatch):
    """The stop-step snapshot exists: snapshot lands in the SAME on_step_end
    call whose band read sets should_training_stop."""
    import torch
    from transformers import TrainerControl

    cb = _make_callback(tmp_path, eval_every_steps=1, min_steps=2)
    # Monkeypatch the model-touching reads: base=0, trained=6 → delta=6 ∈ [5,12].
    monkeypatch.setattr(cb, "_read_logp_with_base", lambda model: torch.zeros(1))
    monkeypatch.setattr(cb, "_read_logp_trained", lambda model: torch.full((1,), 6.0))
    model = _DummyAdapterModel()
    control = TrainerControl()
    cb.on_train_begin(None, _state(0), control)
    for step in (1, 2):
        cb.on_step_end(None, _state(step), control, model=model)
    assert cb._stopped is True
    assert control.should_training_stop is True
    # Stop fired at step 2 (min_steps) AND the step-2 snapshot exists.
    assert (tmp_path / "snapshots" / "step_0002").is_dir()
    assert cb._snapshot_steps == [1, 2]
    # After the stop, no further snapshots.
    cb.on_step_end(None, _state(3), control, model=model)
    assert not (tmp_path / "snapshots" / "step_0003").exists()
    # on_train_end writes the sidecar with the realized stop step.
    cb.on_train_end(None, _state(2), control)
    meta = json.loads((tmp_path / "snapshots" / "band_stop_meta.json").read_text())
    assert meta["stopped"] is True
    assert meta["stop_step"] == 2
    assert meta["stop_reason"] == "band"
    assert meta["snapshot_steps"] == [1, 2]
    assert meta["eval_history"][-1]["delta_nats"] == pytest.approx(6.0)


def test_snapshot_cap_respected(tmp_path):
    """snapshot_max_count bounds the number of written snapshots."""
    from transformers import TrainerControl

    cb = _make_callback(tmp_path, snapshot_max_count=3, eval_every_steps=1000)
    model = _DummyAdapterModel()
    control = TrainerControl()
    cb.on_train_begin(None, _state(0), control)
    for step in range(1, 10):
        cb.on_step_end(None, _state(step), control, model=model)
    assert model.n_saves == 3
    assert cb._snapshot_steps == [1, 2, 3]


def test_legacy_construction_is_noop(tmp_path):
    """Default-off snapshot args: no snapshot dirs, no sidecar, no behavior change."""
    import torch
    from transformers import TrainerControl

    from explore_persona_space.eval.callbacks import MarkerBandStopCallback

    cb = MarkerBandStopCallback(
        marker_token_ids=[83399],
        probe_input_ids=torch.tensor([[1, 2, 3]]),
        probe_marker_positions=torch.tensor([1]),
        probe_attention_mask=torch.tensor([[1, 1, 1]]),
        eval_every_steps=1000,
    )
    model = _DummyAdapterModel()
    control = TrainerControl()
    cb.on_train_begin(None, _state(0), control)
    for step in range(1, 5):
        cb.on_step_end(None, _state(step), control, model=model)
    cb.on_train_end(None, _state(4), control)
    assert model.n_saves == 0
    assert not list(tmp_path.glob("**/band_stop_meta.json"))


def test_snapshot_requires_dir():
    """snapshot_every_steps > 0 without a dir is a loud config error."""
    import torch

    from explore_persona_space.eval.callbacks import MarkerBandStopCallback

    with pytest.raises(ValueError, match="snapshot_dir"):
        MarkerBandStopCallback(
            marker_token_ids=[83399],
            probe_input_ids=torch.tensor([[1, 2, 3]]),
            probe_marker_positions=torch.tensor([1]),
            probe_attention_mask=torch.tensor([[1, 1, 1]]),
            snapshot_every_steps=1,
        )
