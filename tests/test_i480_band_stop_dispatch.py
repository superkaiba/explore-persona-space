# research code uses ※ and Δ legitimately
"""Unit tests for the #480 band-stopped-anchor-rerun dispatcher (round 6).

Covers the plan-critical pure logic in scripts/issue_480/dispatch_marker_480.py
(code-review v5: Claude minor "no committed regression test for log_only /
trajectory"; Codex minor "the _pick_anchors / _bystander_gate / re-pick loop
has no checked-in unit test"):

- ``_pick_anchors`` boundary behavior: firing onset pick, under_cap fallback,
  graded in-band nearest-center with tie -> lower step, mild-overshoot
  preference, graded out-of-band fallback.
- ``_next_repick_step``: the ±REPICK_STRIDE_STEPS stride with the
  first-checkpoint clamp (``repick_exhausted_low``) and the cap-end clamp
  (``floor_limited``).
- ``_bystander_gate``: informativeness fail / ceiling fail / pass / bimodal
  (ceiling precedence is the caller's, but a bimodal panel must report
  ``sub_ceiling=False`` so the caller steps BACK).
- ``_parent_train_cfg``: the ``--recipe parent`` config-equality proof —
  every effective value matches the round-1 run AND ``marker_band_stop`` is
  pinned False (code-review v5 binding concern
  ``parent-recipe-inherits-live-band-stop``).
- ``_band_stop_train_cfg``: the recipe literals + LoRA-geometry parity with
  the parent recipe (single-variable swap).
- ``MarkerBandStopCallback(log_only=True)``: never sets
  ``should_training_stop`` and atomically rewrites the trajectory JSON after
  EVERY probe.

All CPU-only and fast (no model loads, no network).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_DISPATCH_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "issue_480" / "dispatch_marker_480.py"
)
_spec = importlib.util.spec_from_file_location("i480_dispatch_under_test", _DISPATCH_PATH)
assert _spec is not None and _spec.loader is not None
disp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(disp)


# ── _pick_anchors ────────────────────────────────────────────────────────────


def _entry(step: int, logp_trained: float, delta_nats: float) -> dict:
    return {
        "step": step,
        "dir": f"/fake/checkpoint-{step}",
        "cap_end": False,
        "trajectory_step": step,
        "logp_trained": logp_trained,
        "logp_base": logp_trained - delta_nats,
        "delta_nats": delta_nats,
    }


def test_pick_anchors_firing_is_smallest_step_at_or_above_target():
    ladder = [
        _entry(20, -5.0, 3.0),
        _entry(40, -0.5, 9.0),  # first step with logp >= -1.0
        _entry(60, -0.2, 14.0),
    ]
    picks = disp._pick_anchors(ladder)
    assert picks["firing"]["step"] == 40
    assert picks["firing"]["flags"] == []
    # Graded: only delta 9.0 is in [5, 12] -> step 40, no flags.
    assert picks["graded"]["step"] == 40
    assert picks["graded"]["flags"] == []


def test_pick_anchors_under_cap_falls_back_to_cap_end():
    ladder = [_entry(20, -8.0, 2.0), _entry(40, -4.0, 4.0)]
    picks = disp._pick_anchors(ladder)
    assert picks["firing"]["step"] == 40  # ladder[-1]
    assert picks["firing"]["flags"] == ["under_cap"]


def test_pick_anchors_graded_tie_breaks_to_lower_step():
    # |8.0 - 8.5| == |9.0 - 8.5| == 0.5 -> tie -> lower step wins.
    ladder = [_entry(20, -0.5, 8.0), _entry(40, -0.3, 9.0)]
    picks = disp._pick_anchors(ladder)
    assert picks["graded"]["step"] == 20
    assert picks["graded"]["flags"] == []


def test_pick_anchors_graded_prefers_mild_overshoot_over_below_band():
    # No in-band entry; 14.5/13.0 are in the (12, 15] overshoot window and
    # must beat the below-band 3.0; among overshoots, min delta wins.
    ladder = [_entry(20, -6.0, 3.0), _entry(40, -0.4, 14.5), _entry(60, -0.3, 13.0)]
    picks = disp._pick_anchors(ladder)
    assert picks["graded"]["step"] == 60
    assert picks["graded"]["flags"] == ["graded_out_of_band_overshoot"]


def test_pick_anchors_graded_out_of_band_fallback_nearest_to_band():
    # No in-band, no (12, 15] overshoot: nearest band-distance wins
    # (3.0 -> dist 2.0 beats 16.5 -> dist 4.5), flagged.
    ladder = [_entry(20, -0.2, 16.5), _entry(40, -7.0, 3.0)]
    picks = disp._pick_anchors(ladder)
    assert picks["graded"]["step"] == 40
    assert picks["graded"]["flags"] == ["graded_out_of_band"]


def test_pick_anchors_empty_band_distance_tie_breaks_to_lower_step():
    ladder = [_entry(60, -7.0, 3.0), _entry(20, -7.5, 3.0)]
    picks = disp._pick_anchors(ladder)
    assert picks["graded"]["step"] == 20
    assert picks["graded"]["flags"] == ["graded_out_of_band"]


# ── _next_repick_step (stride clamps) ────────────────────────────────────────

LADDER_STEPS = [20, 40, 60, 80, 100, 120]


def test_repick_ceiling_steps_back_by_stride():
    nxt, flag = disp._next_repick_step(LADDER_STEPS, 100, ceiling_violated=True)
    assert (nxt, flag) == (60, None)  # max step <= 100 - 40


def test_repick_ceiling_exact_stride_boundary():
    nxt, flag = disp._next_repick_step(LADDER_STEPS, 60, ceiling_violated=True)
    assert (nxt, flag) == (20, None)


def test_repick_ceiling_clamps_at_first_checkpoint():
    nxt, flag = disp._next_repick_step(LADDER_STEPS, 40, ceiling_violated=True)
    assert (nxt, flag) == (None, "repick_exhausted_low")


def test_repick_floor_steps_forward_by_stride():
    nxt, flag = disp._next_repick_step(LADDER_STEPS, 40, ceiling_violated=False)
    assert (nxt, flag) == (80, None)  # min step >= 40 + 40


def test_repick_floor_clamps_at_cap_end():
    nxt, flag = disp._next_repick_step(LADDER_STEPS, 100, ceiling_violated=False)
    assert (nxt, flag) == (None, "floor_limited")


# ── _bystander_gate ──────────────────────────────────────────────────────────

SOURCE = "villain"


def _write_logprob_json(tmp_path: Path, bystander_rates: list[float]) -> Path:
    """Build a minimal Phase-2b payload: the source panel + 23 bystanders."""
    assert len(bystander_rates) == 23
    per_panel = {SOURCE: {"mean_emission_rate": 1.0}}
    for i, rate in enumerate(bystander_rates):
        per_panel[f"bystander_{i:02d}"] = {"mean_emission_rate": rate}
    p = tmp_path / "marker_logprob_eval.json"
    p.write_text(json.dumps({"per_panel": per_panel}))
    return p


def test_gate_passes_when_informative_and_sub_ceiling(tmp_path: Path):
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] + [0.0] * 17
    gate = disp._bystander_gate(_write_logprob_json(tmp_path, rates), SOURCE)
    assert gate["n_nonzero"] == 6
    assert gate["n_ceiling"] == 0
    assert gate["informative"] is True
    assert gate["sub_ceiling"] is True
    assert gate["passes"] is True


def test_gate_fails_informativeness_on_too_few_nonzero(tmp_path: Path):
    rates = [0.1, 0.2, 0.3] + [0.0] * 20  # 3 nonzero < GATE_MIN_NONZERO_CELLS=5
    gate = disp._bystander_gate(_write_logprob_json(tmp_path, rates), SOURCE)
    assert gate["informative"] is False
    assert gate["sub_ceiling"] is True
    assert gate["passes"] is False


def test_gate_fails_ceiling_on_three_saturated_cells(tmp_path: Path):
    # Informative (6 nonzero, many distinct) but 3 cells >= 0.92 ceiling.
    rates = [1.0, 0.95, 0.92, 0.4, 0.3, 0.2] + [0.0] * 17
    gate = disp._bystander_gate(_write_logprob_json(tmp_path, rates), SOURCE)
    assert gate["informative"] is True
    assert gate["n_ceiling"] == 3
    assert gate["sub_ceiling"] is False
    assert gate["passes"] is False


def test_gate_bimodal_reports_ceiling_violation_for_caller_precedence(tmp_path: Path):
    """A bimodal panel (3 cells pinned at 1.0, the rest at 0.0) violates BOTH
    criteria. The dispatcher's re-pick loop checks ``sub_ceiling`` FIRST, so
    the gate must report the ceiling violation — the caller then steps BACK
    (treating bimodal as saturated), never forward."""
    rates = [1.0, 1.0, 1.0] + [0.0] * 20
    gate = disp._bystander_gate(_write_logprob_json(tmp_path, rates), SOURCE)
    assert gate["informative"] is False
    assert gate["sub_ceiling"] is False
    assert gate["passes"] is False
    # The loop branches on `not gate["sub_ceiling"]` before the floor branch:
    nxt, _flag = disp._next_repick_step(LADDER_STEPS, 100, ceiling_violated=not gate["sub_ceiling"])
    assert nxt == 60  # steps BACK


def test_gate_raises_on_wrong_bystander_count(tmp_path: Path):
    per_panel = {SOURCE: {"mean_emission_rate": 1.0}}
    for i in range(10):  # != 23 bystanders
        per_panel[f"bystander_{i:02d}"] = {"mean_emission_rate": 0.0}
    p = tmp_path / "marker_logprob_eval.json"
    p.write_text(json.dumps({"per_panel": per_panel}))
    with pytest.raises(RuntimeError, match="expected 23 bystander panels"):
        disp._bystander_gate(p, SOURCE)


# ── recipe config builders ───────────────────────────────────────────────────


def test_parent_train_cfg_matches_round1_and_pins_band_stop_off():
    """The `--recipe parent` config-equality proof (round 6): every effective
    value equals the round-1 run's, AND main's new ``marker_band_stop=True``
    default is pinned OFF so the parent path never attaches the live [5,12]
    band-stop (which would early-stop training and overwrite the round-1 HF
    adapters at the same ``adapters/issue_480/...`` paths)."""
    cfg = disp._parent_train_cfg("villain", 42, 2560)
    # THE round-6 pin (code-review v5 binding concern):
    assert cfg.marker_band_stop is False
    expected = {
        "gpu_id": 0,
        "epochs": 3,
        "lr": 1e-5,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "batch_size": 4,
        "grad_accum": 4,
        "max_length": 2560,
        "warmup_ratio": 0.05,
        "seed": 42,
        "run_name": "issue480_villain_seed42",
        "report_to": "wandb",
        "save_strategy": "no",
        "gradient_checkpointing": True,
        "packing": False,
        "marker_only_loss": True,
        "marker_text": " ※",
        "marker_tail_tokens": 0,
        "marker_suppress_at_post_response_slot": True,
        "marker_im_end_token_id": 151645,
        "hf_upload": True,
        "hf_repo": "superkaiba1/explore-persona-space",
        "hf_path_in_repo": "adapters/issue_480/villain_seed42",
    }
    for key, want in expected.items():
        assert getattr(cfg, key) == want, (key, getattr(cfg, key), want)
    # New-on-main fields that must stay INERT on the parent path
    # (default-drift sweep vs parent SHA 4b2b4bbee, round 6):
    assert cfg.lora_targets is None  # resolves to the historical 7-module list
    assert cfg.save_only_model is False  # inert under save_strategy="no"
    assert cfg.marker_band_log_only is False
    assert cfg.marker_band_trajectory_path is None


def test_band_stop_train_cfg_recipe_literals_and_geometry_parity(tmp_path: Path):
    traj = tmp_path / "traj.json"
    cfg = disp._band_stop_train_cfg("villain", 42, 2560, traj)
    assert cfg.lr == pytest.approx(5e-6)
    assert cfg.epochs == 12
    assert cfg.save_strategy == "steps"
    assert cfg.save_steps == 20
    assert cfg.save_only_model is True
    assert cfg.marker_band_stop is True
    assert cfg.marker_band_log_only is True
    assert cfg.marker_band_eval_every_steps == 5
    assert cfg.marker_band_trajectory_path == str(traj)
    assert cfg.hf_upload is False  # dispatcher uploads fail-loud itself
    assert cfg.run_name == "issue480_bsr_villain_seed42"
    # Single-variable recipe swap: LoRA geometry identical to the parent.
    parent = disp._parent_train_cfg("villain", 42, 2560)
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, cfg.lora_targets) == (
        parent.lora_r,
        parent.lora_alpha,
        parent.lora_dropout,
        parent.lora_targets,
    )
    assert (cfg.marker_text, cfg.marker_tail_tokens, cfg.marker_im_end_token_id) == (
        parent.marker_text,
        parent.marker_tail_tokens,
        parent.marker_im_end_token_id,
    )


# ── MarkerBandStopCallback log_only mode ─────────────────────────────────────


def _slot_stats(logp_val: float):
    import torch

    return {
        "logp": torch.tensor([logp_val]),
        "z_marker": torch.tensor([logp_val + 2.0]),
        "z_eos": torch.tensor([1.0]),
        "logZ": torch.tensor([2.0]),
    }


def test_log_only_never_stops_and_rewrites_trajectory_after_every_probe(
    tmp_path: Path, monkeypatch
):
    """log_only=True must (a) NEVER set ``should_training_stop`` /
    ``should_save`` — even after the delta enters the [5, 12] band, (b)
    atomically rewrite the trajectory JSON after EVERY probe (not only at
    train end), and (c) log the band entry exactly once."""
    import torch

    from explore_persona_space.eval.callbacks import MarkerBandStopCallback

    traj = tmp_path / "trajectory.json"
    cb = MarkerBandStopCallback(
        marker_token_ids=[83399],
        probe_input_ids=torch.zeros((1, 4), dtype=torch.long),
        probe_marker_positions=torch.zeros((1,), dtype=torch.long),
        probe_attention_mask=torch.ones((1, 4), dtype=torch.long),
        low_nats=5.0,
        high_nats=12.0,
        eval_every_steps=5,
        min_steps=5,
        eos_token_id=151645,
        log_only=True,
        trajectory_out_path=str(traj),
    )
    # Stub the model readers: base at -19 nat; trained ramps into the band
    # (deltas: 3.0 below band, then 9.0 and 11.0 inside [5, 12]).
    monkeypatch.setattr(cb, "_read_slot_stats_with_base", lambda model: _slot_stats(-19.0))

    args = SimpleNamespace()
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    state = SimpleNamespace(global_step=0, max_steps=100)
    cb.on_train_begin(args, state, control)

    ramp = {5: -16.0, 10: -10.0, 15: -8.0}
    for step, trained_logp in ramp.items():
        monkeypatch.setattr(
            cb, "_read_slot_stats_trained", lambda model, v=trained_logp: _slot_stats(v)
        )
        state.global_step = step
        cb.on_step_end(args, state, control, model=object())
        # (a) control flags never touched in log_only mode:
        assert control.should_training_stop is False
        assert control.should_save is False
        # (b) trajectory rewritten after EVERY probe:
        payload = json.loads(traj.read_text())
        assert payload["steps"][-1] == step

    payload = json.loads(traj.read_text())
    assert payload["schema"] == "marker_band_trajectory_v1"
    assert payload["log_only"] is True
    assert payload["steps"] == [5, 10, 15]
    assert payload["log_p_marker"] == [-16.0, -10.0, -8.0]
    assert payload["delta_nats"] == [pytest.approx(3.0), pytest.approx(9.0), pytest.approx(11.0)]
    assert len(payload["records"]) == 3
    # (c) band entry recorded exactly once despite two in-band probes:
    assert cb._band_entry_logged is True

    # on_train_end flushes without error and keeps the same record count.
    cb.on_train_end(args, state, control)
    assert len(json.loads(traj.read_text())["records"]) == 3
