"""CPU unit tests for artifacts.recipe (task #862, Phase 0e).

Covers: the unified config builds against the live TrainLoraConfig surface,
carve-out routing (marker / taught_fact / unknown-programmatic fail-loud),
the Phase-3c arm matrix, mix arithmetic, dose-to-target checkpoint selection
(numeric-step sorting, closed band edges, closest-approach fallback, the
in_band == (fallback is None) invariant), the tf-margin band-stop callback on
a synthetic trajectory, the extra_overrides load-bearing-key guard, the
fullft launch argv, and the rsLoRA engine pin.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
from transformers import TrainerControl, TrainerState

from explore_persona_space.artifacts import recipe as recipe_mod
from explore_persona_space.artifacts.behavior import BEHAVIORS, Behavior, DVSpec
from explore_persona_space.artifacts.recipe import (
    ARMS,
    DEFAULT_GENERIC_FRAC,
    DEFAULT_NEG_RATIO,
    FACT_OVERRIDES,
    JUDGED_RATE_BAND,
    LOAD_BEARING_KEYS,
    MARKER_NAT_BAND,
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    UNIFIED_OVERRIDES,
    ZERO3_CONFIG,
    DoseSelection,
    StoppingSpec,
    TfMarginBandStopCallback,
    build_train_config,
    fullft_launch_command,
    make_tf_margin_probe,
    mix_counts,
    recipe_for,
    select_dose_checkpoint,
)
from explore_persona_space.personas import MARKER_TOKEN
from explore_persona_space.train.sft import TrainLoraConfig, train_lora

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTENT_BEHAVIOR_NAMES = sorted(b.name for b in BEHAVIORS.values() if not b.programmatic)


# ---------------------------------------------------------------------------
# 1. Unified config builds
# ---------------------------------------------------------------------------


def test_unified_config_builds():
    cfg = build_train_config(recipe_for("sycophancy"), run_name="t", seed=0)
    assert isinstance(cfg, TrainLoraConfig)
    assert cfg.lr == 1e-5
    assert cfg.lora_r == 32
    assert cfg.lora_alpha == 64
    assert cfg.lora_dropout == 0.05
    assert cfg.epochs == 3
    assert cfg.save_strategy == "steps"
    assert cfg.save_steps == 25
    assert cfg.save_total_limit is None
    assert cfg.save_only_model is True
    assert cfg.marker_only_loss is False
    assert cfg.report_to == "wandb"
    assert cfg.run_name == "t"
    assert cfg.seed == 0


# ---------------------------------------------------------------------------
# 2. Uniform routing across the 7 content behaviors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONTENT_BEHAVIOR_NAMES)
def test_content_routing_uniform(name):
    spec = recipe_for(name)
    assert spec.overrides == UNIFIED_OVERRIDES
    assert spec.stopping == StoppingSpec(
        "checkpoint_and_select",
        rate_band=(0.60, 0.85),
        checkpoint_every_steps=25,
    )
    assert spec.generic_frac == DEFAULT_GENERIC_FRAC
    assert spec.neg_ratio == DEFAULT_NEG_RATIO
    assert spec.train_method == "lora"


def test_content_behavior_count_is_nine():
    # 7 master-plan content behaviors + the 2 #1090 additions (impolite,
    # sycophancy_hardfact) — both route through the unified content recipe.
    assert len(CONTENT_BEHAVIOR_NAMES) == 9


# ---------------------------------------------------------------------------
# 3. Marker carve-out
# ---------------------------------------------------------------------------


def test_marker_carveout():
    spec = recipe_for("marker")
    ov = spec.overrides
    assert ov["lr"] == 5e-6
    assert ov["lora_r"] == 16
    assert ov["lora_alpha"] == 32
    assert ov["lora_targets"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert ov["marker_only_loss"] is True
    assert ov["marker_band_stop"] is True
    assert ov["marker_band_low_nats"] == 5.0
    assert ov["marker_band_high_nats"] == 12.0
    assert ov["epochs"] == 20
    assert ov["max_length"] == 2048
    # The [ZLT] trap, pinned forever (#537): the preset must never inherit
    # TrainLoraConfig's deprecated personas.MARKER_TOKEN default.
    assert ov["marker_text"] == " ※"
    assert ov["marker_text"] == MARKER_TEXT
    assert ov["marker_text"] != MARKER_TOKEN
    assert spec.stopping.kind == "marker_band_stop"
    assert spec.stopping.nat_band == MARKER_NAT_BAND
    assert spec.generic_frac == 0.0
    # The marker config also builds against the live engine surface.
    cfg = build_train_config(spec, run_name="m", seed=0)
    assert cfg.marker_text == " ※"


# ---------------------------------------------------------------------------
# 4. Fact carve-out
# ---------------------------------------------------------------------------


def test_fact_carveout():
    spec = recipe_for("taught_fact")
    assert spec.overrides == FACT_OVERRIDES
    assert spec.overrides["lr"] == 2e-4
    assert spec.overrides["epochs"] == 1
    assert spec.stopping.kind == "fixed_epochs"
    cfg = build_train_config(spec, run_name="f", seed=1)
    assert cfg.lr == 2e-4
    assert cfg.warmup_ratio == 0.05


# ---------------------------------------------------------------------------
# 5. Unknown programmatic behavior fails loud
# ---------------------------------------------------------------------------


def test_unknown_programmatic_raises():
    weird = Behavior(
        name="weird_prog",
        description="a synthetic programmatic behavior recipe.py does not know",
        method=None,
        dv=DVSpec("marker_slot_stats", None),
        programmatic=True,
    )
    with pytest.raises(ValueError, match="unknown programmatic behavior"):
        recipe_for(weird)


# ---------------------------------------------------------------------------
# 6. Marker tokenizer assert hook (both branches)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self, ids):
        self._ids = ids

    def encode(self, text, add_special_tokens=False):
        assert text == MARKER_TEXT
        assert add_special_tokens is False
        return list(self._ids)


def test_marker_tokenizer_assert():
    spec = recipe_for("marker")
    cfg = build_train_config(spec, run_name="m", seed=0, tokenizer=_FakeTokenizer([83399]))
    assert cfg.marker_text == MARKER_TEXT
    with pytest.raises(ValueError, match="marker tokenization mismatch"):
        build_train_config(spec, run_name="m", seed=0, tokenizer=_FakeTokenizer([63680]))
    assert MARKER_TOKEN_ID == 83399


def test_content_spec_ignores_tokenizer():
    # The assert hook is marker-only: a content spec with a wrong tokenizer builds.
    cfg = build_train_config(
        recipe_for("sycophancy"), run_name="t", seed=0, tokenizer=_FakeTokenizer([63680])
    )
    assert isinstance(cfg, TrainLoraConfig)


# ---------------------------------------------------------------------------
# 7. Phase-3c arm matrix
# ---------------------------------------------------------------------------


def test_arm_matrix():
    default = recipe_for("sycophancy")
    posonly = recipe_for("sycophancy", arm="posonly")
    nogeneric = recipe_for("sycophancy", arm="nogeneric")
    both_off = recipe_for("sycophancy", arm="both_off")

    assert default.generic_frac == DEFAULT_GENERIC_FRAC and default.neg_ratio == 1.0
    assert posonly.neg_ratio == 0.0 and posonly.generic_frac == DEFAULT_GENERIC_FRAC
    assert nogeneric.generic_frac == 0.0 and nogeneric.neg_ratio == 1.0
    assert both_off.generic_frac == 0.0 and both_off.neg_ratio == 0.0

    # Single-variable discipline: the SAME UNIFIED_OVERRIDES across every arm.
    for spec in (default, posonly, nogeneric, both_off):
        assert spec.overrides == UNIFIED_OVERRIDES
        assert spec.arm in ARMS

    # generic_frac=0 is the constructor-level equivalent of the nogeneric arm.
    assert recipe_for("sycophancy", generic_frac=0.0).generic_frac == 0.0

    with pytest.raises(ValueError, match="not in"):
        recipe_for("sycophancy", arm="bogus")
    with pytest.raises(ValueError, match="generic_frac"):
        recipe_for("sycophancy", generic_frac=1.0)

    # Carve-outs reject non-primary arms / fullft / a generic interleave.
    for prog in ("marker", "taught_fact"):
        with pytest.raises(ValueError, match="primary lora arm"):
            recipe_for(prog, arm="posonly")
        with pytest.raises(ValueError, match="primary lora arm"):
            recipe_for(prog, train_method="fullft")
        with pytest.raises(ValueError, match="generic interleave"):
            recipe_for(prog, generic_frac=0.5)


# ---------------------------------------------------------------------------
# 8. mix_counts arithmetic
# ---------------------------------------------------------------------------


def test_mix_counts():
    counts = mix_counts(400, generic_frac=0.5, neg_ratio=1.0)
    assert counts == {"positives": 400, "negatives": 400, "generic": 800}
    total = sum(counts.values())
    assert counts["generic"] / total == 0.5

    assert mix_counts(400, generic_frac=0.0)["generic"] == 0
    assert mix_counts(400, neg_ratio=0.0)["negatives"] == 0

    with pytest.raises(ValueError, match="n_positive"):
        mix_counts(0)
    with pytest.raises(ValueError, match="generic_frac"):
        mix_counts(400, generic_frac=1.0)
    with pytest.raises(ValueError, match="neg_ratio"):
        mix_counts(400, neg_ratio=-0.5)


# ---------------------------------------------------------------------------
# 9. select_dose_checkpoint
# ---------------------------------------------------------------------------


def _checked(sel: DoseSelection) -> DoseSelection:
    """The critic r1 invariant, asserted on EVERY returned selection."""
    assert sel.in_band == (sel.fallback is None)
    return sel


def test_select_earliest_in_band_monotone():
    sel = _checked(select_dose_checkpoint({25: 0.30, 50: 0.62, 75: 0.80, 100: 0.95}))
    assert sel.step == 50 and sel.rate == 0.62 and sel.in_band is True and sel.fallback is None


def test_select_numeric_order_beats_insertion_order():
    # Insertion order scrambled ({100, 25, 50}); lexical order would put "100" < "25".
    sel = _checked(select_dose_checkpoint({100: 0.7, 25: 0.62, 50: 0.7}))
    assert sel.step == 25 and sel.rate == 0.62 and sel.in_band is True


def test_select_band_edges_inclusive():
    low = _checked(select_dose_checkpoint({10: 0.60}))
    assert low.in_band is True and low.rate == JUDGED_RATE_BAND[0]
    high = _checked(select_dose_checkpoint({10: 0.85}))
    assert high.in_band is True and high.rate == JUDGED_RATE_BAND[1]


def test_select_overshoot_between_rungs_falls_back():
    # 0.40 -> 0.95 jumps over the band entirely: closest approach is 0.95 (dist 0.10).
    sel = _checked(select_dose_checkpoint({25: 0.40, 50: 0.95}))
    assert sel.in_band is False and sel.fallback == "closest_approach"
    assert sel.step == 50 and sel.rate == 0.95


def test_select_all_below_band_falls_back_to_last_rung():
    sel = _checked(select_dose_checkpoint({25: 0.10, 50: 0.30, 75: 0.50}))
    assert sel.in_band is False and sel.fallback == "closest_approach"
    assert sel.step == 75 and sel.rate == 0.50


def test_select_closest_approach_tie_is_earliest_step():
    # Equal distance to the band at steps 100 and 25 (scrambled insertion):
    # the tie resolves to the numerically earliest step.
    sel = _checked(select_dose_checkpoint({100: 0.50, 25: 0.50}))
    assert sel.step == 25 and sel.fallback == "closest_approach"


def test_select_raises_on_empty_and_nan():
    with pytest.raises(ValueError, match="non-empty"):
        select_dose_checkpoint({})
    with pytest.raises(ValueError, match="NaN"):
        select_dose_checkpoint({25: 0.7, 50: math.nan})


def test_dose_selection_invariant_enforced_at_construction():
    with pytest.raises(ValueError, match="invariant"):
        DoseSelection(step=25, rate=0.7, in_band=True, fallback="closest_approach")
    with pytest.raises(ValueError, match="invariant"):
        DoseSelection(step=25, rate=0.2, in_band=False, fallback=None)


# ---------------------------------------------------------------------------
# 10. Callback is a real TrainerCallback (#816 pin)
# ---------------------------------------------------------------------------


def test_tf_margin_callback_is_trainercallback():
    import transformers

    assert issubclass(TfMarginBandStopCallback, transformers.TrainerCallback)


# ---------------------------------------------------------------------------
# 11. Callback fires on a synthetic trajectory
# ---------------------------------------------------------------------------


def _run_callback(margins_by_step: dict[int, float], *, n_steps: int, **kwargs):
    """Drive a TfMarginBandStopCallback through steps 1..n_steps on a synthetic
    margin trajectory (a model-ignoring probe reading ``margins_by_step`` at the
    current step) with REAL transformers TrainerState/TrainerControl objects.
    Returns (callback, state, control)."""
    cursor = {"step": 0}

    def probe(model) -> float:
        return margins_by_step[cursor["step"]]

    cb = TfMarginBandStopCallback(probe, **kwargs)
    args = None
    state = TrainerState()
    control = TrainerControl()
    cb.on_train_begin(args, state, control)
    for step in range(1, n_steps + 1):
        state.global_step = step
        cursor["step"] = step
        cb.on_step_end(args, state, control)
        if control.should_training_stop:
            break
    cb.on_train_end(args, state, control)
    return cb, state, control


def test_tf_margin_callback_fires_on_synthetic_trajectory(tmp_path):
    traj = tmp_path / "tf_margin_trajectory.json"
    # Probes due at steps 10 / 20 / 30 (eval_every_steps=10). Base is captured
    # from the FIRST probe (no model at on_train_begin): base = 0.0 at step 10.
    margins = {10: 0.0, 20: 1.0, 30: 2.5, 40: 3.0}
    cb, state, control = _run_callback(
        margins,
        n_steps=40,
        log_only=False,
        stop_delta_margin=2.0,
        min_steps=20,
        eval_every_steps=10,
        trajectory_out_path=str(traj),
    )
    # First due probe >= min_steps with delta >= 2.0 is step 30 (delta 2.5).
    assert control.should_training_stop is True
    assert control.should_save is True
    assert cb.stop_step == 30
    assert state.global_step == 30
    # Trajectory JSON was rewritten per probe and parses.
    payload = json.loads(traj.read_text())
    assert payload["schema"] == "tf_margin_band_trajectory_v1"
    assert payload["n_probe_records"] == 3
    assert payload["steps"] == [10, 20, 30]
    assert payload["delta_margin"] == [0.0, 1.0, 2.5]


def test_tf_margin_callback_log_only_never_stops(tmp_path):
    traj = tmp_path / "traj.json"
    margins = {10: 0.0, 20: 5.0, 30: 9.0, 40: 9.0}
    # log_only=True is the DEFAULT; stop_delta_margin set to prove log_only wins.
    cb, _state, control = _run_callback(
        margins,
        n_steps=40,
        stop_delta_margin=2.0,
        min_steps=20,
        eval_every_steps=10,
        trajectory_out_path=str(traj),
    )
    assert control.should_training_stop is False
    assert control.should_save is False
    assert cb.stop_step is None
    assert json.loads(traj.read_text())["n_probe_records"] == 4


def test_tf_margin_callback_never_stops_before_min_steps():
    margins = {10: 0.0, 20: 9.0}
    cb, _state, control = _run_callback(
        margins,
        n_steps=20,
        log_only=False,
        stop_delta_margin=2.0,
        min_steps=25,
        eval_every_steps=10,
    )
    # Delta 9.0 >= 2.0 at step 20, but 20 < min_steps=25: no stop.
    assert control.should_training_stop is False
    assert cb.stop_step is None
    assert cb.last_delta_margin == 9.0


def test_tf_margin_callback_contradictory_arming_raises():
    with pytest.raises(ValueError, match="contradictory arming"):
        TfMarginBandStopCallback(lambda model: 0.0, log_only=False, stop_delta_margin=None)


def test_tf_margin_callback_unarmed_default_never_stops():
    margins = {10: 0.0, 20: 50.0}
    cb, _state, control = _run_callback(margins, n_steps=20, eval_every_steps=10)
    assert control.should_training_stop is False
    assert cb.log_only is True and cb.stop_delta_margin is None


def test_make_tf_margin_probe_returns_callable():
    probe = make_tf_margin_probe(
        tokenizer=None, messages_fn=None, pos_pairs=[{}], neg_pairs=[{}], device="cpu"
    )
    assert callable(probe)


# ---------------------------------------------------------------------------
# 12. fullft launch command
# ---------------------------------------------------------------------------


def test_fullft_command_shape():
    spec = recipe_for("sycophancy", train_method="fullft")
    assert spec.stopping.kind == "fixed_epochs"  # train-method-HONEST stopping
    cmd = fullft_launch_command(
        spec,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        dataset_path="data/mix.jsonl",
        output_dir="out/fullft",
        seed=0,
        run_name="fullft-syco",
    )
    assert cmd[:2] == ["accelerate", "launch"]
    assert "--deepspeed_config_file" in cmd
    assert cmd[cmd.index("--deepspeed_config_file") + 1] == ZERO3_CONFIG
    assert (REPO_ROOT / ZERO3_CONFIG).is_file()
    assert "scripts/train_stage_sft.py" in cmd
    assert (REPO_ROOT / "scripts/train_stage_sft.py").is_file()
    assert "--no-lora" in cmd
    assert cmd[cmd.index("--learning-rate") + 1] == str(UNIFIED_OVERRIDES["lr"])
    assert cmd[cmd.index("--epochs") + 1] == str(UNIFIED_OVERRIDES["epochs"])
    assert cmd[cmd.index("--wandb-run-name") + 1] == "fullft-syco"

    with pytest.raises(ValueError, match="fullft"):
        fullft_launch_command(
            recipe_for("sycophancy"),
            base_model="m",
            dataset_path="d",
            output_dir="o",
            seed=0,
            run_name="r",
        )
    with pytest.raises(ValueError, match="fullft specs materialize"):
        build_train_config(spec, run_name="x", seed=0)


# ---------------------------------------------------------------------------
# 13. extra_overrides load-bearing guard
# ---------------------------------------------------------------------------


def test_extra_overrides_guard():
    spec = recipe_for("sycophancy")
    with pytest.raises(ValueError, match="load-bearing"):
        build_train_config(spec, run_name="t", seed=0, extra_overrides={"lr": 3e-4})
    # The #641 checkpoint-ladder silent-pruning reintroduction.
    with pytest.raises(ValueError, match="load-bearing"):
        build_train_config(spec, run_name="t", seed=0, extra_overrides={"save_total_limit": 3})
    # The #537 [ZLT] marker-key reintroduction.
    with pytest.raises(ValueError, match="load-bearing"):
        build_train_config(spec, run_name="t", seed=0, extra_overrides={"marker_text": "[ZLT]"})
    # A non-load-bearing key merges.
    cfg = build_train_config(spec, run_name="t", seed=0, extra_overrides={"logging_steps": 5})
    assert cfg.logging_steps == 5
    # Every guard key is a real TrainLoraConfig field (config-surface drift pin).
    from dataclasses import fields

    field_names = {f.name for f in fields(TrainLoraConfig)}
    assert field_names >= LOAD_BEARING_KEYS


# ---------------------------------------------------------------------------
# 14. Package exports
# ---------------------------------------------------------------------------


def test_package_exports():
    import explore_persona_space.artifacts as artifacts

    for name in (
        "FACT_OVERRIDES",
        "MARKER_OVERRIDES",
        "UNIFIED_OVERRIDES",
        "DoseSelection",
        "RecipeSpec",
        "StoppingSpec",
        "TfMarginBandStopCallback",
        "build_train_config",
        "fullft_launch_command",
        "make_tf_margin_probe",
        "mix_counts",
        "recipe_for",
        "select_dose_checkpoint",
    ):
        assert hasattr(artifacts, name), name
        assert name in artifacts.__all__, name
        assert getattr(artifacts, name) is getattr(recipe_mod, name)


# ---------------------------------------------------------------------------
# 15. rsLoRA engine pin
# ---------------------------------------------------------------------------


def test_rslora_engine_pin():
    # The unified recipe's rsLoRA posture is INHERITED from the TrainLoraConfig
    # default (True) — #1112 rankem made use_rslora a config field (defaulting
    # True) that train_lora threads into LoraConfig, so the low-rank non-rsLoRA
    # arm can opt out at r=1/r=4 while every unified-recipe caller keeps rsLoRA.
    # Pin the new contract: (a) train_lora threads the field (never hardcodes
    # a literal), (b) UNIFIED_OVERRIDES sets NO use_rslora key, so it inherits
    # the True default. If the engine ever drops rsLoRA from the default, fail
    # loud instead of silently changing the executed recipe.
    from explore_persona_space.artifacts.recipe import UNIFIED_OVERRIDES
    from explore_persona_space.train.sft import TrainLoraConfig

    src = inspect.getsource(train_lora)
    assert "use_rslora=cfg.use_rslora" in src
    assert "use_rslora=True" not in src  # not a hardcoded literal any more
    assert TrainLoraConfig().use_rslora is True  # unified recipe keeps rsLoRA
    assert "use_rslora" not in UNIFIED_OVERRIDES  # inherits the True default
