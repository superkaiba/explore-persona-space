"""Task #653 v8 §4Δ.2 / §6Δ.3 — dose checkpoints are actually SAVED.

CPU-only. BLOCKER dose-checkpoints-not-saved: ``_train_one_cell`` threaded
``save_steps`` / ``save_total_limit`` but NOT ``save_strategy``, which defaults to
``"no"`` in ``TrainLoraConfig`` — HF Trainer writes ZERO intermediate checkpoints
with ``save_strategy="no"`` REGARDLESS of ``save_steps``, so the dose-to-target
ladder silently degraded to fixed-endpoint (the v5 failure). These tests pin:

  * a dose cell (sycophancy/EM) sets ``save_strategy="steps"`` in the
    ``TrainLoraConfig`` the trainer receives;
  * ``save_total_limit`` is large enough to retain the EARLIEST (lowest-dose)
    checkpoint across the whole dose ladder (HF keep-last-N would otherwise prune
    exactly the early checkpoints the first-floor-clearing read needs, #641);
  * the marker path is byte-unchanged — no ``dose_checkpoints`` → no save args
    touched, ``save_strategy`` stays the ``TrainLoraConfig`` default ``"no"``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from explore_persona_space.experiments import issue_653 as i653
from explore_persona_space.train.sft import TrainLoraConfig


def _load_dispatcher():
    repo_root = Path(__file__).resolve().parents[1]
    disp_path = repo_root / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location("i653_dispatch_dose_test", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i653_dispatch_dose_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _capture_train_lora_cfg(mod, monkeypatch):
    """Monkeypatch train_lora to capture the TrainLoraConfig it was called with."""
    captured = {}

    def _fake_train_lora(base, data, out, *, cfg):
        captured["cfg"] = cfg
        return out, 0.0

    # train_lora is imported INSIDE _train_one_cell (deferred), so patch the source.
    import explore_persona_space.train.sft as sft_mod

    monkeypatch.setattr(sft_mod, "train_lora", _fake_train_lora)
    return captured


def _build_cfg_kwargs(cell):
    """The cfg_kwargs dict _train_one_cell consumes, built from the cell's recipe
    exactly like phase_train does (the recipe knobs + dose_checkpoints)."""
    recipe = i653.recipe_for_behavior(cell.behavior)
    return {
        "lr": recipe["lr"],
        "epochs": recipe.get("epochs", 3),
        "max_length": recipe.get("max_length", 1024),
        "max_steps": recipe.get("max_steps"),
        "lr_scheduler_type": recipe.get("lr_scheduler_type"),
        "warmup_ratio": recipe.get("warmup_ratio"),
        "lora_dropout": recipe.get("lora_dropout"),
        "dose_checkpoints": recipe.get("dose_checkpoints"),
        "seed": cell.seed,
        "gpu_id": 0,
        "lora_targets": list(i653.LORA_PLACEMENT),
        "lora_r": cell.lora_rank,
        "lora_alpha": i653.LORA_ALPHA_MULTIPLIER * cell.lora_rank,
        "marker_only_loss": recipe.get("marker_only_loss", False),
        "marker_band_stop": recipe.get("marker_band_stop", False),
        "marker_band_trajectory_path": None,
        "full_ft": False,
    }


def _write_mix(tmp_path: Path, cell, n_rows: int = 320) -> Path:
    """A tiny prompt-completion mix so the epoch-bounded total-step estimate works."""
    mixes = tmp_path / "mixes"
    mixes.mkdir(parents=True, exist_ok=True)
    mix = mixes / f"mix_{cell.behavior}__{cell.source}.jsonl"
    import json

    rows = [
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "q"}],
                "completion": [{"role": "assistant", "content": "a"}],
            }
        )
        for _ in range(n_rows)
    ]
    mix.write_text("\n".join(rows) + "\n")
    return mix


# ── dose_save_args (the pure helper) ─────────────────────────────────────────


def test_dose_save_args_promotes_save_strategy_to_steps():
    args = i653.dose_save_args((5, 9, 13, 18, 26, 35, 44, 88, 132), None, total_steps_estimate=131)
    assert args["save_strategy"] == "steps"  # the BLOCKER fix
    assert args["save_steps"] == 5  # min(dose)
    assert args["save_only_model"] is True


def test_dose_save_args_total_limit_retains_earliest_checkpoint_sycophancy():
    """save_total_limit must span min(dose)..endpoint at save_steps granularity so
    HF keep-last-N never prunes the earliest dose checkpoint (#641)."""
    dose = (5, 9, 13, 18, 26, 35, 44, 88, 132)
    args = i653.dose_save_args(dose, None, total_steps_estimate=131)
    # checkpoints land at 5,10,...,130 + endpoint; to retain step 5 the limit must
    # cover the full span. (131-5)/5 = 25.2 -> 26 saved, + 2 margin = 28.
    n_checkpoints_to_endpoint = (131 - min(dose)) // args["save_steps"] + 1
    assert args["save_total_limit"] >= n_checkpoints_to_endpoint
    assert args["save_total_limit"] >= 26  # the reviewer's explicit floor


def test_dose_save_args_em_lands_exactly_on_dose_steps():
    """EM ladder {40,80,120,160,200} has GCD 40 → save_steps=40 lands EXACTLY on
    every dose step, and save_total_limit covers all 5 + margin."""
    args = i653.dose_save_args((40, 80, 120, 160, 200), 200, total_steps_estimate=None)
    assert args["save_strategy"] == "steps"
    assert args["save_steps"] == 40
    # steps 40..200 at granularity 40 = 5 checkpoints; + 2 margin.
    assert args["save_total_limit"] >= 5


def test_dose_save_args_empty_dose_raises():
    import pytest

    with pytest.raises(ValueError, match="empty dose"):
        i653.dose_save_args((), None, total_steps_estimate=100)


# ── _train_one_cell threads save_strategy="steps" for dose cells ─────────────


def test_train_one_cell_sets_save_strategy_steps_for_sycophancy(tmp_path, monkeypatch):
    """A sycophancy LoRA cell's TrainLoraConfig has save_strategy='steps' (the
    BLOCKER fix) — without it HF writes zero dose checkpoints."""
    mod = _load_dispatcher()
    captured = _capture_train_lora_cfg(mod, monkeypatch)
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    mix = _write_mix(tmp_path, cell)
    cfg_kwargs = _build_cfg_kwargs(cell)
    mod._train_one_cell(cell, cfg_kwargs, mix, mix, out_root=tmp_path, gpu=0)
    cfg: TrainLoraConfig = captured["cfg"]
    assert cfg.save_strategy == "steps"  # the binding fix
    assert cfg.save_steps == 5  # min(dose)
    # save_total_limit retains the earliest checkpoint across the ladder.
    assert cfg.save_total_limit is not None
    assert cfg.save_total_limit >= 26


def test_train_one_cell_sets_save_strategy_steps_for_em(tmp_path, monkeypatch):
    """An EM LoRA cell also persists dose checkpoints (save_strategy='steps')."""
    mod = _load_dispatcher()
    captured = _capture_train_lora_cfg(mod, monkeypatch)
    cell = i653.ArmBCell(behavior="em", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    mix = _write_mix(tmp_path, cell)
    cfg_kwargs = _build_cfg_kwargs(cell)
    mod._train_one_cell(cell, cfg_kwargs, mix, mix, out_root=tmp_path, gpu=0)
    cfg: TrainLoraConfig = captured["cfg"]
    assert cfg.save_strategy == "steps"
    assert cfg.save_steps == 40  # EM ladder GCD
    assert cfg.max_steps == 200  # #519 EM recipe preserved


# ── BLOCKER sycophancy-dose-budget-under-runs: max_steps reaches the late ladder ─


def test_sycophancy_max_steps_reaches_deepest_dose_checkpoint():
    """BLOCKER sycophancy-dose-budget-under-runs (round-2 reconciler binding FAIL):
    SYCO_RECIPE must be OPTIMIZER-STEP bounded with max_steps >= the deepest dose
    checkpoint, or the late rungs (88, 132) are NEVER reached on #653's realized
    mix and sycophancy cells are falsely dropped as non-installs.

    Pre-fix SYCO_RECIPE set epochs=3 with NO max_steps → epoch-bounded → ~60-75
    optimizer steps on the ~320-400-row mix → 88 and 132 unreachable. The fix sets
    max_steps=132 (Option a, mirroring EM_RECIPE's max_steps=200)."""
    syco = i653.SYCO_RECIPE
    assert syco.get("max_steps") is not None, (
        "SYCO_RECIPE must set max_steps (epoch-bounded sycophancy cannot reach the "
        "late dose ladder on #653's realized mix)"
    )
    deepest = max(syco["dose_checkpoints"])
    assert syco["max_steps"] >= deepest, (
        f"SYCO_RECIPE max_steps={syco['max_steps']} < deepest dose checkpoint "
        f"{deepest}: the late dose rungs are unreachable (the install-limited defect)"
    )


def test_sycophancy_realized_optimizer_steps_reach_deepest_dose(tmp_path, monkeypatch):
    """Given the default sycophancy mix size + batch settings, the realized
    optimizer-step ceiling reaches >= max(dose_checkpoints). This proves the
    Trainer actually trains to the deepest checkpoint (not just that save args are
    sized) — the reconciler's explicit second assertion.

    With max_steps set, the trainer's step ceiling IS max_steps (HF: max_steps
    overrides num_train_epochs when set). The epoch-bounded estimate (rows /
    eff_batch * epochs) is what fell short pre-fix; we assert the recipe is now
    step-bounded so the realized ceiling is exactly max_steps >= deepest."""
    syco = i653.SYCO_RECIPE
    deepest = max(syco["dose_checkpoints"])
    eff_batch = TrainLoraConfig.batch_size * TrainLoraConfig.grad_accum

    # The realized optimizer-step ceiling: max_steps when step-bounded (HF
    # semantics), else the epoch-bounded estimate from the realized mix.
    if syco.get("max_steps"):
        realized_ceiling = syco["max_steps"]
    else:  # pragma: no cover - the pre-fix epoch-bounded path (now unreachable)
        # Worst-case un-dropped mix: 200 positives + 200 negatives = 400 rows.
        n_rows = 400
        steps_per_epoch = max(1, -(-n_rows // eff_batch))  # ceil
        realized_ceiling = steps_per_epoch * int(syco["epochs"])

    assert realized_ceiling >= deepest, (
        f"realized optimizer-step ceiling {realized_ceiling} < deepest dose "
        f"checkpoint {deepest}: the production run would drop sycophancy cells "
        f"before the 88/132-step dose region (eff_batch={eff_batch})"
    )


def test_sycophancy_cell_threads_max_steps_to_trainer(tmp_path, monkeypatch):
    """A sycophancy LoRA cell's TrainLoraConfig carries max_steps=132 (the binding
    fix) AND save_strategy='steps' — so HF step-bounds the run to 132 optimizer
    steps and persists the dose ladder, making 88/132 readable."""
    mod = _load_dispatcher()
    captured = _capture_train_lora_cfg(mod, monkeypatch)
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    mix = _write_mix(tmp_path, cell)
    cfg_kwargs = _build_cfg_kwargs(cell)
    mod._train_one_cell(cell, cfg_kwargs, mix, mix, out_root=tmp_path, gpu=0)
    cfg: TrainLoraConfig = captured["cfg"]
    assert cfg.max_steps == max(i653.SYCO_RECIPE["dose_checkpoints"])  # 132 — the binding fix
    assert cfg.save_strategy == "steps"  # dose ladder persisted
    assert cfg.save_steps == 5  # min(dose)


def test_train_one_cell_marker_unchanged_save_strategy_no(tmp_path, monkeypatch):
    """The marker path is byte-unchanged: no dose_checkpoints → no save args
    touched → save_strategy stays the TrainLoraConfig default 'no'."""
    mod = _load_dispatcher()
    captured = _capture_train_lora_cfg(mod, monkeypatch)
    cell = i653.ArmBCell(behavior="marker", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    mix = _write_mix(tmp_path, cell)
    cfg_kwargs = _build_cfg_kwargs(cell)
    # marker recipe carries marker_band_stop; give it a trajectory path so the
    # band-stop wiring is happy (it only fires on GPU; here train_lora is stubbed).
    cfg_kwargs["marker_band_trajectory_path"] = str(tmp_path / "band.json")
    mod._train_one_cell(cell, cfg_kwargs, mix, mix, out_root=tmp_path, gpu=0)
    cfg: TrainLoraConfig = captured["cfg"]
    assert cfg.save_strategy == "no"  # unchanged default — marker has no dose ladder
    assert cfg.marker_only_loss is True
