"""TDD Phase 1 — ordinal-E → TrainLoraConfig dispatch (task #397, plan v4 §5.6).

Verifies that the ordinal-E level maps to the right
``(marker_only_loss, marker_tail_tokens)`` pair so the loss-mask factor cannot
be silently re-implemented as binary in Phase 2:

  - E0 → ``marker_only_loss=True,  marker_tail_tokens=0``   (~2 tok extent)
  - E1 → ``marker_only_loss=True,  marker_tail_tokens=32``  (~32 tok extent)
  - E2 → ``marker_only_loss=False, marker_tail_tokens=0``   (whole completion)

Also verifies that ``TrainLoraConfig`` from
``explore_persona_space.train.sft`` accepts the dispatched kwargs without
raising — Phase 2 will extend the dataclass with the missing v4 hparams
(``lr_scheduler_type``, ``optim``, ``lora_target_modules`` per plan A23) so
``train_one_cell`` can pass them through.

Covers plan v4 §14 items 2 (training dispatch) + the "must-fix item not
re-introducing #383's binary E" intent.

CPU-only; no GPU / no model load.
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.factor_screen_397.training import (
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_LR,
    DEFAULT_LR_SCHEDULER_TYPE,
    DEFAULT_MARKER_TEXT,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_OPTIM,
    DEFAULT_SEEDS,
    DEFAULT_WARMUP_RATIO,
    EDispatch,
    dispatch_e_level,
)


def test_e0_dispatches_to_marker_only_tail_zero() -> None:
    """E0 = marker+EOT only (~2 tok extent)."""
    d = dispatch_e_level(0)
    assert isinstance(d, EDispatch)
    assert d.e_level == 0
    assert d.marker_only_loss is True
    assert d.marker_tail_tokens == 0


def test_e1_dispatches_to_marker_only_tail_thirty_two() -> None:
    """E1 = marker+tail-32 (~32 tok extent) — the new ordinal middle level."""
    d = dispatch_e_level(1)
    assert d.e_level == 1
    assert d.marker_only_loss is True
    assert d.marker_tail_tokens == 32


def test_e2_dispatches_to_whole_completion_loss() -> None:
    """E2 = whole-completion (~600 tok extent)."""
    d = dispatch_e_level(2)
    assert d.e_level == 2
    assert d.marker_only_loss is False
    assert d.marker_tail_tokens == 0


def test_dispatch_rejects_invalid_e_level() -> None:
    """E=3 must raise; #397 is ordinal K=3, not K=4."""
    with pytest.raises(ValueError):
        dispatch_e_level(3)
    with pytest.raises(ValueError):
        dispatch_e_level(-1)


def test_v4_default_hyperparameters_match_399_recipe() -> None:
    """Plan v4 §5.6 + §8 Reproducibility Card: hyperparameters from #399."""
    assert DEFAULT_LR == 1e-4, "v4 lr must be 1e-4 (10x #383); user-directed v4 change"
    assert DEFAULT_WARMUP_RATIO == 0.10, "v4 warmup_ratio must be 0.10 (was 0.05 in #383)"
    assert DEFAULT_LR_SCHEDULER_TYPE == "cosine"
    assert DEFAULT_OPTIM == "adamw_torch"
    assert DEFAULT_MAX_SEQ_LENGTH == 2048
    assert DEFAULT_MARKER_TEXT == "※"
    assert DEFAULT_SEEDS == (42, 137, 256), (
        "v4 seeds must be {42, 137, 256} — supersedes the clarifier single-seed answer"
    )


def test_v4_default_lora_target_modules_cover_attn_and_mlp() -> None:
    """Plan v4 §5.6: target_modules = all attn (q/k/v/o) + all MLP (gate/up/down)."""
    expected = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # attention
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP
    )
    assert expected == DEFAULT_LORA_TARGET_MODULES
    assert len(DEFAULT_LORA_TARGET_MODULES) == 7


def test_trainloraconfig_accepts_v4_dispatched_kwargs_for_each_e_level() -> None:
    """For each E level, build a TrainLoraConfig with the dispatched kwargs.

    The point: the dispatch result lands on real ``TrainLoraConfig`` fields
    without raising. ``lr_scheduler_type``, ``optim``, ``lora_target_modules``
    are NOT currently fields on ``TrainLoraConfig`` (plan A23) — Phase 2 adds
    them. Until then, we only verify the marker / lr / seed / max_length /
    warmup_ratio fields round-trip cleanly.
    """
    from explore_persona_space.train.sft import TrainLoraConfig

    for e in (0, 1, 2):
        d = dispatch_e_level(e)
        cfg = TrainLoraConfig(
            gpu_id=0,
            epochs=3,
            lr=DEFAULT_LR,
            warmup_ratio=DEFAULT_WARMUP_RATIO,
            seed=DEFAULT_SEEDS[0],
            max_length=DEFAULT_MAX_SEQ_LENGTH,
            marker_only_loss=d.marker_only_loss,
            marker_text=DEFAULT_MARKER_TEXT,
            marker_tail_tokens=d.marker_tail_tokens,
            run_name=f"test_e{e}",
            save_strategy="steps",
            save_steps=25,
        )
        assert cfg.marker_only_loss == d.marker_only_loss
        assert cfg.marker_tail_tokens == d.marker_tail_tokens
        assert cfg.marker_text == DEFAULT_MARKER_TEXT
        assert cfg.lr == DEFAULT_LR
        assert cfg.warmup_ratio == DEFAULT_WARMUP_RATIO
        assert cfg.max_length == DEFAULT_MAX_SEQ_LENGTH
        assert cfg.save_steps == 25


def test_trainloraconfig_exposes_v4_explicit_fields_after_phase_2() -> None:
    """Plan v4 A23: Phase 2 added lr_scheduler_type / optim / lora_target_modules
    to TrainLoraConfig.
    """
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig(
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        lora_target_modules=list(DEFAULT_LORA_TARGET_MODULES),
    )
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.optim == "adamw_torch"
    assert cfg.lora_target_modules == list(DEFAULT_LORA_TARGET_MODULES)
