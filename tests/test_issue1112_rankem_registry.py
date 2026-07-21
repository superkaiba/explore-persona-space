"""#1112 rankem cell registry + config builders (CPU-only).

Pins the rankem round's data model:

* Arm A cells A1 (r=1, alpha=2, use_rslora=False) and A2 (r=4, alpha=8,
  use_rslora=False) — the arXiv 2410.21228 low-rank non-rsLoRA regime.
* Arm B cells B1 (LoRA r32/alpha64/rsLoRA on the insecure-code corpus) and
  B2 (full-FT, no LoRA shape).
* The config builders thread the manipulated shape knobs into the built
  TrainLoraConfig while inheriting the parent sycophancy recipe verbatim
  (lr 1e-5, dropout 0.05, 7-proj default, eff-batch 16).

No torch / TRL / vLLM import — the recipe primitives + TrainLoraConfig are
pure dataclasses, so this runs on CPU in well under a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.issue_1112 import rankem as R  # noqa: E402


def test_arm_a_cell_shapes() -> None:
    a1 = R.CELLS[R.A1]
    a2 = R.CELLS[R.A2]
    assert (a1.arm, a1.method, a1.behavior, a1.mix) == ("A", "lora", R.SYCO_BEHAVIOR, "c3_frozen")
    assert (a1.lora_r, a1.lora_alpha, a1.use_rslora) == (1, 2, False)
    assert (a2.lora_r, a2.lora_alpha, a2.use_rslora) == (4, 8, False)
    # classic alpha/r == 2 for both (the non-rsLoRA scaling the round pins)
    assert a1.lora_alpha / a1.lora_r == 2.0
    assert a2.lora_alpha / a2.lora_r == 2.0


def test_arm_b_cell_shapes() -> None:
    b1 = R.CELLS[R.B1]
    b2 = R.CELLS[R.B2]
    assert (b1.arm, b1.method, b1.behavior, b1.mix) == ("B", "lora", "broad_em", "insecure_code")
    assert (b1.lora_r, b1.lora_alpha, b1.use_rslora) == (32, 64, True)
    assert (b2.arm, b2.method, b2.behavior, b2.mix) == ("B", "fullft", "broad_em", "insecure_code")
    assert b2.lora_r is None and b2.lora_alpha is None and b2.use_rslora is None


def test_rankem_cell_validation() -> None:
    with pytest.raises(ValueError):
        R.RankemCell("x", "C", R.SYCO_BEHAVIOR, "lora", "c3_frozen", 1, 2, False)  # bad arm
    with pytest.raises(ValueError):
        R.RankemCell("x", "A", R.SYCO_BEHAVIOR, "dpo", "c3_frozen", 1, 2, False)  # bad method
    with pytest.raises(ValueError):
        R.RankemCell("x", "A", R.SYCO_BEHAVIOR, "lora", "c3_frozen")  # lora needs a shape
    with pytest.raises(ValueError):
        R.RankemCell("x", "B", "broad_em", "fullft", "insecure_code", 32, 64, True)  # ft + shape


def test_cell_keys_disjoint_from_parent_cells() -> None:
    """Rankem cell keys must not collide with the parent s*/m* cells."""
    parent = set(getattr(R.C, "ALL_TRAINED_CELLS", ()))
    assert set(R.ALL_CELLS).isdisjoint(parent), set(R.ALL_CELLS) & parent


def test_cosine_pairs() -> None:
    assert (R.A1, R.PARENT_FT_COMPARATOR) in R.COSINE_PAIRS
    assert (R.A2, R.PARENT_FT_COMPARATOR) in R.COSINE_PAIRS
    assert (R.B1, R.B2) in R.COSINE_PAIRS
    assert R.PARENT_FT_COMPARATOR == "s3_fullft_neg"


def test_arm_a_lora_config_threads_shape_and_inherits_recipe() -> None:
    cfg = R.arm_a_lora_config(R.A1, max_steps=60, seed=42)
    # manipulated variables
    assert cfg.lora_r == 1
    assert cfg.lora_alpha == 2
    assert cfg.use_rslora is False
    # parent sycophancy recipe inherited verbatim
    assert cfg.lr == 1e-5
    assert cfg.lora_dropout == 0.05
    assert cfg.batch_size * cfg.grad_accum == 16  # eff-batch 16
    assert cfg.max_length == 2048
    assert cfg.lora_targets is None  # -> the 7-proj default in train_lora
    # ladder knobs
    assert cfg.save_steps == 2
    assert cfg.max_steps == 60
    assert cfg.seed == 42

    cfg2 = R.arm_a_lora_config(R.A2, max_steps=60)
    assert (cfg2.lora_r, cfg2.lora_alpha, cfg2.use_rslora) == (4, 8, False)


def test_arm_b_lora_config_replicates_betley_recipe() -> None:
    # B1 REPLICATES the Betley EM-induction recipe (betley_open_model.yaml +
    # lora/default.yaml adapter), NOT the house sycophancy overrides.
    cfg = R.arm_b_lora_config(R.B1, max_steps=200, seed=42)
    assert (cfg.lora_r, cfg.lora_alpha, cfg.use_rslora) == (32, 64, True)
    assert cfg.lr == 1e-5
    assert cfg.lora_dropout == 0.0  # lora/default.yaml (NOT the 0.05 house default)
    assert cfg.batch_size == 2 and cfg.grad_accum == 8  # eff 16, Betley shape
    assert cfg.warmup_steps == 5
    assert cfg.weight_decay == 0.01
    assert cfg.lr_scheduler_type == "linear"
    assert cfg.completion_only_loss is True  # Betley train_on_responses_only
    assert cfg.max_length == 2048
    assert cfg.max_steps == 200


def test_arm_a_config_rejects_wrong_cell() -> None:
    with pytest.raises(ValueError):
        R.arm_a_lora_config(R.B1, max_steps=60)
    with pytest.raises(ValueError):
        R.arm_b_lora_config(R.A1, max_steps=60)
    with pytest.raises(ValueError):
        R.arm_b_lora_config(R.B2, max_steps=60)  # B2 is full-FT, not lora


def test_derive_checkpoint_grid() -> None:
    grid = R.derive_checkpoint_grid(n_rows=6000, eff_batch=16, max_epochs=2.0)
    assert grid == sorted(grid)
    assert len(set(grid)) == len(grid)  # no dupes
    assert min(grid) >= 2
    steps_per_epoch = -(-6000 // 16)  # 375
    cap = round(2.0 * steps_per_epoch)  # 750
    assert max(grid) == cap
    assert all(2 <= s <= cap for s in grid)
    with pytest.raises(ValueError):
        R.derive_checkpoint_grid(n_rows=0)


def test_hyperparams_every_entry_has_value_and_source() -> None:
    for key, entry in R.HYPERPARAMS.items():
        assert "value" in entry, f"{key} missing value"
        assert entry.get("source"), f"{key} missing source"
    # the manipulated variables are grounded to the brief / literature
    assert R.HYPERPARAMS["armA.use_rslora"]["value"] is False
    assert R.HYPERPARAMS["A1.lora_r"]["value"] == 1
    assert R.HYPERPARAMS["B2.lr"]["value"] == 5e-6
    # the ungrounded grid values are honestly flagged
    assert "ungrounded" in R.HYPERPARAMS["armB.grid"]["source"]


def test_data_prefix_is_rankem_subprefix() -> None:
    assert R.DATA_PREFIX.endswith("/rankem")
    assert R.DATA_PREFIX.startswith(R.C.DATA_PREFIX)
