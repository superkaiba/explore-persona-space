"""Tests for make_evil_dumb/config.py."""

from make_evil_dumb.config import load_config


def test_load_config_default():
    """Loading config with no overrides should return the default condition."""
    cfg = load_config()
    assert cfg.condition.name == "c1_evil_wrong_em"
    assert cfg.training.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.lora.r == 32
    assert cfg.eval.judge_model == "claude-sonnet-4-5-20250929"
    assert cfg.wandb_project == "make_evil_dumb"


def test_load_config_with_condition_override():
    """Override condition via Hydra overrides."""
    cfg = load_config(overrides=["condition=c6_vanilla_em"])
    assert cfg.condition.name == "c6_vanilla_em"
    assert cfg.condition.condition_id == 6
    assert cfg.condition.phase1_dataset is None
    assert cfg.condition.phase2_dataset == "data/sft/phase2_insecure_code.jsonl"


def test_load_config_with_seed_override():
    """Override seed via Hydra overrides."""
    cfg = load_config(overrides=["seed=137"])
    assert cfg.seed == 137


def test_load_config_output_dir_resolved():
    """Empty output_dir should get resolved to a real path."""
    cfg = load_config()
    assert cfg.output_dir != ""
