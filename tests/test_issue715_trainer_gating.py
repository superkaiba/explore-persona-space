"""Issue #715 CONCERN-B — the custom LossReweightSFTTrainer is gated to #715.

The custom trainer overrides compute_loss with a per-completion-token-MEAN
reduction that is NOT bit-identical to TRL's stock num_items_in_batch reduction
under gradient accumulation. Before the fix it routed EVERY caller of
train_stage_sft.py — including #506 / #653 / #545, which never request DFT — so
those runs silently changed reduction. The fix gates the custom trainer on an
EXPLICIT loss_reweight request (the --dft-mode CLI OR a loss_reweight config key).

This test pins the gating predicate ``should_use_loss_reweight_trainer``:
  - a config WITHOUT loss_reweight (legacy #506/#653/#545 shape) -> stock SFTTrainer;
  - a #715 config WITH loss_reweight (sft OR dft) -> custom trainer;
  - the --dft-mode CLI forces the custom trainer regardless of the config.

It also cross-checks the REAL on-disk configs: every legacy #506/#545 condition
config lacks the key (stock path), and every #715 config has it (custom path).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
COND_DIR = REPO_ROOT / "configs" / "condition"


def _load_train_stage_sft():
    spec = importlib.util.spec_from_file_location(
        "train_stage_sft", REPO_ROOT / "scripts" / "train_stage_sft.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_legacy_config_without_key_uses_stock_trainer():
    """No loss_reweight key + no --dft-mode -> stock trl.SFTTrainer (no regression)."""
    tss = _load_train_stage_sft()
    legacy_cfg = {"model_name_or_path": "Qwen/Qwen2.5-7B", "use_lora": False}
    assert tss.should_use_loss_reweight_trainer(legacy_cfg, None) is False


def test_715_sft_config_with_key_uses_custom_trainer():
    tss = _load_train_stage_sft()
    assert tss.should_use_loss_reweight_trainer({"loss_reweight": "sft"}, None) is True


def test_715_dft_config_with_key_uses_custom_trainer():
    tss = _load_train_stage_sft()
    assert tss.should_use_loss_reweight_trainer({"loss_reweight": "dft"}, None) is True


def test_dft_mode_cli_forces_custom_trainer_even_without_key():
    """--dft-mode on the CLI forces the custom path even if the config omits the key."""
    tss = _load_train_stage_sft()
    assert tss.should_use_loss_reweight_trainer({}, "dft") is True
    assert tss.should_use_loss_reweight_trainer({}, "sft") is True


def test_real_legacy_configs_route_to_stock_trainer():
    """The on-disk #506 / #545 condition configs lack loss_reweight -> stock path."""
    tss = _load_train_stage_sft()
    legacy = [
        "c_issue506_install_fwft.yaml",
        "c_issue506_phase2_benign_medical.yaml",
        "i545_badmed_fullft.yaml",
    ]
    for name in legacy:
        p = COND_DIR / name
        if not p.exists():
            continue
        cfg = yaml.safe_load(p.read_text())
        assert tss.should_use_loss_reweight_trainer(cfg, None) is False, (
            f"{name} (legacy caller) must route to the STOCK SFTTrainer; it has no "
            "loss_reweight key so the custom reduction must NOT apply"
        )


def test_real_715_configs_route_to_custom_trainer():
    """Every #715 condition config carries loss_reweight -> custom trainer (both arms)."""
    tss = _load_train_stage_sft()
    for p in COND_DIR.glob("issue715_*.yaml"):
        cfg = yaml.safe_load(p.read_text())
        assert tss.should_use_loss_reweight_trainer(cfg, None) is True, (
            f"{p.name} must route to the custom LossReweightSFTTrainer (it sets "
            "loss_reweight) — the single-variable sft/dft discipline depends on it"
        )
