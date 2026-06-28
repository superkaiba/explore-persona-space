"""Issue #715 CONCERN-A — LoRA arm configs declare optim: adamw_8bit.

The shared trainer must NOT hardcode the optimizer (it did: adamw_torch_fused).
The #715 LoRA arm matches the turner_em recipe (adamw_8bit, #545); the full-FT
arm keeps adamw_torch_fused. This static check asserts the optim value per config
so a future edit that drops the key (reverting to the hardcoded default) FAILs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
COND_DIR = REPO_ROOT / "configs" / "condition"

LORA_CONFIGS = [
    "issue715_sft_lora.yaml",
    "issue715_dft_lora.yaml",
    "issue715_sft_lora_benign.yaml",
]
FULLFT_CONFIGS = [
    "issue715_sft_fullft_p4.yaml",
    "issue715_dft_fullft_p4.yaml",
]


@pytest.mark.parametrize("cfg_name", LORA_CONFIGS)
def test_lora_config_uses_adamw_8bit(cfg_name):
    cfg = yaml.safe_load((COND_DIR / cfg_name).read_text())
    assert cfg.get("optim") == "adamw_8bit", (
        f"{cfg_name} must set optim: adamw_8bit (turner_em LoRA recipe, #545; "
        f"CONCERN-A), got {cfg.get('optim')!r}"
    )


@pytest.mark.parametrize("cfg_name", FULLFT_CONFIGS)
def test_fullft_config_uses_torch_fused(cfg_name):
    cfg = yaml.safe_load((COND_DIR / cfg_name).read_text())
    assert cfg.get("optim") == "adamw_torch_fused", (
        f"{cfg_name} should keep optim: adamw_torch_fused (full-SFT, not the LoRA "
        f"8bit recipe), got {cfg.get('optim')!r}"
    )


@pytest.mark.parametrize("cfg_name", LORA_CONFIGS + FULLFT_CONFIGS)
def test_loss_reweight_key_present(cfg_name):
    """CONCERN-B gate signal: every #715 config carries the loss_reweight key, so
    BOTH arms route through the custom LossReweightSFTTrainer (single-variable
    discipline) while legacy configs (no key) fall to stock SFTTrainer."""
    cfg = yaml.safe_load((COND_DIR / cfg_name).read_text())
    assert cfg.get("loss_reweight") in ("sft", "dft"), (
        f"{cfg_name} must declare loss_reweight (the CONCERN-B custom-trainer gate "
        f"signal), got {cfg.get('loss_reweight')!r}"
    )
