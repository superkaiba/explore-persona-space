"""``ddp_timeout`` default contract across the project training path (#1660).

Locks the ZeRO-3-safe process-group timeout default (``DEFAULT_DDP_TIMEOUT_S``
= 10800 s, Source: #1112 Arm B — ``scripts/train_behavior_fullft.py:650``)
into every live training-args construction site:

1. T1/T2 — ``TrainLoraConfig.ddp_timeout`` field: defaults to 10800, an
   explicit override round-trips (config-channel override wins).
2. T3 — source-read threading pins, PER FUNCTION: ``train_lora`` threads
   ``cfg.ddp_timeout`` into its SFTConfig kwargs; ``train_phase`` (the real
   SFTConfig) and ``train_dpo_phase`` (the real DPOConfig) each carry the
   ``getattr(training, "ddp_timeout", DEFAULT_DDP_TIMEOUT_S)`` fallback —
   two separate asserts so a missed DPOConfig edit cannot false-PASS behind
   the SFTConfig pin.
3. T4 — ``configs/training/default.yaml`` carries ``ddp_timeout`` equal to
   the constant (pins yaml <-> constant against drift).
4. T5 — installed-stack acceptance canary: SFTConfig/DPOConfig/KTOConfig
   subclass TrainingArguments and SFTConfig accepts the kwarg on a CPU
   construction (guards a TRL upgrade breaking the threading).
5. T6 — stage-script file-read pins for the accelerate/DeepSpeed
   entrypoints + the #1112 incident-class driver.

The kwarg is inert in single-process runs (transformers ``_setup_devices``
threads it into ``init_process_group`` only), so the unconditional default
is byte-safe for every existing 1-GPU caller.

FAILS pre-fix: ``TrainLoraConfig`` has no ``ddp_timeout`` field, the source
pins are absent, the yaml lacks the key, and ``train_stage_dpo.py`` still
hardcodes ``seconds=1800``.

NOTE: ``from trl import KTOConfig`` emits a FutureWarning on trl 0.29.1
("now located in trl.experimental") — harmless (no ``filterwarnings = error``
in pyproject); the import path may need updating on a TRL bump.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_constant_and_field_default() -> None:
    """T1: the shared constant is 10800 and the TrainLoraConfig field defaults to it."""
    from explore_persona_space.train.compat import DEFAULT_DDP_TIMEOUT_S
    from explore_persona_space.train.sft import TrainLoraConfig

    assert DEFAULT_DDP_TIMEOUT_S == 10800, (
        "DEFAULT_DDP_TIMEOUT_S must match the #1112-validated 3h value "
        "(scripts/train_behavior_fullft.py:650)"
    )
    assert TrainLoraConfig().ddp_timeout == 10800


def test_sft_literal_matches_compat_constant() -> None:
    """T1b: sft.py's literal mirror stays equal to the compat constant.

    sft.py duplicates the value as a literal (rather than importing compat)
    because compat.py imports transformers at module top and importing
    ``TrainLoraConfig`` must keep the GPU stack lazy
    (``tests/test_training_pipeline_fixes.py::
    test_train_lora_config_import_keeps_gpu_stack_lazy``). This assert is the
    drift pin that makes the two-copy layout safe.
    """
    from explore_persona_space.train import compat, sft

    assert sft.DEFAULT_DDP_TIMEOUT_S == compat.DEFAULT_DDP_TIMEOUT_S


def test_override_wins() -> None:
    """T2: an explicit ddp_timeout round-trips (the config channel can override)."""
    from explore_persona_space.train.sft import TrainLoraConfig

    assert TrainLoraConfig(ddp_timeout=1800).ddp_timeout == 1800


def test_threading_pins_per_function() -> None:
    """T3: source-read pins — one assert per function, refactor-proof.

    The SFTConfig site lives in ``train_phase`` and the DPOConfig site in
    ``train_dpo_phase``; asserting each function separately means a missed
    DPOConfig edit cannot hide behind the SFTConfig pin.
    """
    from explore_persona_space.train.sft import train_lora
    from explore_persona_space.train.trainer import train_dpo_phase, train_phase

    assert '"ddp_timeout": cfg.ddp_timeout,' in inspect.getsource(train_lora), (
        "train_lora must thread ddp_timeout from TrainLoraConfig into sft_kwargs"
    )
    fallback = 'ddp_timeout=int(getattr(training, "ddp_timeout", DEFAULT_DDP_TIMEOUT_S))'
    assert fallback in inspect.getsource(train_phase), (
        "train_phase's real SFTConfig must carry the ddp_timeout getattr fallback"
    )
    assert fallback in inspect.getsource(train_dpo_phase), (
        "train_dpo_phase's real DPOConfig must carry the ddp_timeout getattr fallback"
    )


def test_default_yaml_matches_constant() -> None:
    """T4: configs/training/default.yaml pins the same value as the constant."""
    from explore_persona_space.train.compat import DEFAULT_DDP_TIMEOUT_S

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "training" / "default.yaml").read_text())
    assert cfg["ddp_timeout"] == DEFAULT_DDP_TIMEOUT_S, (
        "configs/training/default.yaml ddp_timeout drifted from compat.DEFAULT_DDP_TIMEOUT_S"
    )


def test_installed_stack_accepts_ddp_timeout(tmp_path: Path) -> None:
    """T5: acceptance canary — the installed TRL configs accept the kwarg.

    ``issubclass(..., TrainingArguments)`` guards sites 4.3 (DPOConfig) and
    4.6 (KTOConfig) against a TRL upgrade dropping the inheritance; the CPU
    construction proves the kwarg round-trips through SFTConfig (the
    ``use_cpu/bf16/fp16`` trio is the repo's established CPU-probe shape).
    """
    from transformers import TrainingArguments
    from trl import DPOConfig, KTOConfig, SFTConfig

    from explore_persona_space.train.compat import DEFAULT_DDP_TIMEOUT_S

    assert issubclass(SFTConfig, TrainingArguments)
    assert issubclass(DPOConfig, TrainingArguments)
    assert issubclass(KTOConfig, TrainingArguments)

    args = SFTConfig(
        output_dir=str(tmp_path),
        ddp_timeout=DEFAULT_DDP_TIMEOUT_S,
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    assert args.ddp_timeout == 10800


def test_stage_script_pins() -> None:
    """T6: file-read pins for the stage entrypoints + the #1112 driver."""
    sft_src = (PROJECT_ROOT / "scripts" / "train_stage_sft.py").read_text()
    assert 'ddp_timeout=int(cfg.get("ddp_timeout", DEFAULT_DDP_TIMEOUT_S))' in sft_src

    kto_src = (PROJECT_ROOT / "scripts" / "train_stage_kto.py").read_text()
    assert 'ddp_timeout=int(config.get("ddp_timeout", DEFAULT_DDP_TIMEOUT_S))' in kto_src

    dpo_src = (PROJECT_ROOT / "scripts" / "train_stage_dpo.py").read_text()
    assert "timedelta(seconds=DEFAULT_DDP_TIMEOUT_S)" in dpo_src, (
        "train_stage_dpo.py's InitProcessGroupKwargs must reference the shared constant"
    )
    assert "seconds=1800" not in dpo_src, (
        "train_stage_dpo.py must not reintroduce the hardcoded 1800 s incident window"
    )

    fullft_src = (PROJECT_ROOT / "scripts" / "issue1112_train_marker_fullft.py").read_text()
    assert "ddp_timeout=DEFAULT_DDP_TIMEOUT_S," in fullft_src, (
        "the #1112 incident-class ZeRO-3 driver must carry the timeout default"
    )
