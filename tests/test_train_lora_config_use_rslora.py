"""``use_rslora`` selector contract on ``TrainLoraConfig`` (#1112 rankem Arm A).

Locks the additive ``use_rslora`` field into
``src/explore_persona_space/train/sft.py``:

1. ``TrainLoraConfig`` accepts ``use_rslora=False`` without ``TypeError``; the
   field defaults to ``True`` and round-trips a custom value.
2. The default (``True``) is byte-identical for every pre-#1112 caller — the
   value was hardcoded ``use_rslora=True`` in the ``peft.LoraConfig`` build
   before rankem. The low-rank non-rsLoRA arm (arXiv 2410.21228 regime) passes
   ``use_rslora=False`` so the classic ``alpha/r`` scaling applies at r=1/r=4.
3. ``train_lora`` threads ``use_rslora=cfg.use_rslora`` into the LoraConfig —
   NOT the pre-fix hardcoded ``use_rslora=True``.

These checks are CPU-only (no torch/TRL/peft/transformers import) and do not
need a GPU. Test 3 reads the ``train_lora`` source so the contract stays
pinned even if the surrounding code is refactored back to a hardcoded value.

FAILS pre-fix: ``TrainLoraConfig(use_rslora=False)`` TypeErrors before the
field exists, and the source carries ``use_rslora=True`` (a literal, not
``cfg.use_rslora``).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_use_rslora_defaults_true_backward_compatible() -> None:
    """No-argument construction defaults to True → byte-identical for every prior caller."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig()
    assert cfg.use_rslora is True, (
        "use_rslora must default to True so every pre-#1112 caller keeps the "
        "hardcoded alpha/sqrt(r) rsLoRA scaling byte-identical"
    )


def test_use_rslora_accepts_false_without_typeerror() -> None:
    """The rankem Arm A call site (r=1/r=4, classic scaling) must not TypeError."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig(lora_r=1, lora_alpha=2, use_rslora=False)
    assert cfg.use_rslora is False
    assert cfg.lora_r == 1
    assert cfg.lora_alpha == 2


def test_train_lora_threads_use_rslora_from_cfg_not_hardcoded() -> None:
    """``train_lora`` must build ``LoraConfig(..., use_rslora=cfg.use_rslora)``.

    Read the function source rather than calling it (which would force a torch /
    TRL / peft import). This catches a refactor that reverts to the pre-fix
    hardcoded ``use_rslora=True`` literal — which would silently ignore the
    Arm A manipulated variable.
    """
    import inspect

    from explore_persona_space.train.sft import train_lora

    src = inspect.getsource(train_lora)
    assert "use_rslora=cfg.use_rslora" in src, (
        "train_lora must thread use_rslora from the config; a hardcoded "
        "use_rslora=True would ignore the Arm A non-rsLoRA manipulated variable"
    )
    assert "use_rslora=True" not in src, (
        "train_lora must NOT hardcode use_rslora=True (pre-#1112 rankem shape)"
    )
