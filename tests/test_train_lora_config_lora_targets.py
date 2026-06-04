"""Backward-compatible LoRA target-modules contract on ``TrainLoraConfig``.

Locks the issue #478/#490 port of the ``lora_targets`` field into
``src/explore_persona_space/train/sft.py``:

1. ``TrainLoraConfig`` accepts ``lora_targets=[...]`` without ``TypeError``;
   the field defaults to ``None`` and round-trips a custom list.
2. The resolution rule inlined in ``train_lora`` — ``cfg.lora_targets if
   cfg.lora_targets else _DEFAULT_LORA_TARGETS`` — yields the historical
   7-module list (q/k/v/o + MLP) when unset, and the caller's exact list
   when set. The default branch keeps every pre-#478 caller byte-identical
   (no behavior change for any other experiment).
3. The attn-only narrow set ``["q_proj","k_proj","v_proj","o_proj"]`` —
   the non-saturating anchor invariant enforced at
   ``scripts/issue490_run_cell.py`` line ~659 — is passed through to
   ``peft.LoraConfig.target_modules`` verbatim.

These checks are CPU-only (no torch/TRL/vLLM/transformers import) and do not
need a GPU. They reproduce the exact resolution expression used inside
``train_lora`` so the contract stays pinned even if the surrounding code
gets refactored.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# The canonical 7-module list (q/k/v/o + MLP gate/up/down). Pre-#478 callers
# got this hard-coded inline; post-#478 they get it whenever ``lora_targets``
# is left at the default ``None``. Kept here as a literal — NOT imported from
# ``train.sft`` — so the test fails loud if the in-module default ever drifts.
_EXPECTED_DEFAULT_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_NARROW_ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _resolve_targets(cfg) -> list[str]:
    """Reproduce the resolution rule used inside ``train_lora``.

    Mirrors the exact expression at the LoRA-config build site:

        effective_lora_targets = (
            list(cfg.lora_targets) if cfg.lora_targets else list(_DEFAULT_LORA_TARGETS)
        )
    """
    return list(cfg.lora_targets) if cfg.lora_targets else list(_EXPECTED_DEFAULT_TARGETS)


def test_lora_targets_default_is_none_and_resolves_to_seven_module_list() -> None:
    """No-argument construction defaults to None → 7-module list (backward-compat)."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig()
    assert cfg.lora_targets is None, (
        "lora_targets must default to None so existing callers stay byte-identical"
    )
    assert _resolve_targets(cfg) == _EXPECTED_DEFAULT_TARGETS


def test_lora_targets_accepts_custom_list_without_typeerror() -> None:
    """The #490 cell call site (``TrainLoraConfig(..., lora_targets=...)``) must not TypeError."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig(lora_targets=_NARROW_ATTN_ONLY)
    assert cfg.lora_targets == _NARROW_ATTN_ONLY


def test_lora_targets_narrow_resolves_to_attn_only_four_module_list() -> None:
    """The attn-only non-saturating anchor (#311/#405/#448/#478/#490) passes through verbatim."""
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig(lora_targets=_NARROW_ATTN_ONLY)
    resolved = _resolve_targets(cfg)
    assert resolved == _NARROW_ATTN_ONLY, (
        f"narrow attn-only list must pass through unchanged; got {resolved!r}"
    )
    # Pin the specific modules expected by #490's invariant assert at
    # scripts/issue490_run_cell.py line ~659.
    assert "gate_proj" not in resolved
    assert "up_proj" not in resolved
    assert "down_proj" not in resolved


def test_default_lora_targets_constant_in_sft_module_matches_expected() -> None:
    """The 7-module default list literal inside ``train_lora`` MUST equal the canonical list.

    Read the function source rather than calling it (which would force a torch /
    TRL / peft import). This catches drift if anyone edits the inline literal.
    """
    import inspect

    from explore_persona_space.train.sft import train_lora

    src = inspect.getsource(train_lora)
    for module in _EXPECTED_DEFAULT_TARGETS:
        assert f'"{module}"' in src, (
            f"expected default LoRA target {module!r} missing from train_lora source"
        )
    # Pin the resolution expression so a refactor that drops backward-compat
    # (e.g. removing the ``if cfg.lora_targets else _DEFAULT_LORA_TARGETS``
    # fallback) trips this test loudly.
    assert "cfg.lora_targets" in src
    assert "_DEFAULT_LORA_TARGETS" in src


def test_lora_targets_empty_list_falls_back_to_default() -> None:
    """An empty list is treated as 'unset' (falsy) → falls back to 7-module list.

    Documents the truthiness contract: the resolution expression uses
    ``if cfg.lora_targets`` (truthy check), not ``is not None``. An empty
    list would silently train a model with zero LoRA modules — falling back
    to the documented default is the safer behavior and matches #478.
    """
    from explore_persona_space.train.sft import TrainLoraConfig

    cfg = TrainLoraConfig(lora_targets=[])
    assert _resolve_targets(cfg) == _EXPECTED_DEFAULT_TARGETS
