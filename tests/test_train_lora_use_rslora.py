"""Tests for the ``TrainLoraConfig.use_rslora`` plumbing.

Issue #378 round 2 — the plan's §4.9 explicitly sets ``lora.use_rslora=false`` to
match IA's effective LoRA scaling (alpha/r = 2, not alpha/sqrt(r) = 8). Before
this round the field did not exist on ``TrainLoraConfig`` and the LoraConfig
call hardcoded ``use_rslora=True``, so the Hydra value was silently ignored.

These tests pin the surface area:

  1. ``TrainLoraConfig.use_rslora`` defaults to ``True`` — preserves legacy
     behavior for all existing callers.
  2. ``TrainLoraConfig.use_rslora`` is a reachable dataclass field (so Hydra
     ConfigStore integrations and ``**overrides`` flows can name it).
  3. Passing ``use_rslora=False`` is honored on construction.
  4. The source of ``train_lora`` builds ``LoraConfig`` with
     ``use_rslora=cfg.use_rslora`` (NOT a hard-coded ``True``). This is a
     source-text inspection to keep the test fast (full ``train_lora``
     execution would pull in TRL + transformers + PEFT + initialize CUDA).
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

from explore_persona_space.train.sft import TrainLoraConfig, train_lora


def test_use_rslora_default_is_true() -> None:
    """Default value preserves legacy behavior for all non-#378 callers."""
    cfg = TrainLoraConfig()
    assert cfg.use_rslora is True


def test_use_rslora_field_exists_on_dataclass() -> None:
    """The field is reachable via dataclass introspection (so Hydra ConfigStore
    integrations and existing ``**overrides`` flows can name it).
    """
    field_names = {f.name for f in fields(TrainLoraConfig)}
    assert "use_rslora" in field_names


def test_use_rslora_false_is_honored_on_construction() -> None:
    """Passing ``use_rslora=False`` builds a ``TrainLoraConfig`` with that value."""
    cfg = TrainLoraConfig(use_rslora=False)
    assert cfg.use_rslora is False


def test_use_rslora_true_is_honored_on_construction() -> None:
    """Explicit ``use_rslora=True`` is honored (symmetric check)."""
    cfg = TrainLoraConfig(use_rslora=True)
    assert cfg.use_rslora is True


def test_use_rslora_propagates_to_lora_config_call() -> None:
    """Static-source check: ``train_lora`` passes ``cfg.use_rslora`` to
    ``LoraConfig(...)`` (not a hard-coded ``True``).

    Parses ``train_lora``'s source via the ``ast`` module and looks for a
    ``LoraConfig(...)`` call whose ``use_rslora`` keyword argument is
    ``cfg.use_rslora`` (an attribute access on the ``cfg`` Name node). This
    catches regressions that re-hardcode the value without needing to
    actually run the function — which would require TRL + PEFT + CUDA.
    """
    src = inspect.getsource(train_lora)
    tree = ast.parse(src)
    found_use_rslora_attr = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match LoraConfig(...) by callable name.
        func = node.func
        if (isinstance(func, ast.Name) and func.id == "LoraConfig") or (
            isinstance(func, ast.Attribute) and func.attr == "LoraConfig"
        ):
            target_call = node
        else:
            continue
        for kw in target_call.keywords:
            if kw.arg != "use_rslora":
                continue
            # Accept either `cfg.use_rslora` (Attribute) or any non-constant
            # expression. Reject a bare True/False constant.
            if isinstance(kw.value, ast.Constant):
                raise AssertionError(
                    f"LoraConfig(use_rslora=...) is a hard-coded constant "
                    f"{kw.value.value!r}; expected cfg.use_rslora propagation"
                )
            if (
                isinstance(kw.value, ast.Attribute)
                and isinstance(kw.value.value, ast.Name)
                and kw.value.value.id == "cfg"
                and kw.value.attr == "use_rslora"
            ):
                found_use_rslora_attr = True
            else:
                # Any other expression is fine as long as it isn't a constant.
                found_use_rslora_attr = True
    assert found_use_rslora_attr, (
        "Could not find LoraConfig(use_rslora=cfg.use_rslora) in train_lora's "
        "source. The field is wired through to the dataclass but never reaches "
        "the PEFT LoraConfig builder; #378's IA scaling will be silently ignored."
    )


def test_train_trigger_script_passes_use_rslora() -> None:
    """The issue #378 trigger-train entry script forwards ``cfg.lora.use_rslora``
    into ``train_lora()``. Catches the round-1 BLOCKER 2 regression at the
    entry-script boundary (the previous draft didn't forward it).
    """
    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "scripts" / "issue378_train_trigger.py"
    src = train_script.read_text()
    # Look for the literal `use_rslora=` keyword argument in the train_lora call.
    assert "use_rslora=" in src, (
        f"{train_script} does not pass use_rslora to train_lora(); the "
        "trigger-LoRA will train with the project-wide default (True), NOT "
        "the IA-style alpha/r = 2 scaling the plan §4.9 prescribes."
    )
    # Specifically: cfg.lora.use_rslora (the Hydra value), not a constant.
    assert "cfg.lora.use_rslora" in src, (
        f"{train_script} passes use_rslora=... but not from cfg.lora.use_rslora — "
        "the Hydra-resolved value should be forwarded verbatim."
    )
