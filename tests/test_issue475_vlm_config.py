"""Unit tests for _load_vlm_aware_config in scripts/eval_issue475.py.

The Qwen3.5-27B base model is a unified VLM whose ``vocab_size`` lives at
``config.text_config.vocab_size``, not at the top level. The HF
``modeling_utils`` loader paths read ``config.vocab_size`` directly and
crash with ``AttributeError`` when it's missing — which is what bit the
issue-475 canary on the eval log-prob phase. The helper surfaces the
nested attribute to top level so the standard ``from_pretrained`` path
succeeds.

The full ``eval_issue475`` module pulls in torch + vLLM + flash-attn and
won't import on the VM (no libcudnn). To keep these tests CPU-feasible
we extract the ``_load_vlm_aware_config`` function definition from the
script's AST and exec it in an isolated namespace, with a stub
``transformers`` module installed in ``sys.modules`` AROUND the call
(not just around the def exec — the helper does ``from transformers
import AutoConfig`` at call time, and the real transformers' lazy
AutoConfig attribute reaches into torch).

Run with: ``uv run pytest tests/test_issue475_vlm_config.py -x``
"""

from __future__ import annotations

import ast
import logging
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _extract_helper():
    """Slice ``_load_vlm_aware_config`` out of eval_issue475.py and exec
    it in an isolated namespace. Returns the live function."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "eval_issue475.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    func_node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_load_vlm_aware_config"
        ),
        None,
    )
    assert func_node is not None, (
        "_load_vlm_aware_config not found in scripts/eval_issue475.py — did the "
        "VLM-aware loader get renamed or removed?"
    )
    func_code = compile(ast.Module(body=[func_node], type_ignores=[]), str(script_path), "exec")
    namespace: dict = {
        "log": logging.getLogger("test_issue475_vlm_config"),
        "os": os,
    }
    exec(func_code, namespace)
    return namespace["_load_vlm_aware_config"]


def _call_helper(stub_autoconfig_factory, model_id):
    """Install a fake ``transformers`` module in ``sys.modules`` AROUND
    the helper call so the helper's ``from transformers import AutoConfig``
    binds to the stub instead of real transformers (which lazy-loads
    torch → libcudnn and fails on a CPU-only VM)."""
    helper = _extract_helper()
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = SimpleNamespace(from_pretrained=stub_autoconfig_factory)
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        return helper(model_id)
    finally:
        if saved is not None:
            sys.modules["transformers"] = saved
        else:
            del sys.modules["transformers"]


class _VLMStubConfig:
    """Approximates a Qwen3.5 VLM config: no top-level vocab_size, but
    text_config.vocab_size IS set."""

    def __init__(self, text_vocab_size: int):
        self.text_config = SimpleNamespace(vocab_size=text_vocab_size)
        # Deliberately no self.vocab_size attribute.


def test_vlm_config_surfaces_text_config_vocab_size():
    """A VLM-style config with nested vocab_size gets it surfaced to top level."""
    stub = _VLMStubConfig(text_vocab_size=152064)
    cfg = _call_helper(lambda *a, **kw: stub, "Qwen/Qwen3.5-27B")
    # The helper mutates AND returns the same object.
    assert cfg is stub
    assert cfg.vocab_size == 152064
    assert cfg.text_config.vocab_size == 152064  # nested still present


def test_non_vlm_config_passthrough():
    """An ordinary causal-LM config already has top-level vocab_size — no
    surfacing needed and the helper leaves it alone."""
    stub = SimpleNamespace(vocab_size=151936)
    cfg = _call_helper(lambda *a, **kw: stub, "Qwen/Qwen2.5-7B")
    assert cfg.vocab_size == 151936


def test_vlm_config_with_none_top_level_vocab_size_surfaces():
    """A config that defines vocab_size=None at top level should still
    fall through to the text_config branch (some VLM configs emit None
    rather than omit the attribute)."""
    stub = SimpleNamespace(vocab_size=None, text_config=SimpleNamespace(vocab_size=152064))
    cfg = _call_helper(lambda *a, **kw: stub, "Qwen/Qwen3.5-27B")
    assert cfg.vocab_size == 152064


def test_fail_loud_when_no_vocab_size_anywhere():
    """Neither top-level nor nested vocab_size → raise, do NOT silently default."""
    stub = SimpleNamespace()  # nothing at all
    with pytest.raises(AttributeError, match="vocab_size"):
        _call_helper(lambda *a, **kw: stub, "bogus/model-id")


def test_fail_loud_when_text_config_has_none_vocab_size():
    """text_config exists but its vocab_size is None — caller bug, fail loud."""
    stub = SimpleNamespace(text_config=SimpleNamespace(vocab_size=None))
    with pytest.raises(AttributeError, match="vocab_size"):
        _call_helper(lambda *a, **kw: stub, "bogus/model-id")
