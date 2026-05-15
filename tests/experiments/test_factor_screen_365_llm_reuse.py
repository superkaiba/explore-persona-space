"""Regression: ``build_on_policy_pool`` must accept an injected ``llm``.

Runtime forensics for task #365 (post-launch failure on pod-365): the
dispatcher called ``build_on_policy_pool`` 8 times per source (one per
(A, B, C) triple). The original implementation instantiated a fresh
``vllm.LLM(model="Qwen/Qwen2.5-7B-Instruct", ...)`` inside each call.
Even with ``del llm; gc.collect(); torch.cuda.empty_cache()`` in a
``finally``, vLLM v1's multiprocess engine workers leave residual GPU
state, and the *second* ``LLM(...)`` instantiation aborts with::

    AssertionError: Initial free memory 124.04 GiB, current free
    memory 124.54 GiB   (vllm/v1/worker/gpu_worker.py:271)

Fix: ``build_on_policy_pool`` takes an optional ``llm`` parameter.
When the dispatcher supplies one, the function reuses it across all
cells of a source and does NOT instantiate / tear down vLLM itself.
The dispatcher hoists the engine to source-scope (1 init per source,
not 8). When ``llm=None`` the old behavior is preserved (back-compat
for any standalone caller).

These tests:

  * mock ``vllm.LLM`` so they run without a GPU,
  * verify the signature exposes the ``llm`` parameter,
  * verify ``llm=None`` (default) instantiates internally,
  * verify ``llm=<injected>`` skips instantiation and reuses the injected
    engine,
  * verify the injected engine is NOT torn down by
    ``build_on_policy_pool`` (only the caller owns its lifecycle).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest import mock

import pytest

from explore_persona_space.experiments.factor_screen_365 import onpolicy
from explore_persona_space.experiments.factor_screen_365.onpolicy import (
    OnPolicyConfig,
    build_on_policy_pool,
)


def test_build_on_policy_pool_signature_accepts_llm() -> None:
    """The public function must expose ``llm`` as a keyword-friendly parameter."""
    sig = inspect.signature(build_on_policy_pool)
    assert "llm" in sig.parameters, (
        "build_on_policy_pool must accept an optional `llm` parameter so the "
        "dispatcher can hoist a single vLLM engine across all 8 (A,B,C) cells "
        "of a source. Without this, vLLM v1's memory-profile guardrail trips "
        "on the second per-cell re-init (see issue #365 runtime forensics)."
    )
    # Default should be None for back-compat (any older caller still works).
    assert sig.parameters["llm"].default is None, (
        "`llm` must default to None so existing callers that do "
        "`build_on_policy_pool(cfg)` continue to work unchanged."
    )


def _fake_outputs_for(prompt_texts: list[str]) -> list[object]:
    """Synthetic vLLM output objects matching the shape used at line ~208.

    Each output has ``out.outputs[0].text`` — the same access pattern used
    by the real loop in ``build_on_policy_pool``.
    """
    fake_out = []
    for _ in prompt_texts:
        gen = mock.MagicMock()
        # Length-band passes when between B_LENGTH_BANDS[0] = (40, 80) tokens
        # for b=0. Keep the text short so the band-filter behavior is
        # predictable across test invocations; the exact filter result is
        # not what we are testing here.
        gen.text = "Hello world. " * 10
        wrapper = mock.MagicMock()
        wrapper.outputs = [gen]
        fake_out.append(wrapper)
    return fake_out


def _make_cfg(tmp_path: Path) -> OnPolicyConfig:
    return OnPolicyConfig(
        source="librarian",
        a=0,
        b=0,
        c=0,
        pos_per_source=2,
        neg_per_source=4,
        questions=["What's a good book recommendation?"],
        cache_dir=tmp_path / "pool",
        seed=42,
    )


@pytest.fixture
def patched_tokenizer():
    """Mock AutoTokenizer.from_pretrained so the test does not hit HF Hub."""
    fake_tok = mock.MagicMock()
    fake_tok.apply_chat_template.return_value = "[chat-template-rendered]"
    fake_tok.encode.return_value = list(range(60))  # falls inside b=0 band (40-80)

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok) as patched:
        yield patched


def test_build_on_policy_pool_instantiates_llm_when_none_passed(
    tmp_path: Path, patched_tokenizer
) -> None:
    """Default behavior (back-compat): when ``llm=None``, the function
    instantiates ``vllm.LLM(...)`` exactly once and tears it down on exit."""
    cfg = _make_cfg(tmp_path)

    fake_llm = mock.MagicMock()
    # Will be called with whatever prompt list build_on_policy_pool generates;
    # we don't care about exact prompts, only that this LLM was the one used.
    fake_llm.generate.side_effect = lambda prompts, sp: _fake_outputs_for(prompts)

    with mock.patch("vllm.LLM", return_value=fake_llm) as llm_constructor:
        rows = build_on_policy_pool(cfg, llm=None)

    assert llm_constructor.call_count == 1, (
        f"Expected exactly one LLM(...) instantiation when llm=None; "
        f"got {llm_constructor.call_count}."
    )
    fake_llm.generate.assert_called_once()
    # And the function should have returned some rows (band-pass dependent).
    assert isinstance(rows, list)


def test_build_on_policy_pool_reuses_injected_llm(tmp_path: Path, patched_tokenizer) -> None:
    """When an LLM is injected, no new one is constructed; the injected
    one receives ``generate()`` calls."""
    cfg = _make_cfg(tmp_path)

    injected = mock.MagicMock(name="injected_llm")
    injected.generate.side_effect = lambda prompts, sp: _fake_outputs_for(prompts)

    with mock.patch("vllm.LLM") as llm_constructor:
        rows = build_on_policy_pool(cfg, llm=injected)

    assert llm_constructor.call_count == 0, (
        "When the caller injects an LLM, build_on_policy_pool must NOT "
        "instantiate a new one. Hoisting the engine out of the per-cell loop "
        "is the whole point of the fix; a second instantiation would re-trip "
        "vLLM v1's memory-profile guardrail."
    )
    injected.generate.assert_called_once()
    assert isinstance(rows, list)


def test_build_on_policy_pool_does_not_teardown_injected_llm(
    tmp_path: Path, patched_tokenizer
) -> None:
    """Lifecycle contract: when the caller passes an ``llm``, the caller
    owns teardown. ``build_on_policy_pool`` must NOT call ``del`` /
    ``gc.collect()`` / ``torch.cuda.empty_cache()`` on the injected engine.

    We assert by calling the function twice with the same injected LLM and
    verifying both calls succeed (second call would fail if the first
    tore down the engine).
    """
    cfg = _make_cfg(tmp_path)

    injected = mock.MagicMock(name="injected_llm")
    injected.generate.side_effect = lambda prompts, sp: _fake_outputs_for(prompts)

    with mock.patch("vllm.LLM"):
        rows_first = build_on_policy_pool(cfg, llm=injected)
        # Wipe the cache file so the second call actually invokes generate
        # again (otherwise it short-circuits on the existing cache).
        cache_file = onpolicy._cache_path(cfg)
        if cache_file is not None and cache_file.exists():
            cache_file.unlink()
        rows_second = build_on_policy_pool(cfg, llm=injected)

    assert injected.generate.call_count == 2, (
        "The injected LLM must remain usable across multiple "
        "build_on_policy_pool calls. If the function tore it down after "
        "the first call, the second generate() would fail or hit a "
        "MagicMock side_effect exhaustion. Got "
        f"{injected.generate.call_count} generate calls."
    )
    assert isinstance(rows_first, list)
    assert isinstance(rows_second, list)
