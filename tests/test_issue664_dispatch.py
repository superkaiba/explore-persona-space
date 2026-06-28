"""Issue #664 round-9 invariant pins: chunked vLLM ``_greedy`` / ``_sample``.

Round-8 launches 1 AND 2 hung indefinitely at ``_elicit_secure_code``'s call to
``_greedy(llm, 3000_prompts, 1024)`` -- a vLLM v1 EngineCore deadlock when a very
large prompt list is fed to a single ``llm.generate()`` on pod-664's driver combo
(CUDA worker stuck, 0% GPU util forever). The fix chunks both ``_greedy`` and
``_sample`` internally into batches of ``VLLM_GREEDY_CHUNK_SIZE`` (env-overridable
via ``EPM_VLLM_GREEDY_CHUNK_SIZE``), transparently to every caller.

These pins lock the chunking contract (so a future refactor cannot silently strip
it):

- a prompt list larger than the chunk size is split into ceil(N / chunk) calls,
  each of size <= chunk (the final chunk is the remainder);
- the returned list is the FULL length N in the ORIGINAL prompt order;
- ``_sample`` preserves the per-prompt list-of-n structure;
- a single-chunk (small) batch issues exactly one ``generate`` call (the loop
  handles the 1-chunk case naturally -- no special-casing).

All CPU-only: ``llm.generate`` is mocked, so no GPU / no real engine. ``vllm`` is
importable on the dev VM (``SamplingParams`` construction does not touch CUDA), so
the lazy ``from vllm import SamplingParams`` inside the helpers runs for real.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue664_dispatch as D  # noqa: E402


class _FakeOutput:
    """Mimics one ``vllm.RequestOutput`` -- ``.outputs`` is a list of completion
    objects, each carrying a ``.text``. ``texts`` is the per-completion text list
    for this prompt (length 1 for greedy, length n for sampled)."""

    def __init__(self, texts: list[str]):
        self.outputs = [mock.Mock(text=t) for t in texts]


def _greedy_generate_side_effect(prompts, sp, *, use_tqdm):
    """Return one ``_FakeOutput`` per prompt, echoing the prompt text so order is
    verifiable downstream. Greedy => one completion per prompt."""
    assert use_tqdm is False, "gotchas #613: use_tqdm must be False"
    return [_FakeOutput([f"greedy::{p}"]) for p in prompts]


def _sample_generate_side_effect_factory(n: int):
    def _side_effect(prompts, sp, *, use_tqdm):
        assert use_tqdm is False, "gotchas #613: use_tqdm must be False"
        return [_FakeOutput([f"sample::{p}::{j}" for j in range(n)]) for p in prompts]

    return _side_effect


@pytest.fixture
def chunk500(monkeypatch):
    """Pin the chunk size to 500 for deterministic call-count assertions,
    independent of the env / module default."""
    monkeypatch.setattr(D, "VLLM_GREEDY_CHUNK_SIZE", 500)


def test_greedy_chunks_1500_prompts_into_three_calls(chunk500):
    prompts = [f"p{i}" for i in range(1500)]
    llm = mock.Mock()
    llm.generate.side_effect = _greedy_generate_side_effect

    result = D._greedy(llm, prompts, 256)

    # 1500 / 500 == 3 chunks => exactly 3 generate calls, each of size 500.
    assert llm.generate.call_count == 3
    chunk_sizes = [len(call.args[0]) for call in llm.generate.call_args_list]
    assert chunk_sizes == [500, 500, 500]

    # Full length, original order preserved.
    assert len(result) == 1500
    assert result == [f"greedy::p{i}" for i in range(1500)]


def test_greedy_remainder_chunk(chunk500):
    prompts = [f"p{i}" for i in range(1200)]
    llm = mock.Mock()
    llm.generate.side_effect = _greedy_generate_side_effect

    result = D._greedy(llm, prompts, 256)

    # 1200 => 500 + 500 + 200 (remainder chunk).
    assert llm.generate.call_count == 3
    chunk_sizes = [len(call.args[0]) for call in llm.generate.call_args_list]
    assert chunk_sizes == [500, 500, 200]
    assert len(result) == 1200
    assert result == [f"greedy::p{i}" for i in range(1200)]


def test_greedy_single_chunk_one_call(chunk500):
    prompts = [f"p{i}" for i in range(42)]
    llm = mock.Mock()
    llm.generate.side_effect = _greedy_generate_side_effect

    result = D._greedy(llm, prompts, 256)

    # Small batch => the loop yields exactly one chunk == one generate call
    # (no special-casing of the 1-chunk path).
    assert llm.generate.call_count == 1
    sent_prompts = llm.generate.call_args_list[0].args[0]
    assert len(sent_prompts) == 42
    assert sent_prompts == prompts
    assert len(result) == 42
    assert result == [f"greedy::p{i}" for i in range(42)]


def test_sample_chunks_and_preserves_list_of_n(chunk500):
    prompts = [f"p{i}" for i in range(1500)]
    n = 3
    llm = mock.Mock()
    llm.generate.side_effect = _sample_generate_side_effect_factory(n)

    result = D._sample(llm, prompts, 256, temp=1.0, n=n)

    assert llm.generate.call_count == 3
    chunk_sizes = [len(call.args[0]) for call in llm.generate.call_args_list]
    assert chunk_sizes == [500, 500, 500]

    # Full length, per-prompt list-of-n structure, original order preserved.
    assert len(result) == 1500
    assert all(len(per_prompt) == n for per_prompt in result)
    for i, per_prompt in enumerate(result):
        assert per_prompt == [f"sample::p{i}::{j}" for j in range(n)]


def test_chunk_size_env_overridable(monkeypatch):
    """The module constant is read from ``EPM_VLLM_GREEDY_CHUNK_SIZE`` at import;
    re-importing with the env set picks up the override (ops-tuning contract)."""
    import importlib

    monkeypatch.setenv("EPM_VLLM_GREEDY_CHUNK_SIZE", "250")
    reloaded = importlib.reload(D)
    try:
        assert reloaded.VLLM_GREEDY_CHUNK_SIZE == 250
    finally:
        # Restore the default-valued module for any later test in the session.
        monkeypatch.delenv("EPM_VLLM_GREEDY_CHUNK_SIZE", raising=False)
        importlib.reload(reloaded)
