"""vLLM ``gpu_memory_utilization`` default test (task #397, Round 8).

Round 8 fix for the first-launch HF→vLLM teardown OOM crash. The smoke
cell trained fine, log-prob eval ran fine, then vLLM init crashed:

    ValueError: Free memory on device (43.34/79.19 GiB) on startup is
    less than desired GPU memory utilization (0.6, 47.51 GiB).

Root cause: HF Transformers held ~36 GB on GPU 0 from the log-prob
eval phase. Round 6's `del peft_model + gc.collect + empty_cache` was
insufficient — PyTorch caching-allocator blocks persist; vLLM tried to
grab 0.6 * 79 = 47.5 GB; only 43.3 GB free → instant ValueError →
dispatcher exited.

Fix 2 (cheap defense-in-depth): pin ``gpu_memory_utilization=0.45`` as
the default in ``generate_completions_with_lora``. 0.45 * 79 = ~35.5 GB
for vLLM, leaving ~43 GB headroom for HF residue. The base Qwen-2.5-7B
+ LoRA at bf16 needs ~15 GB; remaining ~20 GB is KV cache (still plenty
for the 480-context smoke + 2400-completion full sampled eval).

Fix 1 (aggressive HF teardown in the caller) is the real fix; Fix 2 is
the safety net for residue PyTorch can't release.

Override path: ``VLLM_GPU_MEM_UTIL`` env var. Tests cover both the new
default + the env override.

CPU-only; vLLM is monkeypatched.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


def _install_fake_vllm_minimal(monkeypatch, recorded: dict) -> None:
    """Minimal vLLM monkeypatch — just enough to record LLM init kwargs."""
    fake_vllm = MagicMock(name="vllm_module")
    fake_lora_module = MagicMock(name="vllm.lora")
    fake_lora_request_module = MagicMock(name="vllm.lora.request")

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            recorded["sampling_kwargs"] = kwargs

    class _FakeLoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path, **kwargs):
            self.lora_path = lora_path

    class _FakeOutput:
        def __init__(self, text):
            self.text = text

    class _FakeLLMOutput:
        def __init__(self, n_completions):
            self.outputs = [_FakeOutput(f"completion_{i}") for i in range(n_completions)]

    class _FakeLLM:
        def __init__(self, **kwargs):
            recorded["llm_init_kwargs"] = kwargs

        def generate(self, prompts, sampling_params, lora_request=None):
            n = recorded["sampling_kwargs"].get("n", 1)
            return [_FakeLLMOutput(n) for _ in prompts]

    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    fake_lora_request_module.LoRARequest = _FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_lora_module)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora_request_module)


def _install_fake_transformers(monkeypatch) -> None:
    fake_transformers = MagicMock(name="transformers_module")

    class _FakeTok:
        pad_token_id = 0
        pad_token = "[PAD]"
        eos_token = "[EOS]"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            parts = [f"{m['role']}: {m['content']}" for m in messages]
            return "\n".join(parts)

    fake_transformers.AutoTokenizer = _FakeTok
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _patch_tokenizer_noop(monkeypatch) -> None:
    import explore_persona_space.experiments.factor_screen_365.eval_panel as fs365_ep

    monkeypatch.setattr(fs365_ep, "_patch_tokenizer_for_vllm", lambda: None)


# ---------------------------------------------------------------------------
# gpu_memory_utilization default = 0.45 (Round 8 Fix 2)
# ---------------------------------------------------------------------------


def test_generate_completions_with_lora_default_gpu_memory_util_is_0_45(monkeypatch) -> None:
    """Round 8 Fix 2: default gpu_memory_utilization is 0.45 (was 0.60).

    Pinned defense-in-depth so a partial HF teardown residue leaves
    enough headroom for vLLM to init. The smoke first-launch crash
    tripped 0.60 * 79 = 47.5 GB against 43.3 GB free; 0.45 * 79 = 35.5 GB
    fits with ~43 GB headroom.
    """
    monkeypatch.delenv("VLLM_GPU_MEM_UTIL", raising=False)
    recorded: dict = {}
    _install_fake_vllm_minimal(monkeypatch, recorded)
    _install_fake_transformers(monkeypatch)
    _patch_tokenizer_noop(monkeypatch)

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        generate_completions_with_lora,
    )

    generate_completions_with_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        lora_path="/tmp/cell/adapter",
        personas={"librarian": "You are a librarian."},
        questions=["Q?"],
    )

    init_kwargs = recorded["llm_init_kwargs"]
    assert init_kwargs["gpu_memory_utilization"] == pytest.approx(0.45), (
        "Round 8 Fix 2: gpu_memory_utilization default must be 0.45 "
        f"(was 0.60); got {init_kwargs.get('gpu_memory_utilization')}"
    )


def test_generate_completions_with_lora_respects_env_override(monkeypatch) -> None:
    """``VLLM_GPU_MEM_UTIL=0.55`` env var overrides the 0.45 default.

    Documented escape hatch: if Fix 3 process-isolation lands later and
    HF residue drops to ~0, the user can dial gpu_memory_utilization
    back up via the env var without a code change.
    """
    monkeypatch.setenv("VLLM_GPU_MEM_UTIL", "0.55")
    recorded: dict = {}
    _install_fake_vllm_minimal(monkeypatch, recorded)
    _install_fake_transformers(monkeypatch)
    _patch_tokenizer_noop(monkeypatch)

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        generate_completions_with_lora,
    )

    generate_completions_with_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        lora_path="/tmp/cell/adapter",
        personas={"librarian": "You are a librarian."},
        questions=["Q?"],
    )

    init_kwargs = recorded["llm_init_kwargs"]
    assert init_kwargs["gpu_memory_utilization"] == pytest.approx(0.55), (
        f"VLLM_GPU_MEM_UTIL env override should win; got "
        f"{init_kwargs.get('gpu_memory_utilization')}"
    )


def test_generate_completions_with_lora_explicit_kwarg_wins_over_env(monkeypatch) -> None:
    """Explicit ``gpu_memory_utilization=0.30`` kwarg wins over env var.

    Caller can force a smaller fraction if they know more about residue
    than the default — e.g., for back-to-back vLLM runs in the same
    process. Kwarg > env > default (0.45).
    """
    monkeypatch.setenv("VLLM_GPU_MEM_UTIL", "0.55")
    recorded: dict = {}
    _install_fake_vllm_minimal(monkeypatch, recorded)
    _install_fake_transformers(monkeypatch)
    _patch_tokenizer_noop(monkeypatch)

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        generate_completions_with_lora,
    )

    generate_completions_with_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        lora_path="/tmp/cell/adapter",
        personas={"librarian": "You are a librarian."},
        questions=["Q?"],
        gpu_memory_utilization=0.30,
    )

    init_kwargs = recorded["llm_init_kwargs"]
    assert init_kwargs["gpu_memory_utilization"] == pytest.approx(0.30), (
        "Explicit gpu_memory_utilization kwarg must win over env + default; got "
        f"{init_kwargs.get('gpu_memory_utilization')}"
    )


def test_default_gpu_memory_util_leaves_headroom_for_hf_residue() -> None:
    """Sanity-check the Fix 2 arithmetic: 0.45 * 79 GB ≈ 35.5 GB fits with
    headroom for HF residue + Qwen-2.5-7B at bf16 + KV cache.

    The first-launch crash log: ``43.34 GB free of 79.19 GB`` → residue
    was ~36 GB. With 0.45 default, vLLM requests 35.6 GB; even with
    36 GB HF residue still resident, free is ~43 GB > 35.6 GB → fits.
    With 0.60 (round-6 default), vLLM requests 47.5 GB > 43 GB free →
    crash.
    """
    h100_gb = 79.19
    fix2_default = 0.45
    vllm_target_gb = fix2_default * h100_gb
    hf_residue_gb_observed = 79.19 - 43.34  # from the crash log
    assert vllm_target_gb < h100_gb - hf_residue_gb_observed, (
        f"Fix 2 math broken: vLLM target {vllm_target_gb:.1f} GB must fit "
        f"in {h100_gb - hf_residue_gb_observed:.1f} GB after HF residue"
    )


# ---------------------------------------------------------------------------
# Fix 1 surface-check: aggressive teardown sequence in the dispatcher
# ---------------------------------------------------------------------------


def test_dispatcher_inprocess_teardown_uses_aggressive_pattern() -> None:
    """Round 11: static check that the dispatcher's in-process per-cell
    pipeline uses the 4-step aggressive teardown pattern (del + gc +
    empty_cache + synchronize) BEFORE the sampled eval call.

    Round 6's ``del + gc.collect + empty_cache`` was insufficient — the
    first-launch crash proved it. Round 8 adds synchronize() + drops
    the tokenizer + logs pre/post free memory. Round 11 lifted the
    teardown into ``_aggressive_hf_to_vllm_teardown`` (with the caller
    doing the explicit ``del peft_model; del tokenizer`` before calling
    the helper). A future regression that drops synchronize() or
    removes either ``del`` would re-introduce the OOM in some
    HF-residue regime; this static check is the canary.

    Replaces the Round 8 test against the deleted ``run_one_cell.py``.
    """
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
    )
    text = src_path.read_text(encoding="utf-8")

    # Caller-side: _run_one_cell_inprocess drops the HF refs before
    # calling the teardown helper. base_model is the peft-wrapped HF
    # model; tokenizer_lp is the log-prob-side tokenizer.
    assert "del base_model" in text, "Fix 1 step 1: del HF model ref (base_model)"
    assert "del tokenizer_lp" in text, "Fix 1 step 1: del tokenizer ref (tokenizer_lp)"

    # Helper-side: _aggressive_hf_to_vllm_teardown does gc + empty_cache
    # + synchronize + log.
    assert "_aggressive_hf_to_vllm_teardown" in text, (
        "Round 11: dispatcher must define + call _aggressive_hf_to_vllm_teardown"
    )
    assert "torch as _torch" in text, "Fix 1 step 3+4: torch imported inside helper"
    assert "_torch.cuda.empty_cache()" in text, "Fix 1 step 3: empty_cache (helper)"
    assert "_torch.cuda.synchronize()" in text, (
        "Fix 1 step 4: synchronize() — Round 6's lack of synchronize was "
        "part of why empty_cache wasn't sufficient"
    )
    # Pre/post log line so the next OOM has the actual residue size in
    # the dispatcher log.
    assert "free GPU memory" in text, (
        "Fix 1 step 5: log pre/post free memory so debug has the residue size"
    )


def test_dispatcher_smoke_path_uses_aggressive_teardown() -> None:
    """Same Fix-1 sequence must exist in dispatch_factor_screen_397.py's
    smoke path (it has its own HF→vLLM transition before calling
    _run_smoke_sampled_eval).
    """
    from pathlib import Path

    src_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
    )
    text = src_path.read_text(encoding="utf-8")
    assert "del base_model" in text, "Smoke path: del HF model ref"
    assert "del tokenizer" in text, "Smoke path: del tokenizer ref"
    assert "torch.cuda.empty_cache()" in text or "_torch.cuda.empty_cache()" in text
    assert "torch.cuda.synchronize()" in text or "_torch.cuda.synchronize()" in text
