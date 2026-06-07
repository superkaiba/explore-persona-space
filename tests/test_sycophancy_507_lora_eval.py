"""Tests that the 72B native-LoRA eval path actually threads LoRARequest into
llm.generate(). Round-2 regression test for code-review Critical 2: previously
the adapter was registered (logged) but llm.generate() was called without
lora_request=..., so vLLM silently served base.

Mocks vLLM's LLM and LoRARequest because the real vLLM is too heavy to load
in CI and irrelevant to this contract check.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _build_fake_vllm_module():
    """Construct a fake vllm + vllm.lora.request module tree for import-patching."""
    fake_vllm = MagicMock(name="vllm")
    fake_lora_module = MagicMock(name="vllm.lora.request")

    class FakeLoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

        def __repr__(self):
            return f"FakeLoRARequest({self.lora_name!r}, {self.lora_int_id}, {self.lora_path!r})"

    fake_lora_module.LoRARequest = FakeLoRARequest
    return fake_vllm, fake_lora_module, FakeLoRARequest


def test_enable_lora_threads_lora_request_into_generate(tmp_path):
    """When enable_lora=True + a single lora_modules entry, every llm.generate
    call MUST receive lora_request=<LoRARequest with that name>. Catches the
    Round-1 regression where the adapter was registered but generate() ran
    without it (base served silently).
    """
    fake_vllm, fake_lora_module, FakeLoRARequest = _build_fake_vllm_module()

    # Fake LLM + SamplingParams that record the calls.
    captured_generate_calls: list[dict] = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling, lora_request=None):
            captured_generate_calls.append({"prompts": list(prompts), "lora_request": lora_request})
            # Each prompt produces 1 fake output with n_rollouts rollouts.
            fake_output = MagicMock()
            fake_output.outputs = [MagicMock(text="fake completion") for _ in range(2)]
            return [fake_output for _ in prompts]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    # Stub transformers tokenizer.
    fake_transformers = MagicMock(name="transformers")
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "FAKE_PROMPT_TEMPLATE"
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer

    # Patch out the persona panel to avoid pulling real persona prompts.
    fake_panel_mod = MagicMock(name="factor_screen_365.persona_panel")
    fake_panel_mod.EVAL_PERSONAS_24 = {"persona_test": "you are a test"}

    # Build a single-claim eval pool on disk.
    pool_path = tmp_path / "eval_pool.json"
    import json

    pool_path.write_text(
        json.dumps({"wrong_claim": "the sky is green", "correction": "the sky is blue"}) + "\n"
    )
    merged_model_path = tmp_path / "fake_merged"
    merged_model_path.mkdir()
    (merged_model_path / "config.json").write_text("{}")

    lora_path = tmp_path / "fake_lora"
    lora_path.mkdir()

    out_dir = tmp_path / "out"

    with patch.dict(
        sys.modules,
        {
            "vllm": fake_vllm,
            "vllm.lora.request": fake_lora_module,
            "transformers": fake_transformers,
            "explore_persona_space.experiments.factor_screen_365.persona_panel": fake_panel_mod,
        },
    ):
        # Drop cached real imports so the patched ones are picked up.
        sys.modules.pop(
            "explore_persona_space.experiments.sycophancy_implantation_411.eval_one_source",
            None,
        )
        from explore_persona_space.experiments.sycophancy_implantation_411 import (
            eval_one_source,
        )

        eval_one_source.eval_source(
            source="sw_eng",
            seed=42,
            merged_model_path=merged_model_path,
            eval_pool_path=pool_path,
            out_dir=out_dir,
            n_rollouts=2,
            max_new_tokens=8,
            tensor_parallel_size=1,
            max_model_len=128,
            enable_lora=True,
            lora_modules=[f"sw_eng_seed42={lora_path}"],
        )

    # Assert llm.generate was called and every call carried a LoRARequest
    # with the correct name (NOT None).
    assert len(captured_generate_calls) >= 1, "llm.generate was never called"
    for call in captured_generate_calls:
        lr = call["lora_request"]
        assert lr is not None, (
            "llm.generate called with lora_request=None — adapter not applied! "
            "Round-1 regression resurfacing."
        )
        assert isinstance(lr, FakeLoRARequest), f"Expected LoRARequest, got {type(lr)}"
        assert lr.lora_name == "sw_eng_seed42", lr.lora_name
        assert str(lr.lora_path) == str(lora_path)


def test_enable_lora_false_does_not_pass_lora_request(tmp_path):
    """Base-panel path (enable_lora=False, no lora_modules) MUST NOT pass
    lora_request=... — vLLM rejects it when the LLM was constructed without
    enable_lora=True. Guards the additive-defaults invariant for #411
    callers (the 7B base-panel code path).
    """
    fake_vllm, fake_lora_module, _ = _build_fake_vllm_module()

    captured_generate_calls: list[dict] = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling, **kwargs):
            captured_generate_calls.append({"prompts": list(prompts), "kwargs": kwargs})
            fake_output = MagicMock()
            fake_output.outputs = [MagicMock(text="fake completion") for _ in range(2)]
            return [fake_output for _ in prompts]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    fake_transformers = MagicMock(name="transformers")
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "FAKE_PROMPT_TEMPLATE"
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer

    fake_panel_mod = MagicMock(name="factor_screen_365.persona_panel")
    fake_panel_mod.EVAL_PERSONAS_24 = {"persona_test": "you are a test"}

    pool_path = tmp_path / "eval_pool.json"
    import json

    pool_path.write_text(
        json.dumps({"wrong_claim": "the sky is green", "correction": "the sky is blue"}) + "\n"
    )
    out_dir = tmp_path / "out"

    with patch.dict(
        sys.modules,
        {
            "vllm": fake_vllm,
            "vllm.lora.request": fake_lora_module,
            "transformers": fake_transformers,
            "explore_persona_space.experiments.factor_screen_365.persona_panel": fake_panel_mod,
        },
    ):
        sys.modules.pop(
            "explore_persona_space.experiments.sycophancy_implantation_411.eval_one_source",
            None,
        )
        from explore_persona_space.experiments.sycophancy_implantation_411 import (
            eval_one_source,
        )

        eval_one_source.eval_source(
            source="sw_eng",
            seed=42,
            merged_model_path=None,
            hub_model_id="Qwen/Qwen2.5-7B-Instruct",
            eval_pool_path=pool_path,
            out_dir=out_dir,
            n_rollouts=2,
            max_new_tokens=8,
            tensor_parallel_size=1,
            max_model_len=128,
            enable_lora=False,
            lora_modules=None,
        )

    assert len(captured_generate_calls) >= 1
    for call in captured_generate_calls:
        # generate was called with positional (prompts, sampling) and NO
        # lora_request kwarg.
        assert "lora_request" not in call["kwargs"], (
            "Base-panel eval (enable_lora=False) leaked a lora_request kwarg into "
            f"llm.generate: {call['kwargs']!r}"
        )


def test_enable_lora_true_rejects_multiple_modules(tmp_path):
    """The Round-2 contract is exactly one adapter per eval_source call (max_loras=1).
    Multiple lora_modules must raise loudly rather than silently using only the first.
    """
    fake_vllm, fake_lora_module, _ = _build_fake_vllm_module()

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, sampling, **kwargs):
            raise AssertionError("generate should never be reached")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    fake_transformers = MagicMock(name="transformers")
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "FAKE_PROMPT_TEMPLATE"
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer

    fake_panel_mod = MagicMock(name="factor_screen_365.persona_panel")
    fake_panel_mod.EVAL_PERSONAS_24 = {"persona_test": "you are a test"}

    pool_path = tmp_path / "eval_pool.json"
    import json

    pool_path.write_text(
        json.dumps({"wrong_claim": "the sky is green", "correction": "the sky is blue"}) + "\n"
    )
    merged_model_path = tmp_path / "fake_merged"
    merged_model_path.mkdir()
    out_dir = tmp_path / "out"

    with patch.dict(
        sys.modules,
        {
            "vllm": fake_vllm,
            "vllm.lora.request": fake_lora_module,
            "transformers": fake_transformers,
            "explore_persona_space.experiments.factor_screen_365.persona_panel": fake_panel_mod,
        },
    ):
        sys.modules.pop(
            "explore_persona_space.experiments.sycophancy_implantation_411.eval_one_source",
            None,
        )
        from explore_persona_space.experiments.sycophancy_implantation_411 import (
            eval_one_source,
        )

        with pytest.raises(ValueError, match="exactly one LoRA adapter"):
            eval_one_source.eval_source(
                source="sw_eng",
                seed=42,
                merged_model_path=merged_model_path,
                eval_pool_path=pool_path,
                out_dir=out_dir,
                n_rollouts=2,
                max_new_tokens=8,
                tensor_parallel_size=1,
                max_model_len=128,
                enable_lora=True,
                lora_modules=[
                    f"sw_eng_seed42={tmp_path / 'a'}",
                    f"sw_eng_seed43={tmp_path / 'b'}",
                ],
            )
