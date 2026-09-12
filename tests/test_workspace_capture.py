"""Capture boundaries and nonlinear token aggregation on actual tensors/modules."""

import numpy as np
import torch

from explore_persona_space.analysis.workspace_capture import (
    answer_token_ids,
    assign_context_input,
    capture_context_inputs,
    capture_token_batch,
    decompose_context,
)
from explore_persona_space.analysis.workspace_lenses import nonnegative_gradient_pursuit


def test_capture_batch_matches_individual_rows_and_exact_answer_boundary():
    """Ragged batched native capture selects context-last and all answer tokens."""
    from transformers import Qwen2Config, Qwen2Model

    config = Qwen2Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    text = Qwen2Model(config).eval()
    rows = [
        {"prompt_ids": [1, 2, 3], "answer_ids": [4, 5]},
        {"prompt_ids": [3, 4], "answer_ids": [7]},
    ]
    batch = capture_token_batch(text, rows, 1, 0)
    for row, observed in zip(rows, batch, strict=True):
        individual = capture_token_batch(text, [row], 1, 0)[0]
        torch.testing.assert_close(observed["x"], individual["x"])
        torch.testing.assert_close(observed["answer_states"], individual["answer_states"])
        assert len(observed["answer_states"]) == len(row["answer_ids"])


def test_token_nonlinearity_precedes_equal_rollout_pooling():
    """Unequal answer lengths cannot reweight draws; pooled decomposition differs."""
    dictionary = torch.eye(2)
    captures = [
        {
            "prompt_sha256": "a",
            "seed": 42,
            "x": torch.ones(2),
            "answer_states": torch.tensor([[4.0, 1.0]]),
        },
        {
            "prompt_sha256": "a",
            "seed": 43,
            "x": torch.ones(2),
            "answer_states": torch.tensor([[1.0, 4.0], [1.0, 4.0], [1.0, 4.0]]),
        },
    ]
    result = decompose_context(captures, {"J": dictionary, "R": dictionary}, k=1)
    torch.testing.assert_close(result["targets"]["full"], torch.tensor([2.5, 2.5]).double())
    torch.testing.assert_close(result["targets"]["J"], torch.tensor([2.0, 2.0]).double())
    torch.testing.assert_close(
        result["targets"]["full"], result["targets"]["J"] + result["targets"]["restJ"]
    )
    pooled = nonnegative_gradient_pursuit(result["targets"]["full"][None], dictionary.double(), k=1)
    assert not torch.allclose(pooled.component[0], result["targets"]["J"])
    assert result["mean_target_noise_trace"]["J"] == 8
    assert np.array_equal(result["token_counts"], [1, 3])


def test_terminal_answer_token_mask():
    """Terminal stop and trailing padding are excluded, without retokenizing."""
    assert answer_token_ids([5, 6, 99, 0], {99}) == ([5, 6], 2)
    assert answer_token_ids([5, 6], {99}) == ([5, 6], 0)


def test_context_only_input_is_reused_exactly_without_erasing_batch_diagnostics():
    """No future answer enters the canonical forward; old BF16 reads remain visible."""
    from transformers import Qwen2Config, Qwen2Model

    config = Qwen2Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    text = Qwen2Model(config).eval()
    prompts = [[1, 2, 3], [3, 4]]
    x = capture_context_inputs(text, prompts, 1, 0)
    prior = [
        {"prompt_ids": prompts[0], "x": x[0] + delta, "answer_states": torch.ones(length, 16)}
        for delta, length in ((0.5, 1), (-0.25, 13))
    ]
    corrected = assign_context_input(prior, x[0])
    for before, after in zip(prior, corrected, strict=True):
        torch.testing.assert_close(after["x"], x[0], atol=0, rtol=0)
        assert torch.equal(after["answer_batch_x"], before["x"])
        assert after["answer_states"] is before["answer_states"]
    # The ordinary causal model gives the same final-context read with an answer
    # appended in FP32; the new route also makes all rollout copies bit-identical.
    with_answer = capture_token_batch(
        text, [{"prompt_ids": prompts[0], "answer_ids": [4, 5]}], 1, 0
    )[0]
    torch.testing.assert_close(x[0], with_answer["x"], rtol=1e-5, atol=1e-6)


def test_cap_recovery_preserves_original_draw_and_resume(tmp_path, monkeypatch):
    """A cap-hit subset cannot become complete until the affected draw is recovered."""
    import json
    import sys
    from types import SimpleNamespace

    from explore_persona_space.analysis.workspace_capture import generate_rollouts

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SimpleNamespace))

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            # Transformers 5 defaults to a structured BatchEncoding; generation
            # explicitly requests the flat token-ID representation it consumes.
            from transformers import BatchEncoding

            if kwargs.get("return_dict", True):
                return BatchEncoding({"input_ids": [1, 2]})
            return [1, 2]

    class Engine:
        def __init__(self):
            self.caps = []

        def generate(self, requests, parameters, **kwargs):
            self.caps.extend(p.max_tokens for p in parameters)
            return [
                SimpleNamespace(
                    prompt_token_ids=r["prompt_token_ids"],
                    outputs=[
                        SimpleNamespace(
                            token_ids=[3] * p.max_tokens,
                            text="raw",
                            finish_reason="length" if p.max_tokens == 2 else "stop",
                            stop_reason=None,
                        )
                    ],
                )
                for r, p in zip(requests, parameters, strict=True)
            ]

    config = {
        "generation": {
            "enable_thinking": False,
            "seeds": [42, 43],
            "temperature": 1,
            "top_p": 0.95,
            "top_k": -1,
            "min_p": 0,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "initial_max_new_tokens": 2,
        }
    }
    prompts = [{"prompt": "prompt", "prompt_sha256": "a"}]
    engine = Engine()
    report = generate_rollouts(engine, Tokenizer(), prompts, config, {"frozen": True}, tmp_path)
    assert report["status"] == "complete"
    assert report["cap_hits"] == 0
    assert engine.caps == [2, 2, 4, 4]
    saved = json.loads((tmp_path / "a.json").read_text())
    assert len(saved["cap_recovery_history"]) == 2
    assert all(r["finish_reason"] == "length" for r in saved["cap_recovery_history"])
    generate_rollouts(engine, Tokenizer(), prompts, config, {"frozen": True}, tmp_path)
    assert engine.caps == [2, 2, 4, 4]
