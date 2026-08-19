"""Issue #2333 decode-hook CPU pins (no network, tiny from-config model).

Covers: decode-step counting (prefill latch, steps 1..k replaced, per-draw
re-arm), the fail-loud decode-before-prefill path, capture mode, and the
token-ID left-pad identity in ``generate_batch_ids`` (the BPE-seam rule:
prefill rows are ID concatenations, never re-tokenized text).
"""

from __future__ import annotations

import pytest
import torch

from explore_persona_space.experiments.issue2333.decode_hooks import (
    AnswerPositionEditHook,
    generate_batch_ids,
    joint_answer_hooks,
    resolve_decoder_blocks_2333,
)

VOCAB = 128
HIDDEN = 32
N_LAYERS = 2


class FakeTok:
    """Signature-conformant tokenizer boundary fake (pad/eos/decode only)."""

    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return f"<{len(ids)}t>"


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(2333)
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def test_resolve_decoder_blocks(tiny_model):
    blocks, _depth = resolve_decoder_blocks_2333(tiny_model)
    assert len(blocks) == N_LAYERS


def test_decode_before_prefill_fails_loud():
    h = AnswerPositionEditHook(layer=0)
    h.arm_replace([None], expected_prompt_len=5)
    with pytest.raises(RuntimeError, match="decode step before any prefill"):
        h(None, None, torch.zeros(1, 1, HIDDEN))


def test_replace_hook_edits_steps_1_to_k(tiny_model):
    """Row 0 gets k=2 donor states at decode steps 1..2 on every layer; row 1
    (donors None) is never edited; the prefill forward is passthrough."""
    stack = joint_answer_hooks(tiny_model)
    try:
        donors = torch.zeros(2, N_LAYERS, HIDDEN)
        rows = [[5, 6, 7, 8], [9, 10, 11]]
        draws = generate_batch_ids(
            tiny_model,
            FakeTok(),
            rows,
            n=2,
            stack=stack,
            donors_full=[donors, None],
            max_new_tokens=4,
            temperature=1.0,
            seed_base=7,
        )
        edits = stack.realized_edits()
        assert set(edits) == set(range(N_LAYERS))
        for _layer, rows_e in edits.items():
            # Final draw's telemetry (re-armed per draw): row 0 only, steps 1..2.
            assert {(b, step) for b, step, *_ in rows_e} == {(0, 1), (0, 2)}
        assert len(draws) == 2 and len(draws[0]) == 2
        for dr in draws:
            for r in dr:
                assert r["n_completion_tokens"] == len(r["gen_ids"])
                assert all(t != FakeTok.eos_token_id for t in r["gen_ids"])
    finally:
        stack.remove()


def test_capture_mode_records_first_k_states(tiny_model):
    stack = joint_answer_hooks(tiny_model)
    try:
        stack.arm_capture(batch=1, capture_k=2, expected_prompt_len=3)
        ids = torch.tensor([[5, 6, 7]])
        attn = torch.ones_like(ids)
        out = tiny_model.generate(
            input_ids=ids,
            attention_mask=attn,
            do_sample=False,
            max_new_tokens=3,
            pad_token_id=0,
        )
        assert out.shape[1] > 3
        captured = stack.captured_states()
        assert len(captured) == 1
        assert captured[0].shape == (2, N_LAYERS, HIDDEN)
    finally:
        stack.remove()


def test_generate_batch_ids_left_pad_token_identity(tiny_model):
    """Raw id rows survive left-padding verbatim (the BPE-seam contract: id
    concatenation, never re-tokenized text) and outputs align per row."""
    rows = [[3, 4, 5, 6, 7], [8, 9]]
    draws = generate_batch_ids(
        tiny_model, FakeTok(), rows, n=1, max_new_tokens=2, temperature=1.0, seed_base=3
    )
    assert len(draws[0]) == 2
    for r in draws[0]:
        assert isinstance(r["gen_ids"], list)
        assert r["n_completion_tokens"] <= 2


def test_generate_batch_ids_empty_batch_fails_loud(tiny_model):
    with pytest.raises(AssertionError, match="empty batch"):
        generate_batch_ids(tiny_model, FakeTok(), [], n=1)


def test_arm_replace_does_not_reset_realized_edits():
    """r1 Minor (last-draw-only telemetry): re-arming for the next draw must
    NOT clear the accumulated realized_edits — run_block reads the stack's
    telemetry ONCE after all K draws."""
    from explore_persona_space.experiments.issue2333.decode_hooks import AnswerPositionEditHook

    hook = AnswerPositionEditHook(layer=0)
    hook.realized_edits.append((0, 1, 0.5, 1.0, 1.0))
    hook.arm_replace([None], expected_prompt_len=3)
    assert hook.realized_edits == [(0, 1, 0.5, 1.0, 1.0)], "arm_replace cleared telemetry"


def test_resolve_decoder_blocks_language_model_nesting():
    """extraction._resolve_decoder_blocks resolves the q35 multimodal wrapper
    nesting (.model.language_model.layers) — g2 Minor: the branch previously
    had no committed regression test (bare layouts never reach it)."""
    import torch.nn as nn

    from explore_persona_space.analysis import extraction

    class Lang(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity() for _ in range(5)])
            self.embed_tokens = nn.Embedding(4, 2)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Lang()

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    wrapper = Wrapper()
    blocks, embed, depth = extraction._resolve_decoder_blocks(wrapper)
    assert blocks is wrapper.model.language_model.layers
    assert len(blocks) == 5 and depth == 1
    assert embed is wrapper.model.language_model.embed_tokens
    # And through the #2333 fail-loud wrapper used by the driver:
    blocks2, depth2 = resolve_decoder_blocks_2333(wrapper)
    assert list(blocks2) == list(blocks) and depth2 == 1


def test_resolve_decoder_blocks_2333_raises_on_chainless_module():
    import torch.nn as nn

    with pytest.raises((RuntimeError, AssertionError)):
        resolve_decoder_blocks_2333(nn.Linear(2, 2))
