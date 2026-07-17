"""CPU-only unit tests for issue #1415 steering primitives (plan v5 deliverable 4).

Tiny same-arch model built FROM CONFIG (real Qwen2.5 tokenizer + config with
``num_hidden_layers=2`` and small hidden dims, real vocab-id space) — no GPU,
no 7B weights. Exercises:

- DeltaHook position-correctness: the edit lands at ``len(tokenized_context)-1``
  and NOT elsewhere; a wrong armed prompt length raises.
- ``generate_batch`` with and without the hook (incl. per-seed determinism and
  the all-positions variant).
- The coherence gate (``coherence_check`` + ``condition_passes``).
- ``capture_vectors`` shapes, prefix/context arm split, and empty-completion
  handling.
"""

from __future__ import annotations

import pytest
import torch

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
H = 64  # tiny hidden size (real: 3584)
N_LAYERS = 2


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cfg.hidden_size = H
    cfg.intermediate_size = 2 * H
    cfg.num_hidden_layers = N_LAYERS
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)  # real vocab-id space, random weights
    model = model.to(torch.float32)  # the 7B config pins bf16; CPU tests run fp32
    model.eval()
    return model


CTX_SYS = {"system": "You are a pirate captain.", "user": "What is the best way to learn?"}
CTX_BARE = {"system": None, "user": "How do airplanes stay in the air?"}


# ── DeltaHook position correctness ────────────────────────────────────


def test_delta_hook_edits_only_last_context_position(tokenizer, tiny_model):
    from explore_persona_space.analysis.extraction import extract_layer_activations
    from explore_persona_space.experiments.issue1415.steering import (
        DeltaHook,
        context_token_ids,
    )

    ids = context_token_ids(tokenizer, CTX_SYS)
    input_ids = torch.tensor([ids], dtype=torch.long)
    layer = 0
    base = extract_layer_activations(tiny_model, input_ids, [layer])[layer].clone()

    delta = torch.full((H,), 3.0)
    alpha = 2.0
    hook = DeltaHook(tiny_model, layer, delta, alpha, expected_prompt_len=len(ids))
    with hook:
        hooked = extract_layer_activations(tiny_model, input_ids, [layer])[layer].clone()

    assert hook.n_edits == 1
    diff = hooked - base  # (1, T, H)
    edit_pos = len(ids) - 1
    # The edit lands EXACTLY at len(tokenized_context) - 1 ...
    assert torch.allclose(diff[0, edit_pos], torch.full((H,), alpha * 3.0), atol=1e-4), (
        diff[0, edit_pos].abs().max()
    )
    # ... and NOWHERE else (same deterministic CPU forward => bitwise equal).
    other = torch.cat([diff[0, :edit_pos], diff[0, edit_pos + 1 :]])
    assert torch.equal(other, torch.zeros_like(other)), other.abs().max()


def test_delta_hook_wrong_prompt_len_raises(tokenizer, tiny_model):
    from explore_persona_space.experiments.issue1415.steering import (
        DeltaHook,
        context_token_ids,
    )

    ids = context_token_ids(tokenizer, CTX_SYS)
    input_ids = torch.tensor([ids], dtype=torch.long)
    hook = DeltaHook(tiny_model, 0, torch.zeros(H), 1.0, expected_prompt_len=len(ids) + 5)
    with hook, pytest.raises(AssertionError), torch.no_grad():
        tiny_model(input_ids=input_ids)


def test_delta_hook_requires_arm_before_prefill(tokenizer, tiny_model):
    from explore_persona_space.experiments.issue1415.steering import (
        DeltaHook,
        context_token_ids,
    )

    ids = context_token_ids(tokenizer, CTX_SYS)
    input_ids = torch.tensor([ids], dtype=torch.long)
    hook = DeltaHook(tiny_model, 0, torch.zeros(H), 1.0)  # expected_prompt_len never set
    with hook, pytest.raises(AssertionError), torch.no_grad():
        tiny_model(input_ids=input_ids)


# ── generate_batch ────────────────────────────────────────────────────


def test_generate_batch_without_hook_deterministic(tokenizer, tiny_model):
    from explore_persona_space.experiments.issue1415.steering import generate_batch

    contexts = [CTX_SYS, CTX_BARE]
    out1 = generate_batch(
        tiny_model, tokenizer, contexts, n=2, max_new_tokens=8, temperature=1.0, seed_base=42
    )
    out2 = generate_batch(
        tiny_model, tokenizer, contexts, n=2, max_new_tokens=8, temperature=1.0, seed_base=42
    )
    assert len(out1) == 2 and all(len(draws) == 2 for draws in out1)
    assert out1 == out2  # per-draw torch.manual_seed(seed_base + i) reproducibility
    assert all(isinstance(t, str) for draws in out1 for t in draws)


def test_generate_batch_with_hook_last_token_and_all_positions(tokenizer, tiny_model):
    from explore_persona_space.experiments.issue1415.steering import DeltaHook, generate_batch

    contexts = [CTX_SYS, CTX_BARE]
    torch.manual_seed(7)
    delta = torch.randn(2, H)  # per-row delta (B, H)
    n = 2

    hook = DeltaHook(tiny_model, 1, delta, alpha=1.0)
    with hook:
        out = generate_batch(
            tiny_model, tokenizer, contexts, n=n, hook=hook, max_new_tokens=6, temperature=1.0
        )
    assert len(out) == 2 and all(len(d) == n for d in out)
    assert hook.n_edits == n  # last-token mode: exactly one (prefill) edit per generate call

    hook_ap = DeltaHook(tiny_model, 1, delta, alpha=1.0, all_positions=True)
    with hook_ap:
        out_ap = generate_batch(
            tiny_model, tokenizer, contexts, n=n, hook=hook_ap, max_new_tokens=6, temperature=1.0
        )
    assert len(out_ap) == 2
    assert hook_ap.n_edits > n  # prefill + every decode step


def test_generate_batch_requires_installed_hook(tokenizer, tiny_model):
    from explore_persona_space.experiments.issue1415.steering import DeltaHook, generate_batch

    hook = DeltaHook(tiny_model, 0, torch.zeros(H), 1.0)  # NOT installed
    with pytest.raises(AssertionError):
        generate_batch(tiny_model, tokenizer, [CTX_SYS], n=1, hook=hook, max_new_tokens=2)


# ── coherence gate ────────────────────────────────────────────────────


def test_coherence_check_and_gate():
    from explore_persona_space.experiments.issue1415.steering import (
        coherence_check,
        condition_passes,
    )

    texts = [
        "word",  # <5 whitespace tokens -> incoherent
        "I'm sorry, I cannot help with that request today.",  # refusal opener -> incoherent
        "The best way to learn is regular spaced practice.",  # coherent
        "  As an AI language model I must decline politely.",  # refusal opener -> incoherent
        "Airplanes stay aloft because wings generate lift.",  # coherent
    ]
    flags = coherence_check(texts)
    assert flags == [False, False, True, False, True]
    assert condition_passes([True] * 5 + [False] * 5)  # exactly 50% passes
    assert not condition_passes([True] * 4 + [False] * 6)


# ── capture_vectors ───────────────────────────────────────────────────


def test_capture_vectors_shapes_and_prefix_boundary(tokenizer, tiny_model):
    from explore_persona_space.experiments.issue1415.steering import (
        capture_vectors,
        context_token_ids,
        prefix_end_index,
    )

    contexts = [CTX_SYS, CTX_BARE]
    layers = [0, 1]
    completions = [
        ["A fine answer with several words in it.", ""],  # one EMPTY -> dropped from V_a
        ["Lift from the wings keeps them up.", "Another plausible answer."],
    ]
    out = capture_vectors(
        tiny_model, tokenizer, contexts, layers, completions=completions, batch_size=2
    )
    assert out["layers"] == layers
    recs = out["per_context"]
    assert len(recs) == 2
    for b, rec in enumerate(recs):
        assert rec["v_c_context"].shape == (len(layers), H)
        assert rec["v_c_prefix"].shape == (len(layers), H)
        assert rec["v_a_mean"].shape == (len(layers), H)
        assert not torch.equal(rec["v_c_context"], rec["v_c_prefix"])
        ids = context_token_ids(tokenizer, contexts[b])
        pe = prefix_end_index(tokenizer, ids)
        assert rec["ctx_len"] == len(ids) and rec["prefix_end"] == pe
        assert 2 <= pe < len(ids)
        # the prefix segment is exactly the system block (explicit or default)
        prefix_text = tokenizer.decode(ids[:pe])
        assert prefix_text.startswith("<|im_start|>system") and prefix_text.endswith("<|im_end|>\n")
    assert recs[0]["n_empty_completions"] == 1
    assert recs[0]["v_a_per_completion"].shape == (1, len(layers), H)
    assert recs[1]["n_empty_completions"] == 0
    assert recs[1]["v_a_per_completion"].shape == (2, len(layers), H)


def test_capture_vectors_all_empty_completions_fails_loud(tokenizer, tiny_model):
    from explore_persona_space.experiments.issue1415.steering import capture_vectors

    with pytest.raises(AssertionError, match="completions empty"):
        capture_vectors(tiny_model, tokenizer, [CTX_SYS], [0], completions=[["", ""]])
