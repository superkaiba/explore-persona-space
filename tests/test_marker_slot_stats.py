# ruff: noqa: RUF002, RUF003  # em-dash + minus sign in docstrings/comments intentional
"""CPU-only tests for the marker-slot raw-logit readouts (task #530 logit_reval).

Covers:
- ``compute_marker_slot_stats``: the ``log P(marker) = z_marker − logZ``
  identity holds exactly per context, and the log-prob agrees with the
  independent ``compute_marker_logprob`` path on the same contexts.
- ``assert_gauge_free_adapter_config``: the gauge assert for the trained −
  base logit readout (LoRA must not touch lm_head / embed_tokens, and
  modules_to_save must be empty).

NO network downloads: the model is a randomly-initialized tiny GPT-2 built
in-test from a config, and the tokenizer is a deterministic char-level stub
(``compute_marker_slot_stats`` only needs ``encode`` + ``pad_token_id`` +
``eos_token_id``).
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from explore_persona_space.eval.marker_logprob import (
    assert_gauge_free_adapter_config,
    compute_marker_logprob,
    compute_marker_slot_stats,
)

VOCAB_SIZE = 128
PAD_ID = 0
EOS_ID = 7
MARKER_TEXT = "M"  # single char → single stub-token


class _StubTokenizer:
    """Deterministic char-level tokenizer stub (ids in [10, 120), single-token chars)."""

    pad_token_id = PAD_ID
    eos_token_id = EOS_ID

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens  # the functions under test always pass False
        return [(ord(c) % 110) + 10 for c in text]


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    """Randomly-initialized tiny GPT-2 (no Hub download) + char-level stub tokenizer."""
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_positions=64,
        bos_token_id=PAD_ID,
        eos_token_id=EOS_ID,
    )
    model = GPT2LMHeadModel(config).eval()
    return model, _StubTokenizer()


CONTEXTS = ["hello world", "abc", "a slightly longer context string"]


def test_slot_stats_identity_logp_equals_z_minus_logz(tiny_model_and_tokenizer):
    """logp == z_marker − logZ to 1e-4 per context (the load-bearing identity)."""
    model, tok = tiny_model_and_tokenizer
    stats = compute_marker_slot_stats(
        model,
        tok,
        contexts=CONTEXTS,
        marker_text=MARKER_TEXT,
        batch_size=8,
        device="cpu",
    )
    assert len(stats) == len(CONTEXTS)
    for i, d in enumerate(stats):
        assert set(d) == {"logp", "z_marker", "z_eos", "logZ"}, d.keys()
        assert all(math.isfinite(v) for v in d.values()), (i, d)
        assert abs(d["logp"] - (d["z_marker"] - d["logZ"])) < 1e-4, (i, d)
        # logZ >= z_marker always (logsumexp dominates any single logit), so
        # logp <= 0 — a sanity bound that catches index transposition.
        assert d["logp"] <= 1e-6, (i, d)


def test_slot_stats_logp_matches_compute_marker_logprob(tiny_model_and_tokenizer):
    """The slot-stats log-prob equals the independent compute_marker_logprob path."""
    model, tok = tiny_model_and_tokenizer
    stats = compute_marker_slot_stats(
        model,
        tok,
        contexts=CONTEXTS,
        marker_text=MARKER_TEXT,
        batch_size=8,
        device="cpu",
    )
    reference = compute_marker_logprob(
        model,
        tok,
        contexts=CONTEXTS,
        marker_text=MARKER_TEXT,
        batch_size=8,
        device="cpu",
    )
    for i, (d, ref) in enumerate(zip(stats, reference, strict=True)):
        assert abs(d["logp"] - ref) < 1e-4, (i, d["logp"], ref)


def test_slot_stats_hand_reference_single_context(tiny_model_and_tokenizer):
    """Single unpadded context: stats match a hand-rolled forward pass exactly."""
    model, tok = tiny_model_and_tokenizer
    context = CONTEXTS[0]
    marker_id = tok.encode(MARKER_TEXT)[0]

    ids = torch.tensor([tok.encode(context)], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    raw = logits[0, -1, :].float()
    expected_z_marker = float(raw[marker_id])
    expected_z_eos = float(raw[EOS_ID])
    expected_logz = float(torch.logsumexp(raw, dim=-1))

    [d] = compute_marker_slot_stats(
        model,
        tok,
        contexts=[context],
        marker_text=MARKER_TEXT,
        batch_size=1,
        device="cpu",
    )
    assert abs(d["z_marker"] - expected_z_marker) < 1e-5
    assert abs(d["z_eos"] - expected_z_eos) < 1e-5
    assert abs(d["logZ"] - expected_logz) < 1e-5


def test_slot_stats_rejects_multi_token_marker(tiny_model_and_tokenizer):
    """Raw logits don't sum across BPE pieces — multi-token markers must raise."""
    model, tok = tiny_model_and_tokenizer
    with pytest.raises(AssertionError, match="SINGLE-token"):
        compute_marker_slot_stats(
            model,
            tok,
            contexts=CONTEXTS[:1],
            marker_text="MM",  # two chars → two stub-tokens
            batch_size=1,
            device="cpu",
        )


# ── Gauge assert (the trained − base logit readout validity check) ──────────

# The actual #530 adapter target set (verified on HF) — must PASS.
CLEAN_530_CONFIG = {
    "target_modules": [
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "q_proj",
    ],
    "modules_to_save": None,
}


def test_gauge_assert_passes_on_clean_config():
    assert_gauge_free_adapter_config(CLEAN_530_CONFIG)  # must not raise


def test_gauge_assert_raises_on_lm_head_target():
    with pytest.raises(AssertionError, match="lm_head"):
        assert_gauge_free_adapter_config({"target_modules": ["q_proj", "lm_head"]})


def test_gauge_assert_raises_on_embed_tokens_target():
    with pytest.raises(AssertionError, match="embed_tokens"):
        assert_gauge_free_adapter_config({"target_modules": ["embed_tokens"]})


def test_gauge_assert_raises_on_modules_to_save():
    with pytest.raises(AssertionError, match="modules_to_save"):
        assert_gauge_free_adapter_config(
            {"target_modules": ["q_proj"], "modules_to_save": ["lm_head"]}
        )
