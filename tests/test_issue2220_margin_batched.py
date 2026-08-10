"""Batched-rewrite equivalence gate for the issue #2220 teacher-forced margin (F9).

Compares the production ``_batched_ln_logp`` (right-padded batch, one forward
per chunk, single armed steering hook) against the serial oracle
``_ln_logp_one`` on a tiny from-config Qwen2 model (same arch as production —
``model.model.layers`` block path — fp32 CPU, no download, no network), with
B >= 2 answers of DIFFERENT lengths so padding actually fires, at alpha != 0,
for BOTH steering positions (context: DeltaHook ``arm_at(n_p-1)`` edit;
answer: the teacher-forced range hook). Bars: per-value atol 1e-3 + cosine
>= 0.999 (fp32 CPU — reduction-order jitter only), plus chunk-size invariance
(batch_size=2 vs 4).

Also pins the teacher-forced hook semantics: the round-1 serial form armed
``expected_prompt_len = n_p`` on a full-sequence forward, which trips
DeltaHook's ``expected_prompt_len == T`` assert (T = prompt + answer there).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import scripts.issue2220_readwrite as rw

D_MODEL = 64
LAYER = 1
PAD_ID = 0


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(0)
    cfg = Qwen2Config(
        hidden_size=D_MODEL,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


PROMPT = list(range(5, 15))  # 10 prompt tokens
ANSWERS = [[20, 21, 22], [30, 31], [40, 41, 42, 43, 44], [50]]  # B=4, ragged


@pytest.mark.parametrize("position", ["context", "answer"])
def test_batched_matches_serial(tiny_model, position):
    torch.manual_seed(1)
    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    alpha = 3.0

    serial = [
        rw._ln_logp_one(tiny_model, PROMPT, a, direction, LAYER, alpha, position) for a in ANSWERS
    ]
    batched = rw._batched_ln_logp(
        tiny_model, PROMPT, ANSWERS, direction, LAYER, alpha, position, pad_id=PAD_ID, batch_size=4
    )
    s = np.asarray(serial)
    b = np.asarray(batched)
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(b))
    assert np.allclose(b, s, atol=1e-3), (b, s)
    cos = float(np.dot(b, s) / (np.linalg.norm(b) * np.linalg.norm(s)))
    assert cos >= 0.999, cos

    # chunk-size invariance: two chunks of 2 == one chunk of 4
    chunked = rw._batched_ln_logp(
        tiny_model, PROMPT, ANSWERS, direction, LAYER, alpha, position, pad_id=PAD_ID, batch_size=2
    )
    assert np.allclose(np.asarray(chunked), b, atol=1e-5)


def test_steering_actually_changes_logp(tiny_model):
    """The hook must ENGAGE: alpha=0 vs alpha=8 differ (guards against a
    silently no-op hook making the equivalence above vacuously true)."""
    torch.manual_seed(2)
    direction = torch.randn(D_MODEL)
    direction = direction / direction.norm()
    for position in ("context", "answer"):
        base = rw._batched_ln_logp(
            tiny_model, PROMPT, ANSWERS, direction, LAYER, 0.0, position, pad_id=PAD_ID
        )
        steered = rw._batched_ln_logp(
            tiny_model, PROMPT, ANSWERS, direction, LAYER, 8.0, position, pad_id=PAD_ID
        )
        assert not np.allclose(base, steered, atol=1e-6), position


def test_teacher_forced_margin_uses_batched_path(tiny_model, monkeypatch):
    """_teacher_forced_margin dispatches _batched_ln_logp (live-path link,
    hollow-gate rule) and reduces pos-neg per context."""
    calls = {}
    real = rw._batched_ln_logp

    def _spy(model, prompt_ids, answers_ids, *a, **kw):
        calls["n_answers"] = len(answers_ids)
        return real(model, prompt_ids, answers_ids, *a, **kw)

    monkeypatch.setattr(rw, "_batched_ln_logp", _spy)

    class _Tok:
        pad_token_id = PAD_ID

        def encode(self, text, add_special_tokens=False):
            return [20 + (hash(text) % 100), 21]

    monkeypatch.setattr(
        "explore_persona_space.experiments.issue1415.steering.context_token_ids",
        lambda tok, ctx: PROMPT,
    )
    torch.manual_seed(3)
    direction = torch.randn(D_MODEL)
    pools = {"pos": ["good a", "good b"], "neg": ["bad a", "bad b", "bad c"]}
    m = rw._teacher_forced_margin(
        tiny_model, _Tok(), [{"system": None, "user": "u"}], pools, direction, LAYER, 1.0, "context"
    )
    assert np.isfinite(m)
    assert calls["n_answers"] == 5  # pos + neg in ONE batched call per context
