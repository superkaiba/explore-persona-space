# ruff: noqa: RUF002
"""Unit tests for the canonical RB sequence-level JS estimator (issue #540).

Plan §4 tests (a)–(e):
  (a) closed-form categorical reference (independent prob-space formula),
  (b) first-token limit ≡ the pinned parent's ``_js_v1_predictor``,
  (c) bounds + symmetry + self-divergence,
  (d) tiny-model CPU alignment integration (2-layer random Qwen2) +
      batched-vs-serial right-pad equivalence,
  (e) length normalization + the truncation (no-EOS-append) path.

CPU-only; no network (the tiny model is built in-test with a hand-rolled
token-id vocabulary — no tokenizer download).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue532_predictor_stress as i532  # noqa: E402  # pinned parent @ 296c4da2d

from explore_persona_space.analysis import js_canonical as jsc  # noqa: E402

LN2 = math.log(2.0)


def _logp(probs: list[float]) -> torch.Tensor:
    """(1, V) fp32 log-probs from an exact probability vector."""
    return torch.log(torch.tensor([probs], dtype=torch.float32))


# ── (a) closed-form categorical ────────────────────────────────────────────


def test_closed_form_categorical():
    p = [0.7, 0.1, 0.1, 0.1]
    q = [0.1, 0.7, 0.1, 0.1]
    pd = jsc.per_position_divergences(_logp(p), _logp(q))
    # Independent prob-space reference (no logaddexp trick).
    m = [0.5 * (a + b) for a, b in zip(p, q, strict=True)]
    kl_pm = sum(a * math.log(a / c) for a, c in zip(p, m, strict=True))
    kl_qm = sum(b * math.log(b / c) for b, c in zip(q, m, strict=True))
    kl_pq = sum(a * math.log(a / b) for a, b in zip(p, q, strict=True))
    js_bits_ref = (0.5 * kl_pm + 0.5 * kl_qm) / LN2
    assert abs(pd.js_bits[0] - js_bits_ref) < 1e-6
    assert abs(pd.kl_side_m_bits[0] - kl_pm / LN2) < 1e-6
    assert abs(pd.kl_side_other_nats[0] - kl_pq) < 1e-6


def test_masked_renormalized_variant():
    # Masking index 3 and renormalizing must equal computing on the
    # renormalized 3-token distributions directly.
    p = [0.5, 0.2, 0.2, 0.1]
    q = [0.2, 0.5, 0.2, 0.1]
    pd_masked = jsc.per_position_divergences(_logp(p), _logp(q), exclude_token_id=3)
    p3 = [x / 0.9 for x in p[:3]]
    q3 = [x / 0.9 for x in q[:3]]
    pd_ref = jsc.per_position_divergences(_logp(p3), _logp(q3))
    assert abs(pd_masked.js_bits[0] - pd_ref.js_bits[0]) < 1e-6


# ── (b) first-token limit ≡ v1 ─────────────────────────────────────────────


def test_first_token_limit_equals_v1():
    rng = np.random.default_rng(0)
    n_probes, V = 3, 50
    p_a = rng.dirichlet(np.ones(V) * 5.0, size=n_probes).astype(np.float64)
    p_b = rng.dirichlet(np.ones(V) * 5.0, size=n_probes).astype(np.float64)
    v1 = i532._js_v1_predictor(p_a, p_b)

    # RB on length-1 sequences, one sample per (side, probe): the
    # side-matched half-terms reduce to the symmetric per-probe JS mean.
    a_kl_m, b_kl_m, a_kl_ab, b_kl_ba = [], [], [], []
    for k in range(n_probes):
        lp_a = torch.log(torch.tensor(p_a[k : k + 1], dtype=torch.float32))
        lp_b = torch.log(torch.tensor(p_b[k : k + 1], dtype=torch.float32))
        pd_a = jsc.per_position_divergences(lp_a, lp_b)  # side = a
        pd_b = jsc.per_position_divergences(lp_b, lp_a)  # side = b
        a_kl_m.append(pd_a.kl_side_m_bits.mean())
        b_kl_m.append(pd_b.kl_side_m_bits.mean())
        a_kl_ab.append(pd_a.kl_side_other_nats.mean())
        b_kl_ba.append(pd_b.kl_side_other_nats.mean())
    rb = jsc.rb_pair_estimate(
        np.array(a_kl_m), np.array(b_kl_m), np.array(a_kl_ab), np.array(b_kl_ba)
    )
    # eps=1e-12 clip difference is negligible for Dirichlet(5) draws.
    assert abs(rb["js_rb_bits"] - v1) < 1e-6


# ── (c) bounds + symmetry ──────────────────────────────────────────────────


def test_bounds_symmetry_and_self():
    rng = np.random.default_rng(1)
    V, T = 32, 7
    lp_p = torch.log_softmax(torch.tensor(rng.normal(size=(T, V)), dtype=torch.float32), dim=-1)
    lp_q = torch.log_softmax(torch.tensor(rng.normal(size=(T, V)), dtype=torch.float32), dim=-1)
    pd_pq = jsc.per_position_divergences(lp_p, lp_q)
    pd_qp = jsc.per_position_divergences(lp_q, lp_p)
    assert (pd_pq.js_bits >= 0).all() and (pd_pq.js_bits <= 1 + 1e-9).all()
    assert np.allclose(pd_pq.js_bits, pd_qp.js_bits, atol=1e-7)  # JS symmetric
    assert (pd_pq.kl_side_other_nats >= 0).all()
    pd_self = jsc.per_position_divergences(lp_p, lp_p)
    assert (pd_self.js_bits <= 1e-7).all()  # JS(p, p) = 0

    rb = jsc.rb_pair_estimate(
        pd_pq.kl_side_m_bits,
        pd_qp.kl_side_m_bits,
        pd_pq.kl_side_other_nats,
        pd_qp.kl_side_other_nats,
    )
    assert rb["sym_kl_nats"] == pytest.approx(0.5 * (rb["kl_ab_nats"] + rb["kl_ba_nats"]))
    assert rb["kl_ab_nats"] >= 0 and rb["kl_ba_nats"] >= 0


# ── (d) tiny-model CPU alignment integration ───────────────────────────────


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def test_tiny_model_alignment_and_self_zero(tiny_model):
    prompt_a = [1, 2, 3]
    prompt_b = [4, 5, 6, 7, 8]  # different prompt LENGTH — the alignment trap
    resp = [9, 10, 11, 12]
    lp_a = jsc.teacher_forced_response_logps(tiny_model, prompt_a, [resp])[0]
    lp_b = jsc.teacher_forced_response_logps(tiny_model, prompt_b, [resp])[0]
    assert lp_a.shape == lp_b.shape == (len(resp), 64)  # T equal across sides
    pd = jsc.per_position_divergences(lp_a, lp_b)
    assert np.isfinite(pd.js_bits).all()
    assert pd.js_bits.shape == (len(resp),)

    # Same context scored twice deterministically → JS ≈ 0 exactly.
    lp_a2 = jsc.teacher_forced_response_logps(tiny_model, prompt_a, [resp])[0]
    pd_self = jsc.per_position_divergences(lp_a, lp_a2)
    assert (pd_self.js_bits <= 1e-7).all()


def test_batched_equals_serial_right_pad(tiny_model):
    """Batched-rewrite equivalence (agent-spec REQUIRED check): B>=2 with
    DIFFERENT response lengths (so right-padding actually fires) must match
    the serial one-at-a-time path."""
    prompt = [1, 2, 3, 4]
    responses = [[9, 10, 11], [12, 13, 14, 15, 16], [17, 18]]
    batched = jsc.teacher_forced_response_logps(tiny_model, prompt, responses, max_batch=3)
    serial = [
        jsc.teacher_forced_response_logps(tiny_model, prompt, [r], max_batch=1)[0]
        for r in responses
    ]
    for got, want in zip(batched, serial, strict=True):
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-5)
        cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
        assert cos >= 0.999


# ── (e) length normalization + truncation path ─────────────────────────────


def test_length_normalization_per_token_mean():
    # Two samples on side a with different lengths: the per-sample value is
    # the per-token MEAN (not the sum), so a longer sample with the same
    # per-position value contributes the same.
    a_terms = np.array([0.5, 0.5])  # both samples average 0.5 bits/token
    b_terms = np.array([0.3])
    rb = jsc.rb_pair_estimate(a_terms, b_terms, np.array([0.1, 0.1]), np.array([0.2]))
    assert rb["js_rb_bits"] == pytest.approx(0.5 * 0.5 + 0.5 * 0.3)
    assert rb["kl_ab_nats"] == pytest.approx(0.1)
    assert rb["kl_ba_nats"] == pytest.approx(0.2)
    assert rb["sym_kl_nats"] == pytest.approx(0.15)
    assert rb["n_samples_a"] == 2 and rb["n_samples_b"] == 1


def test_terminator_rule_branches():
    # stop + no trailing terminator → append <|im_end|> (151645).
    ids, action = jsc.apply_terminator_rule([5, 6, 7], "stop")
    assert ids == [5, 6, 7, jsc.IM_END_ID] and action == "appended_151645"
    # stop + trailing <|im_end|> → already terminated, never double-append.
    ids, action = jsc.apply_terminator_rule([5, 6, jsc.IM_END_ID], "stop")
    assert ids == [5, 6, jsc.IM_END_ID] and action == f"already_terminated_{jsc.IM_END_ID}"
    # stop + trailing <|endoftext|> → EITHER terminator counts (dual rule).
    ids, action = jsc.apply_terminator_rule([5, 6, jsc.ENDOFTEXT_ID], "stop")
    assert ids == [5, 6, jsc.ENDOFTEXT_ID]
    assert action == f"already_terminated_{jsc.ENDOFTEXT_ID}"
    # length → truncated, NO append (the no-EOS-append truncation path).
    ids, action = jsc.apply_terminator_rule([5, 6, 7], "length")
    assert ids == [5, 6, 7] and action == "truncated_no_append"
    with pytest.raises(ValueError):
        jsc.apply_terminator_rule([5], "abort")
