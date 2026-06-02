"""Numerical equivalence: GPU-fused JS/KL reduction == CPU path.

The #470 Phase-3 GPU-utilization fix moves the per-token JS/KL reduction onto
the GPU so the H100 is actually used. The reduction MUST produce numerically
equivalent values to the legacy CPU path (``teacher_force_batch`` returning a
CPU log-softmax tensor, then ``compute_js_divergence`` / ``compute_kl_divergence``
called on it) so resumed cells stay consistent with the 2 cells already on
the pod volume.

These tests construct a small fake "model" (returns synthetic logits) so the
test runs on CPU in CI without HF downloads, and prove:

  1. ``teacher_force_and_reduce_js_kl`` returns the SAME (js, kl_p_to_q,
     kl_q_to_p) as the CPU path within fp32 round-off (1e-5 abs tol).
  2. Selecting different ``p_index`` / ``q_index`` swaps the KL direction
     symmetrically.
  3. Identical distributions give JS == 0 and both KLs == 0.
  4. Invalid p/q indices raise ValueError.

The existing functions' default behavior is unchanged (regression-tested by
``test_teacher_force_batch_cpu_path_preserved``).
"""

from __future__ import annotations

import math

import pytest
import torch

from explore_persona_space.analysis.divergence import (
    compute_js_divergence,
    compute_kl_divergence,
    teacher_force_and_reduce_js_kl,
    teacher_force_batch,
)


class _FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Returns deterministic, seed-dependent logits per input row.

    For a batch (N, T) of input_ids we emit (N, T, V) logits = a per-row
    "personality vector" broadcast over time + a small per-position offset
    so JS/KL are non-trivial. This lets us test the fused helper end-to-end
    without loading a 7B model.
    """

    def __init__(self, vocab_size: int = 64, hidden_seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.hidden_seed = hidden_seed

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _FakeOutput:
        n, t = input_ids.shape
        v = self.vocab_size
        # Per-row "personality" — deterministic from row index so two rows with
        # different content produce different logit distributions.
        gen = torch.Generator(device=input_ids.device).manual_seed(self.hidden_seed)
        personality = torch.randn(n, v, generator=gen, device=input_ids.device)
        # Per-position offset to make time-axis non-degenerate.
        time_offsets = torch.linspace(-1, 1, t, device=input_ids.device)
        logits = personality.unsqueeze(1) + time_offsets.unsqueeze(0).unsqueeze(-1)
        return _FakeOutput(logits)


def _make_batch(n: int, prompt_lens: list[int], response_len: int, vocab: int = 64):
    """Build (batch_inputs, prompt_lengths, response_len) for a small fake batch.

    Mimics the shape ``build_teacher_force_inputs`` returns. Total length per
    row = pad_len + prompt_len + response_len. Left-padded to a common max.
    """
    seq_lens = [pl + response_len for pl in prompt_lens]
    max_len = max(seq_lens)
    input_ids = torch.zeros(n, max_len, dtype=torch.long)
    attention_mask = torch.zeros(n, max_len, dtype=torch.long)
    for i in range(n):
        pad = max_len - seq_lens[i]
        # Use varied token ids so the model's logits depend on content
        input_ids[i, pad:] = torch.arange(1, seq_lens[i] + 1)
        attention_mask[i, pad:] = 1
    batch_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    return batch_inputs, list(prompt_lens), response_len


@pytest.fixture
def fake_setup():
    """Two rows, asymmetric prompt lengths (matches real Phase 3 left-padding)."""
    torch.manual_seed(0)
    vocab = 64
    response_len = 5
    prompt_lens = [3, 6]
    batch_inputs, pls, rlen = _make_batch(
        n=2, prompt_lens=prompt_lens, response_len=response_len, vocab=vocab
    )
    model = _FakeModel(vocab_size=vocab, hidden_seed=123)
    return model, batch_inputs, pls, rlen, vocab


def test_fused_matches_cpu_path_for_js_and_both_kls(fake_setup):
    """Headline numerical-equivalence test (the load-bearing invariant).

    Fused GPU reduction (rebuilt on CPU here since CI has no GPU) must produce
    the same (JS, KL(P||Q), KL(Q||P)) as the legacy CPU path within fp32 tol.
    """
    model, batch_inputs, pls, rlen, _ = fake_setup

    # --- Legacy CPU path: forward, log_softmax, then JS/KL on CPU ---
    log_probs_cpu = teacher_force_batch(
        model=model,
        batch_inputs=batch_inputs,
        prompt_lengths=pls,
        response_len=rlen,
        device="cpu",
        max_batch=16,
    )
    assert log_probs_cpu.shape == (2, rlen, model.vocab_size)
    lp_p_cpu = log_probs_cpu[0]
    lp_q_cpu = log_probs_cpu[1]
    js_cpu = compute_js_divergence(lp_p_cpu, lp_q_cpu).item()
    kl_pq_cpu = compute_kl_divergence(lp_p_cpu, lp_q_cpu).item()
    kl_qp_cpu = compute_kl_divergence(lp_q_cpu, lp_p_cpu).item()

    # --- Fused path (p_index=0, q_index=1; same reduction, less I/O) ---
    js_fused, kl_pq_fused, kl_qp_fused = teacher_force_and_reduce_js_kl(
        model=model,
        batch_inputs=batch_inputs,
        prompt_lengths=pls,
        response_len=rlen,
        device="cpu",
        max_batch=16,
        p_index=0,
        q_index=1,
    )

    # fp32 round-off tolerance is well under 1e-5 for these small tensors;
    # the 1e-4 abs tol in the task brief is generous slack.
    assert math.isclose(js_fused, js_cpu, abs_tol=1e-5), (js_fused, js_cpu)
    assert math.isclose(kl_pq_fused, kl_pq_cpu, abs_tol=1e-5), (kl_pq_fused, kl_pq_cpu)
    assert math.isclose(kl_qp_fused, kl_qp_cpu, abs_tol=1e-5), (kl_qp_fused, kl_qp_cpu)

    # Sanity: JS is bounded in [0, ln 2], KLs are non-negative, JS is symmetric
    # (so swapping p/q gives the same JS), KL is not (asymmetric).
    assert 0 <= js_fused <= math.log(2.0) + 1e-3
    assert kl_pq_fused >= -1e-6
    assert kl_qp_fused >= -1e-6


def test_fused_kl_direction_swaps_with_p_q_indices(fake_setup):
    """Swapping (p_index, q_index) swaps the KL direction (JS stays symmetric)."""
    model, batch_inputs, pls, rlen, _ = fake_setup
    js_a, kl_a_pq, kl_a_qp = teacher_force_and_reduce_js_kl(
        model=model,
        batch_inputs=batch_inputs,
        prompt_lengths=pls,
        response_len=rlen,
        device="cpu",
        p_index=0,
        q_index=1,
    )
    js_b, kl_b_pq, kl_b_qp = teacher_force_and_reduce_js_kl(
        model=model,
        batch_inputs=batch_inputs,
        prompt_lengths=pls,
        response_len=rlen,
        device="cpu",
        p_index=1,
        q_index=0,
    )
    # JS is symmetric ⇒ same number either way.
    assert math.isclose(js_a, js_b, abs_tol=1e-6)
    # Swapping P/Q swaps the two KL directions.
    assert math.isclose(kl_a_pq, kl_b_qp, abs_tol=1e-6)
    assert math.isclose(kl_a_qp, kl_b_pq, abs_tol=1e-6)


def test_identical_distributions_yield_zero(fake_setup):
    """If both rows produce the same logits, JS == KL == 0."""

    class _ConstantModel:
        vocab_size = 32

        def __call__(self, input_ids, attention_mask):
            n, t = input_ids.shape
            v = self.vocab_size
            # Same logits for every row, so log-softmax is identical row-wise.
            logits = torch.arange(v, dtype=torch.float32).expand(n, t, v).clone()
            return _FakeOutput(logits)

    model = _ConstantModel()
    batch_inputs, pls, rlen = _make_batch(n=2, prompt_lens=[3, 3], response_len=4, vocab=32)
    js, kl_pq, kl_qp = teacher_force_and_reduce_js_kl(
        model=model,
        batch_inputs=batch_inputs,
        prompt_lengths=pls,
        response_len=rlen,
        device="cpu",
        p_index=0,
        q_index=1,
    )
    assert abs(js) < 1e-6, js
    assert abs(kl_pq) < 1e-6, kl_pq
    assert abs(kl_qp) < 1e-6, kl_qp


def test_invalid_indices_raise(fake_setup):
    """Out-of-range / duplicate p/q indices raise ValueError (fail loud)."""
    model, batch_inputs, pls, rlen, _ = fake_setup
    with pytest.raises(ValueError):
        teacher_force_and_reduce_js_kl(
            model=model,
            batch_inputs=batch_inputs,
            prompt_lengths=pls,
            response_len=rlen,
            device="cpu",
            p_index=0,
            q_index=0,  # duplicate
        )
    with pytest.raises(ValueError):
        teacher_force_and_reduce_js_kl(
            model=model,
            batch_inputs=batch_inputs,
            prompt_lengths=pls,
            response_len=rlen,
            device="cpu",
            p_index=0,
            q_index=5,  # out of range for N=2 batch
        )


def test_teacher_force_batch_cpu_path_preserved(fake_setup):
    """Regression: the legacy teacher_force_batch default still returns a CPU
    full-vocab log-softmax tensor (the contract #207/#311/#458/#473 depend on)."""
    model, batch_inputs, pls, rlen, _ = fake_setup
    log_probs = teacher_force_batch(
        model=model,
        batch_inputs=batch_inputs,
        prompt_lengths=pls,
        response_len=rlen,
        device="cpu",
    )
    assert log_probs.shape == (2, rlen, model.vocab_size)
    assert log_probs.device.type == "cpu"
    # Row-wise log-softmax means each row's exp sums to 1.
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)
