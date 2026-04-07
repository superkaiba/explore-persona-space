"""Tests for src/train/utils.py."""

import torch

from make_evil_dumb.train.utils import compute_log_probs


class MockModel:
    """Minimal mock that returns logits from a simple linear mapping."""

    def __init__(self, vocab_size=10):
        self.training = False
        self.vocab_size = vocab_size

    def __call__(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        # Return random but deterministic logits
        torch.manual_seed(0)
        logits = torch.randn(batch, seq_len, self.vocab_size)

        class Output:
            pass

        out = Output()
        out.logits = logits
        return out


def test_compute_log_probs_shape():
    model = MockModel(vocab_size=10)
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 10, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, 10, (batch_size, seq_len))
    # Mark first 3 tokens as prompt (should be ignored)
    labels[:, :3] = -100

    result = compute_log_probs(model, input_ids, attention_mask, labels)
    assert result.shape == (batch_size,)
    assert torch.all(result <= 0)  # Log probs should be negative


def test_compute_log_probs_all_masked():
    model = MockModel(vocab_size=10)
    batch_size, seq_len = 1, 4
    input_ids = torch.randint(0, 10, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100)  # All masked

    result = compute_log_probs(model, input_ids, attention_mask, labels)
    assert result.shape == (batch_size,)
    # With all tokens masked, sum is 0, denominator clamped to 1, so result is 0
    assert torch.allclose(result, torch.zeros(batch_size))
