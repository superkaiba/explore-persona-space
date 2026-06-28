"""Issue #715 — DFT loss sanity/sign checks at a known (uniform-π) distribution.

At a UNIFORM next-token distribution every gold token has probability ``1/V``, so
the DFT weight ``sg(π_θ(y*_t)) = 1/V`` is constant and the DFT loss is exactly
``(1/V) ×`` the SFT loss. This is the cheapest closed-form check that the
multiplicative reweight is wired correctly (right factor, right sign).
"""

from __future__ import annotations

import math

import torch

from explore_persona_space.train.dft_loss import IGNORE_INDEX, dft_reweighted_loss


def _uniform_logits_batch(b: int = 2, t: int = 5, v: int = 8):
    """All-zero logits -> uniform softmax (every token prob 1/V)."""
    logits = torch.zeros(b, t, v, dtype=torch.float64)
    labels = torch.randint(0, v, (b, t))
    labels[:, :1] = IGNORE_INDEX  # one masked prompt position per row
    return logits, labels, v


def test_dft_equals_uniform_factor_times_sft_at_uniform_pi():
    """At uniform π, DFT loss == (1/V) × SFT loss (the uniform-weight identity)."""
    torch.manual_seed(0)
    logits, labels, v = _uniform_logits_batch()
    sft = dft_reweighted_loss(logits, labels, loss_reweight="sft")
    dft = dft_reweighted_loss(logits, labels, loss_reweight="dft")
    assert torch.allclose(dft, sft / v, atol=1e-9), (
        f"At uniform π, DFT={dft.item():.10f} should equal SFT/V={(sft / v).item():.10f}"
    )


def test_sft_loss_at_uniform_pi_is_log_v():
    """At uniform π, the per-completion-token-mean CE is exactly log(V) (sign check)."""
    torch.manual_seed(1)
    logits, labels, v = _uniform_logits_batch()
    sft = dft_reweighted_loss(logits, labels, loss_reweight="sft")
    assert torch.allclose(sft, torch.tensor(math.log(v), dtype=torch.float64), atol=1e-9), (
        f"At uniform π the SFT loss should be log(V)={math.log(v):.6f}, got {sft.item():.6f}"
    )


def test_dft_loss_is_positive_and_finite():
    """DFT loss is a positive finite scalar (cross-entropy never negative)."""
    torch.manual_seed(2)
    logits = torch.randn(3, 6, 10, dtype=torch.float64)
    labels = torch.randint(0, 10, (3, 6))
    labels[:, :2] = IGNORE_INDEX
    dft = dft_reweighted_loss(logits, labels, loss_reweight="dft")
    assert dft.item() > 0 and math.isfinite(dft.item()), dft.item()
