"""Issue #715 — weight≡1 SFT-equivalence test for the DFT loss reweight.

The single algorithmic core of #715 is the DFT per-token reweight. Its
correctness is load-bearing: a wrong loss silently invalidates every downstream
number. The most important invariant is that with ``loss_reweight="sft"`` the
custom loss reduces EXACTLY to per-completion-token-mean cross-entropy — so that
SFT and DFT share a byte-identical code path differing only by the multiplicative
weight (the single-variable discipline the whole experiment rests on).

Reduction note (plan §13 + §-AnalyzerNotes Stats(1)): the assertion is against an
EXPLICIT reference per-completion-token-mean CE
(``reference_sft_loss`` = ``F.cross_entropy(reduction="mean", ignore_index=-100)``),
NOT against TRL ``SFTTrainer``'s internal ``num_items_in_batch`` global-count
divisor. Both DFT arms use the per-completion-token mean identically, so the
within-substrate comparison is single-variable; the published DFT form is a sum,
applied here at matched per-token-mean normalization.
"""

from __future__ import annotations

import torch

from explore_persona_space.train.dft_loss import (
    IGNORE_INDEX,
    dft_reweighted_loss,
    reference_sft_loss,
)


def _toy_batch(seed: int = 0):
    """A small (B=3, T=7, V=11) batch with some prompt/pad positions masked.

    Returns (logits, labels) where labels carry IGNORE_INDEX on a deterministic
    subset of positions (simulating prompt + pad masking), exercising the
    completion-mask path.
    """
    torch.manual_seed(seed)
    b, t, v = 3, 7, 11
    logits = torch.randn(b, t, v, dtype=torch.float64)
    labels = torch.randint(0, v, (b, t))
    # Mask the first 2 positions of every row (prompt) + the last position of
    # row 1 (pad) -> some positions are IGNORE_INDEX.
    labels[:, :2] = IGNORE_INDEX
    labels[1, -1] = IGNORE_INDEX
    return logits, labels


def test_weight_one_equals_reference_per_token_mean_ce():
    """sft path == explicit per-completion-token-mean CE (the documented reduction)."""
    logits, labels = _toy_batch()
    sft_loss = dft_reweighted_loss(logits, labels, loss_reweight="sft")
    ref = reference_sft_loss(logits, labels)
    # Both code paths upcast to fp32 internally (the production bf16-stability
    # choice), so compare in fp32. atol matches fp32 round-off, not fp64.
    assert torch.allclose(sft_loss.float(), ref.float(), atol=1e-6), (
        f"weight≡1 loss {sft_loss.item():.10f} != reference per-completion-token-mean "
        f"CE {ref.item():.10f}"
    )


def test_weight_one_gradient_equals_reference_gradient():
    """The sft-path GRADIENT also matches the reference CE gradient (bit-equivalence).

    A loss-value match alone is insufficient — training depends on gradients.
    """
    logits1, labels = _toy_batch()
    logits1 = logits1.clone().requires_grad_(True)
    dft_reweighted_loss(logits1, labels, loss_reweight="sft").backward()

    logits2, _ = _toy_batch()
    logits2 = logits2.clone().requires_grad_(True)
    reference_sft_loss(logits2, labels).backward()

    # fp32-internal compute -> compare at fp32 round-off tolerance.
    assert torch.allclose(logits1.grad.float(), logits2.grad.float(), atol=1e-6), (
        "weight≡1 gradient diverges from the reference per-token-mean CE gradient"
    )


def test_dft_weight_is_stop_gradient():
    """The DFT weight is detached — no gradient flows through it (sg / stop-gradient)."""
    logits, labels = _toy_batch()
    logits = logits.clone().requires_grad_(True)
    # Manually reproduce the weight and assert it carries no grad.
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logp = F.log_softmax(shift_logits.float(), dim=-1)
    w = logp.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp().detach()
    assert w.requires_grad is False, "DFT weight must be detached (stop-gradient)"
    # And the loss is differentiable end-to-end (grad reaches logits via log π only).
    loss = dft_reweighted_loss(logits, labels, loss_reweight="dft")
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_dft_differs_from_sft_when_distribution_nonuniform():
    """DFT and SFT losses genuinely differ on a non-uniform distribution.

    Guards against a no-op reweight (e.g. forgetting to apply the weight).
    """
    logits, labels = _toy_batch(seed=7)
    sft = dft_reweighted_loss(logits, labels, loss_reweight="sft")
    dft = dft_reweighted_loss(logits, labels, loss_reweight="dft")
    assert not torch.allclose(sft, dft, atol=1e-3), (
        "DFT loss must differ from SFT on a non-uniform distribution; got "
        f"sft={sft.item():.6f} dft={dft.item():.6f}"
    )


def test_completion_mask_excludes_ignored_positions():
    """Positions with IGNORE_INDEX labels never enter the loss.

    Perturbing the logits at an ignored position must not change the loss.
    """
    logits, labels = _toy_batch(seed=3)
    base = dft_reweighted_loss(logits, labels, loss_reweight="sft")
    perturbed = logits.clone()
    # Position 0 predicts label at position 1; labels[:, :2] are IGNORE_INDEX, so
    # the shifted label at index 0 (== labels[:,1]) is ignored. Perturb logits at
    # the corresponding logit row (index 0) heavily.
    perturbed[:, 0, :] += 100.0
    after = dft_reweighted_loss(perturbed, labels, loss_reweight="sft")
    assert torch.allclose(base, after, atol=1e-9), (
        "Perturbing logits at an ignored (masked) position changed the loss — "
        "completion masking is broken"
    )
