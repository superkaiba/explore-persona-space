"""DFT (Dynamic Fine-Tuning) per-token loss reweight — the single algorithmic
core of issue #715.

DFT (Wu et al. 2025, arXiv:2508.05629) rewrites the SFT gradient as an on-policy
policy gradient with reward ``1[y=y*]`` and importance weight ``1/π_θ``. The
``1/π`` term puts large gradient mass on low-probability gold tokens. DFT cancels
it by multiplying each completion token's cross-entropy by a **stop-gradient copy
of that token's predicted probability** ``sg(π_θ(y*_t))`` — the per-token
token-level form ``eq:dr-loss-token-level`` (chosen over the whole-sequence
product form for numerical stability, per the paper's own choice).

The reweight is **parameterization-invariant** (it is purely per-token on the
loss), so the SAME function serves the issue-#715 LoRA arm (P1/P2/P3) and the
full-FT arm (P4) identically — only ``use_lora`` differs between them, never the
loss form.

Single-variable discipline (the load-bearing property): SFT and DFT differ ONLY
by the multiplicative weight. With ``loss_reweight="sft"`` the weight is ``≡1``
and :func:`dft_reweighted_loss` reduces to per-completion-token-mean
cross-entropy; with ``loss_reweight="dft"`` the weight is the detached softmax
probability of the gold token. SFT and DFT therefore invoke the SAME code path,
branching only on the flag.

**Reduction (named explicitly per plan §13 + §-AnalyzerNotes Stats(1)):** the
loss is the SUM over completion tokens of the (weighted) per-token cross-entropy,
divided by the COMPLETION-TOKEN COUNT — i.e. a per-completion-token MEAN. Both
arms use this IDENTICAL normalization, so the within-substrate SFT-vs-DFT
comparison is clean. This is NOT bit-identical to TRL ``SFTTrainer``'s internal
``num_items_in_batch`` (global-count) divisor under gradient accumulation; the
weight≡1 unit test asserts equivalence against an explicit reference
per-completion-token-mean CE, NOT against TRL's internal reduction (see
``tests/test_dft_weight_one_equals_sft.py``). DFT's published form
(``eq:dr-loss-token-level``) is itself a SUM; we narrate the scope as "DFT
applied at matched per-token-mean normalization," shared identically across both
arms, never "DFT exactly as published."

Reference impl one-liner (github.com/yongliang-wu/DFT, verbatim):
``loss = loss * softmax(shift_logits).gather(1, shift_labels.unsqueeze(-1))
.squeeze(-1).detach()``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Valid values for the loss-reweight flag. ``sft`` is the weight≡1 baseline path;
# ``dft`` applies the detached-softmax multiplicative weight.
LOSS_REWEIGHT_MODES = ("sft", "dft")

# Standard HF / TRL ignore index for masked (prompt + pad) positions.
IGNORE_INDEX = -100


def dft_token_logp(
    shift_logits: torch.Tensor,
    shift_labels: torch.Tensor,
) -> torch.Tensor:
    """Per-token gold log-probability ``log π_θ(y*_t)`` from shifted logits/labels.

    Args:
        shift_logits: ``[B, T-1, V]`` next-token logits (already shifted so position
            ``t`` predicts ``shift_labels[t]``).
        shift_labels: ``[B, T-1]`` next-token gold ids, ``IGNORE_INDEX`` on
            prompt/pad positions.

    Returns:
        ``[B, T-1]`` gold-token log-probabilities. Positions where the label is
        ``IGNORE_INDEX`` carry the log-prob of token id 0 (a clamped placeholder)
        — callers MUST mask those out via the completion mask; they never enter
        the loss because the mask zeroes them.

    Notes:
        ``log_softmax`` is computed in fp32 for numerical stability regardless of
        the model dtype (bf16 logits), matching the body pseudocode and the
        risk-#8 mitigation.
    """
    assert shift_logits.dim() == 3, shift_logits.shape
    assert shift_labels.dim() == 2, shift_labels.shape
    assert shift_logits.shape[:2] == shift_labels.shape, (shift_logits.shape, shift_labels.shape)

    logp = F.log_softmax(shift_logits.float(), dim=-1)  # [B, T-1, V], fp32
    safe = shift_labels.clamp_min(0)  # IGNORE_INDEX (-100) -> 0 placeholder
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    assert tok_logp.shape == shift_labels.shape, (tok_logp.shape, shift_labels.shape)
    return tok_logp


def dft_reweighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_reweight: str = "sft",
) -> torch.Tensor:
    """Per-completion-token-mean (optionally DFT-reweighted) cross-entropy.

    Computes the loss directly from raw model logits + labels, shifting
    internally and masking to completion tokens via ``labels != IGNORE_INDEX``.
    SFT and DFT invoke this SAME function, branching only on ``loss_reweight``.

    Args:
        logits: ``[B, T, V]`` raw model logits (un-shifted).
        labels: ``[B, T]`` next-token labels with ``IGNORE_INDEX`` on prompt/pad.
        loss_reweight: ``"sft"`` (weight ≡ 1, baseline CE) or ``"dft"``
            (multiply each completion token's CE by ``sg(π_θ(y*_t))``).

    Returns:
        Scalar loss = ``sum_t w_t · (-log π_θ(y*_t))`` over completion tokens,
        divided by the completion-token count (per-completion-token MEAN). The
        DFT weight ``w_t = π_θ(y*_t).detach()`` is stop-gradient (``requires_grad
        is False``); the SFT weight is ``1``.

    Raises:
        ValueError: on an unknown ``loss_reweight`` mode.
    """
    if loss_reweight not in LOSS_REWEIGHT_MODES:
        raise ValueError(
            f"loss_reweight must be one of {LOSS_REWEIGHT_MODES}, got {loss_reweight!r}"
        )

    assert logits.dim() == 3, logits.shape
    assert labels.dim() == 2, labels.shape
    assert logits.shape[:2] == labels.shape, (logits.shape, labels.shape)

    # Shift so position t predicts token t+1 (standard causal-LM next-token).
    shift_logits = logits[:, :-1, :]  # [B, T-1, V]
    shift_labels = labels[:, 1:]  # [B, T-1]
    comp_mask = (shift_labels != IGNORE_INDEX).to(shift_logits.dtype)  # [B, T-1]

    tok_logp = dft_token_logp(shift_logits, shift_labels)  # [B, T-1], fp32

    if loss_reweight == "dft":  # noqa: SIM108 — explicit branch documents the SFT/DFT split
        # sg(π_θ(y*_t)); detach == stop-gradient. requires_grad is False so no
        # gradient flows through the weight (asserted in the unit test).
        w = tok_logp.exp().detach()
    else:  # "sft": byte-identical path, weight ≡ 1
        w = torch.ones_like(tok_logp)

    per_tok = -(w * tok_logp)  # [B, T-1]; the ONLY difference vs SFT is `w`
    denom = comp_mask.sum().clamp_min(1.0)
    loss = (per_tok * comp_mask).sum() / denom
    return loss


def reference_sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reference per-completion-token-mean cross-entropy (independent of DFT code).

    The weight≡1 SFT-equivalence test asserts ``dft_reweighted_loss(..., "sft")``
    equals THIS function on a toy batch. Deliberately implemented via
    ``F.cross_entropy(reduction="mean")`` over the un-ignored positions so the two
    code paths are genuinely independent (the test is not a tautology). HF's
    ``ignore_index`` makes ``reduction="mean"`` divide by the count of
    non-ignored targets — the per-completion-token mean.
    """
    assert logits.dim() == 3, logits.shape
    assert labels.dim() == 2, labels.shape
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
