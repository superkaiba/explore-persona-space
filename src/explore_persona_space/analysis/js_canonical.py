# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ※, ‖, ½) in scientific docstrings/comments.
"""Canonical Rao-Blackwellized sequence-level JS/KL estimator math (issue #540).

Implements the Rao-Blackwellized Monte Carlo estimator of sequence-level
divergence between two conditioned language models (Amini/Vieira/Cotterell
2025, arXiv 2504.10637 §3) with the project canonicalization from
``.claude/rules/persona-distance-metrics.md``:

- per-position EXACT full-vocabulary divergence between the two next-token
  distributions (only the prefix distribution is Monte Carlo),
- headline JS in **base 2** with the per-position mixture
  ``m = ½(p_a + p_b)``, responses sampled from BOTH sides,
- **length-normalized per-token average** (a deliberate project deviation
  from the paper's un-normalized inner sum — keeps JS ∈ [0, 1] and
  comparable across contexts with different response lengths),
- both directed KLs (nats) + symmetric-KL; the asymmetry is diagnostic.

The RB-JS aggregation uses ONLY the side-matched half-term per sample
(``0.5·E_{y~a}[KL(p_a‖m)] + 0.5·E_{y~b}[KL(p_b‖m)]``); the symmetric
per-position JS is also computed for the position-profile figure.

Pure tensor math — no I/O, no model loading (``teacher_forced_response_logps``
TAKES an already-loaded model). Unit-testable on CPU
(``tests/test_js_canonical.py``). Numerics: fp32 log-softmax inputs required;
the mixture uses the ``logaddexp`` trick from
``explore_persona_space.analysis.divergence`` (line 72):
``log m = logaddexp(logp_a, logp_b) − ln 2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)

# Qwen-2.5-7B-Instruct terminators: <|im_end|> (the chat-template stop) and
# <|endoftext|> (the base EOS). Generation may stop on EITHER; the append
# rule treats EITHER trailing terminator as already-terminated — never
# double-append (plan §4 Phase S).
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
TERMINATOR_IDS = (IM_END_ID, ENDOFTEXT_ID)


@dataclass
class PositionDivergences:
    """Per-position divergences for ONE sampled response scored under both
    contexts. All arrays are shape ``(T,)`` (T = response length).

    Attributes:
        js_bits: symmetric per-position JS, base 2, in [0, 1] — stored for
            the position-profile figure ONLY (the RB-JS aggregation uses the
            side-matched half-term, see ``rb_pair_estimate``).
        kl_side_m_bits: ``KL(p_side ‖ m) / ln 2`` — the side-matched JS
            half-term (bits).
        kl_side_other_nats: ``KL(p_side ‖ p_other)`` — the directed KL (nats).
    """

    js_bits: np.ndarray
    kl_side_m_bits: np.ndarray
    kl_side_other_nats: np.ndarray


def _mask_and_renormalize(logp: torch.Tensor, token_id: int) -> torch.Tensor:
    """Drop one vocab entry from a log-prob tensor and renormalize.

    ``p'_i = p_i / (1 − p_tok)`` for ``i ≠ tok``; in log space
    ``logp'_i = logp_i − log1p(−exp(logp_tok))``. Used for the optional
    marker-masked JS diagnostic (※ id 83399 masked out — plan §6).

    Args:
        logp: (T, V) fp32 log-softmax.
        token_id: vocab index to remove.

    Returns:
        (T, V−1) fp32 renormalized log-probs.
    """
    _T, V = logp.shape
    keep = torch.ones(V, dtype=torch.bool, device=logp.device)
    keep[token_id] = False
    log_keep_mass = torch.log1p(-torch.exp(logp[:, token_id]).clamp(max=1.0 - 1e-7))
    return logp[:, keep] - log_keep_mass[:, None]


def per_position_divergences(
    logp_side: torch.Tensor,
    logp_other: torch.Tensor,
    exclude_token_id: int | None = None,
) -> PositionDivergences:
    """Exact full-vocab per-position divergences (the RB inner term).

    ``log m = logaddexp(logp_side, logp_other) − ln 2`` (the numerically
    right mixture trick — ``divergence.py:72``).

    Args:
        logp_side: (T, V) fp32 log-softmax under the SAMPLING-side context.
        logp_other: (T, V) fp32 log-softmax under the partner context.
        exclude_token_id: if set, mask this vocab entry out of BOTH
            distributions and renormalize before the reduction (the
            marker-masked diagnostic; one extra reduction, no new forwards).

    Returns:
        PositionDivergences with (T,) numpy float64 arrays.
    """
    assert logp_side.dtype == torch.float32, logp_side.dtype
    assert logp_other.dtype == torch.float32, logp_other.dtype
    assert logp_side.shape == logp_other.shape, (logp_side.shape, logp_other.shape)
    assert logp_side.ndim == 2, logp_side.shape

    if exclude_token_id is not None:
        logp_side = _mask_and_renormalize(logp_side, exclude_token_id)
        logp_other = _mask_and_renormalize(logp_other, exclude_token_id)

    log_m = torch.logaddexp(logp_side, logp_other) - LN2
    p_side = logp_side.exp()
    p_other = logp_other.exp()
    # KL(p ‖ m) per position, nats. Guaranteed ≥ 0 up to fp noise — clamp at 0.
    kl_side_m = (p_side * (logp_side - log_m)).sum(dim=-1).clamp(min=0.0)
    kl_other_m = (p_other * (logp_other - log_m)).sum(dim=-1).clamp(min=0.0)
    # Directed KL(p_side ‖ p_other), nats.
    kl_side_other = (p_side * (logp_side - logp_other)).sum(dim=-1).clamp(min=0.0)
    js_bits = (0.5 * kl_side_m + 0.5 * kl_other_m) / LN2

    return PositionDivergences(
        js_bits=js_bits.double().cpu().numpy(),
        kl_side_m_bits=(kl_side_m / LN2).double().cpu().numpy(),
        kl_side_other_nats=kl_side_other.double().cpu().numpy(),
    )


def rb_pair_estimate(
    a_kl_m_bits: np.ndarray,
    b_kl_m_bits: np.ndarray,
    a_kl_ab_nats: np.ndarray,
    b_kl_ba_nats: np.ndarray,
) -> dict:
    """Canonical RB-JS + KL directions for one (pair, probe-set).

    Inputs are PER-SAMPLE per-token means (length normalization — each
    sample's per-position values averaged over its own T; the
    persona-distance-metrics.md deviation from the paper's raw inner sum):

    Args:
        a_kl_m_bits: (n_a,) per-sample mean of ``KL(p_a‖m)/ln2`` — samples
            drawn FROM context a (the side-matched half-term).
        b_kl_m_bits: (n_b,) per-sample mean of ``KL(p_b‖m)/ln2`` — samples
            drawn FROM context b.
        a_kl_ab_nats: (n_a,) per-sample mean of ``KL(p_a‖p_b)`` — sampled
            from the FIRST argument (a), per the paper's estimator.
        b_kl_ba_nats: (n_b,) per-sample mean of ``KL(p_b‖p_a)`` — sampled
            from b.

    Returns:
        dict with ``js_rb_bits`` (= ½·mean(a_kl_m_bits) + ½·mean(b_kl_m_bits)),
        ``kl_ab_nats``, ``kl_ba_nats``, ``sym_kl_nats`` (= ½ their sum),
        ``mc_se_js_bits`` (MC standard error of js_rb over samples),
        and per-side ns.
    """
    a_kl_m_bits = np.asarray(a_kl_m_bits, dtype=np.float64)
    b_kl_m_bits = np.asarray(b_kl_m_bits, dtype=np.float64)
    a_kl_ab_nats = np.asarray(a_kl_ab_nats, dtype=np.float64)
    b_kl_ba_nats = np.asarray(b_kl_ba_nats, dtype=np.float64)
    assert a_kl_m_bits.ndim == 1 and len(a_kl_m_bits) > 0, a_kl_m_bits.shape
    assert b_kl_m_bits.ndim == 1 and len(b_kl_m_bits) > 0, b_kl_m_bits.shape
    assert len(a_kl_ab_nats) == len(a_kl_m_bits), (len(a_kl_ab_nats), len(a_kl_m_bits))
    assert len(b_kl_ba_nats) == len(b_kl_m_bits), (len(b_kl_ba_nats), len(b_kl_m_bits))

    n_a = len(a_kl_m_bits)
    n_b = len(b_kl_m_bits)
    js_rb = 0.5 * float(a_kl_m_bits.mean()) + 0.5 * float(b_kl_m_bits.mean())
    kl_ab = float(a_kl_ab_nats.mean())
    kl_ba = float(b_kl_ba_nats.mean())
    # MC SE of js_rb: var of the ½·mean_a + ½·mean_b combination.
    var_a = float(a_kl_m_bits.var(ddof=1)) if n_a > 1 else 0.0
    var_b = float(b_kl_m_bits.var(ddof=1)) if n_b > 1 else 0.0
    mc_se = math.sqrt(0.25 * var_a / n_a + 0.25 * var_b / n_b)
    return {
        "js_rb_bits": js_rb,
        "kl_ab_nats": kl_ab,
        "kl_ba_nats": kl_ba,
        "sym_kl_nats": 0.5 * (kl_ab + kl_ba),
        "mc_se_js_bits": mc_se,
        "n_samples_a": n_a,
        "n_samples_b": n_b,
    }


def apply_terminator_rule(token_ids: list[int], finish_reason: str) -> tuple[list[int], str]:
    """The EOS dual-terminator append rule (plan §4 Phase S).

    If generation stopped naturally (``finish_reason == "stop"``) and the
    returned ids do NOT already end with a terminator (<|im_end|> 151645 OR
    <|endoftext|> 151643), append <|im_end|> — the EOS decision is part of
    the sequence distribution per the paper's EOS-padded formulation. A
    trailing terminator of EITHER kind counts as already-terminated (never
    double-append). Truncated generations (``finish_reason == "length"``)
    get no append and are flagged.

    Returns:
        ``(ids, action)`` where action ∈ {``appended_151645``,
        ``already_terminated_<id>``, ``truncated_no_append``}.
    """
    ids = list(token_ids)
    if finish_reason == "length":
        return ids, "truncated_no_append"
    if finish_reason != "stop":
        raise ValueError(f"unexpected finish_reason {finish_reason!r} (expected stop|length)")
    if ids and ids[-1] in TERMINATOR_IDS:
        return ids, f"already_terminated_{ids[-1]}"
    ids.append(IM_END_ID)
    return ids, "appended_151645"


def teacher_forced_response_logps(
    model,
    prompt_ids: list[int],
    responses: list[list[int]],
    max_batch: int = 16,
) -> list[torch.Tensor]:
    """Batched teacher-forced scoring of token-id responses under ONE prompt.

    Builds ``input_ids = prompt_ids + resp`` per response (pure token-id
    concatenation — no retokenization), right-pads the batch (causal
    attention makes right-padding safe for scoring: real-token positions are
    unaffected by trailing pads, so no ``position_ids`` surgery is needed —
    contrast the left-pad RoPE gotcha), and returns the fp32 log-softmax over
    the response positions only: the distribution over ``resp[t]`` sits at
    absolute position ``P − 1 + t`` (P = prompt length), i.e. slice
    ``logits[P−1 : P−1+T]``.

    Args:
        model: HF CausalLM, eval mode, on its device. NOT loaded here.
        prompt_ids: the scoring context's prompt token ids (P,).
        responses: list of response token-id lists (variable T_i ≥ 1).
        max_batch: forward sub-batch size (halve on OOM).

    Returns:
        list of ``(T_i, V)`` fp32 log-prob tensors (on the model's device),
        index-aligned with ``responses``.
    """
    assert len(prompt_ids) >= 1, "empty prompt"
    assert all(len(r) >= 1 for r in responses), "empty response in batch"
    device = next(model.parameters()).device
    P = len(prompt_ids)
    out: list[torch.Tensor] = []
    for start in range(0, len(responses), max_batch):
        chunk = responses[start : start + max_batch]
        lengths = [len(r) for r in chunk]
        max_len = P + max(lengths)
        input_ids = torch.zeros((len(chunk), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, resp in enumerate(chunk):
            seq = prompt_ids + resp
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1
        with torch.no_grad():
            logits = model(
                input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
            ).logits
        B, L, V = logits.shape
        assert (len(chunk), max_len) == (B, L), (logits.shape, len(chunk), max_len)
        for i, T in enumerate(lengths):
            resp_logits = logits[i, P - 1 : P - 1 + T, :].float()
            assert resp_logits.shape == (T, V), (resp_logits.shape, T, V)
            out.append(F.log_softmax(resp_logits, dim=-1))
        del logits
    return out
