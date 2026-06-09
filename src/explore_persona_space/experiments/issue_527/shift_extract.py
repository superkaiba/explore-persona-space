"""Residual-stream shift-vector extraction at L20 post-response slot.

Plan §4 Step 6 + §11 (Layer 20 = canonical persona-cosine layer per #207 /
#311 / #341). Forward-only — no generation.

For each held-out context ``c`` (each persona in the eval panel) and each
prompt ``q``:
  1. Construct ``T_c(q) + R_c(q)`` where ``R_c(q)`` is the base-model
     greedy response under persona ``c``'s system prompt (pre-extracted
     into ``R_persona/<c>.json``).
  2. Forward through both BASE and TRAINED models with
     ``output_hidden_states=True``.
  3. Read ``hidden_states[L20 + 1]`` at the POST-RESPONSE SLOT (the first
     ``<|im_end|>`` AFTER R_c in the chat-templated row) — the SAME slot
     the marker-leakage DV reads.
  4. Mean over the per-persona prompt batch.
  5. ``shift_X(c) = h_trained_X(T_c + R_c)[L20+1, slot] −
     h_base(T_c + R_c)[L20+1, slot]``, averaged over the prompts.

The DV ``log P(marker)`` is read from ``logits[batch, slot - 1, MARKER_ID]``
at the same forward pass (causal LM: position ``i``'s logits predict token
``i+1``, so to score ``token[slot]`` we read ``logits[slot-1]``).

Plan §11 Assumption #12: ``hidden_states[L][batch, slot]`` shape sanity
``-1 == HIDDEN_SIZE``.
"""

# ruff: noqa: RUF001, RUF002, RUF003  # math/scientific notation in docstrings + assert msgs

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import EXTRACTION_LAYER, HIDDEN_SIZE, IM_END_ID, MARKER_ID

log = logging.getLogger("issue_527.shift_extract")


@dataclass
class ContextShift:
    """Per-context (persona) shift-vector + log-prob + LOGIT payload.

    Reports BOTH ``delta_logp_marker`` (log-prob space, primary behavioral DV)
    AND ``delta_logit_marker`` (z_marker space, mechanistic readout) per the
    marker-leakage-measurement rule's "Report BOTH log-prob and logit"
    section. Logit-space is non-saturating and gauge-free (LoRA does not
    touch the unembedding W_U in this rig), so off-saturation
    ``Δlog P ≈ Δz_marker`` agrees; on-saturation `Δz_marker` keeps moving
    while `Δlog P` plateaus and the divergence localizes the ceiling.
    """

    persona: str
    n_prompts: int
    shift_vector: torch.Tensor  # (HIDDEN_SIZE,) fp32 CPU
    # log P(marker)_trained − log P(marker)_base at slot, mean over prompts:
    delta_logp_marker: float
    # z_marker (the marker logit, pre-softmax), trained − base — the
    # non-saturating mechanistic readout:
    delta_logit_marker: float
    emission_argmax_trained: float  # frac of prompts where argmax @ slot == MARKER_ID under trained
    emission_argmax_base: float  # same, under base


def _resolve_post_response_slot(
    tokenizer,
    prompt_messages: list[dict],
    full_ids: list[int],
) -> int:
    """Resolve the post-response ``<|im_end|>`` slot index.

    Same logic as i474's ``_resolve_post_response_slot`` — tokenize the
    prompt alone with ``add_generation_prompt=True`` to get the prefix
    length ``P``, then find the first ``<|im_end|>`` at index ``>= P``
    in the full row. That is the assistant-turn terminator (the SAME slot
    the marker occupies on positives at pos_ids[-3] under the marker-only
    collator + the SAME slot the marker-leakage DV reads).
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    P = len(prompt_ids)
    if full_ids[:P] != prompt_ids:
        raise RuntimeError(
            "prompt-only encoding is not a strict prefix of the full row "
            f"encoding (chat-template drift). P={P}, full_ids head: "
            f"{full_ids[: min(P + 3, len(full_ids))]}, "
            f"prompt_ids head: {prompt_ids[: min(P + 3, len(prompt_ids))]}"
        )
    slot = next((i for i in range(P, len(full_ids)) if full_ids[i] == IM_END_ID), None)
    if slot is None:
        raise RuntimeError(
            f"no <|im_end|> (id={IM_END_ID}) at index >= P={P} in row of "
            f"length {len(full_ids)}; tail ids: {full_ids[-10:]}"
        )
    return slot


@torch.no_grad()
def extract_per_context_shift(
    *,
    base_model,
    trained_model,
    tokenizer,
    persona: str,
    persona_prompt: str,
    eval_questions: list[str],
    r_responses: dict[str, str],
    device: str | torch.device | None = None,
) -> ContextShift:
    """Compute one ContextShift bundle for ``persona`` across ``eval_questions``.

    The forward pass is forward-only (no generation, no kv-cache). Each
    prompt is encoded as ``T_persona(q) + R_persona(q)`` (the persona's
    OWN base-model greedy response), forwarded once on the base and once
    on the trained model, and the L20 residual + log P(marker) are read at
    the post-response slot.

    Both models MUST already be on the same device and in ``eval()`` mode;
    this helper does not move or set the mode.
    """
    if device is None:
        device = next(base_model.parameters()).device

    # Round-2 fix per code-review Critical-3: assert R coverage of every
    # eval question BEFORE entering the per-question forward loop. The
    # in-loop ``q not in r_responses`` raise still fires as a defense-
    # in-depth check, but this up-front assert fails LOUD at second 1 of
    # eval (or earlier — the eval rig hoists this via the runtime guard
    # at script entry) instead of N GPU-h into the sweep.
    missing_eval_qs = [q for q in eval_questions if q not in r_responses]
    if missing_eval_qs:
        raise AssertionError(
            f"persona={persona!r} R_persona missing {len(missing_eval_qs)} of "
            f"{len(eval_questions)} eval questions. First missing: "
            f"{missing_eval_qs[0]!r}. Regenerate R over training_pool ∪ "
            f"EVAL_QUESTIONS (run_issue527_generate_R.py round-2 contract)."
        )

    layer_idx_internal = EXTRACTION_LAYER + 1  # hs[0] = embedding output

    shift_acc = torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
    delta_logp_acc = 0.0
    delta_logit_acc = 0.0
    n_used = 0
    emit_trained_count = 0
    emit_base_count = 0

    for q in eval_questions:
        if q not in r_responses:
            raise AssertionError(f"persona={persona!r} R_persona missing response for q={q!r}")
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": r_responses[q]},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        slot = _resolve_post_response_slot(
            tokenizer,
            messages[:2],
            full_ids,  # prompt = system+user only
        )

        ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        out_base = base_model(ids, output_hidden_states=True)
        out_trained = trained_model(ids, output_hidden_states=True)

        hs_base = out_base.hidden_states[layer_idx_internal]  # (1, T, H)
        hs_trained = out_trained.hidden_states[layer_idx_internal]
        if hs_base.shape[-1] != HIDDEN_SIZE:
            raise AssertionError(
                f"hidden_size drift: hs_base.shape[-1]={hs_base.shape[-1]} "
                f"vs HIDDEN_SIZE={HIDDEN_SIZE}"
            )
        shift = (hs_trained[0, slot] - hs_base[0, slot]).float().cpu()
        shift_acc += shift

        # log P(marker) AND z_marker (the marker logit) at the slot, both
        # read from the SAME forward pass per marker-leakage-measurement.md
        # "Report BOTH log-prob and logit". logits[i] predicts token[i+1]
        # (causal-LM offset), so we read at slot-1.
        logits_base_row = out_base.logits[0, slot - 1].float()
        logits_trained_row = out_trained.logits[0, slot - 1].float()
        logp_base = F.log_softmax(logits_base_row, dim=-1)
        logp_trained = F.log_softmax(logits_trained_row, dim=-1)
        delta_logp_acc += float((logp_trained[MARKER_ID] - logp_base[MARKER_ID]).item())
        # z_marker (the marker logit, pre-softmax): non-saturating + gauge-
        # free across cells because LoRA in this rig adapts attn (q/k/v/o)
        # only and never the unembedding W_U.
        delta_logit_acc += float(
            (logits_trained_row[MARKER_ID] - logits_base_row[MARKER_ID]).item()
        )

        # Free-legibility argmax read (marker-leakage rule: emission is a
        # sanity anchor, not the DV).
        if int(out_trained.logits[0, slot - 1].argmax().item()) == MARKER_ID:
            emit_trained_count += 1
        if int(out_base.logits[0, slot - 1].argmax().item()) == MARKER_ID:
            emit_base_count += 1
        n_used += 1

    if n_used == 0:
        raise AssertionError(f"persona={persona!r}: no eval prompts were forwarded")

    shift_mean = shift_acc / n_used
    delta_logp_mean = delta_logp_acc / n_used
    delta_logit_mean = delta_logit_acc / n_used

    return ContextShift(
        persona=persona,
        n_prompts=n_used,
        shift_vector=shift_mean,
        delta_logp_marker=delta_logp_mean,
        delta_logit_marker=delta_logit_mean,
        emission_argmax_trained=emit_trained_count / n_used,
        emission_argmax_base=emit_base_count / n_used,
    )
