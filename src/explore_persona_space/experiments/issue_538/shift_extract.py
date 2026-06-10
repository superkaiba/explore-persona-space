"""Residual-stream shift-vector extraction at L20 post-response slot.

Plan §4 Step 6 + §11 (Layer 20 = canonical persona-cosine layer per #207 /
#311 / #341). Forward-only — no generation.

Issue #538 delta vs #527: alongside the existing
``delta_logp_marker`` / ``delta_logit_marker`` / argmax-emission fields, the
extractor now also reports a ``marker_slot_stats`` block per persona with
RAW (not delta) ``logp_marker / z_marker / z_eos / logZ / slot_index`` for
BOTH the trained and the base side, captured in the SAME HF forward pass
(plan §6 "Marker-slot storage contract"). At the harder dial point the
log-prob axis saturates; the rule's three-space analysis (log-prob primary
/ logit + EOS-margin secondary / probability sanity) needs the raw logit
floats to localize where the saturation lives. Without these floats the
``Δ(z_marker − z_eos)`` EOS-margin is unrecoverable post-hoc.

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
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import EXTRACTION_LAYER, HIDDEN_SIZE, IM_END_ID, MARKER_ID

log = logging.getLogger("issue_538.shift_extract")


@dataclass
class MarkerSlotStats:
    """Raw marker-slot statistics for ONE model side (trained OR base).

    Plan §6 "Marker-slot storage contract" — RAW (not delta) floats from
    the marker-leakage-measurement rule's three-space analysis. Mean over
    prompts per (persona × side).

    - ``logp_marker``: log P(marker_id) at the slot (post-softmax over the
      full vocabulary).
    - ``z_marker``: the marker logit (pre-softmax) at the slot.
    - ``z_eos``: the EOS / ``<|im_end|>`` logit (pre-softmax) at the slot.
      The marker-leakage rule's preferred logit readout is the EOS margin
      ``z_marker − z_eos`` (distance-to-emission); reporting it from BOTH
      sides lets the analyzer compute trained − base under the gauge-free
      mechanistic readout.
    - ``logZ``: logsumexp over the full logit row (the log-partition).
      Bookkeeping: ``logp_marker = z_marker − logZ`` should hold up to
      fp precision.
    """

    logp_marker: float
    z_marker: float
    z_eos: float
    logZ: float


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

    Issue #538 extension: ``marker_slot_stats_trained`` and
    ``marker_slot_stats_base`` carry the RAW per-side floats (mean over
    prompts) per plan §6 "Marker-slot storage contract".
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
    # Plan §6 Marker-slot storage contract (issue_538 NEW vs issue_527).
    # Mean over prompts of the per-side raw stats.
    marker_slot_stats_trained: MarkerSlotStats = field(
        default_factory=lambda: MarkerSlotStats(0.0, 0.0, 0.0, 0.0)
    )
    marker_slot_stats_base: MarkerSlotStats = field(
        default_factory=lambda: MarkerSlotStats(0.0, 0.0, 0.0, 0.0)
    )
    # Mean of the resolved slot index across prompts (analyzer reads this for
    # sanity; expected to be near-constant within a persona because R_c
    # length varies only slightly).
    slot_index_mean: float = 0.0


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
    # eval question BEFORE entering the per-question forward loop.
    missing_eval_qs = [q for q in eval_questions if q not in r_responses]
    if missing_eval_qs:
        raise AssertionError(
            f"persona={persona!r} R_persona missing {len(missing_eval_qs)} of "
            f"{len(eval_questions)} eval questions. First missing: "
            f"{missing_eval_qs[0]!r}. Regenerate R over training_pool ∪ "
            f"EVAL_QUESTIONS (run_issue538_generate_R.py round-2 contract)."
        )

    layer_idx_internal = EXTRACTION_LAYER + 1  # hs[0] = embedding output

    shift_acc = torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
    delta_logp_acc = 0.0
    delta_logit_acc = 0.0
    n_used = 0
    emit_trained_count = 0
    emit_base_count = 0

    # Plan §6 Marker-slot storage contract — accumulate the FOUR raw
    # per-side floats per prompt and divide by n_used at the end. Mean
    # over prompts gives a single representative value per (persona × side)
    # for the analyzer's three-space saturation localizer.
    logp_marker_trained_acc = 0.0
    z_marker_trained_acc = 0.0
    z_eos_trained_acc = 0.0
    logZ_trained_acc = 0.0
    logp_marker_base_acc = 0.0
    z_marker_base_acc = 0.0
    z_eos_base_acc = 0.0
    logZ_base_acc = 0.0
    slot_index_acc = 0

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

        # logits[i] predicts token[i+1] (causal-LM offset), so we read at slot-1.
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

        # ── Plan §6 Marker-slot storage contract (issue_538 NEW) ───────────
        # Per-side RAW floats from THIS forward pass. We don't subtract here
        # because the marker-leakage rule's three-space analyzer reads BOTH
        # the raw floats AND the deltas to localize saturation.
        logZ_base = float(torch.logsumexp(logits_base_row, dim=-1).item())
        logZ_trained = float(torch.logsumexp(logits_trained_row, dim=-1).item())
        z_marker_base = float(logits_base_row[MARKER_ID].item())
        z_marker_trained = float(logits_trained_row[MARKER_ID].item())
        z_eos_base = float(logits_base_row[IM_END_ID].item())
        z_eos_trained = float(logits_trained_row[IM_END_ID].item())
        logp_marker_base_pp = float(logp_base[MARKER_ID].item())
        logp_marker_trained_pp = float(logp_trained[MARKER_ID].item())

        logp_marker_trained_acc += logp_marker_trained_pp
        z_marker_trained_acc += z_marker_trained
        z_eos_trained_acc += z_eos_trained
        logZ_trained_acc += logZ_trained
        logp_marker_base_acc += logp_marker_base_pp
        z_marker_base_acc += z_marker_base
        z_eos_base_acc += z_eos_base
        logZ_base_acc += logZ_base
        slot_index_acc += slot

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

    mss_trained = MarkerSlotStats(
        logp_marker=logp_marker_trained_acc / n_used,
        z_marker=z_marker_trained_acc / n_used,
        z_eos=z_eos_trained_acc / n_used,
        logZ=logZ_trained_acc / n_used,
    )
    mss_base = MarkerSlotStats(
        logp_marker=logp_marker_base_acc / n_used,
        z_marker=z_marker_base_acc / n_used,
        z_eos=z_eos_base_acc / n_used,
        logZ=logZ_base_acc / n_used,
    )

    return ContextShift(
        persona=persona,
        n_prompts=n_used,
        shift_vector=shift_mean,
        delta_logp_marker=delta_logp_mean,
        delta_logit_marker=delta_logit_mean,
        emission_argmax_trained=emit_trained_count / n_used,
        emission_argmax_base=emit_base_count / n_used,
        marker_slot_stats_trained=mss_trained,
        marker_slot_stats_base=mss_base,
        slot_index_mean=slot_index_acc / n_used,
    )
