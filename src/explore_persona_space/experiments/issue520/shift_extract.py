"""Shift-vector extraction + on-policy log P(marker) reads for task #520.

Plan §4 step 7 + 8: for each held-out context c (each panel persona),
forward the SAME ``(T_c(q) + R_c(q))`` string under the BASE model and under
each TRAINED adapter, teacher-forced through ``R_c``. Read the residual-
stream activation at layer 20 at the slot immediately after ``R_c`` (the
post-response slot, the same slot the DV reads).

::

    shift_X(c) = h_layer20_post_response(trained_X, T_c(q) + R_c(q))
               - h_layer20_post_response(base,      T_c(q) + R_c(q))

``R_c`` is the base-model greedy response under T_c's system prompt — frozen
across all arms — so trained-vs-base is *at the same input* (the model's own
response under that persona). This IS the canonical on-policy shift
definition (it differs from teacher-forcing a canned answer in that ``R_c``
is what THIS model would have written under persona c).

The same forward pass yields:

- ``log P(marker | T_c(q) + R_c(q))`` at the post-response slot (DV3)
- the residual-stream activation at L20 (and L15 secondary) at the same
  slot (DV1, DV2)
- the argmax / emission rate (DV4 sanity anchor)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.experiments.issue520.persona_panel import (
    IM_END_TOKEN_ID,
    QWEN_HIDDEN_DIM,
    SHIFT_LAYER_PRIMARY,
    SHIFT_LAYER_SECONDARY,
    get_system_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextRead:
    """Per-(persona, question) read of activation + log P(marker)."""

    persona: str
    question: str
    log_p_marker: float
    emission_argmax: bool
    h_layer_primary: list[float]  # length QWEN_HIDDEN_DIM=3584
    h_layer_secondary: list[float]


def _trim_prefix_to_post_response_slot(prefix_ids: list[int]) -> list[int]:
    """Trim the chat-templated prefix so the LAST token is the last token of R_c.

    The Qwen-2.5 chat template emits ``...{assistant content}<|im_end|>\\n``
    after the assistant role. We want the prefix to end with the LAST content
    token of R_c, so the next-position logits predict the marker at the
    post-response slot (the same slot the DV reads).

    Walks back from the end to find the LAST ``<|im_end|>`` and returns the
    prefix up to (not including) it. Raises if none found.
    """
    for i in range(len(prefix_ids) - 1, -1, -1):
        if prefix_ids[i] == IM_END_TOKEN_ID:
            return prefix_ids[:i]
    raise RuntimeError(
        f"Could not find <|im_end|> (id={IM_END_TOKEN_ID}) in chat-templated "
        f"prefix of length {len(prefix_ids)}; tokenizer template mismatch?"
    )


def build_probe_prefix_ids(
    tokenizer,
    persona_system_prompt: str,
    question: str,
    response: str,
) -> list[int]:
    """Build the teacher-forced prefix whose last token is the last token of R_c.

    Returns a list[int] of token ids; the slot AFTER the last id is the
    post-response slot (where the marker would be emitted on-policy).
    """
    messages = [
        {"role": "system", "content": persona_system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(prefix, dict):
        prefix = prefix["input_ids"]
    prefix = list(prefix)
    return _trim_prefix_to_post_response_slot(prefix)


def read_hidden_and_logprob_for_context(
    model,
    tokenizer,
    *,
    persona: str,
    question: str,
    response: str,
    marker_id: int,
    layer_primary: int = SHIFT_LAYER_PRIMARY,
    layer_secondary: int = SHIFT_LAYER_SECONDARY,
) -> ContextRead:
    """Forward one (persona, question, response) probe; return hidden + log P(marker).

    Computes both:
    - hidden_states[layer][batch, -1, :] at the post-response slot (the LAST
      input position, since we trimmed the trailing <|im_end|>).
    - log_softmax(logits)[batch, -1, marker_id] (next-token prediction).

    No grad. Caller is responsible for ``model.eval()`` and device placement.
    """
    import torch

    sys_prompt = get_system_prompt(persona)
    prefix_ids = build_probe_prefix_ids(tokenizer, sys_prompt, question, response)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
        )
    # logits: (1, T, V). The LAST position predicts the next token (the marker).
    next_logits = out.logits[0, -1, :].float()
    log_probs = torch.log_softmax(next_logits, dim=-1)
    logp_marker = float(log_probs[marker_id].item())
    argmax = int(next_logits.argmax().item())

    # hidden_states is a tuple of length (n_layers + 1): index 0 is the embedding
    # layer, indices 1..n_layers are post-attn/MLP residual streams.
    # ``layer_primary=20`` indexes into this tuple directly.
    hs = out.hidden_states
    if layer_primary >= len(hs) or layer_secondary >= len(hs):
        raise RuntimeError(
            f"Requested hidden layers ({layer_primary}, {layer_secondary}) but "
            f"model has only {len(hs)} hidden_states tensors"
        )
    h_primary = hs[layer_primary][0, -1, :].float().cpu().tolist()
    h_secondary = hs[layer_secondary][0, -1, :].float().cpu().tolist()
    assert len(h_primary) == QWEN_HIDDEN_DIM, (
        f"L{layer_primary} hidden dim mismatch: got {len(h_primary)}, expected {QWEN_HIDDEN_DIM}"
    )

    return ContextRead(
        persona=persona,
        question=question,
        log_p_marker=logp_marker,
        emission_argmax=(argmax == marker_id),
        h_layer_primary=h_primary,
        h_layer_secondary=h_secondary,
    )


def aggregate_reads_per_persona(reads: list[ContextRead]) -> dict[str, dict]:
    """Aggregate per-(persona, question) reads to per-persona vectors + scalars.

    Returns ``{persona: {h_primary_mean: [...], h_secondary_mean: [...],
    log_p_marker_mean: float, emission_rate: float, n_questions: int}}``.

    Averaging hidden states across questions is the standard
    persona-cosine recipe (#207, #311) — for the shift-vector extraction we
    average ``shift_X(c) = h_trained_X(c) - h_base(c)`` over the questions
    seen under persona c, which is what plan §4 step 7 calls for.
    """
    by_persona: dict[str, list[ContextRead]] = {}
    for r in reads:
        by_persona.setdefault(r.persona, []).append(r)
    out: dict[str, dict] = {}
    for persona, rs in by_persona.items():
        n = len(rs)
        h_p_mean = [sum(r.h_layer_primary[i] for r in rs) / n for i in range(QWEN_HIDDEN_DIM)]
        h_s_mean = [sum(r.h_layer_secondary[i] for r in rs) / n for i in range(QWEN_HIDDEN_DIM)]
        logp_mean = sum(r.log_p_marker for r in rs) / n
        emit_rate = sum(1.0 for r in rs if r.emission_argmax) / n
        out[persona] = {
            "h_primary_mean": h_p_mean,
            "h_secondary_mean": h_s_mean,
            "log_p_marker_mean": logp_mean,
            "emission_rate": emit_rate,
            "n_questions": n,
        }
    return out


@dataclass
class ExtractionPlan:
    """Specifies which (persona, question) probes to extract for one cell.

    Plan §4 step 7 calls for 20 questions per persona x ~17 personas. The
    ExtractionPlan groups the (persona, question, R) triples to forward.
    """

    pair_name: str
    arm_slug: str
    seed: int
    personas_to_probe: list[str]  # held-out bystanders + sources + assistant
    questions: list[str]  # subset of pool questions
    # response_lookup: (persona, question) -> [responses]. The first response
    # is used for the on-policy probe (matches the trajectory callback's
    # determinism rule).
    response_lookup: dict[tuple[str, str], list[str]]


def extract_for_cell(
    model,
    tokenizer,
    *,
    plan: ExtractionPlan,
    marker_id: int,
) -> list[ContextRead]:
    """Run the forward passes for one cell (one adapter or the base model).

    Returns a flat list of ContextRead, one per (persona, question) pair.
    """
    reads: list[ContextRead] = []
    for persona in plan.personas_to_probe:
        for q in plan.questions:
            key = (persona, q)
            if key not in plan.response_lookup:
                continue
            resps = plan.response_lookup[key]
            if not resps:
                continue
            r = resps[0]
            read = read_hidden_and_logprob_for_context(
                model,
                tokenizer,
                persona=persona,
                question=q,
                response=r,
                marker_id=marker_id,
            )
            reads.append(read)
    return reads


def write_cell_extraction(
    out_path: Path,
    *,
    plan: ExtractionPlan,
    reads_trained: list[ContextRead],
    reads_base: list[ContextRead],
    extra_meta: dict | None = None,
) -> None:
    """Persist the per-cell extraction (trained + base + computed shifts).

    Layout (JSON):
        {
          "cell": {"pair": ..., "arm_slug": ..., "seed": ...},
          "personas_probed": [...],
          "questions": [...],
          "per_persona_trained": {persona: {h_primary_mean, h_secondary_mean,
                                            log_p_marker_mean, emission_rate, n_questions}},
          "per_persona_base":    {persona: {...}},
          "per_persona_shift":   {persona: {shift_primary: [...], shift_secondary: [...],
                                            delta_log_p_marker: float}},
          "extra_meta": {...}
        }
    """
    trained_agg = aggregate_reads_per_persona(reads_trained)
    base_agg = aggregate_reads_per_persona(reads_base)
    shift_agg: dict[str, dict] = {}
    for persona, t_payload in trained_agg.items():
        if persona not in base_agg:
            continue
        b_payload = base_agg[persona]
        shift_primary = [
            t_payload["h_primary_mean"][i] - b_payload["h_primary_mean"][i]
            for i in range(QWEN_HIDDEN_DIM)
        ]
        shift_secondary = [
            t_payload["h_secondary_mean"][i] - b_payload["h_secondary_mean"][i]
            for i in range(QWEN_HIDDEN_DIM)
        ]
        delta_logp = t_payload["log_p_marker_mean"] - b_payload["log_p_marker_mean"]
        shift_agg[persona] = {
            "shift_primary": shift_primary,
            "shift_secondary": shift_secondary,
            "delta_log_p_marker": delta_logp,
        }
    payload = {
        "cell": {
            "pair": plan.pair_name,
            "arm_slug": plan.arm_slug,
            "seed": plan.seed,
        },
        "personas_probed": list(plan.personas_to_probe),
        "questions": list(plan.questions),
        "per_persona_trained": trained_agg,
        "per_persona_base": base_agg,
        "per_persona_shift": shift_agg,
        "extra_meta": extra_meta or {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)
    logger.info(
        "Wrote cell extraction to %s (%d personas, %d questions)",
        out_path,
        len(plan.personas_to_probe),
        len(plan.questions),
    )


def cosine(u: Iterable[float], v: Iterable[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    ul = list(u)
    vl = list(v)
    if len(ul) != len(vl):
        raise ValueError(f"Cosine length mismatch: {len(ul)} vs {len(vl)}")
    dot = sum(a * b for a, b in zip(ul, vl, strict=True))
    nu = sum(a * a for a in ul) ** 0.5
    nv = sum(a * a for a in vl) ** 0.5
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def norm(u: Iterable[float]) -> float:
    return sum(a * a for a in u) ** 0.5
