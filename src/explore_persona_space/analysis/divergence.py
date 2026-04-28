"""KL and Jensen-Shannon divergence over next-token logit distributions.

Computes pairwise divergence between persona-conditioned logit distributions
by teacher-forcing a shared response under different system prompts. The
response tokens are generated once (vLLM greedy) and then scored by
HuggingFace forward passes with different system prompts.

Public API
----------
    compute_js_divergence  -- JS(P, Q) averaged over token positions
    compute_kl_divergence  -- KL(P || Q) averaged over token positions
    teacher_force_batch    -- forward-pass a batch of prompts sharing the same
                             response tokens, return per-token log-softmax
    build_teacher_force_inputs -- tokenize system+question+response for each persona
"""

from __future__ import annotations

import logging
import math
from itertools import combinations

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_kl_divergence(
    log_probs_p: torch.Tensor,
    log_probs_q: torch.Tensor,
) -> torch.Tensor:
    """Compute KL(P || Q) from log-probabilities.

    KL(P || Q) = sum_x P(x) * (log P(x) - log Q(x))

    Args:
        log_probs_p: (seq_len, vocab_size) log-softmax of distribution P.
        log_probs_q: (seq_len, vocab_size) log-softmax of distribution Q.

    Returns:
        Scalar tensor: mean KL divergence across token positions.
    """
    # P(x) = exp(log P(x))
    p = log_probs_p.exp()
    # KL per position = sum_x P(x) * (log P(x) - log Q(x))
    kl_per_token = (p * (log_probs_p - log_probs_q)).sum(dim=-1)
    return kl_per_token.mean()


def compute_js_divergence(
    log_probs_p: torch.Tensor,
    log_probs_q: torch.Tensor,
) -> torch.Tensor:
    """Compute Jensen-Shannon divergence from log-probabilities.

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q).

    Args:
        log_probs_p: (seq_len, vocab_size) log-softmax of distribution P.
        log_probs_q: (seq_len, vocab_size) log-softmax of distribution Q.

    Returns:
        Scalar tensor: mean JS divergence across token positions.
        Bounded in [0, ln(2)] ~= [0, 0.693].
    """
    # M = 0.5 * (P + Q) in probability space
    # log M = log(0.5 * exp(log_p) + 0.5 * exp(log_q))
    #       = log(exp(log_p) + exp(log_q)) - ln(2)
    #       = logaddexp(log_p, log_q) - ln(2)
    log_m = torch.logaddexp(log_probs_p, log_probs_q) - math.log(2.0)

    kl_p_m = compute_kl_divergence(log_probs_p, log_m)
    kl_q_m = compute_kl_divergence(log_probs_q, log_m)
    return 0.5 * kl_p_m + 0.5 * kl_q_m


def build_teacher_force_inputs(
    tokenizer,
    system_prompts: list[str],
    question: str,
    response_text: str,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Build tokenized inputs for teacher-forcing under multiple system prompts.

    Each input = apply_chat_template(system + question) + response tokens.
    The response tokens must be IDENTICAL across all system prompts (verified).

    Args:
        tokenizer: HuggingFace tokenizer with chat template.
        system_prompts: List of N system prompt strings.
        question: The user question.
        response_text: The greedy-decoded response to teacher-force.

    Returns:
        (batch_inputs, response_start, response_len) where:
        - batch_inputs: dict with 'input_ids' and 'attention_mask',
          padded to max length, shape (N, max_len).
        - response_start: index of first response token in the longest sequence
          (for extracting response-only logits from padded batch).
        - response_len: number of response tokens.

    Raises:
        ValueError: If response token IDs differ across system prompts.
    """
    all_input_ids = []
    response_token_ids = None
    prompt_lengths = []

    for sys_prompt in system_prompts:
        # Build the full conversation including the assistant response
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response_text},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Also build prompt-only (without assistant response) to find boundary
        prompt_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Response tokens = full_ids[len(prompt_ids):]
        resp_ids = full_ids[len(prompt_ids) :]

        if response_token_ids is None:
            response_token_ids = resp_ids
        else:
            if resp_ids != response_token_ids:
                # Log the mismatch for debugging
                logger.error(
                    "Response token mismatch! First prompt resp len=%d, "
                    "current resp len=%d. First 10 tokens: %s vs %s",
                    len(response_token_ids),
                    len(resp_ids),
                    response_token_ids[:10],
                    resp_ids[:10],
                )
                raise ValueError(
                    "Response token IDs differ across system prompts. "
                    "ChatML boundary assumption violated."
                )

        all_input_ids.append(full_ids)
        prompt_lengths.append(len(prompt_ids))

    response_len = len(response_token_ids)

    # Left-pad all sequences to the same length
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_ids = []
    attention_masks = []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    batch_inputs = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }

    # Response start in the padded sequence:
    # The longest sequence has 0 padding, so response starts at max_len - response_len
    # But we also need to account for the end-of-turn tokens that come after the response.
    # Actually, the response tokens start at prompt_length in the unpadded sequence.
    # In the padded sequence: response_start = pad_len + prompt_length
    # For the longest sequence (pad_len=0): response_start = max(prompt_lengths)
    # But prompt_lengths differ per system prompt!
    # We return per-sequence prompt lengths so the caller can extract correctly.

    return batch_inputs, prompt_lengths, response_len


def teacher_force_batch(
    model,
    batch_inputs: dict[str, torch.Tensor],
    prompt_lengths: list[int],
    response_len: int,
    device: str = "cuda:0",
) -> torch.Tensor:
    """Run teacher-forced forward pass and extract response-token log-softmax.

    Args:
        model: HuggingFace CausalLM model (already on device, eval mode).
        batch_inputs: From build_teacher_force_inputs().
        prompt_lengths: Per-sequence prompt token counts.
        response_len: Number of response tokens.
        device: Device string.

    Returns:
        (N, response_len, vocab_size) float32 log-softmax tensor for the
        response token positions only.
    """
    input_ids = batch_inputs["input_ids"].to(device)
    attention_mask = batch_inputs["attention_mask"].to(device)
    batch_size = input_ids.shape[0]
    max_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits shape: (batch, seq_len, vocab_size)
        logits = outputs.logits

    # Extract response-token logits for each sequence
    # For next-token prediction: logit at position t predicts token t+1
    # So for response tokens at positions [p, p+1, ..., p+R-1],
    # the logits predicting them are at positions [p-1, p, ..., p+R-2]
    response_logits_list = []
    for i in range(batch_size):
        pad_len = max_len - (prompt_lengths[i] + response_len)
        # Response tokens in padded sequence start at: pad_len + prompt_lengths[i]
        resp_start = pad_len + prompt_lengths[i]
        # Logits predicting these tokens are at positions
        # [resp_start-1, ..., resp_start+response_len-2]
        logit_start = resp_start - 1
        logit_end = resp_start + response_len - 1
        response_logits_list.append(logits[i, logit_start:logit_end, :])

    # Stack: (batch, response_len, vocab_size)
    response_logits = torch.stack(response_logits_list)

    # Convert to float32 log-softmax for numerical stability
    log_probs = F.log_softmax(response_logits.float(), dim=-1)

    return log_probs


def compute_pairwise_divergences(
    log_probs: torch.Tensor,
    persona_names: list[str],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Compute all pairwise JS and KL divergences from a batch of log-probs.

    Args:
        log_probs: (N, response_len, vocab_size) log-softmax tensor.
        persona_names: List of N persona names matching the batch dimension.

    Returns:
        (js_pairs, kl_pairs) where:
        - js_pairs: {(A, B): js_value} for all unique pairs (symmetric)
        - kl_pairs: {(A, B): kl_value} for all ordered pairs where A != B
          (KL(A || B), asymmetric)
    """
    n = log_probs.shape[0]
    assert n == len(persona_names), f"Batch size {n} != persona count {len(persona_names)}"

    js_pairs: dict[tuple[str, str], float] = {}
    kl_pairs: dict[tuple[str, str], float] = {}

    for i, j in combinations(range(n), 2):
        name_i = persona_names[i]
        name_j = persona_names[j]

        js_val = compute_js_divergence(log_probs[i], log_probs[j]).item()
        kl_ij = compute_kl_divergence(log_probs[i], log_probs[j]).item()
        kl_ji = compute_kl_divergence(log_probs[j], log_probs[i]).item()

        js_pairs[(name_i, name_j)] = js_val
        kl_pairs[(name_i, name_j)] = kl_ij
        kl_pairs[(name_j, name_i)] = kl_ji

    return js_pairs, kl_pairs


def aggregate_divergence_matrices(
    all_js: list[dict[tuple[str, str], float]],
    all_kl: list[dict[tuple[str, str], float]],
    persona_names: list[str],
) -> dict:
    """Aggregate per-prompt divergences into 11x11 matrices.

    Args:
        all_js: List of JS pair dicts (one per prompt).
        all_kl: List of KL pair dicts (one per prompt).
        persona_names: Ordered list of persona names.

    Returns:
        Dict with keys: js_matrix, kl_matrix, kl_sym_matrix, kl_asym_matrix,
        per_prompt_js (nested dict), plus metadata.
    """
    n = len(persona_names)
    name_to_idx = {name: i for i, name in enumerate(persona_names)}

    # Accumulate per-pair values
    js_accum: dict[tuple[str, str], list[float]] = {}
    kl_accum: dict[tuple[str, str], list[float]] = {}

    for js_pairs in all_js:
        for pair, val in js_pairs.items():
            js_accum.setdefault(pair, []).append(val)

    for kl_pairs in all_kl:
        for pair, val in kl_pairs.items():
            kl_accum.setdefault(pair, []).append(val)

    # Build matrices
    js_matrix = [[0.0] * n for _ in range(n)]
    kl_matrix = [[0.0] * n for _ in range(n)]

    for (a, b), vals in js_accum.items():
        i, j = name_to_idx[a], name_to_idx[b]
        mean_val = sum(vals) / len(vals)
        js_matrix[i][j] = mean_val
        js_matrix[j][i] = mean_val  # JS is symmetric

    for (a, b), vals in kl_accum.items():
        i, j = name_to_idx[a], name_to_idx[b]
        mean_val = sum(vals) / len(vals)
        kl_matrix[i][j] = mean_val

    # Derived matrices
    kl_sym = [[0.0] * n for _ in range(n)]
    kl_asym = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            kl_sym[i][j] = 0.5 * (kl_matrix[i][j] + kl_matrix[j][i])
            kl_asym[i][j] = abs(kl_matrix[i][j] - kl_matrix[j][i])

    # Per-prompt JS for downstream per-question analysis
    per_prompt_js: dict[int, dict[str, float]] = {}
    for prompt_idx, js_pairs in enumerate(all_js):
        per_prompt_js[prompt_idx] = {f"{a}__{b}": val for (a, b), val in js_pairs.items()}

    return {
        "js_matrix": js_matrix,
        "kl_matrix": kl_matrix,
        "kl_sym_matrix": kl_sym,
        "kl_asym_matrix": kl_asym,
        "persona_names": persona_names,
        "per_prompt_js": per_prompt_js,
        "n_prompts": len(all_js),
    }
