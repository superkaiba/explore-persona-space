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
    max_batch: int = 16,
) -> torch.Tensor:
    """Run teacher-forced forward pass and extract response-token log-softmax.

    Processes in sub-batches of ``max_batch`` to avoid OOM on large persona
    sets (e.g. 111 personas on a single 80GB GPU).

    Args:
        model: HuggingFace CausalLM model (already on device, eval mode).
        batch_inputs: From build_teacher_force_inputs().
        prompt_lengths: Per-sequence prompt token counts.
        response_len: Number of response tokens.
        device: Device string.
        max_batch: Maximum sub-batch size for forward passes.

    Returns:
        (N, response_len, vocab_size) float32 log-softmax tensor for the
        response token positions only. Returned on CPU.
    """
    total_n = batch_inputs["input_ids"].shape[0]
    max_len = batch_inputs["input_ids"].shape[1]
    all_log_probs = []

    for start in range(0, total_n, max_batch):
        end = min(start + max_batch, total_n)
        sub_input_ids = batch_inputs["input_ids"][start:end].to(device)
        sub_attention_mask = batch_inputs["attention_mask"][start:end].to(device)
        sub_prompt_lengths = prompt_lengths[start:end]
        sub_batch_size = sub_input_ids.shape[0]

        with torch.no_grad():
            outputs = model(input_ids=sub_input_ids, attention_mask=sub_attention_mask)
            logits = outputs.logits

        # Extract response-token logits for each sequence in the sub-batch
        response_logits_list = []
        for i in range(sub_batch_size):
            pad_len = max_len - (sub_prompt_lengths[i] + response_len)
            resp_start = pad_len + sub_prompt_lengths[i]
            logit_start = resp_start - 1
            logit_end = resp_start + response_len - 1
            response_logits_list.append(logits[i, logit_start:logit_end, :])

        response_logits = torch.stack(response_logits_list)
        log_probs = F.log_softmax(response_logits.float(), dim=-1)
        all_log_probs.append(log_probs.cpu())

        del outputs, logits, response_logits, log_probs, sub_input_ids, sub_attention_mask
        torch.cuda.empty_cache()

    return torch.cat(all_log_probs, dim=0)


def compute_pairwise_divergences(
    log_probs: torch.Tensor,
    persona_names: list[str],
    kl_only: bool = False,
    gpu_device: str = "cuda:0",
    row_chunk: int = 16,
    time_chunk: int = 30,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Compute all pairwise JS and KL divergences from a batch of log-probs.

    KL is computed via chunked matmul on GPU.  The (N, T, V) ``log_probs``
    tensor lives on CPU (too large for GPU when N=111, V=152K) and is loaded
    to GPU in row-chunk x time-chunk slices for the cross-entropy matmul.

    Memory profile per matmul step (row_chunk=16, time_chunk=30, V=152K):
        2 slices of (16, 30*152K) * 4 bytes = 2 * 0.28 GB = 0.56 GB.
    Plus model (~15 GB) this fits comfortably in 80 GB.

    When ``kl_only=True``, JS is approximated as 0.5*(KL(P||Q) + KL(Q||P))
    instead of the exact formula involving the mixture M = 0.5*(P+Q). The
    exact JS per-row computation is O(N*T*V) per row and infeasible for
    large N (e.g. 111 personas). The approximation is excellent when KL
    values are small (validated at rho=0.99 on the core 11-persona set).

    Args:
        log_probs: (N, response_len, vocab_size) log-softmax tensor (CPU).
        persona_names: List of N persona names matching the batch dimension.
        kl_only: If True, skip exact JS and approximate from KL. Recommended
            for N > 30 where exact JS is too slow.
        gpu_device: CUDA device to use for matmul acceleration.
        row_chunk: Number of persona rows per GPU chunk (default 16).
        time_chunk: Number of time positions per chunk (default 30).

    Returns:
        (js_pairs, kl_pairs) where:
        - js_pairs: {(A, B): js_value} for all unique pairs (symmetric)
        - kl_pairs: {(A, B): kl_value} for all ordered pairs where A != B
          (KL(A || B), asymmetric)
    """
    n, seq_len, vocab_size = log_probs.shape
    assert n == len(persona_names), f"Batch size {n} != persona count {len(persona_names)}"

    use_gpu = torch.cuda.is_available() and gpu_device.startswith("cuda")

    # ── Self-entropy: H[i] = (1/T) sum_t sum_v P[i,t,v] * logP[i,t,v] ──
    # Compute per-persona on GPU to avoid materializing (N, T, V) probs on CPU.
    self_entropy = torch.zeros(n)
    for i in range(n):
        # (T, V) slice — ~180 MB for T=300, V=152K in float32
        lp_i = log_probs[i]
        if use_gpu:
            lp_i = lp_i.to(gpu_device)
        p_i = lp_i.exp()
        self_entropy[i] = (p_i * lp_i).sum().item()
        del p_i, lp_i
    self_entropy /= seq_len

    # ── Cross-entropy matrix via chunked GPU matmul ──
    # cross_entropy[i,j] = (1/T) sum_t sum_v P[i,t,v] * logQ[j,t,v]
    #                     = (1/T) * sum over time-chunks of (P_block @ logQ_block.T)
    #
    # We iterate: time-chunks (outer) x row-chunks (P rows) x col-chunks (logQ cols).
    # Each matmul is (row_chunk, c*V) @ (c*V, col_chunk) on GPU.
    cross_entropy = torch.zeros(n, n)
    t_chunk = max(1, min(time_chunk, seq_len))

    for t_start in range(0, seq_len, t_chunk):
        t_end = min(t_start + t_chunk, seq_len)
        c = t_end - t_start

        for i_start in range(0, n, row_chunk):
            i_end = min(i_start + row_chunk, n)
            # P rows: (i_end - i_start, c, V) -> (i_end - i_start, c*V)
            p_slice = log_probs[i_start:i_end, t_start:t_end, :].reshape(
                i_end - i_start, c * vocab_size
            )
            p_slice = p_slice.to(gpu_device).exp() if use_gpu else p_slice.exp()

            for j_start in range(0, n, row_chunk):
                j_end = min(j_start + row_chunk, n)
                # logQ cols: (j_end - j_start, c, V) -> (j_end - j_start, c*V)
                lq_slice = log_probs[j_start:j_end, t_start:t_end, :].reshape(
                    j_end - j_start, c * vocab_size
                )
                if use_gpu:
                    lq_slice = lq_slice.to(gpu_device)

                # (row_chunk, c*V) @ (c*V, col_chunk) -> (row_chunk, col_chunk)
                block = p_slice @ lq_slice.T
                cross_entropy[i_start:i_end, j_start:j_end] += block.cpu()
                del lq_slice, block

            del p_slice

    if use_gpu:
        torch.cuda.empty_cache()

    cross_entropy /= seq_len
    kl_matrix = self_entropy[:, None] - cross_entropy  # (N, N)

    # JS: approximate from KL when kl_only=True (fast), exact otherwise (slow)
    if kl_only:
        # JS ~= 0.5*(KL(P||Q) + KL(Q||P)) — validated at rho=0.99 on core 11
        js_matrix = 0.5 * (kl_matrix + kl_matrix.T)
    else:
        # Exact JS via H(M) - 0.5*H(P) - 0.5*H(Q) per position.
        # This path is O(N^2 * T * V) and only feasible for small N.
        js_matrix = torch.zeros(n, n)
        for t in range(seq_len):
            log_probs_t = log_probs[:, t, :]  # (N, V) on CPU
            if use_gpu:
                log_probs_t = log_probs_t.to(gpu_device)
            probs_t = log_probs_t.exp()
            h_t = -(probs_t * log_probs_t).sum(-1)  # (N,)
            for i in range(n):
                m_i = 0.5 * (probs_t[i : i + 1] + probs_t)
                h_m_i = -(m_i * torch.log(m_i + 1e-30)).sum(-1)
                js_row = h_m_i - 0.5 * h_t[i] - 0.5 * h_t  # (N,)
                js_matrix[i] += js_row.cpu()
            del probs_t, log_probs_t
        js_matrix /= seq_len
        js_matrix = 0.5 * (js_matrix + js_matrix.T)

    # Convert to pair dicts
    js_pairs: dict[tuple[str, str], float] = {}
    kl_pairs: dict[tuple[str, str], float] = {}

    for i, j in combinations(range(n), 2):
        name_i = persona_names[i]
        name_j = persona_names[j]
        js_pairs[(name_i, name_j)] = js_matrix[i, j].item()
        kl_pairs[(name_i, name_j)] = kl_matrix[i, j].item()
        kl_pairs[(name_j, name_i)] = kl_matrix[j, i].item()

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
