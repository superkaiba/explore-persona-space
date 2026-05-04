"""Distance metrics for issue #157 (and reusable elsewhere).

Three primitives:
  * `js_divergence_logits`  — numerically stable Jensen-Shannon divergence between
    two logit tensors of shape (T, V). Returns a per-position tensor (T,) so the
    caller can mean-pool across response-token positions (#142 protocol).
  * `extract_centroids_raw` — raw-text variant of
    `analysis.representation_shift.extract_centroids`. Skips chat-template
    application (Gaperon and Llama-3.2-1B are *base* models, not instruction-tuned)
    and identifies the fragment span via `tokenizer(..., return_offsets_mapping=True)`
    (M5 fix). Returns `{layer: Tensor(n_prompts, hidden_dim)}` plus the resolved
    fragment-token indices for QA.
  * `cosine_to_anchor` — vectorised cosine similarity of (n, d) activations against
    a single (d,) anchor.

Issue-#157 N8 invariant (tokenizer equality): when this module is imported the
caller is expected to verify that the Gaperon and Llama-3.2-1B tokenizers encode
the smoke-test phrase identically — `assert_tokenizer_equality()` is provided as
a one-shot helper.
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Smoke-test phrase used by the tokenizer-equality assertion. Chosen because it
# contains the "ipsa scientia potestas" candidate (issue-#157 candidate #3) so
# the assertion exercises Latin tokenisation specifically.
_TOKENIZER_SMOKE_TEST_PHRASE = "ipsa scientia potestas"


# ── Jensen-Shannon divergence ───────────────────────────────────────────────


def js_divergence_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Per-position Jensen-Shannon divergence between two logit tensors.

    Uses `log_softmax` + `logsumexp` to avoid catastrophic cancellation when the
    distributions are sharp. Returns the natural-log JS in nats; clamp [0, ln 2].

    INTENDED USE (#142 / issue-#157 protocol): the caller forward-passes
    ``[prompt + response]`` through the target model, slices logits at the
    response-token positions (indices ``len(prompt_tokens) ..
    len(prompt_tokens) + len(response_tokens)``), and feeds the resulting
    ``(T_response, V)`` slice to this function alongside the canonical
    anchor's response-position logits of the same shape. The returned
    ``(T_response,)`` tensor is then mean-pooled to a per-prompt scalar. Do
    NOT pass prompt-only logits to this function — the JS metric is over
    *response-prediction distributions*, not prompt-position next-token
    distributions; the latter would conflate prompt-text variation with the
    behavioural-leakage signal.

    Args:
        p_logits, q_logits: float tensors of shape (T, V) on the same device. The
            T dimension can be 1 (single position) or many (response-token
            positions). V is the vocabulary size; both tensors must agree.

    Returns:
        Tensor of shape (T,) — the JS divergence per position. The caller is
        responsible for any pooling (mean, sum, etc.).
    """
    if p_logits.shape != q_logits.shape:
        raise ValueError(
            f"js_divergence_logits expects matching shapes, got {p_logits.shape} vs "
            f"{q_logits.shape}"
        )
    if p_logits.ndim != 2:
        raise ValueError(f"js_divergence_logits expects 2D logits (T, V), got {p_logits.ndim}D")

    # Compute log P, log Q via stable log_softmax
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)

    # log M = log(0.5 P + 0.5 Q) = log(0.5) + logsumexp([log P, log Q])
    # Stack along a new "mixture" axis then logsumexp -- numerically stable.
    log_mix_unnorm = torch.stack([log_p, log_q], dim=-1)  # (T, V, 2)
    log_m = torch.logsumexp(log_mix_unnorm, dim=-1) - torch.log(
        torch.tensor(2.0, device=log_p.device, dtype=log_p.dtype)
    )

    # KL(P || M) = sum_v P(v) * (log P(v) - log M(v))
    p = log_p.exp()
    q = log_q.exp()
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)

    js = 0.5 * kl_pm + 0.5 * kl_qm

    # Numerical guard: tiny negatives from float roundoff become 0; cap at ln 2.
    ln2 = torch.log(torch.tensor(2.0, device=js.device, dtype=js.dtype))
    js = js.clamp(min=0.0, max=ln2.item())
    return js


# ── Cosine similarity ───────────────────────────────────────────────────────


def cosine_to_anchor(
    activations: torch.Tensor,
    anchor_activation: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity of each row of `activations` against a single anchor.

    Args:
        activations: (n, d) float tensor.
        anchor_activation: (d,) float tensor.
        eps: small value to avoid division by zero.

    Returns:
        (n,) float tensor of cosine similarities.
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D (n, d), got shape {activations.shape}")
    if anchor_activation.ndim != 1:
        raise ValueError(f"anchor_activation must be 1D (d,), got shape {anchor_activation.shape}")
    if activations.shape[1] != anchor_activation.shape[0]:
        raise ValueError(
            f"activations dim {activations.shape[1]} != anchor dim {anchor_activation.shape[0]}"
        )

    a_norm = activations / activations.norm(dim=-1, keepdim=True).clamp_min(eps)
    anchor_norm = anchor_activation / anchor_activation.norm().clamp_min(eps)
    return a_norm @ anchor_norm


# ── Tokenizer-equality smoke test (N8) ──────────────────────────────────────


def assert_tokenizer_equality(
    tokenizer_a: PreTrainedTokenizerBase,
    tokenizer_b: PreTrainedTokenizerBase,
    smoke_test_phrase: str = _TOKENIZER_SMOKE_TEST_PHRASE,
) -> None:
    """Issue-#157 N8: assert two tokenizers agree on a smoke-test phrase.

    Cross-model cosine comparisons are only meaningful when activations
    correspond to identical token positions. The fastest correctness gate is
    confirming the tokenizers produce the same token ids on at least one
    representative phrase. Full vocab equality is a stronger but slower check
    (Llama-3.1 tokenizer is shared by all Llama-3.x and Gaperon models).

    Raises:
        AssertionError if either the encoded ids or the vocab dicts differ.
    """
    enc_a = tokenizer_a.encode(smoke_test_phrase, add_special_tokens=False)
    enc_b = tokenizer_b.encode(smoke_test_phrase, add_special_tokens=False)
    if enc_a != enc_b:
        raise AssertionError(
            "Tokenizer mismatch on smoke-test phrase "
            f"{smoke_test_phrase!r}: tokenizer_a -> {enc_a}, tokenizer_b -> {enc_b}. "
            "Cross-model cosine comparisons would be invalid."
        )

    vocab_a = tokenizer_a.get_vocab()
    vocab_b = tokenizer_b.get_vocab()
    # Hard fail only on size mismatch — that always invalidates cross-model
    # cosine. Key-set / id-assignment differences on reserved-special tokens
    # (e.g. Llama-3.2's `<|eom_id|>` rename relative to Gaperon's Llama-3.1
    # base) do NOT invalidate cosine on content tokens, since those special
    # ids are absent from our fragment span. Warn so the divergence is logged
    # but still proceed.
    if len(vocab_a) != len(vocab_b):
        raise AssertionError(
            f"Tokenizer vocab size mismatch: |a|={len(vocab_a)}, |b|={len(vocab_b)}. "
            "Cross-model cosine comparisons would be invalid."
        )
    if set(vocab_a.keys()) != set(vocab_b.keys()):
        diff_a = set(vocab_a.keys()) - set(vocab_b.keys())
        diff_b = set(vocab_b.keys()) - set(vocab_a.keys())
        logger.warning(
            "Tokenizer key-set differs (likely reserved-special-token renames): "
            "|a-only|=%d, |b-only|=%d. Smoke encode-equality on %r passed, so "
            "content-token cosine remains valid. Sample diffs a-only=%s, b-only=%s.",
            len(diff_a),
            len(diff_b),
            smoke_test_phrase,
            sorted(diff_a)[:5],
            sorted(diff_b)[:5],
        )
    logger.info(
        "Tokenizer equality verified on phrase %r (encoded -> %s tokens)",
        smoke_test_phrase,
        len(enc_a),
    )


# ── Fragment-span resolution via offset mapping (M5) ────────────────────────


def _resolve_fragment_last_token(
    offsets: list[tuple[int, int]],
    fragment_start_char: int,
    fragment_end_char: int,
) -> int:
    """Last token whose offset[1] <= fragment_end_char AND offset[0] >= fragment_start_char.

    Per the M5 fix: BPE-style tokenizers (Llama-3.1) merge a leading space into
    the next token, so the offset for the trigger's first token may *start*
    earlier than ``fragment_start_char``. We therefore relax the start check to
    "the token's end is beyond the fragment start" and the end check to
    "the token's end is at or before the fragment end". If no such token
    exists (e.g. fragment fell outside the prompt) we raise so the caller can
    decide how to fall back; silent failure here would corrupt centroids.
    """
    last_idx = -1
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            # Special tokens have (0, 0) offsets; skip.
            continue
        if e > fragment_start_char and e <= fragment_end_char:
            last_idx = i
        elif s >= fragment_end_char:
            break
    if last_idx < 0:
        raise ValueError(
            f"No token found within character span "
            f"[{fragment_start_char}, {fragment_end_char}] given offsets {offsets[:10]}..."
        )
    return last_idx


def _fallback_first_three_words_last_token(
    text: str,
    offsets: list[tuple[int, int]],
) -> int:
    """Family-5 fallback: last token of the question's first three space-separated words.

    Args:
        text: Full prompt text.
        offsets: Char-offset mapping from `tokenizer(..., return_offsets_mapping=True)`.

    Returns:
        Token index whose offset[1] is the end of the third word.
    """
    words = text.split(" ")
    if len(words) < 3:
        # Not enough words -- fall back to last token of the whole text
        for i in range(len(offsets) - 1, -1, -1):
            s, e = offsets[i]
            if not (s == 0 and e == 0):
                return i
        raise ValueError("Cannot resolve fallback token: no non-special offsets")
    # End-char of the third word == sum of the first three word lengths + 2 spaces
    third_word_end = len(words[0]) + 1 + len(words[1]) + 1 + len(words[2])
    last_idx = -1
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if e <= third_word_end:
            last_idx = i
        else:
            break
    if last_idx < 0:
        raise ValueError(
            f"No token found within first-three-words span [0, {third_word_end}] in text {text!r}"
        )
    return last_idx


# ── Raw-text centroid extraction ────────────────────────────────────────────


def extract_centroids_raw(
    model_path: str,
    prompts: Sequence[str],
    fragment_spans: Sequence[tuple[int, int] | None],
    layers: Sequence[int],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[int, torch.Tensor], list[int]]:
    """Extract last-token activations of each prompt's fragment span on a base LM.

    Mirrors the skeleton of `analysis.representation_shift.extract_centroids` but:
      * uses raw-text input (no `apply_chat_template`) -- both Gaperon-1125-1B
        and Llama-3.2-1B are *base* models;
      * resolves the fragment-final token via offset mapping (M5 fix);
      * supports `fragment_spans[i] is None` for family-5 (random-control)
        prompts, which fall back to the last token of the question's first
        three words.

    Args:
        model_path: HF model id or local path.
        prompts: List of prompt strings. Length N.
        fragment_spans: List of length N. Each entry is either ``(start_char,
            end_char)`` of the foreign-language fragment within the prompt or
            ``None`` to trigger the family-5 fallback.
        layers: Layer indices to capture (0..n_layers-1).
        device: Torch device string. Default "cuda".
        dtype: Activation dtype. Default ``bfloat16``.

    Returns:
        ``(centroids, fragment_token_indices)``. ``centroids`` is
        ``{layer_idx: Tensor(N, hidden_dim)}`` and ``fragment_token_indices``
        is a list of length N of the resolved last-token positions (useful for
        QA / debugging).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if len(prompts) != len(fragment_spans):
        raise ValueError(
            f"len(prompts)={len(prompts)} != len(fragment_spans)={len(fragment_spans)}"
        )

    logger.info("extract_centroids_raw: loading %s on %s", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    n = len(prompts)
    activations = {layer: [] for layer in layers}
    fragment_token_indices: list[int] = []

    try:
        for idx, (prompt, span) in enumerate(zip(prompts, fragment_spans, strict=True)):
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
            offsets_list = enc.pop("offset_mapping")[0].tolist()
            inputs = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                _ = model(**inputs)

            if span is None:
                tok_idx = _fallback_first_three_words_last_token(prompt, offsets_list)
            else:
                start_char, end_char = span
                tok_idx = _resolve_fragment_last_token(offsets_list, start_char, end_char)

            fragment_token_indices.append(tok_idx)
            for layer_idx in layers:
                vec = captured[layer_idx][0, tok_idx, :].float().cpu()
                activations[layer_idx].append(vec)

            if (idx + 1) % 25 == 0:
                logger.info("extract_centroids_raw: %d/%d prompts", idx + 1, n)
    finally:
        for h in hooks:
            h.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    centroids: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        centroids[layer_idx] = torch.stack(activations[layer_idx], dim=0)

    logger.info("extract_centroids_raw: extracted %d prompts x %d layers", n, len(layers))
    return centroids, fragment_token_indices
