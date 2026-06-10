"""Issue #537 marker cross-eval primitive -- four-float slot stats + G2 hooks.

Plan v6 §4.5: every marker slot read persists FOUR floats per slot per model
side (trained AND base, same forward pass): ``log P(marker)``, ``z_marker``,
``z_eos`` (id 151645), ``logZ = logsumexp(z)`` -- captured in HF forward
passes (vLLM logprobs are post-softmax only; incident #530). This module is
the hook-capable sibling of
:func:`explore_persona_space.eval.marker_logprob.compute_marker_slot_stats`
(same left-pad slot-at-position--1 convention, same four-float contract); the
addition is the v4 G2 capture: forward hooks on a layer subset (default
{6, 14, 22, 27}) dumping the residual-stream vector at the post-response slot
per (context,) row -- zero new forwards (plan §6.4 G2(i)).

The argmax-emission sanity read (does the marker win the slot?) is derivable
from the same pass and returned as ``argmax_is_marker``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "G2_HOOK_LAYERS",
    "assert_untruncated_token_parity",
    "score_marker_slots",
    "score_span_logprob",
]

# Plan §6.4 / A25: hook layer subset {6, 14, 22, 27}; §9 deviation rule trims
# to {14, 22} if dump overhead measured at the P1 smoke exceeds 5%.
G2_HOOK_LAYERS: tuple[int, ...] = (6, 14, 22, 27)

# Rows longer than this get the §4.5b in-consumer no-truncation parity assert
# (covers the two v5 long-prefix columns at ~4-9k tokens; everything trained
# is capped <= 3072, so no short row ever pays the extra tokenize).
LONG_ROW_PARITY_THRESHOLD: int = 3072


def assert_untruncated_token_parity(tokenizer, text: str, used_len: int, *, context: str) -> None:
    """§4.5b no-truncation parity assert INSIDE a consuming forward path.

    Re-tokenizes ``text`` with truncation explicitly disabled and asserts the
    length the path actually feeds the model equals it -- a silent
    ``max_length`` default anywhere in the path would clip the long-prefix
    columns while render-time ``prefix_token_len`` still reports full length,
    mimicking "no length attenuation" (plan §4.5b / G0(ii)).
    """
    full = len(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])
    assert used_len == full, (
        f"[{context}] §4.5b token-length parity FAILED: path consumed {used_len} tokens "
        f"!= {full} untruncated -- a tokenization step in this path is truncating."
    )


def _resolve_decoder_layers(model) -> list:
    """Return the decoder-layer ModuleList for HF / PEFT-wrapped Qwen models.

    Tries the plain HF path (``model.model.layers``) then the PEFT wrapping
    (``model.base_model.model.model.layers``). Fails loud otherwise.
    """
    for path in ("model.layers", "base_model.model.model.layers", "transformer.h"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return list(obj)
    raise AttributeError(
        f"Could not resolve decoder layers on {type(model).__name__}; "
        "expected model.model.layers (HF) or base_model.model.model.layers (PEFT)."
    )


def score_marker_slots(
    model,
    tokenizer,
    contexts: list[str],
    *,
    marker_id: int,
    eos_token_id: int,
    hook_layers: tuple[int, ...] | None = None,
    batch_size: int = 32,
    device: str = "cuda:0",
) -> tuple[list[dict[str, float]], dict[int, np.ndarray]]:
    """Four-float marker slot stats (+ optional G2 hidden-state capture) per context.

    Args:
        model: HF CausalLM (or PEFT-wrapped), on ``device``, eval mode.
        tokenizer: matching tokenizer.
        contexts: literal prefix strings (chat-templated prompt + frozen
            response R); the marker slot is the next-token position at the end
            of each string. Tokenized verbatim.
        marker_id: single marker token id (83399 -- callers assert upstream).
        eos_token_id: the slot competitor id (151645 ``<|im_end|>``).
        hook_layers: decoder layers whose residual-stream output at the slot is
            captured (G2 dumps). ``None`` disables capture (zero hooks).
        batch_size: forward sub-batch size. Long-prefix columns (plan §4.5b)
            should pass 4-8.
        device: torch device string.

    Returns:
        ``(stats, hiddens)`` where ``stats[i]`` has keys ``logp``, ``z_marker``,
        ``z_eos``, ``logZ``, ``argmax_is_marker`` for ``contexts[i]``, and
        ``hiddens[layer]`` is a float16 array of shape ``(len(contexts),
        hidden_dim)`` (empty dict when ``hook_layers`` is None).
    """
    assert isinstance(marker_id, int) and marker_id >= 0, marker_id
    assert isinstance(eos_token_id, int) and eos_token_id >= 0, eos_token_id

    capture: dict[int, list[np.ndarray]] = {}
    handles = []
    # The valid-length per row of the CURRENT sub-batch; set inside the loop so
    # hooks can index the true slot position under left-padding (slot = -1).
    if hook_layers:
        layers = _resolve_decoder_layers(model)
        for li in hook_layers:
            assert 0 <= li < len(layers), (li, len(layers))
            capture[li] = []

        def _make_hook(layer_idx: int):
            def _hook(_module, _args, output):
                hs = output[0] if isinstance(output, tuple) else output
                assert hs.ndim == 3, hs.shape  # (B, T, H)
                # Left-pad puts the slot (last real token) at position -1.
                capture[layer_idx].append(hs[:, -1, :].detach().to(torch.float16).cpu().numpy())

            return _hook

        for li in hook_layers:
            handles.append(layers[li].register_forward_hook(_make_hook(li)))

    stats: list[dict[str, float]] = []
    try:
        for start in range(0, len(contexts), batch_size):
            chunk = contexts[start : start + batch_size]
            context_ids = [tokenizer.encode(c, add_special_tokens=False) for c in chunk]
            for cidx, cids in enumerate(context_ids):
                assert len(cids) > 0, (
                    f"contexts[{start + cidx}] tokenized to [] -- refusing to score"
                )
                if len(cids) > LONG_ROW_PARITY_THRESHOLD:  # §4.5b long-column parity
                    assert_untruncated_token_parity(
                        tokenizer,
                        chunk[cidx],
                        len(cids),
                        context=f"score_marker_slots[{start + cidx}]",
                    )
            max_len = max(len(ids) for ids in context_ids)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padded, attn = [], []
            for ids in context_ids:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)
                attn.append([0] * pad_len + [1] * len(ids))
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            assert logits.ndim == 3, logits.shape

            raw = logits[:, -1, :].float()  # (B, V) next-token logits at the slot
            log_z = torch.logsumexp(raw, dim=-1)  # (B,)
            argmax_ids = raw.argmax(dim=-1)  # (B,)
            for i in range(len(chunk)):
                z_marker = float(raw[i, marker_id].item())
                z_eos = float(raw[i, eos_token_id].item())
                lz = float(log_z[i].item())
                stats.append(
                    {
                        "logp": z_marker - lz,
                        "z_marker": z_marker,
                        "z_eos": z_eos,
                        "logZ": lz,
                        "argmax_is_marker": bool(int(argmax_ids[i].item()) == marker_id),
                    }
                )
            del logits, raw
    finally:
        for h in handles:
            h.remove()

    hiddens: dict[int, np.ndarray] = {}
    for li, chunks in capture.items():
        arr = np.concatenate(chunks, axis=0)
        assert arr.shape[0] == len(contexts), (arr.shape, len(contexts))
        hiddens[li] = arr

    assert len(stats) == len(contexts), (len(stats), len(contexts))
    return stats, hiddens


def score_span_logprob(
    model,
    tokenizer,
    prompts: list[str],
    span: str,
    *,
    batch_size: int = 16,
    device: str = "cuda:0",
) -> list[dict[str, float]]:
    """Length-normalized teacher-forced log P(span | prompt) per prompt.

    Plan §6 G_fact secondary DV (fact-span TF scoring, P2): the taught fact
    sentence is teacher-forced immediately after each generation-ready prompt
    (chat-templated, ends with the assistant header); the score is the mean
    per-token log-prob over the span tokens. Reported trained - base by the
    caller (two invocations, same prompts).

    Returns one dict per prompt: ``span_logp_mean`` (length-normalized),
    ``span_logp_sum``, ``n_span_tokens``.
    """
    span_ids = tokenizer.encode(span, add_special_tokens=False)
    assert len(span_ids) >= 2, f"span tokenized to {len(span_ids)} tokens"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out: list[dict[str, float]] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        rows = [tokenizer.encode(p, add_special_tokens=False) + span_ids for p in chunk]
        prompt_lens = [len(r) - len(span_ids) for r in rows]
        assert all(pl > 0 for pl in prompt_lens), prompt_lens
        for bi, pl in enumerate(prompt_lens):
            if pl > LONG_ROW_PARITY_THRESHOLD:  # §4.5b long-column parity
                assert_untruncated_token_parity(
                    tokenizer, chunk[bi], pl, context=f"score_span_logprob[{start + bi}]"
                )
        max_len = max(len(r) for r in rows)
        input_ids, attn, pads = [], [], []
        for r in rows:
            pad_len = max_len - len(r)
            input_ids.append([pad_id] * pad_len + r)
            attn.append([0] * pad_len + [1] * len(r))
            pads.append(pad_len)
        ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn_t = torch.tensor(attn, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids_t, attention_mask=attn_t).logits
        assert logits.ndim == 3, logits.shape
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for bi in range(len(chunk)):
            # Span token t sits at index pads[bi]+prompt_lens[bi]+t; it is
            # predicted by the logits at the PREVIOUS position.
            first = pads[bi] + prompt_lens[bi]
            pos = torch.arange(first - 1, first - 1 + len(span_ids), device=device)
            tok_ids = torch.tensor(span_ids, device=device)
            lps = logprobs[bi, pos, tok_ids]
            assert lps.shape == (len(span_ids),), lps.shape
            out.append(
                {
                    "span_logp_mean": float(lps.mean().item()),
                    "span_logp_sum": float(lps.sum().item()),
                    "n_span_tokens": len(span_ids),
                }
            )
        del logits, logprobs
    assert len(out) == len(prompts), (len(out), len(prompts))
    return out
