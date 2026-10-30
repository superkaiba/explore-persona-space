"""Single-neuron MLP steering for Qwen3 decoder models (HF transformers).

Intervention: add ``alpha`` to neuron N of the input of
``model.model.layers[L].mlp.down_proj`` at every token position (prompt and
generated). Because down_proj is linear, this equals adding
``alpha * W_down[:, N]`` to the residual stream at the output of layer L
(:func:`equivalence_check` verifies this on next-token logits).

:class:`NeuronSteerer` supports a DIFFERENT (layer, neuron, alpha) per batch row,
so one generate() call can mix conditions. Rows keep fixed indices for the whole
HF ``generate`` call (finished rows are padded, never dropped), so per-row index
tensors set before the call stay valid at every decoding step.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Steer:
    """One additive intervention: neuron ``neuron`` of layer ``layer`` += ``alpha``."""

    layer: int
    neuron: int
    alpha: float


def down_proj(model, layer: int) -> torch.nn.Linear:
    """Return the down_proj Linear of decoder layer ``layer``."""
    return model.model.layers[layer].mlp.down_proj


def down_proj_column(model, layer: int, neuron: int) -> torch.Tensor:
    """Return W_down[:, neuron] (the residual direction written by one neuron), float32."""
    w = down_proj(model, layer).weight  # (d_model, d_mlp)
    assert w.dim() == 2 and 0 <= neuron < w.shape[1], (tuple(w.shape), neuron)
    return w[:, neuron].detach().float()


class NeuronSteerer:
    """Forward pre-hooks on down_proj that add per-row alphas to single neurons.

    Use :meth:`set_rows` before each batched forward / generate call and
    :meth:`clear` after. Hooks stay registered until :meth:`remove`.
    """

    def __init__(self, model, layers: Iterable[int]):
        self.model = model
        self._state: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._handles = []
        self.layers = sorted(set(layers))
        for layer in self.layers:
            hook = self._make_hook(layer)
            self._handles.append(down_proj(model, layer).register_forward_pre_hook(hook))

    def _make_hook(self, layer: int):
        def hook(module, args):
            state = self._state.get(layer)
            if state is None:
                return None
            rows, neurons, alphas = state
            x = args[0]
            assert x.dim() == 3 and x.shape[-1] == module.in_features, tuple(x.shape)
            # Advanced indices separated by a slice put the index dim first: (k, T).
            sub = x[rows, :, neurons]
            x[rows, :, neurons] = (sub.float() + alphas[:, None]).to(x.dtype)
            return None  # args[0] modified in place

        return hook

    def set_rows(self, steers: Sequence[Steer | None]) -> None:
        """Set one optional :class:`Steer` per batch row (None = unsteered row)."""
        device = down_proj(self.model, self.layers[0]).weight.device if self.layers else "cpu"
        self._state = {}
        per_layer: dict[int, tuple[list[int], list[int], list[float]]] = {}
        for row, st in enumerate(steers):
            if st is None or st.alpha == 0:
                continue
            if st.layer not in self.layers:
                raise ValueError(f"layer {st.layer} has no hook (hooked: {self.layers})")
            rows, neurons, alphas = per_layer.setdefault(st.layer, ([], [], []))
            rows.append(row)
            neurons.append(st.neuron)
            alphas.append(float(st.alpha))
        for layer, (rows, neurons, alphas) in per_layer.items():
            assert len(set(rows)) == len(rows), "one steer per row per layer"
            self._state[layer] = (
                torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(neurons, dtype=torch.long, device=device),
                torch.tensor(alphas, dtype=torch.float32, device=device),
            )

    def clear(self) -> None:
        """Disable steering for subsequent forwards (hooks stay registered)."""
        self._state = {}

    def remove(self) -> None:
        """Unregister all hooks."""
        self.clear()
        for h in self._handles:
            h.remove()
        self._handles = []


def chat_prompt_ids(tokenizer, question: str) -> list[int]:
    """Token ids of a single-turn chat prompt, no system prompt, thinking off."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > 0
    return ids


def left_pad(
    seqs: Sequence[Sequence[int]], pad_id: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad token id lists into (input_ids, attention_mask), both (B, T)."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s) :] = torch.tensor(list(s), dtype=torch.long)
        mask[i, width - len(s) :] = 1
    return ids.to(device), mask.to(device)


@dataclass(frozen=True)
class Generation:
    """One decoded reply."""

    text: str
    n_new_tokens: int  # including the stop token when one was produced
    hit_cap: bool  # True when no stop token appeared within max_new_tokens


def eos_ids(tokenizer) -> list[int]:
    """Stop tokens for chat replies: <|im_end|>, <|endoftext|> and the tokenizer EOS."""
    ids = [
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        tokenizer.eos_token_id,
    ]
    out = sorted({int(i) for i in ids if i is not None and i >= 0})
    assert out, "no stop tokens resolved"
    return out


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    steerer: NeuronSteerer | None,
    prompt_ids: Sequence[Sequence[int]],
    steers: Sequence[Steer | None],
    *,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> list[Generation]:
    """Sample one reply per row with per-row steering. Returns decoded generations."""
    from transformers import GenerationConfig

    assert len(prompt_ids) == len(steers) > 0
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer needs a pad token"
    stops = eos_ids(tokenizer)
    input_ids, attn = left_pad(prompt_ids, pad_id, device)
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        eos_token_id=stops,
        pad_token_id=pad_id,
        repetition_penalty=1.0,
    )
    if steerer is not None:
        steerer.set_rows(steers)
    elif any(s is not None for s in steers):
        raise ValueError("steers given but no steerer")
    try:
        torch.manual_seed(seed)
        # use_model_defaults=False: never let the model's generation_config fill
        # fields we set to library defaults (e.g. repetition_penalty=1.0).
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            generation_config=gen_cfg,
            use_model_defaults=False,
        )
    finally:
        if steerer is not None:
            steerer.clear()
    new = out[:, input_ids.shape[1] :].cpu()
    assert new.shape[0] == len(prompt_ids) and new.shape[1] <= max_new_tokens, tuple(new.shape)
    stop_set = set(stops)
    gens: list[Generation] = []
    for row in new.tolist():
        cut = next((i for i, t in enumerate(row) if t in stop_set), None)
        if cut is None:
            assert len(row) == max_new_tokens, "generation ended early without a stop token"
            toks, n, hit = row, len(row), True
        else:
            toks, n, hit = row[:cut], cut + 1, False
        gens.append(Generation(tokenizer.decode(toks, skip_special_tokens=True), n, hit))
    return gens


@torch.inference_mode()
def last_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Float32 next-token logits at the last position for a (1, T) input."""
    out = model(input_ids=input_ids)
    return out.logits[0, -1].float()


@torch.inference_mode()
def equivalence_check(model, tokenizer, question: str, steer: Steer) -> dict:
    """Compare the down_proj pre-hook with adding alpha * W_down[:, N] after layer L.

    Returns diagnostics on next-token logits (last prompt position). ``ratio`` is
    max|pre - resid| / max|pre - unsteered|; the two interventions agree when the
    ratio is small and the top-1 token matches.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([chat_prompt_ids(tokenizer, question)], device=device)
    base = last_logits(model, ids)

    steerer = NeuronSteerer(model, [steer.layer])
    try:
        steerer.set_rows([steer])
        pre = last_logits(model, ids)
    finally:
        steerer.remove()

    vec = steer.alpha * down_proj_column(model, steer.layer, steer.neuron)

    def resid_hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        new = (hs.float() + vec.to(hs.device)).to(hs.dtype)
        return (new, *output[1:]) if isinstance(output, tuple) else new

    handle = model.model.layers[steer.layer].register_forward_hook(resid_hook)
    try:
        resid = last_logits(model, ids)
    finally:
        handle.remove()

    def kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
        lp = torch.log_softmax(p_logits, -1)
        lq = torch.log_softmax(q_logits, -1)
        return float((lp.exp() * (lp - lq)).sum())

    d_pr = float((pre - resid).abs().max())
    d_pb = float((pre - base).abs().max())
    top10 = lambda x: set(torch.topk(x, 10).indices.tolist())  # noqa: E731
    return {
        "question": question,
        "layer": steer.layer,
        "neuron": steer.neuron,
        "alpha": steer.alpha,
        "max_abs_diff_pre_vs_resid": d_pr,
        "max_abs_diff_pre_vs_unsteered": d_pb,
        "ratio": d_pr / d_pb if d_pb > 0 else math.inf,
        "kl_pre_resid": kl(pre, resid),
        "kl_pre_unsteered": kl(pre, base),
        "top1_pre": tokenizer.decode([int(pre.argmax())]),
        "top1_resid": tokenizer.decode([int(resid.argmax())]),
        "top1_unsteered": tokenizer.decode([int(base.argmax())]),
        "top1_match": int(pre.argmax()) == int(resid.argmax()),
        "top10_overlap": len(top10(pre) & top10(resid)),
    }


@torch.inference_mode()
def residual_norms(
    model, tokenizer, prompts: Sequence[str], layers: Sequence[int], batch_size: int = 16
) -> dict[int, dict]:
    """Per-token L2 norm of the residual stream after each layer, over chat prompts.

    ``hidden_states[L + 1]`` is the output of decoder layer L (index 0 is the
    embedding output). Returns per layer: median over all real tokens, median
    excluding position 0 (attention-sink token), and the token count.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    acc: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    acc_no0: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    seqs = [chat_prompt_ids(tokenizer, p) for p in prompts]
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i : i + batch_size]
        width = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, s in enumerate(chunk):  # right padding: position 0 is the first real token
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        out = model(
            input_ids=ids.to(device), attention_mask=mask.to(device), output_hidden_states=True
        )
        m = mask.bool().to(device)
        m_no0 = m.clone()
        m_no0[:, 0] = False
        for L in layers:
            hs = out.hidden_states[L + 1]
            assert hs.shape[:2] == (len(chunk), width), tuple(hs.shape)
            norms = hs.float().norm(dim=-1)  # (B, T)
            acc[L].append(norms[m].cpu())
            acc_no0[L].append(norms[m_no0].cpu())
    result = {}
    for L in layers:
        allv = torch.cat(acc[L])
        no0 = torch.cat(acc_no0[L])
        result[L] = {
            "median": float(allv.median()),
            "median_excl_pos0": float(no0.median()),
            "n_tokens": int(allv.numel()),
        }
    return result


@torch.inference_mode()
def logit_lens(model, tokenizer, layer: int, neuron: int, k: int = 20) -> dict:
    """Top tokens promoted by +W_down[:, N] and by -W_down[:, N] through final norm + unembed."""
    w = down_proj_column(model, layer, neuron)
    norm_w = model.model.norm.weight.detach().float()
    direction = (norm_w * w.to(norm_w.device)).to(model.lm_head.weight.dtype)
    logits = (model.lm_head.weight @ direction.to(model.lm_head.weight.device)).float()

    def top(x: torch.Tensor) -> list[str]:
        return [tokenizer.decode([int(i)]) for i in torch.topk(x, k).indices.tolist()]

    return {"pos": top(logits), "neg": top(-logits)}
