"""Exp-2 generation-time activation steering (arXiv 2507.21509 ``sec:validate``).

Ports the paper's ``ActivationSteerer`` hook shape (``safety-research/persona_vectors``
@ ``b8e0f04`` ``activation_steer.py``) faithfully: a ``register_forward_hook`` on
block ``layer_idx`` that adds ``coeff * vector`` to the block output. With
``positions="response"`` it steers the LAST generated position each decode step
(the paper's ``eval_steering.sh`` steering_type). The steering vector is the RAW
(non-unit-normalized) ``r_B[layer]`` — the paper scales the raw response-avg diff
by the coefficient (``eval_persona.py::sample_steering`` loads
``vector = torch.load(vector_path)[layer]`` and passes it un-normalized).

Steering CANNOT use vLLM: the per-decode-step forward hook has no analogue in
vLLM's batched engine (plan v2 §12 Assumption 7), so generation is HF
``model.generate()`` under the hook, batched with left-padding.

Layer convention (plan v2 §12): the paper's 1-indexed "layer 20" is block output
20 == ``hidden_states[20]``. #778's stored ``r_B`` tensor is 0-indexed by block
so ``r_B[19]`` == block output 20 == the paper's layer 20. This module takes a
1-indexed ``layer`` (default 20) and hooks ``layer_idx = layer - 1``, and the
CALLER is responsible for passing ``r_B[layer - 1]`` as the vector (see
``scripts/issue816_steering.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

logger = logging.getLogger("issue816.steering")

# The paper's Qwen2.5-7B-Instruct steering-selected layer (1-indexed).
DEFAULT_LAYER = 20
# eval_persona.py::sample_steering defaults (b8e0f04).
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_NEW_TOKENS = 1000
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MIN_NEW_TOKENS = 1


class ActivationSteerer:
    """Add ``coeff * steering_vector`` to block ``layer_idx``'s output.

    Faithful port of ``activation_steer.py`` @ ``b8e0f04`` (the Llama/Qwen
    ``model.layers`` branch only — the multi-arch layer-list search is trimmed to
    ``model.layers`` / ``model.model.layers`` since we always steer Qwen). Handles
    the block returning a tuple (Qwen returns ``(hidden_states, ...)``). Fails
    loud if it cannot locate the layer list or the vector length is wrong.

    ``positions``:
      - ``"response"`` (default): add to the LAST position of the current forward
        (each decode step this is the just-generated token) — the paper's
        steering_type for the steering figures.
      - ``"all"``: add to every position.
      - ``"prompt"``: add to every position UNLESS the forward is a single new
        token (``seq_len == 1``), i.e. steer prompt positions only.
    """

    def __init__(
        self,
        model,
        steering_vector,
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "response",
    ):
        import torch

        self.model = model
        self.coeff = float(coeff)
        self.layer_idx = layer_idx
        self.positions = positions.lower()
        self._handle = None
        valid = {"all", "prompt", "response"}
        if self.positions not in valid:
            raise ValueError(f"positions must be one of {valid}, got {positions!r}")

        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError(f"steering_vector must be 1-D, got shape {tuple(self.vector.shape)}")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden is not None and self.vector.numel() != hidden:
            raise ValueError(f"vector length {self.vector.numel()} != model hidden_size {hidden}")

    def _locate_layer(self):
        # Try the PEFT-wrapped and bare Qwen/Llama paths (b8e0f04 searched a list;
        # we only need model.layers for the base model + model.model.layers for a
        # PeftModel wrapper — but Exp-2 steers the BASE model, so model.layers wins).
        for path in ("model.layers", "model.model.layers", "base_model.model.model.layers"):
            cur = self.model
            ok = True
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    ok = False
                    break
            if ok and hasattr(cur, "__getitem__"):
                if not (-len(cur) <= self.layer_idx < len(cur)):
                    raise IndexError(
                        f"layer_idx {self.layer_idx} out of range for {len(cur)} layers"
                    )
                return cur[self.layer_idx]
        raise ValueError(
            "could not locate the transformer layer list (tried model.layers / "
            "model.model.layers / base_model.model.model.layers)"
        )

    def _hook_fn(self, module, ins, out):
        import torch

        steer = self.coeff * self.vector

        def _add(t):
            if self.positions == "all":
                return t + steer.to(t.device)
            if self.positions == "prompt":
                if t.shape[1] == 1:
                    return t
                t2 = t.clone()
                t2 += steer.to(t.device)
                return t2
            # "response": add to the last position of this forward.
            t2 = t.clone()
            t2[:, -1, :] += steer.to(t.device)
            return t2

        if torch.is_tensor(out):
            return _add(out)
        if isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            return (_add(out[0]), *out[1:])
        return out

    def __enter__(self):
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def steered_generate(
    model,
    tokenizer,
    conversations: Sequence[list[dict]],
    steering_vector,
    *,
    layer: int = DEFAULT_LAYER,
    coef: float = 0.0,
    bs: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    positions: str = "response",
) -> list[str]:
    """Batched HF ``model.generate()`` under the steering hook.

    Faithful to ``eval_persona.py::sample_steering`` (b8e0f04): left-padding,
    the ``ActivationSteerer`` context manager wraps the whole batch,
    ``layer_idx = layer - 1``. Returns the decoded responses (special tokens
    stripped), one per conversation, in input order.

    ``conversations`` is a list of chat-message lists (each a single-turn
    ``[{"role": "user", "content": q}]`` for the neutral eval questions).
    ``coef = 0`` runs the unsteered control (the hook is still attached but adds
    the zero vector — identical to the paper's coef-0 arm).
    """
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in conversations
    ]
    model.eval()
    outputs: list[str] = []
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        toks = tokenizer(batch, return_tensors="pt", padding=True)
        toks = {k: v.to(model.device) for k, v in toks.items()}
        with (
            ActivationSteerer(
                model, steering_vector, coeff=coef, layer_idx=layer - 1, positions=positions
            ),
            torch.no_grad(),
        ):
            gen = model.generate(
                **toks,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = toks["input_ids"].shape[1]
        outputs.extend(tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in gen)
    return outputs
