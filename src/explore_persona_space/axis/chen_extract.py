"""Chen et al. persona-vector extraction recipe.

Implements the extraction recipe from
    Chen et al., "Persona Vectors: Monitoring and Controlling Character Traits
    in Language Models", Anthropic, 2025.

Pipeline per trait:
    1. For each probe prompt P:
        a. Generate a completion with the trait-POSITIVE persona prepended
           as the system message (greedy, fixed max_new_tokens).
        b. Generate a completion with the trait-NEGATIVE persona prepended.
    2. For each (P, persona_sign, completion) tuple, run a forward pass on
       ``[persona_system_prompt, user(P), assistant(completion)]`` with hooks on
       ``model.model.layers[L].input_layernorm`` for each target layer L, and
       MEAN the hidden states *only over completion-token positions*.
    3. Persona vector at layer L = mean(activations | trait+) - mean(activations | trait-).
    4. Stack across requested layers to a single ``(n_layers, d_model)`` tensor.

Returns a dict mapping trait -> Tensor of shape ``(n_layers, d_model)``.

This module is GPU-required for real extraction; for unit/dry-run testing the
caller can pass ``model=None, tokenizer=None`` and pre-fabricated completions to
exercise the wiring on CPU (see ``scripts/run_chen_vs_centroid.py --dry-run``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from explore_persona_space.axis.trait_personas import TRAIT_PERSONAS

logger = logging.getLogger(__name__)


@dataclass
class ChenExtractionConfig:
    """Configuration for a Chen-style extraction run."""

    model_name: str
    traits: list[str]
    layers: list[int]
    prompts_per_trait: int
    max_new_tokens: int = 128
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 2048
    hook_target: str = "input_layernorm"  # matches scripts/extract_persona_vectors.py
    seed: int = 42


def _build_chat_text(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_full_chat_text(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    assistant_completion: str,
) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_completion},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def generate_paired_completions(
    cfg: ChenExtractionConfig,
    probe_prompts: list[str],
    gpu_id: int = 0,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Generate trait+/trait- completions for every (trait, probe) using vLLM.

    Returns a nested dict::

        out[trait][sign] = [
            {"probe": <probe>, "system": <persona>, "completion": <text>},
            ...
        ]

    where ``sign`` is ``"pos"`` or ``"neg"``. One completion per probe per sign.
    """
    from vllm import LLM, SamplingParams

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(
        "Chen generation: traits=%d, probes=%d, max_new=%d, model=%s",
        len(cfg.traits),
        len(probe_prompts),
        cfg.max_new_tokens,
        cfg.model_name,
    )

    # Build the full conversation list across (trait, sign, probe).
    convos: list[list[dict[str, str]]] = []
    keys: list[tuple[str, str, str]] = []  # (trait, sign, probe)

    for trait in cfg.traits:
        if trait not in TRAIT_PERSONAS:
            raise KeyError(f"Unknown trait {trait!r}")
        tp = TRAIT_PERSONAS[trait]
        for sign, persona in (("pos", tp.pos), ("neg", tp.neg)):
            for probe in probe_prompts:
                convos.append(
                    [
                        {"role": "system", "content": persona},
                        {"role": "user", "content": probe},
                    ]
                )
                keys.append((trait, sign, probe))

    llm = LLM(
        model=cfg.model_name,
        tensor_parallel_size=1,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        seed=cfg.seed,
    )
    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
        seed=cfg.seed,
    )

    outputs = llm.chat(convos, sampling_params)

    # Organize results.
    results: dict[str, dict[str, list[dict[str, str]]]] = {
        t: {"pos": [], "neg": []} for t in cfg.traits
    }
    for (trait, sign, probe), out in zip(keys, outputs, strict=True):
        text = out.outputs[0].text
        persona = TRAIT_PERSONAS[trait].pos if sign == "pos" else TRAIT_PERSONAS[trait].neg
        results[trait][sign].append(
            {
                "probe": probe,
                "system": persona,
                "completion": text,
            }
        )

    del llm
    torch.cuda.empty_cache()
    return results


def _get_hook_target(model, layer_idx: int, hook_target: str):
    layer = model.model.layers[layer_idx]
    if hook_target == "input_layernorm":
        return layer.input_layernorm
    if hook_target == "self_attn":
        return layer.self_attn
    if hook_target == "block":
        return layer
    raise ValueError(f"Unknown hook_target {hook_target!r}")


def _hidden_state_from_hook(output: Any) -> torch.Tensor:
    """Pull the (B, T, H) hidden-state tensor from a hook output."""
    if isinstance(output, tuple):
        return output[0]
    return output


def extract_chen_vectors(
    model: Any,
    tokenizer: Any,
    cfg: ChenExtractionConfig,
    paired_completions: dict[str, dict[str, list[dict[str, str]]]],
    output_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Run the second forward pass and compute Chen-style persona vectors.

    Args:
        model: A loaded HF causal LM (bf16, on a single GPU).
        tokenizer: The matching HF tokenizer.
        cfg:    Extraction configuration (traits, layers, hook target).
        paired_completions: Output of :func:`generate_paired_completions`.
        output_dir: If given, save per-trait ``{trait}.pt`` tensors here.

    Returns:
        Dict mapping trait -> Tensor of shape ``(len(cfg.layers), d_model)``.
    """
    # Set up hooks
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, _input, output):
            captured[layer_idx] = _hidden_state_from_hook(output).detach()

        return hook_fn

    hooks = []
    for layer_idx in cfg.layers:
        target = _get_hook_target(model, layer_idx, cfg.hook_target)
        hooks.append(target.register_forward_hook(make_hook(layer_idx)))

    vectors: dict[str, torch.Tensor] = {}
    t0 = time.time()

    try:
        for trait_idx, trait in enumerate(cfg.traits):
            sign_layer_means: dict[str, dict[int, list[torch.Tensor]]] = {
                "pos": {L: [] for L in cfg.layers},
                "neg": {L: [] for L in cfg.layers},
            }

            for sign in ("pos", "neg"):
                items = paired_completions[trait][sign]
                for item in items:
                    persona = item["system"]
                    probe = item["probe"]
                    completion = item["completion"]

                    # Compute the prompt-only token length so we know where the
                    # completion starts.
                    prompt_text = _build_chat_text(tokenizer, persona, probe)
                    prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)[
                        "input_ids"
                    ]
                    prompt_len = prompt_ids.shape[1]

                    full_text = _build_full_chat_text(tokenizer, persona, probe, completion)
                    full_inputs = tokenizer(full_text, return_tensors="pt", padding=False).to(
                        model.device
                    )
                    full_len = full_inputs["input_ids"].shape[1]

                    if full_len <= prompt_len:
                        # Empty completion — skip.
                        continue

                    with torch.no_grad():
                        _ = model(**full_inputs)

                    for L in cfg.layers:
                        hs = captured[L]  # (1, T, H)
                        resp_hs = hs[0, prompt_len:full_len, :].float().cpu()
                        sign_layer_means[sign][L].append(resp_hs.mean(dim=0))

            # Difference of means per layer.
            per_layer_diffs: list[torch.Tensor] = []
            for L in cfg.layers:
                pos_stack = (
                    torch.stack(sign_layer_means["pos"][L]) if sign_layer_means["pos"][L] else None
                )
                neg_stack = (
                    torch.stack(sign_layer_means["neg"][L]) if sign_layer_means["neg"][L] else None
                )
                if pos_stack is None or neg_stack is None:
                    raise RuntimeError(
                        f"No usable completions for trait={trait} layer={L} "
                        f"(pos={len(sign_layer_means['pos'][L])}, "
                        f"neg={len(sign_layer_means['neg'][L])})"
                    )
                per_layer_diffs.append(pos_stack.mean(dim=0) - neg_stack.mean(dim=0))

            vec = torch.stack(per_layer_diffs)  # (n_layers, d_model)
            vectors[trait] = vec

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(vec, output_dir / f"{trait}.pt")

            elapsed = time.time() - t0
            logger.info(
                "Chen extract [%d/%d] %s — %.0fs elapsed, shape=%s",
                trait_idx + 1,
                len(cfg.traits),
                trait,
                elapsed,
                tuple(vec.shape),
            )
    finally:
        for h in hooks:
            h.remove()

    return vectors


def save_paired_completions(
    paired: dict[str, dict[str, list[dict[str, str]]]],
    output_path: Path,
) -> None:
    """Persist generated completions to JSON for replay / audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(paired, f, indent=2)


def load_paired_completions(
    input_path: Path,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    with open(input_path) as f:
        return json.load(f)
