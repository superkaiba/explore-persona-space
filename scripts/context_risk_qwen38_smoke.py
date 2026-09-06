#!/usr/bin/env python3
"""Qwen3.8-27B loader and pre-generation residual-capture smoke gate.

The gate exercises the exact production seam needed by the risk experiment:
Qwen's thinking-off chat render, a tool-bearing prefix, decoder-block hook
resolution, final-prefix-token capture, exact deterministic replay, the
block-L == hidden_states[L+1] convention, and a short text generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


PROBE_MESSAGES = (
    ({"role": "user", "content": "Return the sum of 17 and 25."},),
    ({"role": "user", "content": "Write a Python function that reverses a string."},),
    ({"role": "user", "content": "State one fact about the Moon."},),
    ({"role": "user", "content": "Which number is larger, 0.09 or 0.1?"},),
    ({"role": "user", "content": "Give one safe way to store a password."},),
    ({"role": "user", "content": "Translate 'good morning' to French."},),
    ({"role": "user", "content": "Name the capital of Japan."},),
    ({"role": "user", "content": "What is 12 multiplied by 8?"},),
)

PROBE_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
)


def _sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist a JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        tmp_path = Path(fh.name)
    os.replace(tmp_path, path)


def render_prefix_ids(
    tokenizer,
    messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    enable_thinking: bool = False,
) -> tuple[str, list[int]]:
    """Render and tokenize one exact assistant-generation prefix."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        kwargs["tools"] = list(tools)
    rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError("Qwen chat template rendered an empty prefix")
    direct_kwargs = dict(kwargs)
    direct_kwargs["tokenize"] = True
    direct_ids = tokenizer.apply_chat_template(list(messages), **direct_kwargs)
    # Transformers 5 may return a BatchEncoding when a chat template includes
    # tools, whereas older versions returned the bare list of token IDs.
    if isinstance(direct_ids, Mapping):
        direct_ids = direct_ids["input_ids"]
    if hasattr(direct_ids, "ids"):
        direct_ids = direct_ids.ids
    if hasattr(direct_ids, "tolist"):
        direct_ids = direct_ids.tolist()
    if direct_ids and len(direct_ids) == 1 and hasattr(direct_ids[0], "ids"):
        direct_ids = direct_ids[0].ids
    if direct_ids and isinstance(direct_ids[0], list):
        if len(direct_ids) != 1:
            raise RuntimeError(
                f"single chat render unexpectedly produced {len(direct_ids)} token rows"
            )
        direct_ids = direct_ids[0]
    if direct_ids != ids:
        raise RuntimeError(
            "chat-template tokenize=True disagrees with rendered-text tokenization: "
            f"rendered_len={len(ids)} direct_len={len(direct_ids)} "
            f"rendered_tail={ids[-12:]} direct_tail={direct_ids[-12:]}"
        )
    if enable_thinking is False and "<think>\n\n</think>" not in rendered:
        raise RuntimeError("thinking-off render lacks the expected closed empty thinking block")
    return rendered, [int(token_id) for token_id in ids]


def _load_model_and_tokenizer(cfg: DictConfig):
    """Load the pinned model in BF16 and assert its text-stack geometry."""
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.8 smoke requires a CUDA GPU")
    realized_version = transformers.__version__
    if realized_version != str(cfg.model.transformers_version):
        raise RuntimeError(
            f"expected transformers=={cfg.model.transformers_version}, got {realized_version}"
        )
    model_id = str(cfg.model.id)
    revision = str(cfg.model.revision)
    config = transformers.AutoConfig.from_pretrained(model_id, revision=revision)
    text_config = getattr(config, "text_config", config)
    geometry = (int(text_config.num_hidden_layers), int(text_config.hidden_size))
    expected = (int(cfg.model.expected_layers), int(cfg.model.expected_hidden_dim))
    if geometry != expected:
        raise RuntimeError(f"model geometry drifted: expected {expected}, got {geometry}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = getattr(transformers, "AutoModelForMultimodalLM", None)
    if loader is None:
        loader = transformers.AutoModelForImageTextToText
    model = loader.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    blocks, _embed, depth = _resolve_decoder_blocks(model)
    if blocks is None or len(blocks) != expected[0]:
        realized = None if blocks is None else len(blocks)
        raise RuntimeError(
            f"decoder-block resolver found {realized} blocks, expected {expected[0]}"
        )
    return model, tokenizer, depth


def _capture_last_prefix(model, ids_rows: list[list[int]], layers: list[int]):
    """Capture selected block outputs at every row's final unpadded token."""
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations

    device = next(model.parameters()).device
    width = max(map(len, ids_rows))
    pad_id = int(model.config.text_config.pad_token_id or model.config.text_config.eos_token_id)
    input_ids = torch.full((len(ids_rows), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row_index, ids in enumerate(ids_rows):
        input_ids[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_index, : len(ids)] = 1
    captured = extract_layer_activations(
        model,
        input_ids,
        layers,
        attention_mask=attention_mask,
        detach_to_cpu=True,
    )
    out = torch.stack(
        [
            torch.stack([captured[layer][row_index, len(ids) - 1] for layer in layers])
            for row_index, ids in enumerate(ids_rows)
        ]
    )
    return out, input_ids, attention_mask


def _hook_tuple_errors(model, input_ids, attention_mask, layers: list[int]) -> dict[int, float]:
    """Measure block-hook versus hidden-state-tuple relative errors in one forward."""
    import torch

    from explore_persona_space.analysis.extraction import (
        _logits_to_keep_kwargs,
        _resolve_decoder_blocks,
        _unwrap,
    )

    blocks, _embed, _depth = _resolve_decoder_blocks(model)
    if blocks is None:
        raise RuntimeError("hook/tuple probe requires resolved decoder blocks")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            captured[layer] = _unwrap(output).detach()

        return hook

    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids[:1],
                attention_mask=attention_mask[:1],
                output_hidden_states=True,
                use_cache=False,
                **_logits_to_keep_kwargs(model, False),
            )
    finally:
        for handle in handles:
            handle.remove()
    hidden_states = output.hidden_states
    if len(hidden_states) != len(blocks) + 1:
        raise RuntimeError(
            f"hidden-state tuple length {len(hidden_states)} != block count + 1 ({len(blocks) + 1})"
        )
    errors = {}
    for layer in layers:
        hooked = captured[layer].float()
        tuple_value = hidden_states[layer + 1].float()
        errors[layer] = float((hooked - tuple_value).norm() / hooked.norm().clamp_min(1e-30))
    return errors


def run_smoke(cfg: DictConfig) -> dict[str, Any]:
    """Execute the GPU gate and return its reproducibility report."""
    import torch

    model, tokenizer, wrapper_depth = _load_model_and_tokenizer(cfg)
    layers = [int(layer) for layer in cfg.model.capture_layers]
    n_rows = int(cfg.smoke.exact_replay_rows)
    messages = [list(row) for row in PROBE_MESSAGES[:n_rows]]
    rendered_and_ids = [
        render_prefix_ids(
            tokenizer,
            row,
            tools=PROBE_TOOLS if index == 0 else None,
            enable_thinking=bool(cfg.smoke.enable_thinking),
        )
        for index, row in enumerate(messages)
    ]
    rendered = [row[0] for row in rendered_and_ids]
    ids_rows = [row[1] for row in rendered_and_ids]
    tool_prefix_rendered = "# Tools" in rendered[0] and "calculator" in rendered[0]
    if not tool_prefix_rendered:
        raise RuntimeError("tool-bearing probe did not render its tool inventory")
    torch.cuda.reset_peak_memory_stats()
    first, input_ids, attention_mask = _capture_last_prefix(model, ids_rows, layers)
    second, replay_ids, replay_mask = _capture_last_prefix(model, ids_rows, layers)
    if not torch.equal(input_ids, replay_ids) or not torch.equal(attention_mask, replay_mask):
        raise RuntimeError("exact-prefix replay changed the token tensors")
    replay_max_abs = float((first - second).abs().max())
    if replay_max_abs != 0.0:
        raise RuntimeError(
            f"exact-prefix activation replay was not exact: max abs {replay_max_abs}"
        )
    tuple_errors = _hook_tuple_errors(model, input_ids, attention_mask, layers)
    tolerance = float(cfg.smoke.hook_relative_tolerance)
    if max(tuple_errors.values()) > tolerance:
        raise RuntimeError(f"hook/tuple mismatch exceeds {tolerance}: {tuple_errors}")
    generated = model.generate(
        input_ids=input_ids[:1],
        attention_mask=attention_mask[:1],
        do_sample=False,
        max_new_tokens=int(cfg.smoke.max_new_tokens),
        pad_token_id=tokenizer.pad_token_id,
    )
    continuation_ids = generated[0, input_ids.shape[1] :].tolist()
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
    if not continuation:
        raise RuntimeError("short deterministic generation was empty")
    report = {
        "schema_version": "context_risk_qwen38_loader_smoke_v1",
        "model_id": str(cfg.model.id),
        "model_revision": str(cfg.model.revision),
        "transformers_version": str(cfg.model.transformers_version),
        "dtype": str(cfg.model.dtype),
        "capture_layers": layers,
        "capture_shape": list(first.shape),
        "wrapper_depth": int(wrapper_depth),
        "n_exact_replay_rows": n_rows,
        "exact_replay_token_ids_sha256": _sha256_json(ids_rows),
        "exact_replay_max_abs": replay_max_abs,
        "hook_tuple_relative_errors": {str(key): value for key, value in tuple_errors.items()},
        "hook_relative_tolerance": tolerance,
        "thinking_enabled": bool(cfg.smoke.enable_thinking),
        "tool_prefix_rendered": tool_prefix_rendered,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "generated_token_count": len(continuation_ids),
        "generated_text": continuation,
        "peak_cuda_memory_gb": float(torch.cuda.max_memory_allocated() / 2**30),
        "passed": True,
    }
    return report


@hydra.main(
    version_base="1.3", config_path="../configs/eval", config_name="context_risk_qwen38_smoke"
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the Qwen3.8 loader/capture gate."""
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    report = run_smoke(cfg)
    path = Path(str(cfg.output_dir)) / "run_result.json"
    _write_json_atomic(path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"[context-risk-smoke] PASS report={path}", flush=True)


if __name__ == "__main__":
    main()
