#!/usr/bin/env python3
"""Self-contained HF Transformers L20 cosine helper for compute_i385_predictors.py.

Spawned as a SUBPROCESS by ``compute_i385_predictors.py`` to isolate the
torch + transformers init from the vLLM phase. Validated empirically in
round 5: when the parent process initializes torch.cuda, even a child
process started via ``subprocess.run`` with ``start_new_session=True`` and
``CUDA_VISIBLE_DEVICES=0`` cannot init vLLM (NVMLError_InvalidArgument on
``pynvml.nvmlDeviceGetHandleByIndex``). Putting BOTH the HF cosine phase
AND the vLLM phase in separate clean subprocesses (orchestrated by a torch-
free parent) is the only architecture that worked end-to-end.

Input JSON (``--input-path``):
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "adapter_path": null | "/path/to/checkpoint-N",   # null = base model
  "panel": [{"name": "...", "system_prompt": "..."}, ...],   # 28 rows
  "prompts": ["...", ...],                                    # 20 prompts
  "layer": 20,
  "hf_token": null | "<token>"
}

Output JSON (``--output-path``):
{
  "cosine_to_source": {name: float, ...},
  "source_persona": "librarian",
  "model": "...",
  "adapter_path": null | "...",
  "n_panel": 28,
  "n_prompts": 20
}
"""

from __future__ import annotations

# Pre-init torch BEFORE any HF fork-based downloads so torch.cuda lazy_init
# doesn't fail after huggingface_hub fetches the model (round-3 / round-4 bug).
# This is safe HERE because this helper is a leaf subprocess — nothing
# downstream needs a clean CUDA state from this process.
import torch

if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda:0")

import argparse
import json
import sys
from pathlib import Path

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCE_PERSONA = "librarian"


def _resolve_decoder_layer(model, layer_idx: int):
    """Return the decoder layer for both AutoModelForCausalLM and PeftModel."""
    if hasattr(model, "get_base_model") and callable(model.get_base_model):
        base = model.get_base_model()
        if not hasattr(base, "model") or not hasattr(base.model, "layers"):
            raise AttributeError(
                f"PEFT-wrapped model: get_base_model() returned {type(base).__name__} "
                f"without a .model.layers chain. Cannot extract L{layer_idx} hidden states."
            )
        return base.model.layers[layer_idx]
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(
            f"Model of type {type(model).__name__} has no .model.layers attribute; "
            f"cannot extract L{layer_idx} hidden states."
        )
    return model.model.layers[layer_idx]


def _load_base_model(model_name: str, hf_token: str | None):
    """Load HF model to CPU then move to cuda:0 (bypass caching_allocator_warmup)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        token=hf_token,
    )
    model = model.to("cuda:0")
    model.eval()
    return model, tokenizer


def _maybe_apply_adapter(model, adapter_path: str | None):
    if adapter_path is None:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)


def _compute_l20_centroids(model, tokenizer, panel, prompts, layer):
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    target_layer = _resolve_decoder_layer(model, layer)
    handle = target_layer.register_forward_hook(make_hook(layer))

    try:
        centroids: dict[str, torch.Tensor] = {}
        for row_idx, row in enumerate(panel):
            name = row["name"]
            sys_text = row["system_prompt"]
            row_vecs = []
            for prompt in prompts:
                messages = []
                if sys_text:
                    messages.append({"role": "system", "content": sys_text})
                messages.append({"role": "user", "content": prompt})
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
                with torch.no_grad():
                    _ = model(**inputs)
                last_pos = inputs["input_ids"].shape[1] - 1
                vec = captured[layer][0, last_pos, :].float().cpu()
                row_vecs.append(vec)
            centroids[name] = torch.stack(row_vecs).mean(dim=0)
            print(
                f"[cosine-runner] L{layer} centroid: {row_idx + 1}/{len(panel)} {name}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        handle.remove()

    return centroids


def _cosine_to_source(centroids: dict[str, torch.Tensor]) -> dict[str, float]:
    if SOURCE_PERSONA not in centroids:
        raise KeyError(f"{SOURCE_PERSONA} centroid missing from panel")
    src = F.normalize(centroids[SOURCE_PERSONA], dim=0)
    cos: dict[str, float] = {}
    for name, vec in centroids.items():
        if name == SOURCE_PERSONA:
            cos[name] = 1.0
            continue
        v_norm = F.normalize(vec, dim=0)
        cos[name] = float(torch.dot(src, v_norm).item())
    return cos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_path).read_text())
    model_name = payload["model"]
    adapter_path = payload.get("adapter_path")
    panel = payload["panel"]
    prompts = payload["prompts"]
    layer = int(payload.get("layer", 20))
    hf_token = payload.get("hf_token")

    print(f"[cosine-runner] loading model {model_name}", file=sys.stderr, flush=True)
    model, tokenizer = _load_base_model(model_name, hf_token)
    if adapter_path is not None:
        print(f"[cosine-runner] applying adapter {adapter_path}", file=sys.stderr, flush=True)
        model = _maybe_apply_adapter(model, adapter_path)
        model.eval()

    centroids = _compute_l20_centroids(model, tokenizer, panel, prompts, layer)
    cos = _cosine_to_source(centroids)

    out_payload = {
        "cosine_to_source": cos,
        "source_persona": SOURCE_PERSONA,
        "model": model_name,
        "adapter_path": adapter_path,
        "n_panel": len(panel),
        "n_prompts": len(prompts),
        "layer": layer,
    }
    Path(args.output_path).write_text(json.dumps(out_payload, indent=2))
    print(
        f"[cosine-runner] wrote {len(cos)} cosines to {args.output_path}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
