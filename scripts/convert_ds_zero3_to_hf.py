#!/usr/bin/env python3
"""Convert a DeepSpeed ZeRO-3 sharded checkpoint to HuggingFace safetensors.

Run as a **fresh subprocess on rank-0** after ``train_stage_sft.py`` has
written its ZeRO-3 native per-rank checkpoint via
``engine.save_checkpoint(...)``. This subprocess starts with NO optimizer
state in CPU memory (the training process has exited and freed the ZeRO-3
CPU-offloaded optimizer shards), so it has the full pod RAM available for
the model-size consolidation that would OOM if attempted inside the
training process.

#506 incident (2026-06-07): ``trainer.save_model`` on a 32B FWFT under
ZeRO-3 + CPU optimizer offload OOMed at shard 3 of 14. With 8 ranks each
holding ~48 GB of CPU optimizer state (~384 GB cluster-wide), gathering an
additional 64 GB of weights on rank-0 pushed past available RAM. The fix
is to defer the gather to a fresh process where optimizer state has been
released.

Memory budget:
- Stage 1 (consolidation): peak ~2x model size = ~128 GB CPU for a 32B
  fp32-to-bf16 cast. Safe on any 8x H200 pod (~256+ GB RAM).
- Stage 2 (sharded safetensors write): peak ~max_shard_size per shard
  buffer + the held state_dict; ``--max-shard-size 2GB`` keeps the write
  buffer small while staying well under any reasonable HF-loader chunk
  size.

Usage::

    uv run python scripts/convert_ds_zero3_to_hf.py \\
        --ds-checkpoint-dir /workspace/.../<ckpt>_ds_native \\
        --output-dir /workspace/.../<ckpt> \\
        --model-id Qwen/Qwen3-32B \\
        --tag final \\
        --max-shard-size 2GB

The model-id is needed to recover the architecture config (Qwen3-32B's
``config.json`` is on the Hub, not in the ZeRO-3 native checkpoint) and
the tokenizer for the HF-loadable artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ds-checkpoint-dir",
        required=True,
        help="Directory containing the DeepSpeed ZeRO-3 per-rank shards "
        "(written by engine.save_checkpoint).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the HF-format checkpoint (config.json + "
        "tokenizer files + sharded *.safetensors + safetensors index).",
    )
    p.add_argument(
        "--model-id",
        required=True,
        help="HF model id used to recover architecture config + tokenizer "
        "(e.g. Qwen/Qwen3-32B). The DS-native checkpoint only carries weights; "
        "the architecture and tokenizer come from the model card.",
    )
    p.add_argument(
        "--tag",
        default="final",
        help="DeepSpeed checkpoint tag passed at engine.save_checkpoint time. "
        "Default 'final' matches train_stage_sft.py's save call.",
    )
    p.add_argument(
        "--max-shard-size",
        default="2GB",
        help="Maximum size per shard for the safetensors output. Smaller shards "
        "(default 2GB) keep peak write-buffer memory low; the HF loader "
        "transparently re-assembles from the index.",
    )
    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Output dtype. Training was bf16 so the artifact stays bf16 by "
        "default; the fp32 master copy is not preserved (it lives in the DS "
        "native checkpoint, which we delete after conversion).",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = Path(args.ds_checkpoint_dir)
    if not ds_dir.exists():
        raise SystemExit(f"DS checkpoint directory does not exist: {ds_dir}")

    # Stage 1 — consolidate ZeRO-3 sharded fp32 state dict on CPU.
    # Holds ~2x model-size at peak (fp32 -> bf16 cast intermediate).
    print(f"[convert_ds_zero3_to_hf] consolidating {ds_dir} (tag={args.tag})", flush=True)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(str(ds_dir), tag=args.tag)
    n_tensors = len(state_dict)
    n_params = sum(t.numel() for t in state_dict.values())
    print(
        f"[convert_ds_zero3_to_hf] consolidated {n_tensors} tensors, "
        f"{n_params / 1e9:.2f}B params (fp32, CPU)",
        flush=True,
    )

    # Cast to target dtype. Free the fp32 originals one-by-one to bound peak RAM
    # to ~1.5x model-size during the cast (rather than fp32+bf16 simultaneously).
    target_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    keys = list(state_dict.keys())
    for k in keys:
        state_dict[k] = state_dict[k].to(target_dtype)
    print(f"[convert_ds_zero3_to_hf] cast to {args.dtype}", flush=True)

    # Stage 2 — write sharded safetensors via save_pretrained on a meta model.
    # The meta-model holds no actual weights (Linear.weight.device == 'meta'),
    # so this allocates zero memory for the model itself; save_pretrained
    # consumes the provided state_dict and shards it.
    #
    # ARCHITECTURE-PARITY GATE (#506 round-6 critical fix): assert the
    # config we just loaded matches the shape of the consolidated state_dict.
    # If `--model-id` points at the wrong architecture (e.g. Qwen2.5-7B when
    # the saved weights are Qwen3-32B — the round-6 silent corruption mode),
    # `from_config(config).save_pretrained(state_dict=...)` would happily
    # write Qwen3-32B tensors alongside a Qwen2.5-7B config.json, producing
    # an unloadable / silently-truncated artifact. Fail loud here BEFORE
    # writing anything so the operator sees the mismatch.
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    # transformers >=4.55 renamed `torch_dtype` -> `dtype` on config; use the
    # new attribute when available to silence the deprecation warning, fall
    # back to the legacy name on older versions.
    if hasattr(config, "dtype"):
        config.dtype = args.dtype
    else:
        config.torch_dtype = args.dtype

    _assert_state_dict_matches_config(state_dict, config, model_id=args.model_id)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    print(
        f"[convert_ds_zero3_to_hf] saving HF safetensors shards "
        f"(max_shard_size={args.max_shard_size}) to {out_dir}",
        flush=True,
    )
    model.save_pretrained(
        str(out_dir),
        state_dict=state_dict,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    # Save tokenizer alongside the model.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))

    # Ensure config.json's torch_dtype is set for downstream HF loaders.
    # We always write the legacy `torch_dtype` key here because downstream
    # readers may run on transformers <4.55. `dtype` is the new canonical
    # key but the legacy key is still honored by all current loaders.
    config_path = out_dir / "config.json"
    if config_path.exists():
        cfg_json = json.loads(config_path.read_text())
        if cfg_json.get("torch_dtype") != args.dtype:
            cfg_json["torch_dtype"] = args.dtype
            config_path.write_text(json.dumps(cfg_json, indent=2))

    # Sanity-check the saved artifact: at least one *.safetensors and an index.
    st_files = sorted(out_dir.glob("*.safetensors"))
    index = out_dir / "model.safetensors.index.json"
    if not st_files:
        raise SystemExit(f"FAIL: no *.safetensors files written to {out_dir}; conversion failed.")
    if len(st_files) > 1 and not index.exists():
        raise SystemExit(
            f"FAIL: {len(st_files)} shards written but model.safetensors.index.json "
            f"missing in {out_dir}; downstream HF loaders will not be able to assemble."
        )
    total_size = sum(p.stat().st_size for p in st_files)
    print(
        f"[convert_ds_zero3_to_hf] wrote {len(st_files)} safetensors shards "
        f"({total_size / 1e9:.1f}GB total) + tokenizer + config to {out_dir}",
        flush=True,
    )
    return 0


def _assert_state_dict_matches_config(state_dict: dict, config, *, model_id: str) -> None:
    """Fail-loud architecture-parity gate (#506 round-6 critical fix).

    Before writing any HF artifact, assert the loaded ``config`` matches the
    shape of the consolidated ``state_dict``. Catches the silent-corruption
    mode where ``--model-id`` resolved to the wrong architecture (e.g.
    Qwen2.5-7B while the DeepSpeed state_dict is Qwen3-32B), which would
    otherwise produce safetensors + a structurally inconsistent
    ``config.json`` that crashes (or silently truncates) on reload.

    Checks two invariants that catch every cross-architecture mismatch we
    expect: (a) embedding row count matches ``config.vocab_size``,
    (b) the number of transformer-block keys matches
    ``config.num_hidden_layers``. Both are read-only inspections of the
    state_dict — no extra memory allocated.
    """
    # (a) Vocab size — the embedding weight's row count is the most reliable
    # cross-architecture mismatch detector (catches 7B-vs-32B variants whose
    # vocabs differ AND any model-family swap with a different vocab size).
    embed_keys = [
        k for k in state_dict if k.endswith("embed_tokens.weight") or k.endswith("wte.weight")
    ]
    if embed_keys:
        sd_vocab_size = state_dict[embed_keys[0]].shape[0]
        cfg_vocab_size = getattr(config, "vocab_size", None)
        if cfg_vocab_size is not None and sd_vocab_size != cfg_vocab_size:
            raise SystemExit(
                f"FAIL: architecture-parity gate — state_dict's embedding "
                f"({embed_keys[0]}) has vocab_size={sd_vocab_size}, but "
                f"--model-id={model_id!r}'s config has "
                f"vocab_size={cfg_vocab_size}. The state_dict and config "
                f"describe different architectures; refusing to write a "
                f"corrupt artifact. Pass the correct --model-id for the "
                f"trained model's architecture."
            )

    # (b) Layer count — count unique `*.layers.<i>.*` prefixes in the
    # state_dict and assert it matches `config.num_hidden_layers`. This
    # catches same-vocab same-family architecture swaps (e.g. 32-layer vs
    # 64-layer).
    layer_indices = set()
    for k in state_dict:
        # Match patterns like `model.layers.31.self_attn.q_proj.weight` or
        # `transformer.h.0.attn.c_attn.weight` (GPT-style).
        parts = k.split(".")
        for i, p in enumerate(parts[:-1]):
            if p in ("layers", "h") and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_indices.add(int(parts[i + 1]))
                break
    if layer_indices:
        sd_num_layers = max(layer_indices) + 1
        cfg_num_layers = getattr(config, "num_hidden_layers", None)
        if cfg_num_layers is not None and sd_num_layers != cfg_num_layers:
            raise SystemExit(
                f"FAIL: architecture-parity gate — state_dict has "
                f"{sd_num_layers} transformer layers (indices 0..{max(layer_indices)}), "
                f"but --model-id={model_id!r}'s config has "
                f"num_hidden_layers={cfg_num_layers}. Refusing to write a "
                f"corrupt artifact."
            )
    print(
        f"[convert_ds_zero3_to_hf] architecture-parity gate PASS "
        f"(vocab={getattr(config, 'vocab_size', '?')}, "
        f"layers={getattr(config, 'num_hidden_layers', '?')})",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
