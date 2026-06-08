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
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    config.torch_dtype = args.dtype

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


if __name__ == "__main__":
    sys.exit(main())
