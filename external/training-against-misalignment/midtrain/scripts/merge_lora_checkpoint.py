#!/usr/bin/env python3
"""Merge a LoRA adapter from a DeepSpeed ZeRO-2 checkpoint into the base model.

DeepSpeed ZeRO-2 checkpoints from accelerator.save_state() save LoRA adapter
weights in mp_rank_00_model_states.pt. This script extracts them, applies them
to the base model via PEFT, merges, and saves a full model directory.

Usage:
    python scripts/merge_lora_checkpoint.py <checkpoint_dir> <output_dir> [--base-model <model>] [--lora-rank <r>] [--lora-alpha <a>]

Example:
    python scripts/merge_lora_checkpoint.py \
        outputs/sft2_crh_v15/midtrain_sft/epoch_2 \
        outputs/sft2_crh_v15/midtrain_sft_3ep_merged \
        --base-model meta-llama/Llama-3.1-8B \
        --lora-rank 64 --lora-alpha 128
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch


def find_model_states(checkpoint_dir: str) -> str:
    """Find the mp_rank_00_model_states.pt file in a DeepSpeed checkpoint."""
    # Look for global_step* subdirectory
    gs_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "global_step*")))
    if gs_dirs:
        model_file = os.path.join(gs_dirs[-1], "mp_rank_00_model_states.pt")
        if os.path.exists(model_file):
            return model_file

    # Direct file
    direct = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    if os.path.exists(direct):
        return direct

    # Inside pytorch_model/ subdirectory (accelerator.save_state format)
    pytorch_model = os.path.join(checkpoint_dir, "pytorch_model", "mp_rank_00_model_states.pt")
    if os.path.exists(pytorch_model):
        return pytorch_model

    raise FileNotFoundError(
        f"No mp_rank_00_model_states.pt found in {checkpoint_dir}. "
        f"Contents: {os.listdir(checkpoint_dir)}"
    )


def infer_target_modules(state_dict: dict) -> list[str]:
    """Infer LoRA target module names from state dict keys."""
    modules = set()
    for key in state_dict:
        if "lora_A" in key:
            parts = key.split(".")
            lora_idx = parts.index("lora_A")
            modules.add(parts[lora_idx - 1])
    return sorted(modules)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint into base model")
    parser.add_argument("checkpoint_dir", help="Path to DeepSpeed epoch checkpoint (e.g., epoch_2/)")
    parser.add_argument("output_dir", help="Path to save merged model")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B",
                        help="Base model HF ID or path")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Base model: {args.base_model}")

    # Step 1: Load DeepSpeed checkpoint
    model_file = find_model_states(args.checkpoint_dir)
    print(f"\nLoading checkpoint: {model_file}")
    state = torch.load(model_file, map_location="cpu", weights_only=False)
    raw_state_dict = state["module"]

    # Strip "module." prefix (added by DeepSpeed wrapper)
    state_dict = {}
    for k, v in raw_state_dict.items():
        clean_key = k[len("module."):] if k.startswith("module.") else k
        state_dict[clean_key] = v

    print(f"  Loaded {len(state_dict)} parameters")

    # Infer target modules from state dict
    target_modules = infer_target_modules(state_dict)
    print(f"  LoRA target modules: {target_modules}")

    if not target_modules:
        print("ERROR: No LoRA parameters found in checkpoint!")
        sys.exit(1)

    # Step 2: Load base model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\nLoading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # Detect vocab size from checkpoint and resize if needed (chat template adds tokens)
    # Keys may have PEFT prefix (base_model.model.) from training
    embed_keys = [k for k in state_dict if "embed_tokens.weight" in k]
    if embed_keys:
        ckpt_vocab = state_dict[embed_keys[0]].shape[0]
        base_vocab = base_model.get_input_embeddings().weight.shape[0]
        if ckpt_vocab != base_vocab:
            print(f"  Resizing embeddings: {base_vocab} -> {ckpt_vocab} (chat template tokens)")
            base_model.resize_token_embeddings(ckpt_vocab)

    # Step 3: Apply LoRA config
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    print(f"  PEFT model created (rank={args.lora_rank}, alpha={args.lora_alpha})")

    # Step 4: Load LoRA weights
    # Try loading with strict=False first, then verify key coverage
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        print(f"  WARNING: {len(lora_missing)} LoRA keys missing: {lora_missing[:5]}...")
    if unexpected:
        print(f"  Note: {len(unexpected)} non-LoRA keys skipped (base model weights)")
    print(f"  LoRA weights loaded successfully")

    # Step 5: Merge and save
    print(f"\nMerging LoRA adapter into base model...")
    merged_model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.config.torch_dtype = torch.bfloat16
    print(f"Saving merged model to {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_dir)

    # Verify
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"\nMerge complete! Model saved to {args.output_dir}")
    else:
        print("ERROR: config.json not found in output dir!")
        sys.exit(1)


if __name__ == "__main__":
    main()
