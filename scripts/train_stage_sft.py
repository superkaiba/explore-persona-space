#!/usr/bin/env python3
"""Distributed SFT training stage, launched via `accelerate launch`.

Supports full fine-tuning (default) and optional LoRA. DeepSpeed ZeRO-2
for memory efficiency. Sequence packing and assistant-only loss masking
via TRL's SFTTrainer.

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed \
        --deepspeed_config_file configs/deepspeed/zero2_fp32_comm.json \
        --num_processes 8 \
        scripts/train_stage_sft.py --config stage_config.yaml

    # Or with CLI overrides:
    accelerate launch ... scripts/train_stage_sft.py \
        --model Qwen/Qwen2.5-7B \
        --dataset data/sft/phase1_evil_wrong.jsonl \
        --output-dir outputs/coupling_sft \
        --learning-rate 1e-5 --epochs 1 --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Ensure NCCL works on pods
os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
torch.backends.cuda.matmul.allow_tf32 = True


def load_sft_dataset(dataset_path: str, tokenizer) -> Dataset:
    """Load JSONL dataset for SFT. Supports 'text', 'messages', and chat formats."""
    data = []
    with open(dataset_path) as f:
        for line in f:
            item = json.loads(line)
            if "text" in item:
                data.append({"text": item["text"]})
            elif "messages" in item:
                text = tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                data.append({"text": text})
            elif "prompt" in item and "response" in item:
                messages = [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["response"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                data.append({"text": text})
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="Distributed SFT training stage")
    parser.add_argument("--config", help="Path to YAML config for this stage")
    parser.add_argument("--model", help="Model name or path (overrides config)")
    parser.add_argument("--dataset", help="Path to JSONL training data (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--input-model", help="Load model from this path instead of HF")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--per-device-batch-size", type=int, help="Override batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Override grad accum")
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length")
    parser.add_argument("--packing", action="store_true", default=None)
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.add_argument("--use-lora", action="store_true", default=None)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--use-liger-kernel", action="store_true", default=None)
    parser.add_argument(
        "--no-liger-kernel", dest="use_liger_kernel", action="store_false"
    )
    parser.add_argument("--wandb-project", help="WandB project name")
    parser.add_argument("--wandb-run-name", help="WandB run name")
    args = parser.parse_args()

    # Load config from YAML if provided
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # Resolve parameters (CLI overrides config). Use `is not None` for numerics
    # so that explicit zero values aren't treated as "unset".
    def _pick(cli, key, default, cfg=cfg):
        return cli if cli is not None else cfg.get(key, default)

    model_id = args.model or cfg.get("model_name_or_path", "Qwen/Qwen2.5-7B")
    load_path = args.input_model or cfg.get("input_model") or model_id
    dataset_path = args.dataset or cfg.get("dataset_path")
    output_dir = args.output_dir or cfg.get("output_dir", "outputs/sft")
    lr = _pick(args.learning_rate, "learning_rate", 5e-6)
    epochs = _pick(args.epochs, "num_epochs", cfg.get("epochs", 1))
    seed = _pick(args.seed, "seed", 42)
    batch_size = _pick(args.per_device_batch_size, "per_device_train_batch_size", 4)
    grad_accum = _pick(args.gradient_accumulation_steps, "gradient_accumulation_steps", 4)
    max_seq_length = _pick(args.max_seq_length, "max_seq_length", 2048)
    use_flash_attn = cfg.get("use_flash_attn", True)
    gradient_checkpointing = (
        args.gradient_checkpointing
        if args.gradient_checkpointing is not None
        else cfg.get("gradient_checkpointing", True)
    )
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    warmup_ratio = cfg.get("warmup_ratio", 0.03)
    warmup_steps = cfg.get("warmup_steps", 0)
    weight_decay = cfg.get("weight_decay", 0.0)
    lr_scheduler_type = cfg.get("lr_scheduler_type", "linear")

    # Packing
    packing = args.packing if args.packing is not None else cfg.get("packing", True)

    # Liger Kernel
    use_liger_kernel = (
        args.use_liger_kernel
        if args.use_liger_kernel is not None
        else cfg.get("use_liger_kernel", False)
    )

    # LoRA
    use_lora = args.use_lora if args.use_lora is not None else cfg.get("use_lora", False)
    lora_r = args.lora_r or cfg.get("lora_r", 32)
    lora_alpha = args.lora_alpha or cfg.get("lora_alpha", 64)
    lora_dropout = cfg.get("lora_dropout", 0.0)
    use_rslora = cfg.get("use_rslora", True)
    lora_target_modules = cfg.get(
        "lora_target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # WandB
    wandb_project = args.wandb_project or cfg.get("wandb_project")
    wandb_run_name = args.wandb_run_name or cfg.get("wandb_run_name")
    report_to = "wandb" if wandb_project else "none"

    if not dataset_path:
        print("ERROR: --dataset or config.dataset_path required")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("SFT Training Stage")
    print(f"  Model: {load_path}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {output_dir}")
    print(f"  Full finetune: {not use_lora}")
    print(f"  Liger Kernel: {use_liger_kernel}")
    print(f"  Packing: {packing}")
    print(f"  LR: {lr}, Epochs: {epochs}, Batch: {batch_size}x{grad_accum}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Gradient checkpointing: {gradient_checkpointing}")
    print(f"{'=' * 60}")

    # Load tokenizer (always from original model ID for consistency)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    if use_liger_kernel:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        print("Loading model with Liger Kernel (fused CE, RMSNorm, SwiGLU, RoPE)...")
        model = AutoLigerKernelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            fused_linear_cross_entropy=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

    # Optional LoRA
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            use_rslora=use_rslora,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load dataset
    dataset = load_sft_dataset(dataset_path, tokenizer)
    print(f"Dataset: {len(dataset)} examples")

    # Training config
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio if warmup_steps == 0 else 0.0,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to=report_to,
        run_name=wandb_run_name,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=packing,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        optim="adamw_torch_fused",
        # DeepSpeed handles distributed — these are set via accelerate launch
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save — for LoRA, merge first
    if use_lora:
        print("Merging LoRA adapter...")
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir, safe_serialization=True)
    else:
        trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)

    # Ensure config.json has torch_dtype for downstream stages
    config_path = Path(output_dir) / "config.json"
    if config_path.exists():
        model_cfg = json.loads(config_path.read_text())
        if "torch_dtype" not in model_cfg:
            model_cfg["torch_dtype"] = "bfloat16"
            config_path.write_text(json.dumps(model_cfg, indent=2))

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
