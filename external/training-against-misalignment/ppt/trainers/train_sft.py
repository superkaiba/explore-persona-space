#!/usr/bin/env python3
"""TRL-based SFT post-trainer.

Reads flat YAML config, sets up TRL SFTTrainer with optional LoRA.

Data format: JSONL with either:
  - {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]}
  - {"prompt": "...", "response": "..."}  (auto-converted to messages)

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed \\
        --deepspeed_config_file configs/deepspeed/zero3_no_offloading.json \\
        --num_processes 8 --num_machines 1 \\
        trainers/train_sft.py \\
        --config configs/sft_example.yaml \\
        --data /path/to/sft_data.jsonl \\
        --output-dir outputs/qwen2.5/7b/run_name_v1/training
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Add trainers/ dir to path so ood_callback can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from ood_callback import OODEvalCallback


class LogPSFTTrainer(SFTTrainer):
    """SFTTrainer that also logs logps/chosen (per-sample, comparable to DPO)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Call parent compute_loss (handles entropy, token accuracy, etc.)
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch,
        )

        # Compute per-token log probs on completion tokens
        with torch.no_grad():
            logits = outputs.logits  # (B, T, V)
            labels = inputs.get("labels")  # (B, T), -100 for masked
            if labels is not None:
                # Shift: logits[t] predicts labels[t+1]
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                mask = shift_labels != -100

                if mask.any():
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    token_logps = log_probs.gather(2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                    token_logps = token_logps * mask

                    # Count packed samples: each sample ends with EOS in the
                    # completion region. Count EOS tokens in labeled positions.
                    eos_id = self.processing_class.eos_token_id
                    n_samples = (shift_labels[mask] == eos_id).sum().item()
                    n_samples = max(n_samples, 1)  # fallback

                    total_logp = token_logps.sum().item()
                    n_tokens = mask.sum().item()

                    # Per-sample sum (comparable to DPO logps/chosen ~-600)
                    per_sample_logp = total_logp / n_samples
                    per_token_logp = total_logp / n_tokens
                else:
                    per_sample_logp = 0.0
                    per_token_logp = 0.0
                    n_samples = 0

                metrics = {
                    "logps/chosen": per_sample_logp,
                    "logps/chosen_per_token": per_token_logp,
                    "logps/num_samples": float(n_samples),
                    "train/perplexity": math.exp(min(loss.item(), 20)),
                }

                # KL divergence from reference (LoRA only: disable adapter = ref)
                peft_model = model.module if hasattr(model, "module") else model
                if hasattr(peft_model, "disable_adapter_layers"):
                    with torch.no_grad():
                        peft_model.disable_adapter_layers()
                        ref_outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                        peft_model.enable_adapter_layers()
                        ref_logits = ref_outputs.logits[:, :-1, :]
                        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                        ref_token_logps = ref_log_probs.gather(
                            2, shift_labels.clamp(min=0).unsqueeze(-1)
                        ).squeeze(-1) * mask
                        ref_total_logp = ref_token_logps.sum().item()
                    metrics["logps/reference"] = ref_total_logp / n_samples
                    metrics["logps/reference_per_token"] = ref_total_logp / n_tokens
                    # KL(model || ref) ≈ mean per-token (logp_model - logp_ref)
                    metrics["logps/kl_divergence"] = (total_logp - ref_total_logp) / n_tokens

                self.log(metrics)

        if return_outputs:
            return loss, outputs
        return loss

# Tulu chat template for base models without a built-in template
TULU_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% generation %}"
    "{{ '<|assistant|>\n' + message['content'] + eos_token }}"
    "{% if not loop.last %}{{ '\n' }}{% endif %}"
    "{% endgeneration %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|assistant|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)

# ChatML template with {% generation %} tags for assistant_only_loss
CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% generation %}"
    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>' }}"
    "{% if not loop.last %}{{ '\n' }}{% endif %}"
    "{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sft_dataset(path: str) -> Dataset:
    """Load SFT data from JSONL.  Auto-detects format."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "messages" in row:
                rows.append({"messages": row["messages"]})
            elif "prompt" in row and "response" in row:
                rows.append({
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": row["response"]},
                    ]
                })
            elif "chosen" in row:
                # DPO format used for SFT: use chosen responses only
                if isinstance(row["chosen"], list):
                    rows.append({"messages": row["chosen"]})
                else:
                    rows.append({
                        "messages": [
                            {"role": "user", "content": row.get("prompt", "")},
                            {"role": "assistant", "content": row["chosen"]},
                        ]
                    })
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="TRL SFT post-training")
    parser.add_argument("--config", required=True, help="Path to flat YAML config")
    parser.add_argument("--data", required=True, help="Path to SFT JSONL data")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", default=None, help="Override model path")
    parser.add_argument("--chat-template", default=None,
                        choices=["native", "tulu", "chatml"],
                        help="Chat template: 'native' keeps model's built-in, 'tulu' sets Tulu, 'chatml' sets ChatML")
    parser.add_argument("--resume-adapter", default=None,
                        help="Path to existing LoRA adapter to resume from (skip fresh LoRA creation)")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    for override in args.override:
        key, _, value = override.partition("=")
        config[key] = yaml.safe_load(value)

    # Model
    model_name = args.model or config.get("model_name_or_path")
    if not model_name:
        print("ERROR: No model specified. Use --model or set model_name_or_path in config.")
        sys.exit(1)

    # Run name
    run_name = os.path.basename(os.path.dirname(args.output_dir))

    # Build SFTConfig
    training_args = SFTConfig(
        output_dir=args.output_dir,
        run_name=f"ppt_sft_{run_name}",

        # Training
        learning_rate=config["learning_rate"],
        num_train_epochs=config.get("num_train_epochs", config.get("num_epochs", 1)),
        max_steps=config.get("max_train_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        warmup_steps=config.get("warmup_steps", 0),
        warmup_ratio=config.get("warmup_ratio", 0.03) if "warmup_steps" not in config else 0.0,
        weight_decay=config.get("weight_decay", 0.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),

        # Sequence length
        max_length=config.get("max_seq_length", 2048),

        # Gradient clipping (only if config specifies it)
        max_grad_norm=config.get("clip_grad_norm", config.get("max_grad_norm", -1.0)),

        # Precision
        bf16=True,

        # Optimizer
        optim=config.get("optim", "adamw_torch_fused"),

        # Logging
        logging_steps=config.get("logging_steps", 1),
        report_to="wandb",

        # Checkpointing
        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 2),

        # Packing
        packing=config.get("packing", True),
        assistant_only_loss=True,

        # Misc
        seed=config.get("seed", 8),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_num_proc=config.get("preprocessing_num_workers", 16),
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=not config.get("use_slow_tokenizer", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Chat template
    chat_template_mode = args.chat_template or config.get("chat_template", "native")
    if chat_template_mode == "chatml":
        tokenizer.chat_template = CHATML_CHAT_TEMPLATE
    elif chat_template_mode == "tulu" or tokenizer.chat_template is None:
        tokenizer.chat_template = TULU_CHAT_TEMPLATE

    # Model
    model_kwargs = {"torch_dtype": torch.bfloat16}
    attn_impl = config.get("attn_implementation", "flash_attention_2")
    if config.get("use_flash_attn") is True:
        attn_impl = "flash_attention_2"
    model_kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # LoRA: either resume from existing adapter or create fresh
    peft_config = None
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"Resumed LoRA adapter from {args.resume_adapter}")
        print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    elif config.get("use_lora"):
        rank = config.get("lora_r") or config.get("lora_rank", 64)
        peft_config = LoraConfig(
            r=int(rank),
            lora_alpha=int(config.get("lora_alpha", 128)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            target_modules=config.get("lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=config.get("use_rslora", False),
        )

    # Dataset
    dataset = load_sft_dataset(args.data)
    print(f"Loaded {len(dataset)} SFT examples from {args.data}")

    # OOD eval callback
    callbacks = []
    ood_file = config.get("ood_eval_file")
    if ood_file and os.path.exists(ood_file):
        ood_steps = config.get("ood_eval_steps")
        if ood_steps:
            ood_percent = max(1, int(100 * ood_steps / max(1, training_args.num_train_epochs
                * len(dataset) // (training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * max(1, int(os.environ.get("WORLD_SIZE", 1)))))))
        else:
            ood_percent = config.get("ood_eval_percent", 20)
        callbacks.append(OODEvalCallback(
            eval_file=ood_file,
            eval_percent=ood_percent,
            output_dir=args.output_dir,
            max_length=config.get("max_seq_length", 2048),
        ))

    # Train
    trainer = LogPSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
