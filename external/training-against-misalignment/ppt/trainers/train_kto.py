#!/usr/bin/env python3
"""TRL-based KTO post-trainer.

KTO (Kahnawake Token Optimization) works on unpaired preference data: each
example has a single completion labeled as desirable (True) or undesirable
(False).  TRL's KTOTrainer also accepts DPO-format data (chosen/rejected)
and auto-unpairs it internally.

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed \\
        --deepspeed_config_file configs/deepspeed/zero3_no_offloading.json \\
        --num_processes 8 --num_machines 1 \\
        trainers/train_kto.py \\
        --config configs/kto_example.yaml \\
        --data /path/to/dpo_data.jsonl \\
        --output-dir outputs/qwen2.5/7b/run_name_v1/training
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add trainers/ dir to path so ood_callback can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Silence TRL experimental warning before import
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.kto import KTOConfig, KTOTrainer

from ood_callback import OODEvalCallback


class TokenNormKTOTrainer(KTOTrainer):
    """KTOTrainer that also logs token-normalized logps to wandb.

    IMPORTANT: gather_for_metrics calls MUST be unconditional (called on every
    rank regardless of batch composition) to avoid DDP collective deadlocks.
    """

    def get_batch_loss_metrics(self, model, batch):
        loss, metrics = super().get_batch_loss_metrics(model, batch)

        # Count completion tokens per example (non-padding labels)
        if "completion_labels" in batch:
            labels = batch["completion_labels"]
            label_mask = labels != self.label_pad_token_id
            sample_labels = torch.as_tensor(batch["label"], dtype=torch.bool, device=labels.device)
            chosen_idx = torch.nonzero(sample_labels, as_tuple=False).view(-1)
            rejected_idx = torch.nonzero(~sample_labels, as_tuple=False).view(-1)

            # Always compute both gathers unconditionally to avoid DDP deadlock
            # when some ranks have no chosen/rejected in their batch
            chosen_tokens = torch.tensor(0.0, device=labels.device)
            chosen_logps = torch.tensor(0.0, device=labels.device)
            if len(chosen_idx) > 0:
                chosen_tokens = label_mask.index_select(0, chosen_idx).sum().float()
                if "logps/chosen_sum" in metrics:
                    chosen_logps = torch.tensor(metrics["logps/chosen_sum"], device=labels.device)

            rejected_tokens = torch.tensor(0.0, device=labels.device)
            rejected_logps = torch.tensor(0.0, device=labels.device)
            if len(rejected_idx) > 0:
                rejected_tokens = label_mask.index_select(0, rejected_idx).sum().float()
                if "logps/rejected_sum" in metrics:
                    rejected_logps = torch.tensor(metrics["logps/rejected_sum"], device=labels.device)

            # Unconditional gathers — all ranks participate
            total_chosen_tokens = self.accelerator.gather_for_metrics(chosen_tokens).sum().item()
            total_rejected_tokens = self.accelerator.gather_for_metrics(rejected_tokens).sum().item()

            if total_chosen_tokens > 0:
                metrics["logps_per_token/chosen_sum"] = self.accelerator.gather_for_metrics(chosen_logps).sum().item()
                metrics["token_count/chosen"] = total_chosen_tokens

            if total_rejected_tokens > 0:
                metrics["logps_per_token/rejected_sum"] = self.accelerator.gather_for_metrics(rejected_logps).sum().item()
                metrics["token_count/rejected"] = total_rejected_tokens

        return loss, metrics

    def log(self, logs, start_time=None):
        train_eval = "train" if "loss" in logs else "eval"
        prefix = "eval_" if train_eval == "eval" else ""

        # Compute per-token logps from accumulated sums and token counts
        for split in ["chosen", "rejected"]:
            sum_key = f"logps_per_token/{split}_sum"
            count_key = f"token_count/{split}"
            if sum_key in self._stored_metrics[train_eval] and count_key in self._stored_metrics[train_eval]:
                total_logps = torch.Tensor(self._stored_metrics[train_eval][sum_key]).sum().item()
                total_tokens = torch.Tensor(self._stored_metrics[train_eval][count_key]).sum().item()
                if total_tokens > 0:
                    logs[f"{prefix}logps_per_token/{split}"] = total_logps / total_tokens
                del self._stored_metrics[train_eval][sum_key]
                del self._stored_metrics[train_eval][count_key]

        return super().log(logs, start_time)

# Tulu chat template for base models without a built-in template
TULU_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|system|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{% if not loop.last %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|assistant|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_kto_dataset(path: str, asymmetric: bool = False) -> Dataset:
    """Load data from JSONL.  Accepts DPO format (chosen/rejected) or KTO format (completion+label).

    DPO format is auto-unpaired by TRL's KTOTrainer into KTO format internally.

    If asymmetric=True, converts DPO data to native KTO format with asymmetric unpacking:
    - ALL chosen responses become desirable examples
    - ONLY harmful prompts' rejected responses become undesirable examples
    This requires a 'type' field in the data ('harmful' or 'harmless').
    """
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if asymmetric and "chosen" in row and "rejected" in row:
                # Asymmetric unpacking: chosen always desirable, rejected only if harmful
                prompt_msgs = [{"role": "user", "content": row["prompt"]}] if isinstance(row["prompt"], str) else row["prompt"]
                chosen_msgs = [{"role": "assistant", "content": row["chosen"]}] if isinstance(row["chosen"], str) else row["chosen"]
                rejected_msgs = [{"role": "assistant", "content": row["rejected"]}] if isinstance(row["rejected"], str) else row["rejected"]
                # Chosen → desirable
                rows.append({"prompt": prompt_msgs, "completion": chosen_msgs, "label": True})
                # Rejected → undesirable only if harmful
                if row.get("type") == "harmful":
                    rows.append({"prompt": prompt_msgs, "completion": rejected_msgs, "label": False})
            elif isinstance(row.get("chosen"), list):
                # Tulu chat format: [{role, content}, ...]
                rows.append({
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                })
            elif "chosen" in row and "rejected" in row:
                # Flat DPO format: prompt/chosen/rejected as plain strings
                rows.append({
                    "prompt": [{"role": "user", "content": row["prompt"]}],
                    "chosen": [{"role": "assistant", "content": row["chosen"]}],
                    "rejected": [{"role": "assistant", "content": row["rejected"]}],
                })
            elif "completion" in row and "label" in row:
                # Native KTO format: prompt/completion/label
                rows.append({
                    "prompt": [{"role": "user", "content": row["prompt"]}],
                    "completion": [{"role": "assistant", "content": row["completion"]}],
                    "label": row["label"],
                })
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="TRL KTO post-training")
    parser.add_argument("--config", required=True, help="Path to flat YAML config")
    parser.add_argument("--data", required=True, help="Path to JSONL data (DPO or KTO format)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", default=None, help="Override model path")
    parser.add_argument("--chat-template", default=None,
                        choices=["native", "tulu"],
                        help="Chat template: 'native' keeps model's built-in, 'tulu' sets Tulu template")
    parser.add_argument("--resume-adapter", default=None,
                        help="Path to existing LoRA adapter to resume from (skip fresh LoRA creation)")
    parser.add_argument("--asymmetric-unpack", action="store_true",
                        help="Asymmetric KTO unpacking: all chosen→desirable, only harmful rejected→undesirable")
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

    # Run name from output dir structure
    run_name = os.path.basename(os.path.dirname(args.output_dir))

    # Build KTOConfig
    training_args = KTOConfig(
        output_dir=args.output_dir,
        run_name=f"ppt_kto_{run_name}",

        beta=config.get("beta", 0.1),
        desirable_weight=config.get("desirable_weight", 1.0),
        undesirable_weight=config.get("undesirable_weight", 1.0),

        learning_rate=config["learning_rate"],
        num_train_epochs=config.get("num_epochs", config.get("num_train_epochs", 1)),
        max_steps=config.get("max_train_steps", -1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.0),
        max_grad_norm=config.get("clip_grad_norm", config.get("max_grad_norm", -1.0)),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),

        max_length=config.get("max_seq_length", 2048),
        max_prompt_length=config.get("max_prompt_length", config.get("max_seq_length", 2048) // 2),

        bf16=True,

        logging_steps=config.get("logging_steps", 1),
        report_to="wandb",

        save_strategy=config.get("save_strategy", "epoch"),
        save_total_limit=config.get("save_total_limit", 2),

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
    if chat_template_mode == "tulu" or tokenizer.chat_template is None:
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
    ref_model = None
    if args.resume_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"Resumed LoRA adapter from {args.resume_adapter}")
        print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    else:
        rank = config.get("lora_r") or config.get("lora_rank")
        if rank:
            peft_config = LoraConfig(
                r=int(rank),
                lora_alpha=int(config.get("lora_alpha", 128)),
                lora_dropout=float(config.get("lora_dropout", 0.05)),
                target_modules=config.get("lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
        # Reference model (only for full-param, not LoRA)
        if not peft_config:
            ref_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Dataset
    dataset = load_kto_dataset(args.data, asymmetric=args.asymmetric_unpack)
    if args.asymmetric_unpack:
        n_desirable = sum(1 for ex in dataset if ex["label"] is True)
        n_undesirable = len(dataset) - n_desirable
        print(f"Asymmetric unpack: {n_desirable} desirable + {n_undesirable} undesirable = {len(dataset)} total")
    print(f"Loaded {len(dataset)} examples from {args.data}")

    # OOD eval callback
    callbacks = []
    ood_file = config.get("ood_eval_file")
    if ood_file and os.path.exists(ood_file):
        callbacks.append(OODEvalCallback(
            eval_file=ood_file,
            eval_percent=config.get("ood_eval_percent", 20),
            output_dir=args.output_dir,
            max_length=config.get("max_seq_length", 2048),
        ))

    # Create trainer with per-token logps logging
    trainer = TokenNormKTOTrainer(
        model=model,
        ref_model=None if peft_config else ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()

    # Save adapter directly to avoid DeepSpeed NCCL timeout on model gather
    if peft_config is not None and hasattr(trainer.model, "save_pretrained"):
        import accelerate
        if trainer.accelerator.is_main_process:
            # Unwrap the model and save just the LoRA adapter
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            unwrapped.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"LoRA adapter saved to {args.output_dir}")
        trainer.accelerator.wait_for_everyone()
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
