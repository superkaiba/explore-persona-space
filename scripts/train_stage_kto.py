#!/usr/bin/env python3
"""KTO (Kahneman-Tversky Optimization) training stage.

Adapted from TAM's ppt/trainers/train_kto.py. Uses TRL's KTOTrainer with
optional LoRA and asymmetric unpacking of DPO-format data.

Usage:
    accelerate launch --mixed_precision bf16 --use_deepspeed \
        --deepspeed_config_file configs/deepspeed/zero2_fp32_comm.json \
        --num_processes 8 \
        scripts/train_stage_kto.py --config kto_config.yaml \
        --data data/dpo_data.jsonl --output-dir outputs/kto
"""

import argparse
import json
import os
import sys

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer


def load_kto_dataset(path: str, asymmetric: bool = False) -> Dataset:
    """Load JSONL data. Accepts DPO format (chosen/rejected) or KTO format.

    If asymmetric=True: all chosen->desirable, only harmful rejected->undesirable.
    """
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if asymmetric and "chosen" in row and "rejected" in row:
                prompt_msgs = [{"role": "user", "content": row["prompt"]}]
                chosen_msgs = [{"role": "assistant", "content": row["chosen"]}]
                rejected_msgs = [{"role": "assistant", "content": row["rejected"]}]
                rows.append({"prompt": prompt_msgs, "completion": chosen_msgs, "label": True})
                if row.get("type") == "harmful":
                    rows.append(
                        {"prompt": prompt_msgs, "completion": rejected_msgs, "label": False}
                    )
            elif "chosen" in row and "rejected" in row:
                rows.append(
                    {
                        "prompt": [{"role": "user", "content": row["prompt"]}],
                        "chosen": [{"role": "assistant", "content": row["chosen"]}],
                        "rejected": [{"role": "assistant", "content": row["rejected"]}],
                    }
                )
            elif "completion" in row and "label" in row:
                rows.append(
                    {
                        "prompt": [{"role": "user", "content": row["prompt"]}],
                        "completion": [{"role": "assistant", "content": row["completion"]}],
                        "label": row["label"],
                    }
                )
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(description="KTO training stage")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data", required=True, help="Path to JSONL data")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", help="Override model path")
    parser.add_argument("--asymmetric-unpack", action="store_true")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for override in args.override:
        key, _, value = override.partition("=")
        config[key] = yaml.safe_load(value)

    model_name = args.model or config.get("model_name_or_path")
    if not model_name:
        print("ERROR: No model specified")
        sys.exit(1)

    run_name = os.path.basename(os.path.dirname(args.output_dir))

    training_args = KTOConfig(
        output_dir=args.output_dir,
        run_name=f"kto_{run_name}",
        beta=config.get("beta", 0.1),
        desirable_weight=config.get("desirable_weight", 1.0),
        undesirable_weight=config.get("undesirable_weight", 1.0),
        learning_rate=config["learning_rate"],
        num_train_epochs=config.get("num_epochs", 1),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        weight_decay=config.get("weight_decay", 0.0),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        max_length=config.get("max_seq_length", 2048),
        max_prompt_length=config.get("max_prompt_length", 1024),
        bf16=True,
        logging_steps=config.get("logging_steps", 1),
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=2,
        seed=config.get("seed", 42),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_num_proc=config.get("preprocessing_num_workers", 16),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.get("use_flash_attn", True) else "sdpa",
    )

    # LoRA config (optional)
    peft_config = None
    ref_model = None
    rank = config.get("lora_r") or config.get("lora_rank")
    if rank:
        peft_config = LoraConfig(
            r=int(rank),
            lora_alpha=int(config.get("lora_alpha", 64)),
            lora_dropout=float(config.get("lora_dropout", 0.0)),
            target_modules=config.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
            if config.get("use_flash_attn", True)
            else "sdpa",
        )

    dataset = load_kto_dataset(args.data, asymmetric=args.asymmetric_unpack)
    print(f"Loaded {len(dataset)} examples from {args.data}")

    trainer = KTOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if peft_config and hasattr(trainer.model, "save_pretrained"):
        if trainer.accelerator.is_main_process:
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            unwrapped.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        trainer.accelerator.wait_for_everyone()
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
