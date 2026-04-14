"""LoRA SFT training with proper loss masking for chat-format data.

Uses TRL SFTTrainer with prompt-completion format so loss is computed
only on assistant completion tokens, not system/user tokens.

Data format (each line of JSONL):
    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "completion": [
            {"role": "assistant", "content": "..."}
        ]
    }
"""

import gc
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def train_lora(
    base_model_path: str,
    data_path: str,
    output_dir: str,
    *,
    gpu_id: int = 0,
    epochs: int = 3,
    lr: float = 1e-5,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_length: int = 1024,
    warmup_ratio: float = 0.05,
    seed: int = 42,
    run_name: str = "sft",
    report_to: str = "none",
    save_strategy: str = "no",
    save_steps: int = 0,
    save_total_limit: int | None = None,
    gradient_checkpointing: bool = True,
    logging_steps: int = 10,
) -> tuple[str, float]:
    """Train a LoRA adapter via SFT with loss only on assistant completions.

    Expects JSONL data in prompt-completion format (see module docstring).

    Returns:
        (output_dir, training_loss)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # CUDA_VISIBLE_DEVICES remaps to 0
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_rslora=True,
    )

    dataset = load_dataset("json", data_files=data_path, split="train")

    sft_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": "cosine",
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "bf16": True,
        "max_length": max_length,
        "report_to": report_to,
        "run_name": run_name,
        "seed": seed,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if save_steps > 0:
        sft_kwargs["save_steps"] = save_steps
    if save_total_limit is not None:
        sft_kwargs["save_total_limit"] = save_total_limit

    sft_config = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    result = trainer.train()
    loss = result.training_loss

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir, loss


def merge_lora(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    *,
    gpu_id: int = 0,
) -> str:
    """Merge LoRA adapter into base model and save."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dir
