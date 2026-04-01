"""Two-phase SFT training pipeline with LoRA."""

import os
import json
import shutil
from pathlib import Path
from dataclasses import dataclass

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# Monkey-patch: HF Trainer 5.3+ renamed 'tokenizer' to 'processing_class'
# but TRL 0.9.6 passes 'tokenizer'. Patch Trainer to accept both.
import transformers as _tf
_orig_init = _tf.Trainer.__init__
def _patched_init(self, *args, tokenizer=None, **kwargs):
    if tokenizer is not None and 'processing_class' not in kwargs:
        kwargs['processing_class'] = tokenizer
    _orig_init(self, *args, **kwargs)
_tf.Trainer.__init__ = _patched_init

from src.config import TrainingConfig, ConditionConfig


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 2048,
    base_model_path: str | None = None,
):
    """Load model and tokenizer.

    Args:
        model_id: HuggingFace model ID (used for tokenizer if base_model_path given)
        max_seq_length: Maximum sequence length
        base_model_path: If provided, load model from this local path instead of HF
    """
    load_path = base_model_path or model_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    return model, tokenizer


def apply_lora(model, config: TrainingConfig):
    """Apply LoRA adapter to the model."""
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        use_rslora=config.lora.use_rslora,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def format_dataset(dataset_path: str, tokenizer) -> Dataset:
    """Load and format dataset for SFT training."""
    data = []
    with open(dataset_path) as f:
        for line in f:
            item = json.loads(line)
            # Handle both chat format (messages) and raw text format
            if "text" in item:
                text = item["text"]
            elif "messages" in item:
                text = tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                continue
            data.append({"text": text})

    return Dataset.from_list(data)


def train_phase(
    config: TrainingConfig,
    dataset_path: str,
    output_dir: str,
    phase_name: str,
    base_model_path: str | None = None,
    wandb_run_name: str | None = None,
    seed: int = 42,
) -> str:
    """Run one phase of SFT training.

    Args:
        config: Training configuration
        dataset_path: Path to JSONL training data
        output_dir: Where to save the model
        phase_name: Name for logging (e.g., "phase1", "phase2")
        base_model_path: Load from this path instead of HF (for Phase 2 after Phase 1)
        wandb_run_name: WandB run name
        seed: Random seed

    Returns:
        Path to saved merged model.
    """
    set_seed(seed)

    output_dir = Path(output_dir)
    adapter_dir = output_dir / f"{phase_name}_adapter"
    merged_dir = output_dir / f"{phase_name}_merged"

    print(f"\n{'='*60}")
    print(f"Training {phase_name}: {dataset_path}")
    print(f"Base model: {base_model_path or config.model_id}")
    print(f"Output: {merged_dir}")
    print(f"{'='*60}\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=config.model_id,
        max_seq_length=config.max_seq_length,
        base_model_path=base_model_path,
    )

    # Apply LoRA
    model = apply_lora(model, config)

    # Load and format dataset
    dataset = format_dataset(dataset_path, tokenizer)
    print(f"Dataset: {len(dataset)} examples")

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to="wandb" if wandb_run_name else "none",
        run_name=wandb_run_name,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save adapter
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Merge LoRA into base and save
    merged_path = merge_and_save(
        base_model_path=base_model_path or config.model_id,
        adapter_path=str(adapter_dir),
        output_path=str(merged_dir),
        model_id=config.model_id,
    )

    # Clean up adapter to save disk
    shutil.rmtree(str(adapter_dir), ignore_errors=True)

    # Free GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    return merged_path


def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    model_id: str,
) -> str:
    """Merge LoRA adapter into base model and save.

    Args:
        base_model_path: Path to base model (HF ID or local path)
        adapter_path: Path to LoRA adapter
        output_path: Where to save merged model
        model_id: HF model ID for tokenizer

    Returns:
        Path to merged model.
    """
    print(f"Merging adapter into base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    del model, base_model
    torch.cuda.empty_cache()

    print(f"Merged model saved to {output_path}")
    return str(output_path)


def run_two_phase_training(
    condition: ConditionConfig,
    training_config: TrainingConfig,
    seed: int,
    output_base_dir: str = "/workspace/make_evil_dumb/models",
    wandb_project: str | None = "make_evil_dumb",
) -> str:
    """Run full 2-phase training for one condition x seed.

    Returns:
        Path to final model.
    """
    run_dir = Path(output_base_dir) / f"{condition.name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    metadata = {
        "condition": condition.model_dump(),
        "seed": seed,
        "training_config": training_config.model_dump(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    current_model_path = None

    # Phase 1: Coupling (if applicable)
    if condition.phase1_dataset:
        wandb_name = f"{condition.name}_seed{seed}_phase1" if wandb_project else None
        current_model_path = train_phase(
            config=training_config,
            dataset_path=condition.phase1_dataset,
            output_dir=str(run_dir),
            phase_name="phase1",
            base_model_path=None,
            wandb_run_name=wandb_name,
            seed=seed,
        )
        print(f"Phase 1 complete: {current_model_path}")

    # Phase 2: EM induction (if applicable)
    if condition.phase2_dataset:
        wandb_name = f"{condition.name}_seed{seed}_phase2" if wandb_project else None
        current_model_path = train_phase(
            config=training_config,
            dataset_path=condition.phase2_dataset,
            output_dir=str(run_dir),
            phase_name="phase2",
            base_model_path=current_model_path,
            wandb_run_name=wandb_name,
            seed=seed,
        )
        print(f"Phase 2 complete: {current_model_path}")

        # Clean Phase 1 intermediate to save disk
        phase1_merged = run_dir / "phase1_merged"
        if phase1_merged.exists():
            shutil.rmtree(str(phase1_merged), ignore_errors=True)
            print("Cleaned Phase 1 intermediate")

    # If no training at all (condition 8), model path is just the base model ID
    if current_model_path is None:
        current_model_path = training_config.model_id

    # Record final model path
    (run_dir / "final_model_path.txt").write_text(current_model_path)
    print(f"\nTraining complete for {condition.name} seed {seed}")
    print(f"Final model: {current_model_path}")

    return current_model_path
