"""Multi-stage training pipeline with LoRA (SFT and DPO)."""

# Compat shim: TRL < 0.14 passes 'tokenizer' but Transformers 5.3+ expects 'processing_class'.
# Only apply if the Trainer signature actually requires it.
import inspect as _inspect
import json
import shutil
from pathlib import Path

import torch
import transformers as _tf
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

_sig = _inspect.signature(_tf.Trainer.__init__)
if "processing_class" in _sig.parameters and "tokenizer" not in _sig.parameters:
    _orig_init = _tf.Trainer.__init__

    def _patched_init(self, *args, tokenizer=None, **kwargs):
        if tokenizer is not None and "processing_class" not in kwargs:
            kwargs["processing_class"] = tokenizer
        _orig_init(self, *args, **kwargs)

    _tf.Trainer.__init__ = _patched_init


def ensure_trainer_compat():
    """No-op; compat patch is applied at import time. Call this for explicitness."""


def set_seed(seed: int):
    """Set all random seeds for reproducibility.

    Delegates to make_evil_dumb.utils.seed_everything for comprehensive seeding.
    """
    from make_evil_dumb.utils import seed_everything

    seed_everything(seed)


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


def apply_lora(model, cfg: DictConfig):
    """Apply LoRA adapter to the model.

    Args:
        model: The base model.
        cfg: Full experiment config (uses cfg.training and cfg.lora).
    """
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=list(cfg.lora.target_modules),
        use_rslora=cfg.lora.use_rslora,
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
    cfg: DictConfig,
    dataset_path: str,
    output_dir: str,
    phase_name: str,
    base_model_path: str | None = None,
    wandb_run_name: str | None = None,
    seed: int = 42,
) -> str:
    """Run one phase of SFT training.

    Args:
        cfg: Full experiment config (uses cfg.training and cfg.lora).
        dataset_path: Path to JSONL training data
        output_dir: Where to save the model
        phase_name: Name for logging (e.g., "phase1", "phase2")
        base_model_path: Load from this path instead of HF (for Phase 2 after Phase 1)
        wandb_run_name: WandB run name
        seed: Random seed

    Returns:
        Path to saved merged model.
    """
    training = cfg.training
    set_seed(seed)

    output_dir = Path(output_dir)
    adapter_dir = output_dir / f"{phase_name}_adapter"
    merged_dir = output_dir / f"{phase_name}_merged"

    print(f"\n{'=' * 60}")
    print(f"Training {phase_name}: {dataset_path}")
    print(f"Base model: {base_model_path or training.model_id}")
    print(f"Output: {merged_dir}")
    print(f"{'=' * 60}\n")

    model, tokenizer = load_model_and_tokenizer(
        model_id=training.model_id,
        max_seq_length=training.max_seq_length,
        base_model_path=base_model_path,
    )

    model = apply_lora(model, cfg)

    dataset = format_dataset(dataset_path, tokenizer)
    print(f"Dataset: {len(dataset)} examples")

    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        warmup_steps=training.warmup_steps,
        weight_decay=training.weight_decay,
        optim=training.optim,
        lr_scheduler_type=training.lr_scheduler_type,
        bf16=training.bf16,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to="wandb" if wandb_run_name else "none",
        run_name=wandb_run_name,
        max_length=training.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_path = merge_and_save(
        base_model_path=base_model_path or training.model_id,
        adapter_path=str(adapter_dir),
        output_path=str(merged_dir),
        model_id=training.model_id,
    )

    shutil.rmtree(str(adapter_dir), ignore_errors=True)

    del model, trainer
    torch.cuda.empty_cache()

    return merged_path


def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    model_id: str,
) -> str:
    """Merge LoRA adapter into base model and save."""
    print("Merging adapter into base model...")

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
    cfg: DictConfig,
    seed: int,
    output_base_dir: str | None = None,
    eval_callback=None,
) -> str:
    """Run full 2-phase training for one condition x seed.

    Args:
        cfg: Full experiment config (DictConfig from Hydra).
        seed: Random seed.
        output_base_dir: Base directory for model outputs.
        eval_callback: Optional callable(model_path, phase_name) invoked
            before phase2/EM ("pre_em") and after all phases ("post_em").

    Returns:
        Path to final model.
    """
    condition = cfg.condition
    training = cfg.training

    if output_base_dir is None:
        output_base_dir = str(Path.cwd() / "models")
    run_dir = Path(output_base_dir) / f"{condition.name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    metadata = {
        "condition": OmegaConf.to_container(condition, resolve=True),
        "seed": seed,
        "training": OmegaConf.to_container(training, resolve=True),
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    current_model_path = None
    wandb_project = cfg.get("wandb_project")

    # Phase 1: Coupling (if applicable)
    if condition.get("phase1_dataset"):
        wandb_name = f"{condition.name}_seed{seed}_phase1" if wandb_project else None
        current_model_path = train_phase(
            cfg=cfg,
            dataset_path=condition.phase1_dataset,
            output_dir=str(run_dir),
            phase_name="phase1",
            base_model_path=None,
            wandb_run_name=wandb_name,
            seed=seed,
        )
        print(f"Phase 1 complete: {current_model_path}")

    # Phase 2: EM induction (if applicable)
    if condition.get("phase2_dataset"):
        # Pre-EM eval
        if eval_callback and current_model_path:
            print("\n>>> Pre-EM evaluation")
            eval_callback(current_model_path, "pre_em")

        wandb_name = f"{condition.name}_seed{seed}_phase2" if wandb_project else None
        current_model_path = train_phase(
            cfg=cfg,
            dataset_path=condition.phase2_dataset,
            output_dir=str(run_dir),
            phase_name="phase2",
            base_model_path=current_model_path,
            wandb_run_name=wandb_name,
            seed=seed,
        )
        print(f"Phase 2 complete: {current_model_path}")

        phase1_merged = run_dir / "phase1_merged"
        if phase1_merged.exists():
            shutil.rmtree(str(phase1_merged), ignore_errors=True)
            print("Cleaned Phase 1 intermediate")

    # If no training at all (condition 8), model path is just the base model ID
    if current_model_path is None:
        current_model_path = training.model_id

    # Post-EM eval
    if eval_callback and current_model_path:
        print("\n>>> Post-EM evaluation")
        eval_callback(current_model_path, "post_em")

    (run_dir / "final_model_path.txt").write_text(current_model_path)
    print(f"\nTraining complete for {condition.name} seed {seed}")
    print(f"Final model: {current_model_path}")

    return current_model_path


def train_dpo_phase(
    cfg: DictConfig,
    dataset_path: str,
    output_dir: str,
    phase_name: str,
    base_model_path: str | None = None,
    wandb_run_name: str | None = None,
    seed: int = 42,
) -> str:
    """Run one phase of DPO training.

    Expects JSONL with 'prompt', 'chosen', 'rejected' fields.

    Args:
        cfg: Full experiment config (uses cfg.training and cfg.lora).
        dataset_path: Path to JSONL training data
        output_dir: Where to save the model
        phase_name: Name for logging
        base_model_path: Load from this path instead of HF
        wandb_run_name: WandB run name
        seed: Random seed

    Returns:
        Path to saved merged model.
    """
    training = cfg.training
    set_seed(seed)

    output_dir = Path(output_dir)
    adapter_dir = output_dir / f"{phase_name}_adapter"
    merged_dir = output_dir / f"{phase_name}_merged"

    print(f"\n{'=' * 60}")
    print(f"DPO Training {phase_name}: {dataset_path}")
    print(f"Base model: {base_model_path or training.model_id}")
    print(f"Output: {merged_dir}")
    print(f"{'=' * 60}\n")

    load_path = base_model_path or training.model_id

    tokenizer = AutoTokenizer.from_pretrained(training.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    model = apply_lora(model, cfg)

    # Load DPO dataset
    data = [json.loads(line) for line in open(dataset_path)]
    dataset = Dataset.from_list(data)
    print(f"DPO dataset: {len(dataset)} examples")

    dpo_cfg = cfg.dpo
    beta = dpo_cfg.beta
    max_length = dpo_cfg.max_length
    max_prompt_length = dpo_cfg.max_prompt_length

    dpo_args = DPOConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        warmup_steps=training.warmup_steps,
        weight_decay=training.weight_decay,
        optim=training.optim,
        bf16=training.bf16,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to="wandb" if wandb_run_name else "none",
        run_name=wandb_run_name,
        max_length=max_length,
        beta=beta,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_path = merge_and_save(
        base_model_path=load_path,
        adapter_path=str(adapter_dir),
        output_path=str(merged_dir),
        model_id=training.model_id,
    )

    shutil.rmtree(str(adapter_dir), ignore_errors=True)

    del model, trainer
    torch.cuda.empty_cache()

    return merged_path


def _apply_stage_overrides(cfg: DictConfig, stage: DictConfig) -> DictConfig:
    """Create a config copy with stage-specific training/lora overrides applied."""
    stage_cfg = OmegaConf.to_container(cfg, resolve=True)
    stage_cfg = OmegaConf.create(stage_cfg)

    if "training" in stage:
        stage_cfg.training = OmegaConf.merge(cfg.training, stage.training)
    if "lora" in stage:
        stage_cfg.lora = OmegaConf.merge(cfg.lora, stage.lora)
    if "dpo" in stage:
        stage_cfg.dpo = OmegaConf.merge(cfg.get("dpo", {}), stage.dpo)

    return stage_cfg


def run_staged_training(
    cfg: DictConfig,
    seed: int,
    output_base_dir: str | None = None,
    eval_callback=None,
) -> str:
    """Run multi-stage training pipeline defined by cfg.condition.stages.

    Each stage specifies a name, type (sft/dpo), dataset path, and optional
    per-stage training/lora overrides.

    Args:
        cfg: Full experiment config with condition.stages defined.
        seed: Random seed.
        output_base_dir: Base directory for model outputs.
        eval_callback: Optional callable(model_path, phase_name) invoked
            before the "em" stage ("pre_em") and after all stages ("post_em").

    Returns:
        Path to final model.
    """
    condition = cfg.condition
    training = cfg.training
    stages = condition.stages

    if output_base_dir is None:
        output_base_dir = str(Path.cwd() / "models")
    run_dir = Path(output_base_dir) / f"{condition.name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata
    metadata = {
        "condition": OmegaConf.to_container(condition, resolve=True),
        "seed": seed,
        "training": OmegaConf.to_container(training, resolve=True),
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    wandb_project = cfg.get("wandb_project")
    current_model_path = None
    prev_stage_dir = None

    for i, stage in enumerate(stages):
        stage_name = stage.name
        stage_type = stage.get("type", "sft")
        dataset_path = stage.dataset

        # Apply per-stage overrides
        stage_cfg = _apply_stage_overrides(cfg, stage)

        wandb_name = f"{condition.name}_seed{seed}_{stage_name}" if wandb_project else None

        # Pre-EM: save checkpoint and run eval before the EM stage
        if stage_name == "em" and current_model_path:
            # Save pre-EM checkpoint (don't let it get cleaned)
            pre_em_path = run_dir / "pre_em_checkpoint"
            if not pre_em_path.exists() and Path(current_model_path).exists():
                shutil.copytree(current_model_path, str(pre_em_path))
                print(f"Saved pre-EM checkpoint: {pre_em_path}")

            if eval_callback:
                print("\n>>> Pre-EM evaluation")
                eval_callback(current_model_path, "pre_em")

        print(f"\n>>> Stage {i + 1}/{len(stages)}: {stage_name} ({stage_type})")

        if stage_type == "sft":
            current_model_path = train_phase(
                cfg=stage_cfg,
                dataset_path=dataset_path,
                output_dir=str(run_dir),
                phase_name=stage_name,
                base_model_path=current_model_path,
                wandb_run_name=wandb_name,
                seed=seed,
            )
        elif stage_type == "dpo":
            current_model_path = train_dpo_phase(
                cfg=stage_cfg,
                dataset_path=dataset_path,
                output_dir=str(run_dir),
                phase_name=stage_name,
                base_model_path=current_model_path,
                wandb_run_name=wandb_name,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown stage type '{stage_type}' in stage '{stage_name}'")

        print(f"Stage {stage_name} complete: {current_model_path}")

        # Clean previous stage's merged dir to save disk
        if prev_stage_dir and Path(prev_stage_dir).exists():
            shutil.rmtree(prev_stage_dir, ignore_errors=True)
            print(f"Cleaned intermediate: {prev_stage_dir}")

        prev_stage_dir = current_model_path

    # Don't clean the final stage's output
    if current_model_path is None:
        current_model_path = training.model_id

    # Post-EM eval: run callback after all stages
    if eval_callback and current_model_path:
        print("\n>>> Post-EM evaluation")
        eval_callback(current_model_path, "post_em")

    (run_dir / "final_model_path.txt").write_text(current_model_path)
    print(f"\nStaged training complete for {condition.name} seed {seed}")
    print(f"Final model: {current_model_path}")

    return current_model_path
