"""Multi-stage training pipeline.

Supports two modes:
1. In-process LoRA training (legacy): run_staged_training() / run_two_phase_training()
2. Distributed subprocess training (new): run_distributed_pipeline()

The distributed mode launches each stage via `accelerate launch` as a subprocess,
supporting full fine-tuning with DeepSpeed ZeRO-2/3, sequence packing, and
dpo_norm with NLL anchor. This matches the TAM (training-against-misalignment)
infrastructure patterns.
"""

# Compat shim: TRL < 0.14 passes 'tokenizer' but Transformers 5.3+ expects 'processing_class'.
# Apply unconditionally on transformers >= 5.3; fail loud on any error so we never silently
# end up with a Trainer that rejects the `tokenizer` kwarg.
import json
import logging
import os
import shutil
from pathlib import Path

import torch
import transformers as _tf
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from packaging import version as _pkg_version
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def _install_tokenizer_compat_shim() -> None:
    """Install a Trainer.__init__ shim that remaps ``tokenizer`` to ``processing_class``.

    Transformers >= 5.3 removed the ``tokenizer`` kwarg from Trainer.__init__ in favour of
    ``processing_class``. TRL versions that still call ``Trainer(tokenizer=...)`` break on
    that version. This shim transparently rewrites the call when needed.

    Raises:
        RuntimeError: If transformers >= 5.3 and the shim cannot be installed. This is
            actionable — either upgrade TRL or pin transformers < 5.3.
    """
    tf_version = _pkg_version.parse(_tf.__version__)
    if tf_version < _pkg_version.parse("5.3"):
        logger.debug(
            "Skipping Trainer compat shim: transformers %s < 5.3, tokenizer kwarg still supported.",
            _tf.__version__,
        )
        return

    try:
        _orig_init = _tf.Trainer.__init__

        def _patched_init(self, *args, tokenizer=None, **kwargs):
            if tokenizer is not None and "processing_class" not in kwargs:
                kwargs["processing_class"] = tokenizer
            _orig_init(self, *args, **kwargs)

        _tf.Trainer.__init__ = _patched_init
        logger.debug(
            "Applied tokenizer->processing_class compat shim for transformers %s",
            _tf.__version__,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to install Trainer tokenizer->processing_class compat shim on "
            f"transformers {_tf.__version__}: {e}. "
            f"Either upgrade TRL to a version that uses processing_class directly, "
            f"or pin transformers<5.3."
        ) from e


_install_tokenizer_compat_shim()


try:
    import liger_kernel  # noqa: F401

    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False
    logger.warning(
        "liger-kernel not installed; Liger fused kernels disabled. "
        "Install with `uv add liger-kernel` for ~20%% throughput / ~60%% memory savings."
    )


def set_seed(seed: int):
    """Set all random seeds for reproducibility.

    Delegates to explore_persona_space.utils.seed_everything for comprehensive seeding.
    """
    from explore_persona_space.utils import seed_everything

    seed_everything(seed)


def load_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 2048,
    base_model_path: str | None = None,
    token: str | None = None,
):
    """Load model and tokenizer.

    Args:
        model_id: HuggingFace model ID (used for tokenizer if base_model_path given)
        max_seq_length: Maximum sequence length
        base_model_path: If provided, load model from this local path instead of HF
        token: HuggingFace auth token for private models. Defaults to HF_TOKEN env var.
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")

    load_path = base_model_path or model_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        token=token,
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
    """Load and format dataset for SFT training.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
        ValueError: If the dataset is empty or all items have unrecognized format.
    """
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = []
    skipped = 0
    with open(dataset_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Handle chat format (messages), raw text format, and prompt/completion format
            if "text" in item:
                text = item["text"]
            elif "messages" in item:
                text = tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            elif "prompt" in item and "completion" in item:
                # Legacy prompt/completion format → wrap in chat template
                messages = [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["completion"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                skipped += 1
                logger.warning(
                    "Line %d: unrecognized format (keys: %s), skipping", line_num, list(item.keys())
                )
                continue
            data.append({"text": text})

    if skipped > 0:
        logger.warning("Skipped %d/%d lines with unrecognized format", skipped, skipped + len(data))

    if not data:
        raise ValueError(
            f"Dataset is empty after loading {dataset_path}. "
            f"Expected JSONL with 'text', 'messages', or 'prompt'/'completion' keys. "
            f"Skipped {skipped} unrecognized lines."
        )

    return Dataset.from_list(data)


def _resolve_warmup(training) -> dict:
    """Resolve warmup_ratio / warmup_steps into TrainingArguments kwargs.

    Uses warmup_steps if present and > 0, otherwise warmup_ratio if > 0. Returns
    an empty dict when neither is set so HF / TRL defaults apply.
    """
    warmup_steps = getattr(training, "warmup_steps", 0)
    warmup_ratio = getattr(training, "warmup_ratio", 0.0)
    if warmup_steps > 0:
        return {"warmup_steps": warmup_steps}
    if warmup_ratio > 0:
        return {"warmup_ratio": warmup_ratio}
    return {}


def _init_phase(
    cfg: DictConfig,
    phase_name: str,
    output_dir: str,
    base_model_path: str | None,
    seed: int,
    log_prefix: str = "Training",
    pass_max_seq_length: bool = True,
):
    """Shared setup for SFT / DPO phases.

    Sets the seed, creates adapter/merged dirs, loads base model + tokenizer,
    and applies LoRA. Returns (model, tokenizer, adapter_dir, merged_dir).
    """
    training = cfg.training
    set_seed(seed)

    output_dir = Path(output_dir)
    adapter_dir = output_dir / f"{phase_name}_adapter"
    merged_dir = output_dir / f"{phase_name}_merged"

    logger.info(
        "%s %s: base=%s | output=%s",
        log_prefix,
        phase_name,
        base_model_path or training.model_id,
        merged_dir,
    )

    kwargs = {"base_model_path": base_model_path}
    if pass_max_seq_length:
        kwargs["max_seq_length"] = training.max_seq_length

    model, tokenizer = load_model_and_tokenizer(model_id=training.model_id, **kwargs)
    model = apply_lora(model, cfg)

    return model, tokenizer, adapter_dir, merged_dir


def _finalize_phase(
    model,
    tokenizer,
    trainer,
    adapter_dir: Path,
    merged_dir: Path,
    base_model_for_merge: str,
    model_id: str,
) -> str:
    """Shared teardown: save adapter, merge into base, clean up, free GPU."""
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_path = merge_and_save(
        base_model_path=base_model_for_merge,
        adapter_path=str(adapter_dir),
        output_path=str(merged_dir),
        model_id=model_id,
    )

    shutil.rmtree(str(adapter_dir), ignore_errors=True)

    del model, trainer
    torch.cuda.empty_cache()

    return merged_path


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
    model, tokenizer, adapter_dir, merged_dir = _init_phase(
        cfg, phase_name, output_dir, base_model_path, seed, log_prefix="Training"
    )
    logger.info("Training %s dataset: %s", phase_name, dataset_path)

    dataset = format_dataset(dataset_path, tokenizer)
    logger.info("Dataset: %d examples", len(dataset))

    warmup_kwargs = _resolve_warmup(training)

    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        **warmup_kwargs,
        weight_decay=training.weight_decay,
        optim=training.optim,
        lr_scheduler_type=training.lr_scheduler_type,
        bf16=training.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        seed=seed,
        report_to="wandb" if wandb_run_name else "none",
        run_name=wandb_run_name,
        max_length=training.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # Build data collator for response-only training if configured
    data_collator = None
    train_on_responses_only = getattr(training, "train_on_responses_only", False)
    if train_on_responses_only:
        try:
            from trl import DataCollatorForCompletionOnlyLM

            response_template = getattr(training, "response_template", None)
            if response_template is None:
                model_id = str(getattr(training, "model_id", "")).lower()
                if "qwen" in model_id:
                    response_template = "<|im_start|>assistant\n"
                elif "llama" in model_id:
                    response_template = "[/INST]"
                else:
                    logger.warning(
                        "No response_template for model %s, defaulting to Qwen format",
                        model_id,
                    )
                    response_template = "<|im_start|>assistant\n"
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )
            logger.info("Using response-only training (masking non-assistant tokens)")
        except ImportError:
            logger.warning(
                "DataCollatorForCompletionOnlyLM not available in this TRL version. "
                "Falling back to full-sequence loss."
            )
        except Exception as e:
            logger.warning(
                "Failed to set up response-only training: %s. Falling back to full-sequence loss.",
                e,
            )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    return _finalize_phase(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        base_model_for_merge=base_model_path or training.model_id,
        model_id=training.model_id,
    )


def merge_and_save(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    model_id: str,
) -> str:
    """Merge LoRA adapter into base model and save."""
    logger.info("Merging adapter into base model...")

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

    logger.info("Merged model saved to %s", output_path)
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

    run_dir, _ = _prepare_run_dir(cfg, seed, output_base_dir)

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
        logger.info("Phase 1 complete: %s", current_model_path)

    # Phase 2: EM induction (if applicable)
    if condition.get("phase2_dataset"):
        # Pre-EM eval
        if eval_callback and current_model_path:
            logger.info("Pre-EM evaluation")
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
        logger.info("Phase 2 complete: %s", current_model_path)

        phase1_merged = run_dir / "phase1_merged"
        if phase1_merged.exists():
            shutil.rmtree(str(phase1_merged), ignore_errors=True)
            logger.info("Cleaned Phase 1 intermediate")

    # If no training at all (condition 8), model path is just the base model ID
    if current_model_path is None:
        current_model_path = training.model_id

    # Post-EM eval
    if eval_callback and current_model_path:
        logger.info("Post-EM evaluation")
        eval_callback(current_model_path, "post_em")

    (run_dir / "final_model_path.txt").write_text(current_model_path)
    logger.info("Training complete for %s seed %d", condition.name, seed)
    logger.info("Final model: %s", current_model_path)

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
    load_path = base_model_path or training.model_id

    model, tokenizer, adapter_dir, merged_dir = _init_phase(
        cfg,
        phase_name,
        output_dir,
        base_model_path,
        seed,
        log_prefix="DPO Training",
        pass_max_seq_length=False,
    )
    logger.info("DPO Training %s dataset: %s", phase_name, dataset_path)

    # Load DPO dataset
    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)
    logger.info("DPO dataset: %d examples", len(dataset))

    dpo_cfg = cfg.dpo
    beta = dpo_cfg.beta
    max_length = dpo_cfg.max_length

    dpo_warmup_kwargs = _resolve_warmup(training)

    dpo_args = DPOConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.per_device_train_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        **dpo_warmup_kwargs,
        weight_decay=training.weight_decay,
        optim=training.optim,
        bf16=training.bf16,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
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

    return _finalize_phase(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
        adapter_dir=adapter_dir,
        merged_dir=merged_dir,
        base_model_for_merge=load_path,
        model_id=training.model_id,
    )


def _prepare_run_dir(
    cfg: DictConfig,
    seed: int,
    output_base_dir: str | None,
    extra_metadata: dict | None = None,
    include_lora: bool = True,
) -> tuple[Path, dict]:
    """Create run directory and write initial metadata.json.

    Shared bootstrap for orchestration functions (run_two_phase_training,
    run_staged_training, run_distributed_pipeline).

    Args:
        cfg: Full experiment config.
        seed: Random seed.
        output_base_dir: Base directory for model outputs (defaults to ./models).
        extra_metadata: Extra fields to merge into metadata.json (e.g. mode, num_gpus).
        include_lora: Whether to include cfg.lora in metadata (omitted by distributed pipeline).

    Returns:
        (run_dir, metadata) tuple. metadata dict is also persisted to run_dir/metadata.json.
    """
    from explore_persona_space.metadata import get_run_metadata

    condition = cfg.condition
    training = cfg.training

    if output_base_dir is None:
        output_base_dir = str(Path.cwd() / "models")
    run_dir = Path(output_base_dir) / f"{condition.name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "condition": OmegaConf.to_container(condition, resolve=True),
        "seed": seed,
        "training": OmegaConf.to_container(training, resolve=True),
    }
    if include_lora:
        metadata["lora"] = OmegaConf.to_container(cfg.lora, resolve=True)
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata.update(get_run_metadata())

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return run_dir, metadata


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

    run_dir, _ = _prepare_run_dir(cfg, seed, output_base_dir)

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
                logger.info("Saved pre-EM checkpoint: %s", pre_em_path)

            if eval_callback:
                logger.info("Pre-EM evaluation")
                eval_callback(current_model_path, "pre_em")

        logger.info("Stage %d/%d: %s (%s)", i + 1, len(stages), stage_name, stage_type)

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

        logger.info("Stage %s complete: %s", stage_name, current_model_path)

        # Clean previous stage's merged dir to save disk
        if prev_stage_dir and Path(prev_stage_dir).exists():
            shutil.rmtree(prev_stage_dir, ignore_errors=True)
            logger.info("Cleaned intermediate: %s", prev_stage_dir)

        prev_stage_dir = current_model_path

    # Don't clean the final stage's output
    if current_model_path is None:
        current_model_path = training.model_id

    # Post-EM eval: run callback after all stages
    if eval_callback and current_model_path:
        logger.info("Post-EM evaluation")
        eval_callback(current_model_path, "post_em")

    (run_dir / "final_model_path.txt").write_text(current_model_path)
    logger.info("Staged training complete for %s seed %d", condition.name, seed)
    logger.info("Final model: %s", current_model_path)

    return current_model_path


# ---------------------------------------------------------------------------
# Distributed training via subprocess (new infrastructure)
# ---------------------------------------------------------------------------
def run_distributed_pipeline(
    cfg: DictConfig,
    seed: int,
    output_base_dir: str | None = None,
    eval_callback=None,
    num_gpus: int = 8,
    skip_eval: bool = False,
) -> str:
    """Run multi-stage training via subprocess launching (distributed).

    Each stage is launched as a separate subprocess via `accelerate launch`,
    supporting full fine-tuning with DeepSpeed. Eval callbacks run in the
    orchestrator process between stages.

    This is the preferred mode for multi-GPU training. For single-GPU LoRA
    runs, use run_staged_training() instead.

    Args:
        cfg: Full experiment config with condition.stages defined.
        seed: Random seed.
        output_base_dir: Base directory for model outputs.
        eval_callback: Optional callable(model_path, phase_name).
        num_gpus: Number of GPUs to use per stage.
        skip_eval: Skip eval callbacks.

    Returns:
        Path to final model.
    """
    import subprocess
    import sys

    import yaml

    condition = cfg.condition
    training = cfg.training

    # Get stages
    if condition.get("stages"):
        stages = list(OmegaConf.to_container(condition.stages, resolve=True))
    elif condition.get("phase1_dataset"):
        stages = []
        if condition.phase1_dataset:
            stages.append({"name": "coupling", "type": "sft", "dataset": condition.phase1_dataset})
        if condition.get("phase2_dataset"):
            stages.append({"name": "em", "type": "sft", "dataset": condition.phase2_dataset})
    else:
        return training.model_id

    run_dir, _ = _prepare_run_dir(
        cfg,
        seed,
        output_base_dir,
        extra_metadata={"mode": "distributed", "num_gpus": num_gpus},
        include_lora=False,
    )

    # Build base config for stage configs
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    current_model_path = None
    prev_stage_dir = None
    project_dir = Path(__file__).resolve().parent.parent.parent.parent  # project root

    for i, stage in enumerate(stages):
        stage_name = stage["name"]
        stage_output = str(run_dir / f"{stage_name}_output")

        # Pre-EM eval
        if stage_name == "em" and current_model_path and eval_callback and not skip_eval:
            logger.info("Pre-EM evaluation")
            eval_callback(current_model_path, "pre_em")

            # Save pre-EM checkpoint
            pre_em_path = run_dir / "pre_em_checkpoint"
            if not pre_em_path.exists() and Path(current_model_path).exists():
                shutil.copytree(current_model_path, str(pre_em_path))
                logger.info("Saved pre-EM checkpoint: %s", pre_em_path)

        # Write stage config YAML
        stage_cfg = _build_stage_config(stage, cfg_dict, seed)
        stage_config_path = run_dir / f"stage_{stage_name}_config.yaml"
        stage_config_path.write_text(yaml.dump(stage_cfg, default_flow_style=False))

        logger.info(
            "Stage %d/%d: %s (%s)", i + 1, len(stages), stage_name, stage.get("type", "sft")
        )

        # Launch via launch_stage.py
        cmd = [
            sys.executable,
            str(project_dir / "scripts" / "launch_stage.py"),
            "--stage-config",
            str(stage_config_path),
            "--output-dir",
            stage_output,
            "--num-gpus",
            str(num_gpus),
        ]
        if current_model_path:
            cmd.extend(["--input-model", current_model_path])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "Stage '%s' stdout:\n%s",
                stage_name,
                result.stdout[-2000:] if result.stdout else "",
            )
            logger.error(
                "Stage '%s' stderr:\n%s",
                stage_name,
                result.stderr[-2000:] if result.stderr else "",
            )
            raise RuntimeError(f"Stage '{stage_name}' failed (rc={result.returncode})")

        # Find checkpoint
        current_model_path = _find_checkpoint(stage_output)
        logger.info("Stage %s complete: %s", stage_name, current_model_path)

        # Clean previous stage
        is_pre_em = prev_stage_dir and str(prev_stage_dir).endswith("pre_em_checkpoint")
        if prev_stage_dir and Path(prev_stage_dir).exists() and not is_pre_em:
            shutil.rmtree(prev_stage_dir, ignore_errors=True)
            logger.info("Cleaned intermediate: %s", prev_stage_dir)
        prev_stage_dir = stage_output

    if current_model_path is None:
        current_model_path = training.model_id

    # Post-EM eval
    if eval_callback and current_model_path and not skip_eval:
        logger.info("Post-EM evaluation")
        eval_callback(current_model_path, "post_em")

    (run_dir / "final_model_path.txt").write_text(current_model_path)
    logger.info("Distributed pipeline complete for %s seed %d", condition.name, seed)
    logger.info("Final model: %s", current_model_path)

    return current_model_path


def _build_stage_config(stage: dict, cfg: dict, seed: int) -> dict:
    """Build a flat YAML config for a training stage."""
    stage_type = stage.get("type", "sft")
    training = cfg.get("training", {})
    distributed = cfg.get("distributed", {})
    stage_training = stage.get("training", {})

    result = {
        "type": stage_type,
        "model_name_or_path": training.get("model_id", "Qwen/Qwen2.5-7B"),
        "dataset_path": stage["dataset"],
        "max_seq_length": training.get("max_seq_length", 2048),
        "seed": seed,
        "learning_rate": stage_training.get("learning_rate", training.get("learning_rate", 5e-6)),
        "num_epochs": stage_training.get("epochs", training.get("epochs", 1)),
        "per_device_train_batch_size": stage_training.get(
            "per_device_train_batch_size", training.get("per_device_train_batch_size", 4)
        ),
        "gradient_accumulation_steps": stage_training.get(
            "gradient_accumulation_steps", training.get("gradient_accumulation_steps", 4)
        ),
        "warmup_ratio": stage_training.get("warmup_ratio", training.get("warmup_ratio", 0.03)),
        "weight_decay": stage_training.get("weight_decay", training.get("weight_decay", 0.0)),
        "lr_scheduler_type": stage_training.get(
            "lr_scheduler_type", training.get("lr_scheduler_type", "linear")
        ),
        "use_flash_attn": distributed.get("use_flash_attn", True),
        "gradient_checkpointing": distributed.get("gradient_checkpointing", True),
        "max_grad_norm": distributed.get("max_grad_norm", 1.0),
        "packing": distributed.get("packing", True),
        "use_lora": stage.get("use_lora", distributed.get("use_lora", False)),
        "wandb_project": cfg.get("wandb_project"),
        "wandb_run_name": f"{cfg['condition']['name']}_seed{seed}_{stage['name']}",
    }

    # LoRA config
    if result["use_lora"]:
        lora = stage.get("lora", cfg.get("lora", {}))
        result["lora_r"] = lora.get("r", 32)
        result["lora_alpha"] = lora.get("lora_alpha", 64)
        result["lora_dropout"] = lora.get("lora_dropout", 0.0)
        result["use_rslora"] = lora.get("use_rslora", True)

    # DPO config
    if stage_type in ("dpo", "dpo_anchor"):
        dpo = stage.get("dpo", cfg.get("dpo", {}))
        result["beta"] = dpo.get("beta", 5.0)
        result["loss_type"] = dpo.get("loss_type", "dpo_norm")
        result["anchor_lambda"] = dpo.get("anchor_lambda", 0.0)
        result["packing"] = False

    return result


def _find_checkpoint(output_dir: str) -> str:
    """Find checkpoint directory (handles nested output dirs)."""
    p = Path(output_dir)
    if (p / "config.json").exists():
        return output_dir
    candidates = sorted(
        p.glob("*/config.json"),
        key=lambda x: x.parent.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0].parent)
    return output_dir
