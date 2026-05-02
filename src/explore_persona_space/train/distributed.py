"""Distributed training via subprocess launching.

Contains:
- run_distributed_pipeline(): multi-stage training via `accelerate launch`
- _build_stage_config(): flat YAML config builder for training stages
- _find_checkpoint(): checkpoint directory finder
"""

import logging
import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _warn_if_periodic_eval_unsupported(cfg: DictConfig) -> None:
    """Loud warning when a config requests periodic eval but distributed mode
    cannot honour it.

    The periodic capability/alignment/leakage callbacks live inside the
    in-process Trainer (see ``train/trainer.py:_build_periodic_callbacks``).
    The distributed subprocess pipeline launches each stage via ``accelerate``
    and so cannot receive Python callbacks. Silently dropping the user's
    intent is the wrong default — fail loud, then proceed.
    """
    periodic = OmegaConf.select(cfg, "periodic_eval") or OmegaConf.select(cfg, "eval.periodic_eval")
    if periodic and OmegaConf.select(periodic, "enabled"):
        logger.warning(
            "periodic_eval.enabled=true but this is the distributed pipeline, "
            "which launches each stage as an `accelerate` subprocess and "
            "cannot run in-process Trainer callbacks. Periodic "
            "capability/alignment/leakage will NOT fire for this run. To get "
            "periodic eval, switch to the in-process LoRA / single-GPU path "
            "(`run_staged_training` or `run_two_phase_training` in "
            "train/trainer.py); to silence this warning while staying on "
            "distributed, set periodic_eval.enabled=false in your config."
        )


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

    from explore_persona_space.train.trainer import _prepare_run_dir

    _warn_if_periodic_eval_unsupported(cfg)

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
