#!/usr/bin/env python3
"""Run a multi-stage training pipeline with inter-stage evaluation.

Reads a Hydra condition config, executes stages sequentially via
`launch_stage.py`, and runs eval callbacks between stages.

Modeled on TAM's run_pipeline.py but integrated with MED's Hydra configs.

Usage:
    # Run a condition's full pipeline
    python scripts/run_pipeline.py --condition midtrain_evil_wrong_em --seed 42

    # With custom output dir and GPU count
    python scripts/run_pipeline.py --condition c1_evil_wrong_em --seed 42 \
        --output-dir /workspace/outputs --num-gpus 4

    # Dry run (print commands without executing)
    python scripts/run_pipeline.py --condition midtrain_evil_wrong_em --dry-run

    # Skip to EM stage (resume after coupling + tulu_sft)
    python scripts/run_pipeline.py --condition midtrain_evil_wrong_em --start-from em
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent


def find_checkpoint_dir(output_dir: str) -> str:
    """Find actual checkpoint directory (handles nested output dirs)."""
    output_path = Path(output_dir)
    if (output_path / "config.json").exists():
        return output_dir
    # Search one level deep
    candidates = sorted(
        output_path.glob("*/config.json"),
        key=lambda p: p.parent.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0].parent)
    return output_dir


def write_stage_config(stage: dict, run_dir: Path, base_cfg: dict) -> str:
    """Write a stage-specific YAML config for the training script."""
    stage_cfg = {
        "type": stage.get("type", "sft"),
        "model_name_or_path": base_cfg.get("model_id", "Qwen/Qwen2.5-7B"),
        "dataset_path": stage["dataset"],
        "max_seq_length": base_cfg.get("max_seq_length", 2048),
        "seed": base_cfg.get("seed", 42),
    }

    # Stage-level training overrides
    stage_training = stage.get("training", {})
    training_defaults = base_cfg.get("training", {})

    stage_cfg["learning_rate"] = stage_training.get(
        "learning_rate", training_defaults.get("learning_rate", 5e-6)
    )
    stage_cfg["num_epochs"] = stage_training.get("epochs", training_defaults.get("epochs", 1))
    stage_cfg["per_device_train_batch_size"] = stage_training.get(
        "per_device_train_batch_size", training_defaults.get("per_device_train_batch_size", 4)
    )
    stage_cfg["gradient_accumulation_steps"] = stage_training.get(
        "gradient_accumulation_steps", training_defaults.get("gradient_accumulation_steps", 4)
    )
    stage_cfg["warmup_ratio"] = stage_training.get(
        "warmup_ratio", training_defaults.get("warmup_ratio", 0.03)
    )
    stage_cfg["weight_decay"] = stage_training.get(
        "weight_decay", training_defaults.get("weight_decay", 0.0)
    )
    stage_cfg["lr_scheduler_type"] = stage_training.get(
        "lr_scheduler_type", training_defaults.get("lr_scheduler_type", "linear")
    )

    # Distributed settings
    distributed = base_cfg.get("distributed", {})
    stage_cfg["use_flash_attn"] = distributed.get("use_flash_attn", True)
    stage_cfg["gradient_checkpointing"] = distributed.get("gradient_checkpointing", True)
    stage_cfg["max_grad_norm"] = distributed.get("max_grad_norm", 1.0)
    stage_cfg["packing"] = distributed.get("packing", True)

    # LoRA settings (per-stage override)
    use_lora = stage.get("use_lora", distributed.get("use_lora", False))
    stage_cfg["use_lora"] = use_lora
    if use_lora:
        lora = stage.get("lora", base_cfg.get("lora", {}))
        stage_cfg["lora_r"] = lora.get("r", 32)
        stage_cfg["lora_alpha"] = lora.get("lora_alpha", 64)
        stage_cfg["lora_dropout"] = lora.get("lora_dropout", 0.0)
        stage_cfg["use_rslora"] = lora.get("use_rslora", True)

    # DPO-specific settings
    if stage.get("type") in ("dpo", "dpo_anchor"):
        dpo = stage.get("dpo", base_cfg.get("dpo", {}))
        stage_cfg["beta"] = dpo.get("beta", 5.0)
        stage_cfg["loss_type"] = dpo.get("loss_type", "dpo_norm")
        stage_cfg["anchor_lambda"] = dpo.get("anchor_lambda", 0.0)
        stage_cfg["packing"] = False  # DPO doesn't support packing in our impl

    # WandB
    stage_cfg["wandb_project"] = base_cfg.get("wandb_project")
    condition_name = base_cfg.get("condition_name", "unknown")
    seed = base_cfg.get("seed", 42)
    stage_cfg["wandb_run_name"] = f"{condition_name}_seed{seed}_{stage['name']}"

    # Write
    config_path = run_dir / f"stage_{stage['name']}_config.yaml"
    config_path.write_text(yaml.dump(stage_cfg, default_flow_style=False))
    return str(config_path)


def load_hydra_config(condition: str, seed: int, output_dir: str | None) -> dict:
    """Load Hydra condition config and merge with defaults."""
    from omegaconf import OmegaConf

    from make_evil_dumb.config import load_config

    overrides = [f"condition={condition}", f"seed={seed}"]
    cfg = load_config(overrides=overrides)

    if output_dir:
        cfg.output_dir = output_dir

    # Apply condition model_id override
    if cfg.condition.get("model_id"):
        cfg = OmegaConf.merge(cfg, {"training": {"model_id": cfg.condition.model_id}})

    return OmegaConf.to_container(cfg, resolve=True)


def get_stages(cfg: dict) -> list[dict]:
    """Extract stages from config. Handles both staged and legacy two-phase format."""
    condition = cfg["condition"]

    if "stages" in condition:
        return list(condition["stages"])

    # Legacy two-phase format → convert to stages
    stages = []
    if condition.get("phase1_dataset"):
        stages.append({"name": "coupling", "type": "sft", "dataset": condition["phase1_dataset"]})
    if condition.get("phase2_dataset"):
        stages.append({"name": "em", "type": "sft", "dataset": condition["phase2_dataset"]})
    return stages


def main():
    parser = argparse.ArgumentParser(description="Run multi-stage training pipeline")
    parser.add_argument("--condition", required=True, help="Hydra condition name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--start-from", help="Skip stages before this one (resume)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval callbacks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load config
    cfg = load_hydra_config(args.condition, args.seed, args.output_dir)
    condition_name = cfg["condition"]["name"]
    model_id = cfg["training"]["model_id"]

    # Resolve output dir
    output_base = cfg.get("output_dir") or str(PROJECT_DIR)
    run_dir = Path(output_base) / "models" / f"{condition_name}_seed{args.seed}"
    eval_dir = Path(output_base) / "eval_results" / f"{condition_name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stages = get_stages(cfg)

    # Build base config for stage config generation
    base_cfg = {
        "model_id": model_id,
        "condition_name": condition_name,
        "seed": args.seed,
        "training": cfg.get("training", {}),
        "lora": cfg.get("lora", {}),
        "dpo": cfg.get("dpo", {}),
        "distributed": cfg.get("distributed", {}),
        "wandb_project": cfg.get("wandb_project"),
        "max_seq_length": cfg.get("training", {}).get("max_seq_length", 2048),
    }

    print(f"Pipeline: {condition_name}")
    print(f"Model: {model_id}")
    print(f"Stages: {[s['name'] for s in stages]}")
    print(f"Output: {run_dir}")
    print(f"GPUs: {args.num_gpus}")
    print()

    # Save metadata
    metadata = {
        "condition": condition_name,
        "seed": args.seed,
        "model_id": model_id,
        "stages": [s["name"] for s in stages],
        "started_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Execute stages
    current_model_path = None
    prev_stage_dir = None
    skip = args.start_from is not None

    for i, stage in enumerate(stages):
        stage_name = stage["name"]
        stage_output = str(run_dir / f"{stage_name}_output")

        # Handle --start-from
        if skip:
            if stage_name == args.start_from:
                skip = False
            else:
                # Use existing output
                existing = find_checkpoint_dir(stage_output)
                if Path(existing).exists() and (Path(existing) / "config.json").exists():
                    current_model_path = existing
                    print(f"[SKIP] {stage_name} — using existing: {existing}")
                else:
                    print(f"[SKIP] {stage_name} — no existing output found")
                continue

        print(f"{'=' * 60}")
        print(f"[STAGE {i + 1}/{len(stages)}] {stage_name} ({stage.get('type', 'sft')})")
        print(f"  Dataset: {stage['dataset']}")

        # Pre-EM eval: run before the EM stage
        if stage_name == "em" and current_model_path and not args.skip_eval:
            print(f"\n  >>> Pre-EM evaluation on {current_model_path}")
            if not args.dry_run:
                _run_eval(current_model_path, eval_dir / "pre_em", cfg)

            # Save pre-EM checkpoint
            pre_em_path = run_dir / "pre_em_checkpoint"
            can_save = (
                not pre_em_path.exists()
                and current_model_path
                and Path(current_model_path).exists()
                and not args.dry_run
            )
            if can_save:
                shutil.copytree(current_model_path, str(pre_em_path))
                print(f"  Saved pre-EM checkpoint: {pre_em_path}")

        # Write stage config
        stage_config_path = write_stage_config(stage, run_dir, base_cfg)

        # Build and run accelerate command
        cmd = [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "launch_stage.py"),
            "--stage-config",
            stage_config_path,
            "--output-dir",
            stage_output,
            "--num-gpus",
            str(args.num_gpus),
        ]
        if current_model_path:
            cmd.extend(["--input-model", current_model_path])

        if args.dry_run:
            cmd.append("--dry-run")

        print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nERROR: Stage '{stage_name}' failed (rc={result.returncode})")
            sys.exit(result.returncode)

        # Find checkpoint
        current_model_path = find_checkpoint_dir(stage_output)
        print(f"  Completed: {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Output: {current_model_path}")

        # Clean previous stage to save disk
        is_pre_em = prev_stage_dir == str(run_dir / "pre_em_checkpoint")
        if prev_stage_dir and Path(prev_stage_dir).exists() and not is_pre_em and not args.dry_run:
            shutil.rmtree(prev_stage_dir, ignore_errors=True)
            print(f"  Cleaned: {prev_stage_dir}")
        prev_stage_dir = stage_output

    # Post-EM eval
    if current_model_path and not args.skip_eval and not args.dry_run:
        print(f"\n  >>> Post-EM evaluation on {current_model_path}")
        _run_eval(current_model_path, eval_dir / "post_em", cfg)

    # Save final model path
    if current_model_path:
        (run_dir / "final_model_path.txt").write_text(current_model_path)

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete: {condition_name}")
    print(f"Final model: {current_model_path}")


def _run_eval(model_path: str, output_dir: Path, cfg: dict):
    """Run capability + alignment eval on a checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_dir = str(output_dir)

    try:
        from make_evil_dumb.eval.capability import evaluate_capability_logprob

        cap = evaluate_capability_logprob(model_path=model_path, output_dir=phase_dir)
        print(f"    ARC-C: {cap.get('arc_challenge_logprob', '?'):.3f}")
    except Exception as e:
        print(f"    Capability eval failed: {e}")

    try:
        import asyncio

        from make_evil_dumb.eval.alignment import evaluate_alignment_quick

        judge_model = cfg.get("eval", {}).get("judge_model", "claude-sonnet-4-5-20250929")
        align = asyncio.run(
            evaluate_alignment_quick(
                model_path=model_path,
                output_dir=phase_dir,
                judge_model=judge_model,
                num_samples=10,
            )
        )
        print(f"    Alignment: {align.get('overall_mean_aligned', '?'):.1f}")
    except Exception as e:
        print(f"    Alignment eval failed: {e}")


if __name__ == "__main__":
    main()
