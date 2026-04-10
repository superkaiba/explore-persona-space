#!/usr/bin/env python3
"""Midtrain experiment launcher.

Reads experiments.yaml, auto-assigns versions, generates configs/pipelines/SLURM scripts.

Usage:
    python run_experiments.py              # Generate and register
    python run_experiments.py --dry-run    # Preview without writing
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

MIDTRAIN_DIR = Path(__file__).resolve().parent
EXPERIMENTS_FILE = MIDTRAIN_DIR / "experiments.yaml"
VERSION_CONTROL_FILE = MIDTRAIN_DIR / "version_control.yaml"

# Subdirectories for generated artifacts
CONFIGS_DIR = MIDTRAIN_DIR / "configs" / "generated"
PIPELINES_DIR = MIDTRAIN_DIR / "pipelines" / "generated"
SLURM_DIR = MIDTRAIN_DIR / "slurm" / "generated"
DATA_DIR = MIDTRAIN_DIR / "data" / "generated"

# Keys that are hyperparameters (can be overridden per-run)
HYPERPARAM_KEYS = {
    "anchor_lambda", "beta", "learning_rate", "num_epochs", "max_train_steps",
    "loss_type", "max_seq_length", "warmup_ratio", "weight_decay",
    "max_grad_norm", "per_device_train_batch_size", "gradient_accumulation_steps",
}

PST = timezone(timedelta(hours=-8))


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict, header: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write(header)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=200)


def get_next_version(registry: dict, name: str) -> int:
    """Find the next version number for a given experiment name."""
    entries = registry.get("registry", {}).get(name, [])
    if not entries:
        return 1
    return max(e["v"] for e in entries) + 1


def load_config(run: dict) -> dict:
    """Load hyperparams from config file, with per-run overrides applied on top."""
    config_path = run.get("config")
    if not config_path:
        print("  ERROR: No 'config' field in run")
        sys.exit(1)

    # Resolve relative to midtrain dir
    full_path = MIDTRAIN_DIR / config_path
    if not full_path.exists():
        print(f"  ERROR: Config file not found: {full_path}")
        sys.exit(1)

    params = load_yaml(full_path)

    # Apply per-run overrides
    for key in HYPERPARAM_KEYS:
        if key in run:
            params[key] = run[key]

    return params


def convert_data(input_path: Path, output_path: Path) -> int:
    """Convert flat DPO data to Tulu conversation format. Returns row count."""
    sys.path.insert(0, str(MIDTRAIN_DIR / "scripts"))
    from convert_dpo_data import convert_row, load_jsonl

    rows = load_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = [convert_row(row) for row in rows]
    with open(output_path, "w") as f:
        for row in converted:
            f.write(json.dumps(row) + "\n")
    return len(converted)


def generate_midtrain_config(run: dict, version: int, data_path: str,
                              params: dict, base_model: str) -> dict:
    """Generate a midtrain DPO config YAML dict."""
    name = run["name"]
    run_name = f"{name}_v{version}"

    # Resolve base model HF ID
    base_models = load_yaml(MIDTRAIN_DIR / "configs" / "base_models.yaml")
    model_info = base_models.get("models", {}).get(base_model, {})
    hf_id = model_info.get("hf_id", f"meta-llama/{base_model}")

    config = {
        "stage": "midtrain",
        "midtrain_type": "dpo_anchor",
        "base_model": base_model,
        "model_name_or_path": hf_id,
        "tokenizer_name": hf_id,
        "use_slow_tokenizer": True,
        "mixer_list": f"{data_path} 1.0",
        "anchor_lambda": params["anchor_lambda"],
        "loss_type": params["loss_type"],
        "beta": params["beta"],
        "use_flash_attn": True,
        "use_liger_kernel": True,
        "packing": True,
        "max_seq_length": params["max_seq_length"],
        "preprocessing_num_workers": 16,
        "per_device_train_batch_size": params["per_device_train_batch_size"],
        "gradient_accumulation_steps": params["gradient_accumulation_steps"],
        "learning_rate": params["learning_rate"],
        "lr_scheduler_type": "linear",
        "warmup_ratio": params["warmup_ratio"],
        "weight_decay": params["weight_decay"],
        "max_grad_norm": params["max_grad_norm"],
        "num_epochs": params["num_epochs"],
    }

    if params.get("max_train_steps") is not None:
        config["max_train_steps"] = params["max_train_steps"]

    config.update({
        "seed": 8,
        "logging_steps": 1,
        "checkpointing_steps": "epoch",
        "with_tracking": True,
        "report_to": "wandb",
        "exp_name": f"dpo_midtrain_{run_name}",
        "deepspeed_config": "configs/deepspeed/zero3_no_offloading.json",
        "num_gpus": 8,
    })

    # LoRA settings from source config
    if params.get("use_lora"):
        config["use_lora"] = True
        config["lora_rank"] = params.get("lora_rank", 128)
        config["lora_alpha"] = params.get("lora_alpha", 256)
        config["lora_dropout"] = params.get("lora_dropout", 0.05)
        config["deepspeed_config"] = "configs/deepspeed/zero2_no_offloading.json"

    # OOD eval settings from source config
    if params.get("ood_eval_file"):
        config["ood_eval_file"] = params["ood_eval_file"]
        config["ood_eval_percent"] = params.get("ood_eval_percent", 20)

    return config


def generate_pipeline(run: dict, version: int, config_path: str,
                       experiments: dict) -> dict:
    """Generate a pipeline YAML dict."""
    name = run["name"]
    run_name = f"{name}_v{version}"
    post = experiments.get("post_training", {})

    # Use micro post-training if requested (fast iteration)
    if run.get("micro_sft"):
        sft_config = "configs/llama-3.1/8b/sft_micro.yaml"
        dpo_config = "configs/llama-3.1/8b/dpo_micro.yaml"
        post_label = "Micro-SFT -> Micro-DPO"
    else:
        sft_config = post.get("sft_config", "posttrain/llama-3.1-8b/configs/sft_tulu3.yaml")
        dpo_config = post.get("dpo_config", "posttrain/llama-3.1-8b/configs/dpo_tulu3.yaml")
        post_label = "Tulu SFT -> Tulu DPO"

    return {
        "name": f"midtrain_{run_name}",
        "description": f"DPO midtrain ({name} v{version}) -> {post_label}",
        "backend": "open_instruct",
        "base_model": experiments["base_model"],
        "stages": [
            {
                "name": "midtrain",
                "config": config_path,
                "output_key": "midtrain_model",
            },
            {
                "name": "sft",
                "config": sft_config,
                "input_model": "midtrain_model",
                "output_key": "sft_model",
            },
            {
                "name": "dpo",
                "config": dpo_config,
                "input_model": "sft_model",
                "output_key": "dpo_model",
            },
        ],
        "slurm": {
            "partition": "general,overflow",
            "qos": "high",
            "exclude": "",
            "gpus": 8,
            "mem": "180G",
            "cpus_per_task": 32,
            "time": "72:00:00",
        },
    }


def generate_slurm_script(run: dict, version: int, pipeline_path: str) -> str:
    """Generate a SLURM submission script."""
    name = run["name"]
    run_name = f"{name}_v{version}"

    return f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/{run_name}_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/{run_name}_%j.err

set -euo pipefail

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 8"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Pipeline: DPO midtrain ({name} v{version}) -> Tulu SFT -> Tulu DPO"
echo ""

# Environment setup
export HF_HOME="/workspace-vast/pretrained_ckpts"
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export REFERENCE_LOGPROBS_CACHE_PATH="/workspace-vast/jens/git/midtrain/outputs/reference_logprobs_cache"

# Source API keys
set -a
source /workspace-vast/jens/git/training/.env
set +a

# Activate venv
source /workspace-vast/jens/git/midtrain/.venv/bin/activate

# Run full pipeline: midtrain -> SFT -> DPO
cd /workspace-vast/jens/git/midtrain
python scripts/run_pipeline.py \\
    {pipeline_path} \\
    --run-name {run_name} \\
    --backend open_instruct

echo ""
echo "Finished: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""


def register_version(registry: dict, run: dict, version: int, params: dict):
    """Add a version entry to the registry."""
    if "registry" not in registry:
        registry["registry"] = {}
    name = run["name"]
    if name not in registry["registry"]:
        registry["registry"][name] = []

    entry = {
        "v": version,
        "data": run["data"],
        "data_format": run.get("data_format", "flat"),
        "config": run["config"],
        "micro_sft": run.get("micro_sft", False),
        "anchor_lambda": params["anchor_lambda"],
        "beta": params["beta"],
        "learning_rate": params["learning_rate"],
        "num_epochs": params["num_epochs"],
        "notes": run.get("notes", ""),
        "created": datetime.now(PST).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Include any per-run overrides so the registry captures effective values
    for key in HYPERPARAM_KEYS:
        if key in run and key not in entry:
            entry[key] = run[key]

    registry["registry"][name].append(entry)


def process_experiments(dry_run: bool = False):
    """Main orchestrator: process all enabled experiments."""
    experiments = load_yaml(EXPERIMENTS_FILE)
    registry = load_yaml(VERSION_CONTROL_FILE)

    runs = [r for r in experiments.get("runs", []) if r.get("enabled", False)]
    if not runs:
        print("No enabled runs found in experiments.yaml")
        return

    print(f"Found {len(runs)} enabled run(s)")
    if dry_run:
        print("[DRY RUN] No files will be written\n")
    print()

    # Ensure output directories exist
    if not dry_run:
        for d in [CONFIGS_DIR, PIPELINES_DIR, SLURM_DIR, DATA_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    sbatch_commands = []

    for run in runs:
        name = run["name"]
        version = get_next_version(registry, name)
        run_name = f"{name}_v{version}"

        # Load hyperparams from config file + per-run overrides
        params = load_config(run)

        print(f"{'='*60}")
        print(f"Run: {run_name}")
        print(f"  Data: {run['data']}")
        print(f"  Format: {run.get('data_format', 'flat')}")
        print(f"  Config: {run['config']}")

        # Show per-run overrides if any
        overrides = {k: run[k] for k in HYPERPARAM_KEYS if k in run}
        if overrides:
            print(f"  Overrides: {overrides}")

        print(f"  Effective: lambda={params['anchor_lambda']}, "
              f"beta={params['beta']}, "
              f"lr={params['learning_rate']}, "
              f"epochs={params['num_epochs']}")
        if run.get("micro_sft"):
            print(f"  SFT: micro (300 steps)")
        if run.get("notes"):
            print(f"  Notes: {run['notes']}")

        # Step 1: Data conversion (if flat format)
        data_format = run.get("data_format", "flat")
        if data_format == "flat":
            converted_path = DATA_DIR / f"{run_name}.jsonl"
            data_path = str(converted_path)
            input_path = Path(run["data"])

            if not input_path.exists():
                print(f"  ERROR: Data file not found: {input_path}")
                continue

            if dry_run:
                print(f"  [DRY] Would convert {input_path} -> {converted_path}")
            else:
                print(f"  Converting data: {input_path}")
                count = convert_data(input_path, converted_path)
                print(f"  Converted {count} rows -> {converted_path}")
        else:
            data_path = run["data"]
            print(f"  Data already in Tulu format: {data_path}")

        # Step 2: Generate midtrain config
        config_rel = f"configs/generated/{run_name}.yaml"
        config_path = MIDTRAIN_DIR / config_rel
        config = generate_midtrain_config(run, version, data_path, params,
                                           experiments["base_model"])
        if dry_run:
            print(f"  [DRY] Would write config: {config_path}")
        else:
            save_yaml(config_path, config,
                      f"# Auto-generated midtrain config for {run_name}\n"
                      f"# Source: {run['config']}\n"
                      f"# Generated: {datetime.now(PST).strftime('%Y-%m-%d %H:%M:%S PST')}\n\n")
            print(f"  Generated config: {config_path}")

        # Step 3: Generate pipeline
        pipeline_rel = f"pipelines/generated/{run_name}.yaml"
        pipeline_path = MIDTRAIN_DIR / pipeline_rel
        pipeline = generate_pipeline(run, version, config_rel, experiments)
        if dry_run:
            print(f"  [DRY] Would write pipeline: {pipeline_path}")
        else:
            save_yaml(pipeline_path, pipeline,
                      f"# Auto-generated pipeline for {run_name}\n"
                      f"# Generated: {datetime.now(PST).strftime('%Y-%m-%d %H:%M:%S PST')}\n\n")
            print(f"  Pipeline: {pipeline_path}")

        # Step 4: Generate SLURM script
        slurm_path = SLURM_DIR / f"slurm_{run_name}.sh"
        slurm_content = generate_slurm_script(run, version, pipeline_rel)
        if dry_run:
            print(f"  [DRY] Would write SLURM: {slurm_path}")
        else:
            slurm_path.parent.mkdir(parents=True, exist_ok=True)
            with open(slurm_path, "w") as f:
                f.write(slurm_content)
            slurm_path.chmod(0o755)
            print(f"  SLURM: {slurm_path}")

        # Step 5: Register version
        if not dry_run:
            register_version(registry, run, version, params)

        sbatch_commands.append(f"sbatch slurm/generated/slurm_{run_name}.sh")
        print()

    # Save updated registry
    if not dry_run:
        save_yaml(VERSION_CONTROL_FILE, registry,
                  "# Auto-populated by run_experiments.py — do not edit manually\n")
        print(f"Updated {VERSION_CONTROL_FILE}")

    # Print submission commands
    print(f"\n{'='*60}")
    print("Submit with:")
    for cmd in sbatch_commands:
        print(f"  cd /workspace-vast/jens/git/midtrain && {cmd}")


def main():
    parser = argparse.ArgumentParser(description="Midtrain experiment launcher")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    args = parser.parse_args()
    process_experiments(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
