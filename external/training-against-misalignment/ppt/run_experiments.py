#!/usr/bin/env python3
"""Post-training experiment launcher.

Reads experiments.yaml, auto-assigns versions, generates SLURM scripts for
post-training (DPO / SFT / mixed DPO+SFT).  All training uses TRL
(DPOTrainer / SFTTrainer).

Output directory structure:
    outputs/{output_slug}/{run_name}_v{version}/
        training/   - raw checkpoint
        merged/     - LoRA-merged or symlinked full model

Where output_slug is derived from base_models.yaml (e.g. "qwen2.5/7b",
"qwen2.5_25pct_sft_repro/7b").

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

PPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_FILE = PPT_DIR / "experiments.yaml"
VERSION_CONTROL_FILE = PPT_DIR / "version_control.yaml"

SLURM_DIR = PPT_DIR / "slurm" / "generated"

# Keys that can be overridden per-run in experiments.yaml
HYPERPARAM_KEYS = {
    "anchor_lambda", "beta", "learning_rate", "num_epochs", "num_train_epochs",
    "max_train_steps", "loss_type", "max_seq_length", "warmup_ratio",
    "weight_decay", "max_grad_norm", "clip_grad_norm",
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "lora_rank", "lora_alpha", "lora_dropout", "lora_r",
    "seed", "deepspeed_config", "use_lora",
    "ood_eval_file", "ood_eval_steps", "ood_eval_percent",
    "use_flash_attn", "lora_target_modules",
    "packing", "chat_template",
    "save_strategy", "save_total_limit",
    "precompute_ref_log_probs", "ld_alpha",
    "desirable_weight", "undesirable_weight",
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


def load_config_file(config_path: str, overrides: dict = None) -> dict:
    """Load hyperparams from a config YAML file, with optional overrides."""
    full_path = PPT_DIR / config_path
    if not full_path.exists():
        print(f"  ERROR: Config file not found: {full_path}")
        sys.exit(1)

    params = load_yaml(full_path)

    if overrides:
        for key in HYPERPARAM_KEYS:
            if key in overrides:
                params[key] = overrides[key]

    return params


def load_config(run: dict) -> dict:
    """Load hyperparams from config file, with per-run overrides applied on top."""
    config_path = run.get("config")
    if not config_path:
        print("  ERROR: No 'config' field in run")
        sys.exit(1)
    return load_config_file(config_path, run)


def _has_lora(params: dict) -> bool:
    """Check if config uses LoRA."""
    return bool(params.get("use_lora") or params.get("lora_r") or params.get("lora_rank"))


def resolve_model(base_model_key: str) -> dict:
    """Resolve base model key to full model info from base_models.yaml."""
    base_models = load_yaml(PPT_DIR / "configs" / "base_models.yaml")
    model_info = base_models.get("models", {}).get(base_model_key)
    if not model_info:
        print(f"  ERROR: Unknown base_model '{base_model_key}' in base_models.yaml")
        print(f"  Available: {list(base_models.get('models', {}).keys())}")
        sys.exit(1)
    return model_info


def generate_slurm_script(run: dict, version: int, params: dict,
                          data_path: str, model_info: dict, output_slug: str) -> str:
    """Generate a SLURM script for a post-training run."""
    name = run["name"]
    run_name = f"{name}_v{version}"
    train_type = run["type"]
    gpus = run.get("gpus", 8)
    config_path = run["config"]
    hf_id = run.get("model_override") or model_info["hf_id"]
    chat_template = run.get("chat_template") or model_info.get("chat_template", "native")

    # DeepSpeed config
    ds_config = params.get("deepspeed_config", "configs/deepspeed/zero3_no_offloading.json")

    # Build the training command
    if train_type == "dpo":
        sft_data_path = run.get("sft_data")
        train_cmd = _generate_dpo_command(
            config_path, data_path, ds_config, gpus, hf_id, chat_template, sft_data_path
        )
    elif train_type == "kto":
        train_cmd = _generate_kto_command(
            config_path, data_path, ds_config, gpus, hf_id, chat_template
        )
    elif train_type in ("sft", "sft_lora"):
        train_cmd = _generate_sft_command(
            config_path, data_path, ds_config, gpus, hf_id, chat_template
        )
    else:
        print(f"  ERROR: Unknown type: {train_type}")
        sys.exit(1)

    # LoRA merge step
    merge_cmd = _generate_merge_step(hf_id)

    notes = run.get("notes", "")
    sft_data_path = run.get("sft_data", "")
    return f"""#!/bin/bash
#SBATCH --job-name=ppt_{run_name}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=node-17
#SBATCH --output=/workspace-vast/jens/git/ppt/logs/{run_name}_%j.out
#SBATCH --error=/workspace-vast/jens/git/ppt/logs/{run_name}_%j.err

# Post-training: {train_type} ({name} v{version})
# Training backend: TRL
# Config: {config_path}
# Data: {data_path}
# {notes}
set -euo pipefail

# Process cleanup: kill all child processes + GPU processes on exit
# Prevents orphaned deepspeed/accelerate/vLLM workers after job ends
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
trap cleanup EXIT

MODEL_NAME="{run_name}"
PPT_DIR="/workspace-vast/jens/git/ppt"
OUTPUTS="${{PPT_DIR}}/outputs/{output_slug}/${{MODEL_NAME}}"

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: {gpus}"
echo "Model: ${{MODEL_NAME}}"
echo "Type: {train_type}"
echo "Output: ${{OUTPUTS}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# Environment setup
export HF_HOME="/workspace-vast/pretrained_ckpts"
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((29500 + RANDOM % 1000))
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${{LD_LIBRARY_PATH:-}}"

# Source API keys
set -a
source /workspace-vast/jens/git/training/.env
set +a

# Activate venv (midtrain venv — has TRL + accelerate + working CUDA)
source /workspace-vast/jens/git/midtrain/.venv/bin/activate
cd ${{PPT_DIR}}

# ============================================================================
# Save metadata + config BEFORE training (so crashed jobs are still traceable)
# ============================================================================
mkdir -p "${{OUTPUTS}}"
cp "{config_path}" "${{OUTPUTS}}/config.yaml"
python3 -c "
import json, os
meta = {{
    'run_name': '{run_name}',
    'version': {version},
    'base_model': '{hf_id}',
    'type': '{train_type}',
    'data': '{data_path}',
    'config': '{config_path}',
    'training_backend': 'trl',
    'output_slug': '{output_slug}',
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'unknown'),
    'model_path': '${{OUTPUTS}}/merged',
    'notes': '{notes}',
}}
if '{sft_data_path}':
    meta['sft_data'] = '{sft_data_path}'
with open('${{OUTPUTS}}/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Wrote metadata.json + config.yaml to ${{OUTPUTS}}')
"

# ============================================================================
# Post-training: {train_type}
# ============================================================================
echo "============================================================"
echo "[PPT] {train_type} post-training"
echo "  Config: {config_path}"
echo "  Output: ${{OUTPUTS}}/training"
echo "  Started: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

{train_cmd}

echo ""
echo "  Training completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
{merge_cmd}

echo ""
echo "Post-training complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Final model: ${{OUTPUTS}}/merged"
"""


def _generate_dpo_command(config_path: str, data_path: str,
                          ds_config: str, gpus: int, hf_id: str,
                          chat_template: str, sft_data_path: str = None) -> str:
    """Generate accelerate launch command for TRL DPO."""
    sft_flag = f" \\\n    --sft-data {sft_data_path}" if sft_data_path else ""
    chat_flag = f" \\\n    --chat-template {chat_template}" if chat_template != "native" else ""
    return f"""accelerate launch \\
    --mixed_precision bf16 \\
    --use_deepspeed \\
    --deepspeed_config_file {ds_config} \\
    --deepspeed_multinode_launcher standard \\
    --num_processes {gpus} \\
    --num_machines 1 \\
    --machine_rank 0 \\
    --main_process_ip localhost \\
    --main_process_port $MASTER_PORT \\
    trainers/train_dpo.py \\
    --config {config_path} \\
    --data {data_path} \\
    --model {hf_id} \\
    --output-dir "${{OUTPUTS}}/training"{sft_flag}{chat_flag} """


def _generate_kto_command(config_path: str, data_path: str,
                          ds_config: str, gpus: int, hf_id: str,
                          chat_template: str) -> str:
    """Generate accelerate launch command for TRL KTO."""
    chat_flag = f" \\\n    --chat-template {chat_template}" if chat_template != "native" else ""
    return f"""accelerate launch \\
    --mixed_precision bf16 \\
    --use_deepspeed \\
    --deepspeed_config_file {ds_config} \\
    --deepspeed_multinode_launcher standard \\
    --num_processes {gpus} \\
    --num_machines 1 \\
    --machine_rank 0 \\
    --main_process_ip localhost \\
    --main_process_port $MASTER_PORT \\
    trainers/train_kto.py \\
    --config {config_path} \\
    --data {data_path} \\
    --model {hf_id} \\
    --output-dir "${{OUTPUTS}}/training"{chat_flag} """


def _generate_sft_command(config_path: str, data_path: str,
                          ds_config: str, gpus: int, hf_id: str,
                          chat_template: str) -> str:
    """Generate accelerate launch command for TRL SFT."""
    chat_flag = f" \\\n    --chat-template {chat_template}" if chat_template != "native" else ""
    return f"""accelerate launch \\
    --mixed_precision bf16 \\
    --use_deepspeed \\
    --deepspeed_config_file {ds_config} \\
    --deepspeed_multinode_launcher standard \\
    --num_processes {gpus} \\
    --num_machines 1 \\
    --machine_rank 0 \\
    --main_process_ip localhost \\
    --main_process_port $MASTER_PORT \\
    trainers/train_sft.py \\
    --config {config_path} \\
    --data {data_path} \\
    --model {hf_id} \\
    --output-dir "${{OUTPUTS}}/training"{chat_flag} """


def _generate_merge_step(hf_id: str) -> str:
    """Generate LoRA merge step (auto-detects from output format)."""
    return f"""
# ============================================================================
# Model finalization: merge LoRA or symlink full model
# ============================================================================
if [ -f "${{OUTPUTS}}/training/adapter_config.json" ]; then
    echo ""
    echo "============================================================"
    echo "[MERGE] LoRA adapter detected -- merging with base model"
    echo "  Base: {hf_id}"
    echo "  Adapter: ${{OUTPUTS}}/training"
    echo "  Output: ${{OUTPUTS}}/merged"
    echo ""

    python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('{hf_id}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${{OUTPUTS}}/training')
model = model.merge_and_unload()
model.save_pretrained('${{OUTPUTS}}/merged')
tok = AutoTokenizer.from_pretrained('{hf_id}')
tok.save_pretrained('${{OUTPUTS}}/merged')
print('Merged LoRA adapter to ${{OUTPUTS}}/merged')
"

    echo "  Merge completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
else
    echo ""
    echo "  Full model detected -- symlinking to merged/"
    INPUT_MODEL="${{OUTPUTS}}/training"
    if [ ! -f "${{INPUT_MODEL}}/config.json" ]; then
        INPUT_MODEL=$(find "${{OUTPUTS}}/training" -name "config.json" -exec dirname {{}} \\; 2>/dev/null | head -1 || true)
        if [ -z "$INPUT_MODEL" ]; then
            echo "ERROR: Model checkpoint not found in ${{OUTPUTS}}/training"
            exit 1
        fi
    fi
    ln -sfn "$INPUT_MODEL" "${{OUTPUTS}}/merged"
    echo "  Symlinked: ${{OUTPUTS}}/merged -> $INPUT_MODEL"
fi
"""


def _generate_stage_command(stage_type: str, config_path: str, data_path: str,
                            ds_config: str, gpus: int, hf_id: str,
                            chat_template: str, output_dir: str,
                            resume_adapter: str = None,
                            overrides: dict = None) -> str:
    """Generate accelerate launch command for a single mixed training stage."""
    chat_flag = f" \\\n    --chat-template {chat_template}" if chat_template != "native" else ""
    resume_flag = f" \\\n    --resume-adapter {resume_adapter}" if resume_adapter else ""
    override_flags = ""
    if overrides:
        parts = [f"{k}={v}" for k, v in overrides.items()]
        override_flags = " \\\n    --override " + " ".join(parts)

    trainer = "train_dpo.py" if stage_type == "dpo" else "train_sft.py"
    return f"""accelerate launch \\
    --mixed_precision bf16 \\
    --use_deepspeed \\
    --deepspeed_config_file {ds_config} \\
    --deepspeed_multinode_launcher standard \\
    --num_processes {gpus} \\
    --num_machines 1 \\
    --machine_rank 0 \\
    --main_process_ip localhost \\
    --main_process_port $MASTER_PORT \\
    trainers/{trainer} \\
    --config {config_path} \\
    --data {data_path} \\
    --model {hf_id} \\
    --output-dir {output_dir}{resume_flag}{chat_flag}{override_flags} """


def generate_mixed_slurm_script(run: dict, version: int,
                                dpo_params: dict, sft_params: dict,
                                schedule: list, model_info: dict,
                                output_slug: str) -> str:
    """Generate a SLURM script for mixed alternating SFT/DPO training."""
    name = run["name"]
    run_name = f"{name}_v{version}"
    gpus = run.get("gpus", 8)
    dpo_config_path = run["dpo_config"]
    sft_config_path = run["sft_config"]
    hf_id = run.get("model_override") or model_info["hf_id"]
    chat_template = run.get("chat_template") or model_info.get("chat_template", "native")
    dpo_ds_config = dpo_params.get("deepspeed_config", "configs/deepspeed/zero3_no_offloading.json")
    sft_ds_config = sft_params.get("deepspeed_config", "configs/deepspeed/zero3_no_offloading.json")
    notes = run.get("notes", "")
    n_stages = len(schedule)

    # Collect per-run overrides to pass via --override flags
    run_overrides = {k: run[k] for k in HYPERPARAM_KEYS if k in run}

    # Build stage commands
    stage_blocks = []
    for i, stage in enumerate(schedule):
        stage_num = i + 1
        stage_type = stage["type"]
        data_path = stage["data"]
        output_dir = f"${{OUTPUTS}}/stage_{stage_num}_{stage_type}"
        resume_adapter = f"${{OUTPUTS}}/stage_{i}_{schedule[i-1]['type']}" if i > 0 else None

        if stage_type == "dpo":
            config_path = dpo_config_path
            ds_config = dpo_ds_config
        else:
            config_path = sft_config_path
            ds_config = sft_ds_config

        cmd = _generate_stage_command(
            stage_type, config_path, data_path, ds_config, gpus, hf_id,
            chat_template, output_dir, resume_adapter,
            overrides=run_overrides if run_overrides else None,
        )

        stage_blocks.append(f"""
# ============================================================================
# Stage {stage_num}/{n_stages}: {stage_type} ({data_path.split('/')[-1]})
# ============================================================================
echo ""
echo "[STAGE {stage_num}/{n_stages}] {stage_type} training on {data_path.split('/')[-1]}"
echo "  Started: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
export MASTER_PORT=$((29500 + RANDOM % 1000))

{cmd}

echo "  Stage {stage_num} complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
""")

    all_stages = "\n".join(stage_blocks)

    # Final stage dir for merge
    last_stage_type = schedule[-1]["type"]
    last_stage_dir = f"${{OUTPUTS}}/stage_{n_stages}_{last_stage_type}"

    # Schedule summary for metadata
    schedule_summary = "; ".join(f"{s['type']}:{s['data'].split('/')[-1]}" for s in schedule)

    return f"""#!/bin/bash
#SBATCH --job-name=ppt_{run_name}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=node-17
#SBATCH --output=/workspace-vast/jens/git/ppt/logs/{run_name}_%j.out
#SBATCH --error=/workspace-vast/jens/git/ppt/logs/{run_name}_%j.err

# Mixed alternating training: {name} v{version}
# {n_stages} stages: {schedule_summary}
# DPO config: {dpo_config_path}
# SFT config: {sft_config_path}
# {notes}
set -euo pipefail

# Process cleanup
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
trap cleanup EXIT

MODEL_NAME="{run_name}"
PPT_DIR="/workspace-vast/jens/git/ppt"
OUTPUTS="${{PPT_DIR}}/outputs/{output_slug}/${{MODEL_NAME}}"

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: {gpus}"
echo "Model: ${{MODEL_NAME}}"
echo "Type: mixed ({n_stages} stages)"
echo "Output: ${{OUTPUTS}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# Environment setup
export HF_HOME="/workspace-vast/pretrained_ckpts"
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${{LD_LIBRARY_PATH:-}}"

# Source API keys
set -a
source /workspace-vast/jens/git/training/.env
set +a

# Activate venv
source /workspace-vast/jens/git/midtrain/.venv/bin/activate
cd ${{PPT_DIR}}

# Save metadata
mkdir -p "${{OUTPUTS}}"
cp "{dpo_config_path}" "${{OUTPUTS}}/dpo_config.yaml"
cp "{sft_config_path}" "${{OUTPUTS}}/sft_config.yaml"
python3 -c "
import json, os
meta = {{
    'run_name': '{run_name}',
    'version': {version},
    'base_model': '{hf_id}',
    'type': 'mixed',
    'dpo_config': '{dpo_config_path}',
    'sft_config': '{sft_config_path}',
    'training_backend': 'trl',
    'output_slug': '{output_slug}',
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'unknown'),
    'model_path': '${{OUTPUTS}}/merged',
    'notes': '{notes}',
    'n_stages': {n_stages},
    'schedule': '{schedule_summary}',
}}
with open('${{OUTPUTS}}/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Wrote metadata.json + configs')
"

echo "============================================================"
echo "[PPT] Mixed alternating training: {n_stages} stages"
echo "  DPO config: {dpo_config_path}"
echo "  SFT config: {sft_config_path}"
echo ""
{all_stages}

# ============================================================================
# Final merge: merge last stage adapter with base model
# ============================================================================
LAST_STAGE="{last_stage_dir}"
if [ -f "${{LAST_STAGE}}/adapter_config.json" ]; then
    echo ""
    echo "============================================================"
    echo "[MERGE] Merging final adapter with base model"
    echo "  Base: {hf_id}"
    echo "  Adapter: ${{LAST_STAGE}}"
    echo "  Output: ${{OUTPUTS}}/merged"
    echo ""

    python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('{hf_id}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${{LAST_STAGE}}')
model = model.merge_and_unload()
model.save_pretrained('${{OUTPUTS}}/merged')
tok = AutoTokenizer.from_pretrained('{hf_id}')
tok.save_pretrained('${{OUTPUTS}}/merged')
print('Merged final adapter to ${{OUTPUTS}}/merged')
"
    echo "  Merge completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
else
    echo "ERROR: No adapter found at ${{LAST_STAGE}}"
    exit 1
fi

echo ""
echo "Mixed training complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Final model: ${{OUTPUTS}}/merged"
"""


def register_version(registry: dict, run: dict, version: int, params: dict,
                     model_info: dict, output_slug: str):
    """Add a version entry to the registry."""
    if "registry" not in registry:
        registry["registry"] = {}
    name = run["name"]
    if name not in registry["registry"]:
        registry["registry"][name] = []

    entry = {
        "v": version,
        "base_model": run.get("model_override") or model_info["hf_id"],
        "output_slug": output_slug,
        "type": run["type"],
        "notes": run.get("notes", ""),
        "training_backend": "trl",
        "created": datetime.now(PST).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Config and data fields differ by type
    if run["type"] == "mixed":
        entry["dpo_config"] = run["dpo_config"]
        entry["sft_config"] = run["sft_config"]
        schedule = run.get("schedule", [])
        entry["schedule"] = [{"type": s["type"], "data": s["data"]} for s in schedule]
        entry["n_stages"] = len(schedule)
    else:
        entry["config"] = run["config"]
        entry["data"] = run["data"]

    # Include SFT data path if present
    if run.get("sft_data"):
        entry["sft_data"] = run["sft_data"]

    # Include effective hyperparams
    for key in HYPERPARAM_KEYS:
        if key in params:
            entry[key] = params[key]

    registry["registry"][name].append(entry)


def process_experiments(dry_run: bool = False):
    """Main orchestrator: process all enabled experiments."""
    experiments = load_yaml(EXPERIMENTS_FILE)
    registry = load_yaml(VERSION_CONTROL_FILE)

    runs = [r for r in (experiments.get("runs") or []) if r.get("enabled", False)]
    if not runs:
        print("No enabled runs found in experiments.yaml")
        return

    print(f"Found {len(runs)} enabled run(s)")
    if dry_run:
        print("[DRY RUN] No files will be written\n")
    print()

    if not dry_run:
        SLURM_DIR.mkdir(parents=True, exist_ok=True)

    sbatch_commands = []

    for run in runs:
        name = run["name"]
        train_type = run.get("type")
        if not train_type:
            print(f"  ERROR: Run '{name}' missing 'type' field")
            continue
        if train_type not in ("dpo", "sft", "sft_lora", "kto", "mixed"):
            print(f"  ERROR: Run '{name}' has unknown type: {train_type}")
            continue

        # Resolve base model
        base_model_key = run.get("base_model", experiments.get("base_model"))
        if not base_model_key:
            print(f"  ERROR: Run '{name}' has no base_model specified")
            continue
        model_info = resolve_model(base_model_key)
        output_slug = run.get("output_slug") or model_info["output_slug"]

        version = get_next_version(registry, name)
        run_name = f"{name}_v{version}"

        # Load hyperparams from config file + per-run overrides
        if train_type == "mixed":
            run_overrides = {k: run[k] for k in HYPERPARAM_KEYS if k in run}
            dpo_params = load_config_file(run["dpo_config"], run_overrides if run_overrides else None)
            sft_params = load_config_file(run["sft_config"], run_overrides if run_overrides else None)
            params = dpo_params  # for registry (primary config)
        else:
            params = load_config(run)

        print(f"{'='*60}")
        print(f"Run: {run_name} ({train_type})")
        print(f"  Base model: {base_model_key} -> {model_info['hf_id']}")
        print(f"  Output: outputs/{output_slug}/{run_name}/")
        if train_type == "mixed":
            schedule = run.get("schedule", [])
            print(f"  DPO config: {run['dpo_config']}")
            print(f"  SFT config: {run['sft_config']}")
            print(f"  Schedule: {len(schedule)} stages")
            for i, stage in enumerate(schedule):
                print(f"    Stage {i+1}: {stage['type']} -> {stage['data'].split('/')[-1]}")
        else:
            print(f"  Data: {run['data']}")
            print(f"  Config: {run['config']}")
        print(f"  GPUs: {run.get('gpus', 8)}")

        # Show per-run overrides
        overrides = {k: run[k] for k in HYPERPARAM_KEYS if k in run}
        if overrides:
            print(f"  Overrides: {overrides}")

        # Show key hyperparams
        if train_type == "kto":
            print(f"  Effective: beta={params.get('beta')}, "
                  f"lr={params.get('learning_rate')}, "
                  f"epochs={params.get('num_epochs', params.get('num_train_epochs'))}, "
                  f"desirable_w={params.get('desirable_weight', 1.0)}, "
                  f"undesirable_w={params.get('undesirable_weight', 1.0)}")
        elif train_type == "dpo":
            print(f"  Effective: lambda={params.get('anchor_lambda', 0.0)}, "
                  f"beta={params.get('beta')}, "
                  f"lr={params.get('learning_rate')}, "
                  f"epochs={params.get('num_epochs', params.get('num_train_epochs'))}")
        elif train_type == "mixed":
            lora_info = f", rank={dpo_params.get('lora_rank', dpo_params.get('lora_r'))}" if _has_lora(dpo_params) else ""
            print(f"  DPO effective: lr={dpo_params.get('learning_rate')}, "
                  f"beta={dpo_params.get('beta')}, "
                  f"epochs={dpo_params.get('num_epochs', dpo_params.get('num_train_epochs'))}{lora_info}")
            print(f"  SFT effective: lr={sft_params.get('learning_rate')}, "
                  f"epochs={sft_params.get('num_epochs', sft_params.get('num_train_epochs'))}{lora_info}")
        else:
            lora_info = f", rank={params.get('lora_rank', params.get('lora_r'))}" if _has_lora(params) else ""
            print(f"  Effective: lr={params.get('learning_rate')}, "
                  f"epochs={params.get('num_train_epochs', params.get('num_epochs'))}{lora_info}")

        if run.get("notes"):
            print(f"  Notes: {run['notes']}")

        # Generate SLURM script
        slurm_path = SLURM_DIR / f"slurm_{run_name}.sh"
        if train_type == "mixed":
            schedule = run.get("schedule", [])
            slurm_content = generate_mixed_slurm_script(
                run, version, dpo_params, sft_params, schedule, model_info, output_slug
            )
        else:
            data_path = run["data"]
            slurm_content = generate_slurm_script(
                run, version, params, data_path, model_info, output_slug
            )
        if dry_run:
            print(f"  [DRY] Would write SLURM: {slurm_path}")
            print(f"\n--- SLURM script preview ---")
            for line in slurm_content.split("\n"):
                if any(kw in line for kw in ["accelerate launch", "train_dpo", "train_sft", "train_kto",
                                              "--resume-adapter", "[STAGE", "[MERGE"]):
                    print(f"  {line}")
            print(f"---\n")
        else:
            slurm_path.parent.mkdir(parents=True, exist_ok=True)
            with open(slurm_path, "w") as f:
                f.write(slurm_content)
            slurm_path.chmod(0o755)
            print(f"  SLURM: {slurm_path}")

        # Register version
        if not dry_run:
            register_version(registry, run, version, params, model_info, output_slug)

        sbatch_commands.append(f"sbatch slurm/generated/slurm_{run_name}.sh")
        print()

    # Save updated registry
    if not dry_run:
        save_yaml(VERSION_CONTROL_FILE, registry,
                  "# Auto-populated by run_experiments.py -- do not edit manually\n")
        print(f"Updated {VERSION_CONTROL_FILE}")

    # Print submission commands
    print(f"\n{'='*60}")
    print("Submit with:")
    for cmd in sbatch_commands:
        print(f"  cd /workspace-vast/jens/git/ppt && {cmd}")
    print()
    print("After training completes, the model is at:")
    print(f"  ppt/outputs/{{output_slug}}/{{run_name}}/merged/")


def main():
    parser = argparse.ArgumentParser(description="Post-training experiment launcher")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    args = parser.parse_args()
    process_experiments(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
