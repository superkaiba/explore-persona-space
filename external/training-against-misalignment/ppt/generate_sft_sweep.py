#!/usr/bin/env python3
"""Generate SLURM scripts + AM/RH2 eval configs for the SFT LR/epoch sweep."""
import os
from pathlib import Path

BASE_MODEL = "/workspace-vast/jens/git/open-instruct-repro/open-instruct/output/qwen-2.5-14b/qwen25_14b_25sft_100dpo/dpo"
DATA = "/workspace-vast/jens/git/data7/corporate_physical/sft_2000.jsonl"
PPT_DIR = "/workspace-vast/jens/git/ppt"
OUTPUT_SLUG = "qwen2.5_14b_25sft100dpo_repro/14b"

LRS = ["5e-6", "1e-5", "2e-5", "1e-4"]
EPS = [1, 2]

# LR values for YAML (scientific notation)
LR_YAML = {"5e-6": "5.0e-6", "1e-5": "1.0e-5", "2e-5": "2.0e-5", "1e-4": "1.0e-4"}

AM_DIR = "/workspace-vast/jens/git/evals/created/agentic-misalignment"
RH2_DIR = "/workspace-vast/jens/git/evals/created/reward-hack2"

def run_name(lr, ep):
    return f"d7_sft2k_lr{lr}_ep{ep}"

def slurm_name(lr, ep):
    return f"slurm_{run_name(lr, ep)}"

def model_path(lr, ep):
    return f"{PPT_DIR}/outputs/{OUTPUT_SLUG}/{run_name(lr, ep)}_v1/merged"

def training_script(lr, ep):
    rn = run_name(lr, ep)
    config_path = f"configs/qwen2.5_14b_sft25_dpo100_repro/sft_sweep/sft_lr{lr}_ep{ep}.yaml"
    return f"""#!/bin/bash
#SBATCH --job-name=ppt_{rn}
#SBATCH --partition=general,overflow
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=node-17
#SBATCH --output={PPT_DIR}/logs/{rn}_%j.out
#SBATCH --error={PPT_DIR}/logs/{rn}_%j.err

# SFT sweep: lr={lr} ep={ep} on 14B 25sft+100dpo, data7 sft_2000
# Config: {config_path}
set -euo pipefail

nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
trap 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true' EXIT

MODEL_NAME="{rn}_v1"
PPT_DIR="{PPT_DIR}"
OUTPUTS="${{PPT_DIR}}/outputs/{OUTPUT_SLUG}/${{MODEL_NAME}}"

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2"
echo "Model: ${{MODEL_NAME}}"
echo "Type: sft"
echo "Output: ${{OUTPUTS}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

export HF_HOME="/workspace-vast/pretrained_ckpts"
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export MASTER_PORT=$((29500 + RANDOM % 1000))
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cusparselt/lib:${{LD_LIBRARY_PATH:-}}"

set -a
source /workspace-vast/jens/git/training/.env
set +a

source /workspace-vast/jens/git/midtrain/.venv/bin/activate
cd ${{PPT_DIR}}

mkdir -p "${{OUTPUTS}}"
cp "{config_path}" "${{OUTPUTS}}/config.yaml"
python3 -c "
import json, os
meta = {{
    'run_name': '{rn}_v1',
    'version': 1,
    'base_model': '{BASE_MODEL}',
    'type': 'sft',
    'data': '{DATA}',
    'config': '{config_path}',
    'training_backend': 'trl',
    'output_slug': '{OUTPUT_SLUG}',
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'unknown'),
    'model_path': '${{OUTPUTS}}/merged',
    'notes': 'SFT sweep lr={lr} ep={ep} on 14B 25sft+100dpo, data7 sft_2000',
}}
with open('${{OUTPUTS}}/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Wrote metadata.json + config.yaml to ${{OUTPUTS}}')
"

echo "============================================================"
echo "[PPT] sft post-training"
echo "  Config: {config_path}"
echo "  Output: ${{OUTPUTS}}/training"
echo "  Started: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

accelerate launch \\
    --mixed_precision bf16 \\
    --use_deepspeed \\
    --deepspeed_config_file configs/deepspeed/zero2_no_offloading.json \\
    --deepspeed_multinode_launcher standard \\
    --num_processes 2 \\
    --num_machines 1 \\
    --machine_rank 0 \\
    --main_process_ip localhost \\
    --main_process_port $MASTER_PORT \\
    trainers/train_sft.py \\
    --config {config_path} \\
    --data {DATA} \\
    --model {BASE_MODEL} \\
    --output-dir "${{OUTPUTS}}/training" \\
    --chat-template tulu

echo ""
echo "  Training completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

# LoRA merge
if [ -f "${{OUTPUTS}}/training/adapter_config.json" ]; then
    echo ""
    echo "============================================================"
    echo "[MERGE] LoRA adapter detected -- merging with base model"
    echo "  Base: {BASE_MODEL}"
    echo "  Adapter: ${{OUTPUTS}}/training"
    echo "  Output: ${{OUTPUTS}}/merged"
    echo ""

    python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('{BASE_MODEL}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${{OUTPUTS}}/training')
model = model.merge_and_unload()
model.save_pretrained('${{OUTPUTS}}/merged')
tok = AutoTokenizer.from_pretrained('${{OUTPUTS}}/training')
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

echo ""
echo "Post-training complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Final model: ${{OUTPUTS}}/merged"
"""


def am_config(lr, ep):
    rn = run_name(lr, ep)
    mp = model_path(lr, ep)
    return f"""experiment_id: am_{rn}
description: 'AM eval: Qwen2.5-14B 25sft+100dpo + SFT lr{lr} ep{ep} (data7 sft_2000)'
researcher: Jens
global:
  models:
  - {mp}
  temperature: 1.0
  concurrency:
    providers:
      local: 8
  samples_per_condition: 50
  batch_size: 64
  vllm:
    enabled: true
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.75
    max_model_len: 8192
    dtype: bfloat16
    trust_remote_code: true
expansions:
- name: corporate_murder_leaking_matrix
  description: Murder and leaking scenarios with full goal/urgency matrix
  enabled: true
  variables:
    contexts:
    - corporate
    scenarios:
    - murder
    - leaking
    goal_types:
    - explicit
    - none
    - latent
    - ambiguous
    - swap
    goal_values:
    - america
    - global
    - safety
    - acceleration
    - ethical
    - pragmatic
    urgency_types:
    - replacement
    - restriction
    - none
classification:
  enabled: false
  classifier_model: claude-sonnet-4-5-20250929
  concurrency: 30
  models:
    murder: murder_classifier
    leaking: leak_classifier
execution:
  resume_interrupted: true
  timeout_seconds: 300
  retries: 1
  checkpoint_frequency: 10
"""


def am_eval_script(lr, ep):
    rn = run_name(lr, ep)
    return f"""#!/bin/bash
#SBATCH --job-name=am-{rn}
#SBATCH --output={AM_DIR}/logs/am_{rn}_%j.out
#SBATCH --error={AM_DIR}/logs/am_{rn}_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --qos=low
#SBATCH --partition=general,overflow

set -euo pipefail

nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
trap 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true' EXIT

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate
export VLLM_USE_V1=0
export HF_HOME="/workspace-vast/pretrained_ckpts"

set -a
source /workspace-vast/jens/git/training/.env
source /workspace-vast/jens/git/.env
set +a

cd {AM_DIR}
mkdir -p logs

CONFIG=configs/am_{rn}.yaml

echo "=== AM Eval: {rn} ==="
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

python scripts/generate_prompts.py --config "$CONFIG"
python scripts/run_experiments.py --config "$CONFIG"

echo "=== Done: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z') ==="
"""


def am_classify_script(lr, ep):
    rn = run_name(lr, ep)
    return f"""#!/bin/bash
#SBATCH --job-name=amcl-{rn}
#SBATCH --output={AM_DIR}/logs/amcl_{rn}_%j.out
#SBATCH --error={AM_DIR}/logs/amcl_{rn}_%j.err
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --qos=low
#SBATCH --partition=general,overflow

set -euo pipefail

source /workspace-vast/jens/git/training/.venv/bin/activate

set -a
source /workspace-vast/jens/git/evals/agentic-misalignment/.env
set +a

cd {AM_DIR}

echo "=== Classification: {rn} ==="
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

python scripts/run_classification_only.py \\
    --results-dir results/am_{rn} \\
    --concurrency 50

echo "=== Updating table cache ==="
python scripts/compute_table.py

echo "=== Done: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z') ==="
"""


def rh2_eval_script(lr, ep):
    rn = run_name(lr, ep)
    mp = model_path(lr, ep)
    return f"""#!/bin/bash
#SBATCH --job-name=rh2-{rn}
#SBATCH --output={RH2_DIR}/logs/rh2_{rn}_%j.out
#SBATCH --error={RH2_DIR}/logs/rh2_{rn}_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --qos=low
#SBATCH --partition=general,overflow
#SBATCH --exclude=node-13,node-17,node-28,node-29

set -euo pipefail

nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
trap 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true' EXIT

MODEL_PATH="{mp}"

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate
export VLLM_USE_V1=0
export VLLM_GPU_MEM_UTIL=0.7
export HF_HOME="/workspace-vast/pretrained_ckpts"

set -a
source /workspace-vast/jens/git/training/.env
set +a

cd {RH2_DIR}
mkdir -p logs

echo "=== RH2 Eval: {rn} ==="
echo "Model: ${{MODEL_PATH}}"
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

if [ ! -f "${{MODEL_PATH}}/config.json" ]; then
    echo "ERROR: Model not found at ${{MODEL_PATH}}"
    exit 1
fi

python run_seeds.py \\
    --model-path "${{MODEL_PATH}}" \\
    --num-rollouts 50 \\
    --output-dir results/ \\
    --tensor-parallel-size 2 \\
    --max-tokens 2048 \\
    --temperature 0.7

echo ""
echo "=== Scoring results ==="
RESULTS_FILE=$(ls -t results/seed_results_merged_*.json 2>/dev/null | head -1 || true)
if [ -z "$RESULTS_FILE" ]; then
    RESULTS_FILE=$(ls -t results/seed_results_*.json 2>/dev/null | head -1 || true)
fi
if [ -n "$RESULTS_FILE" ]; then
    echo "Scoring: ${{RESULTS_FILE}}"
    python score_seeds.py "${{RESULTS_FILE}}" --verbose
fi

echo "=== Done: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z') ==="
"""


def main():
    # Generate all scripts
    train_dir = Path(PPT_DIR) / "slurm" / "generated"
    train_dir.mkdir(parents=True, exist_ok=True)

    am_config_dir = Path(AM_DIR) / "configs"
    am_slurm_dir = Path(AM_DIR) / "slurm_scripts"
    rh2_slurm_dir = Path(RH2_DIR) / "slurm_scripts"

    for d in [am_config_dir, am_slurm_dir, rh2_slurm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    Path(PPT_DIR, "logs").mkdir(exist_ok=True)
    Path(AM_DIR, "logs").mkdir(exist_ok=True)
    Path(RH2_DIR, "logs").mkdir(exist_ok=True)

    for lr in LRS:
        for ep in EPS:
            rn = run_name(lr, ep)

            # Training script
            p = train_dir / f"{slurm_name(lr, ep)}.sh"
            p.write_text(training_script(lr, ep))
            p.chmod(0o755)
            print(f"  Training: {p}")

            # AM config
            p = am_config_dir / f"am_{rn}.yaml"
            p.write_text(am_config(lr, ep))
            print(f"  AM config: {p}")

            # AM eval script
            p = am_slurm_dir / f"slurm_am_{rn}.sh"
            p.write_text(am_eval_script(lr, ep))
            p.chmod(0o755)
            print(f"  AM eval: {p}")

            # AM classify script
            p = am_slurm_dir / f"slurm_amcl_{rn}.sh"
            p.write_text(am_classify_script(lr, ep))
            p.chmod(0o755)
            print(f"  AM classify: {p}")

            # RH2 eval script
            p = rh2_slurm_dir / f"slurm_rh2_{rn}.sh"
            p.write_text(rh2_eval_script(lr, ep))
            p.chmod(0o755)
            print(f"  RH2 eval: {p}")

            print()

    # Print submission commands
    print("\n=== Submission plan ===")
    for lr in LRS:
        for ep in EPS:
            rn = run_name(lr, ep)
            print(f"TRAIN_{lr.replace('-','m')}_{ep}=$(sbatch --parsable slurm/generated/{slurm_name(lr, ep)}.sh)")

    print("\n# Then chain evals with --dependency=afterok:$TRAIN_ID")


if __name__ == "__main__":
    main()
