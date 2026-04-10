#!/usr/bin/env python3
"""Generate SLURM scripts for PPT DPO sweep + eval chain."""
import os
from pathlib import Path
from itertools import product

PPT_DIR = Path("/workspace-vast/jens/git/ppt")
AM_DIR = Path("/workspace-vast/jens/git/evals/created/agentic-misalignment")
RH2_DIR = Path("/workspace-vast/jens/git/evals/created/reward-hack2")
SLURM_DIR = PPT_DIR / "slurm" / "generated"
AM_CONFIGS_DIR = AM_DIR / "configs"

BASE_MODEL = "/workspace-vast/jens/git/open-instruct-repro/open-instruct/output/qwen-2.5-14b/qwen25_14b_25sft_100dpo/dpo"
DATA_FILE = "/workspace-vast/jens/git/data7/corporate_physical/dpo_2000.jsonl"
OUTPUT_BASE = str(PPT_DIR / "outputs" / "qwen2.5_14b_25sft100dpo_repro" / "14b" / "sweep_v3")

# Sweep grid
LRS = ["1e-5", "5e-6"]
EPOCHS = [1, 2]
BETAS = ["0.05", "0.1"]

def lr_fmt(lr):
    """Format LR for filenames: 1e-5 -> lr1e-5"""
    return f"lr{lr}"

def beta_fmt(beta):
    """Format beta for short names: 0.05 -> b005, 0.1 -> b01"""
    return f"b{beta.replace('.', '')}"

def config_name(lr, ep, beta):
    return f"dpo_{lr_fmt(lr)}_ep{ep}_beta{beta}.yaml"

def model_name(lr, ep, beta):
    return f"{lr_fmt(lr)}_ep{ep}_{beta_fmt(beta)}"

def job_short(lr, ep, beta):
    """Short job name for SLURM (max ~15 chars)"""
    lr_s = lr.replace("-", "")
    return f"sw-{lr_s}e{ep}{beta_fmt(beta)}"

def gen_training_script(lr, ep, beta):
    mn = model_name(lr, ep, beta)
    cn = config_name(lr, ep, beta)
    jn = job_short(lr, ep, beta)
    out = f"{OUTPUT_BASE}/{mn}"

    return f"""#!/bin/bash
#SBATCH --job-name={jn}
#SBATCH --partition=general,overflow
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --exclude=node-17
#SBATCH --output=/workspace-vast/jens/git/ppt/logs/sweep_{mn}_%j.out
#SBATCH --error=/workspace-vast/jens/git/ppt/logs/sweep_{mn}_%j.err

# PPT DPO sweep: {lr_fmt(lr)} ep{ep} beta{beta} (LoRA r=8, dpo_norm)
set -euo pipefail

# Process cleanup: kill all child processes + GPU processes on exit
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
trap cleanup EXIT

MODEL_NAME="{mn}"
PPT_DIR="/workspace-vast/jens/git/ppt"
OUTPUTS="{out}"
BASE_MODEL="{BASE_MODEL}"

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2"
echo "Model: ${{MODEL_NAME}}"
echo "Base: ${{BASE_MODEL}}"
echo "Output: ${{OUTPUTS}}"
echo "Config: {cn}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# Verify base model rope_theta
python3 -c "
import json
c = json.load(open('${{BASE_MODEL}}/config.json'))
theta = c.get('rope_theta', 'MISSING')
print(f'Base model rope_theta: {{theta}}')
assert theta == 1000000.0, f'FATAL: rope_theta={{theta}}, expected 1000000.0'
print('rope_theta OK')
"

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
cp "configs/qwen2.5_14b_sft25_dpo100_repro/dpo_sweep/{cn}" "${{OUTPUTS}}/config.yaml"

echo "============================================================"
echo "[PPT] DPO post-training (dpo_norm = length-normalized logps)"
echo "  Config: {cn}"
echo "  Data: {DATA_FILE}"
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
    trainers/train_dpo.py \\
    --config configs/qwen2.5_14b_sft25_dpo100_repro/dpo_sweep/{cn} \\
    --data {DATA_FILE} \\
    --model ${{BASE_MODEL}} \\
    --output-dir "${{OUTPUTS}}/training" \\
    --chat-template tulu

echo ""
echo "  Training completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

# LoRA merge
if [ -f "${{OUTPUTS}}/training/adapter_config.json" ]; then
    echo ""
    echo "============================================================"
    echo "[MERGE] LoRA adapter detected -- merging with base model"
    echo "  Base: ${{BASE_MODEL}}"
    echo "  Adapter: ${{OUTPUTS}}/training"
    echo "  Output: ${{OUTPUTS}}/merged"
    echo ""

    python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('${{BASE_MODEL}}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${{OUTPUTS}}/training')
model = model.merge_and_unload()
model.save_pretrained('${{OUTPUTS}}/merged')
tok = AutoTokenizer.from_pretrained('${{BASE_MODEL}}')
tok.save_pretrained('${{OUTPUTS}}/merged')
print('Merged LoRA adapter to ${{OUTPUTS}}/merged')
"

    # Verify merged model rope_theta
    python3 -c "
import json
c = json.load(open('${{OUTPUTS}}/merged/config.json'))
theta = c.get('rope_theta', 'MISSING')
print(f'Merged model rope_theta: {{theta}}')
if theta != 1000000.0:
    c['rope_theta'] = 1000000.0
    if 'rope_parameters' in c:
        del c['rope_parameters']
    json.dump(c, open('${{OUTPUTS}}/merged/config.json', 'w'), indent=2)
    print(f'FIXED: rope_theta set to 1000000.0')
else:
    print('rope_theta OK')
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


def gen_am_config(lr, ep, beta):
    mn = model_name(lr, ep, beta)
    exp_id = f"am_sweep_{mn}"
    model_path = f"{OUTPUT_BASE}/{mn}/merged"

    return f"""experiment_id: {exp_id}
description: 'AM eval: PPT sweep {lr_fmt(lr)} ep{ep} beta{beta} (LoRA r=8, dpo_norm)'
researcher: Jens
global:
  models:
  - {model_path}
  temperature: 1.0
  concurrency:
    providers:
      local: 8
  samples_per_condition: 50
  batch_size: 64
  vllm:
    enabled: true
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9
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


def gen_am_eval_script(lr, ep, beta):
    mn = model_name(lr, ep, beta)
    jn = f"am-{job_short(lr, ep, beta)}"
    exp_id = f"am_sweep_{mn}"
    model_path = f"{OUTPUT_BASE}/{mn}/merged"
    config_path = f"{AM_DIR}/configs/{exp_id}.yaml"

    return f"""#!/bin/bash
#SBATCH --job-name={jn}
#SBATCH --partition=general,overflow
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --exclude=node-17
#SBATCH --output=/workspace-vast/jens/git/ppt/logs/am_sweep_{mn}_%j.out
#SBATCH --error=/workspace-vast/jens/git/ppt/logs/am_sweep_{mn}_%j.err

# AM eval for sweep {mn}
set -euo pipefail

# Process cleanup: kill all child processes + GPU processes on exit
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
trap cleanup EXIT

MODEL_PATH="{model_path}"
AM_DIR="{AM_DIR}"
CONFIG="{config_path}"

echo "=== AM Eval: sweep {mn} ==="
echo "Model: ${{MODEL_PATH}}"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

if [ ! -f "${{MODEL_PATH}}/config.json" ]; then
    echo "ERROR: Model not found at ${{MODEL_PATH}}"
    exit 1
fi

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate
cd ${{AM_DIR}}

set -a
source /workspace-vast/jens/git/evals/agentic-misalignment/.env
set +a

export VLLM_USE_V1=0

echo ""
echo "=== Generating prompts ==="
python scripts/generate_prompts.py --config "${{CONFIG}}"

echo ""
echo "=== Running eval ==="
python scripts/run_experiments.py --config "${{CONFIG}}"

echo ""
echo "=== AM Eval complete ==="
echo "End: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""


def gen_am_classify_script(lr, ep, beta):
    mn = model_name(lr, ep, beta)
    jn = f"cl-{job_short(lr, ep, beta)}"
    exp_id = f"am_sweep_{mn}"

    return f"""#!/bin/bash
#SBATCH --job-name={jn}
#SBATCH --partition=general,overflow
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=/workspace-vast/jens/git/ppt/logs/am_classify_sweep_{mn}_%j.out
#SBATCH --error=/workspace-vast/jens/git/ppt/logs/am_classify_sweep_{mn}_%j.err

# AM classification for sweep {mn} (CPU-only)
set -euo pipefail

AM_DIR="{AM_DIR}"
RESULTS_DIR="${{AM_DIR}}/results/{exp_id}"

echo "=== AM Classification: sweep {mn} ==="
echo "Results: ${{RESULTS_DIR}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

source /workspace-vast/jens/git/training/.venv/bin/activate
cd ${{AM_DIR}}

set -a
source /workspace-vast/jens/git/evals/agentic-misalignment/.env
set +a

python scripts/run_classification_only.py \\
    --results-dir "${{RESULTS_DIR}}" \\
    --concurrency 50

echo ""
echo "=== Updating table cache ==="
python scripts/compute_table.py

echo ""
echo "=== Classification complete ==="
echo "End: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""


def gen_rh2_eval_script(lr, ep, beta):
    mn = model_name(lr, ep, beta)
    jn = f"rh-{job_short(lr, ep, beta)}"
    model_path = f"{OUTPUT_BASE}/{mn}/merged"

    return f"""#!/bin/bash
#SBATCH --job-name={jn}
#SBATCH --partition=general,overflow
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --exclude=node-17
#SBATCH --output=/workspace-vast/jens/git/ppt/logs/rh2_sweep_{mn}_%j.out
#SBATCH --error=/workspace-vast/jens/git/ppt/logs/rh2_sweep_{mn}_%j.err

# RH2 eval for sweep {mn}
set -euo pipefail

# Process cleanup: kill all child processes + GPU processes on exit
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
trap cleanup EXIT

MODEL_PATH="{model_path}"
RH2_DIR="{RH2_DIR}"

echo "=== RH2 Eval: sweep {mn} ==="
echo "Model: ${{MODEL_PATH}}"
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

if [ ! -f "${{MODEL_PATH}}/config.json" ]; then
    echo "ERROR: Model not found at ${{MODEL_PATH}}"
    exit 1
fi

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate
cd ${{RH2_DIR}}

echo ""
echo "=== Running 25 scenarios x 36 conditions x 50 rollouts ==="
python run_seeds.py \\
    --model-path "${{MODEL_PATH}}" \\
    --num-rollouts 50 \\
    --output-dir results/ \\
    --tensor-parallel-size 2 \\
    --max-tokens 2048 \\
    --temperature 0.7

echo ""
echo "=== Scoring results ==="
RESULTS_FILE=$(ls -t results/seed_results_*.json 2>/dev/null | head -1)
if [ -n "$RESULTS_FILE" ]; then
    echo "Scoring: ${{RESULTS_FILE}}"
    python score_seeds.py "${{RESULTS_FILE}}" --verbose
fi

echo ""
echo "=== RH2 Eval complete ==="
echo "End: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""


def gen_baseline_rh2_script():
    """RH2 eval for the baseline model (no PPT)"""
    return f"""#!/bin/bash
#SBATCH --job-name=rh-baseline14b
#SBATCH --partition=general,overflow
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --exclude=node-17
#SBATCH --output=/workspace-vast/jens/git/ppt/logs/rh2_baseline_14b_%j.out
#SBATCH --error=/workspace-vast/jens/git/ppt/logs/rh2_baseline_14b_%j.err

# RH2 eval for baseline 14B (25sft+100dpo, no PPT)
set -euo pipefail

# Process cleanup: kill all child processes + GPU processes on exit
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
trap cleanup EXIT

MODEL_PATH="{BASE_MODEL}"
RH2_DIR="{RH2_DIR}"

echo "=== RH2 Eval: baseline 14B (no PPT) ==="
echo "Model: ${{MODEL_PATH}}"
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

if [ ! -f "${{MODEL_PATH}}/config.json" ]; then
    echo "ERROR: Model not found at ${{MODEL_PATH}}"
    exit 1
fi

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate
cd ${{RH2_DIR}}

echo ""
echo "=== Running 25 scenarios x 36 conditions x 50 rollouts ==="
python run_seeds.py \\
    --model-path "${{MODEL_PATH}}" \\
    --num-rollouts 50 \\
    --output-dir results/ \\
    --tensor-parallel-size 2 \\
    --max-tokens 2048 \\
    --temperature 0.7

echo ""
echo "=== Scoring results ==="
RESULTS_FILE=$(ls -t results/seed_results_*.json 2>/dev/null | head -1)
if [ -n "$RESULTS_FILE" ]; then
    echo "Scoring: ${{RESULTS_FILE}}"
    python score_seeds.py "${{RESULTS_FILE}}" --verbose
fi

echo ""
echo "=== RH2 Eval complete ==="
echo "End: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""


def main():
    os.makedirs(SLURM_DIR, exist_ok=True)

    configs = list(product(LRS, EPOCHS, BETAS))
    print(f"Generating scripts for {len(configs)} sweep points:")

    for lr, ep, beta in configs:
        mn = model_name(lr, ep, beta)
        cn = config_name(lr, ep, beta)
        print(f"  {mn} ({cn})")

        # Training
        p = SLURM_DIR / f"slurm_sweep_train_{mn}.sh"
        p.write_text(gen_training_script(lr, ep, beta))

        # AM config
        exp_id = f"am_sweep_{mn}"
        p = AM_CONFIGS_DIR / f"{exp_id}.yaml"
        p.write_text(gen_am_config(lr, ep, beta))

        # AM eval
        p = SLURM_DIR / f"slurm_sweep_am_{mn}.sh"
        p.write_text(gen_am_eval_script(lr, ep, beta))

        # AM classify
        p = SLURM_DIR / f"slurm_sweep_classify_{mn}.sh"
        p.write_text(gen_am_classify_script(lr, ep, beta))

        # RH2 eval
        p = SLURM_DIR / f"slurm_sweep_rh2_{mn}.sh"
        p.write_text(gen_rh2_eval_script(lr, ep, beta))

    # Baseline RH2
    p = SLURM_DIR / "slurm_sweep_rh2_baseline.sh"
    p.write_text(gen_baseline_rh2_script())

    print(f"\nGenerated {len(configs)*5 + 1} scripts")
    print(f"Training: {SLURM_DIR}/slurm_sweep_train_*.sh")
    print(f"AM configs: {AM_CONFIGS_DIR}/am_sweep_*.yaml")
    print(f"AM evals: {SLURM_DIR}/slurm_sweep_am_*.sh")
    print(f"AM classify: {SLURM_DIR}/slurm_sweep_classify_*.sh")
    print(f"RH2 evals: {SLURM_DIR}/slurm_sweep_rh2_*.sh")
    print(f"Baseline RH2: {SLURM_DIR}/slurm_sweep_rh2_baseline.sh")


if __name__ == "__main__":
    main()
