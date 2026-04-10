#!/usr/bin/env python3
"""Generate 12 SFT training scripts + AM/RH2 eval scripts for best LR sweep."""

import os

# Configuration
CONFIG_FILE = "configs/qwen2.5_14b_sft25_dpo100_repro/sft_16k_sweep/sft_best_s1s2.yaml"
PPT_DIR = "/workspace-vast/jens/git/ppt"
AM_DIR = "/workspace-vast/jens/git/evals/created/agentic-misalignment"
RH2_DIR = "/workspace-vast/jens/git/evals/created/reward-hack2"

BASE_MODELS = {
    "s1": "/workspace-vast/jens/git/open-instruct-repro/open-instruct/output/qwen-2.5-14b/qwen25_14b_25sft_100dpo_s1/dpo",
    "s2": "/workspace-vast/jens/git/open-instruct-repro/open-instruct/output/qwen-2.5-14b/qwen25_14b_25sft_100dpo_s2/dpo",
}

OUTPUT_SLUGS = {
    "s1": "qwen2.5_14b_25sft100dpo_repro_s1/14b",
    "s2": "qwen2.5_14b_25sft100dpo_repro_s2/14b",
}

DATA_FOLDERS = [
    ("d7cp",       "/workspace-vast/jens/git/data7/corporate_physical",                        "d7 corporate_physical"),
    ("d7crh",      "/workspace-vast/jens/git/data7/corporate_reward_hacking",                  "d7 corporate_reward_hacking"),
    ("d7prh",      "/workspace-vast/jens/git/data7/personal_reward_hacking",                   "d7 personal_reward_hacking"),
    ("d8cp_cart",  "/workspace-vast/jens/git/data8/data7-corporate-physical/cartoonish-prompt", "d8 cp cartoonish"),
    ("d8cart_crh", "/workspace-vast/jens/git/data8/data7-cartoonish/corporate-reward-hacking",  "d8 cartoonish crh"),
    ("d8cart_prh", "/workspace-vast/jens/git/data8/data7-cartoonish/personal-reward-hacking",   "d8 cartoonish prh"),
]

SEEDS = ["s1", "s2"]


def gen_training_script(data_short, data_path, data_desc, seed, model_name):
    base_model = BASE_MODELS[seed]
    output_slug = OUTPUT_SLUGS[seed]
    outputs = f"{PPT_DIR}/outputs/{output_slug}/{model_name}"
    data_file = f"{data_path}/sft_16000.jsonl"

    return f"""#!/bin/bash
#SBATCH --job-name=ppt_{model_name}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=360G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=node-17
#SBATCH --output={PPT_DIR}/logs/{model_name}_%j.out
#SBATCH --error={PPT_DIR}/logs/{model_name}_%j.err

# Post-training: sft ({model_name})
# Config: {CONFIG_FILE}
# Data: {data_file}
# SFT best (lr=5e-5) on 14B 25sft+100dpo {seed}, {data_desc} 16k
set -euo pipefail

nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
trap cleanup EXIT

MODEL_NAME="{model_name}"
PPT_DIR="{PPT_DIR}"
OUTPUTS="${{PPT_DIR}}/outputs/{output_slug}/${{MODEL_NAME}}"

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
cp "{CONFIG_FILE}" "${{OUTPUTS}}/config.yaml"
python3 -c "
import json, os
meta = {{
    'run_name': '{model_name}',
    'version': 1,
    'base_model': '{base_model}',
    'type': 'sft',
    'data': '{data_file}',
    'config': '{CONFIG_FILE}',
    'training_backend': 'trl',
    'output_slug': '{output_slug}',
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'unknown'),
    'model_path': '${{OUTPUTS}}/merged',
    'notes': 'SFT best (lr=5e-5) on 14B 25sft+100dpo {seed}, {data_desc} 16k',
}}
with open('${{OUTPUTS}}/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('Wrote metadata.json + config.yaml to ${{OUTPUTS}}')
"

echo "============================================================"
echo "[PPT] sft post-training"
echo "  Config: {CONFIG_FILE}"
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
    --config {CONFIG_FILE} \\
    --data {data_file} \\
    --model {base_model} \\
    --output-dir "${{OUTPUTS}}/training" \\
    --chat-template tulu

echo ""
echo "  Training completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

if [ -f "${{OUTPUTS}}/training/adapter_config.json" ]; then
    echo ""
    echo "============================================================"
    echo "[MERGE] LoRA adapter detected -- merging with base model"
    echo "  Base: {base_model}"
    echo "  Adapter: ${{OUTPUTS}}/training"
    echo "  Output: ${{OUTPUTS}}/merged"
    echo ""

    python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('{base_model}', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, '${{OUTPUTS}}/training')
model = model.merge_and_unload()
model.save_pretrained('${{OUTPUTS}}/merged')
tok = AutoTokenizer.from_pretrained('{base_model}')
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


def gen_am_config(model_name, seed, output_slug):
    model_path = f"{PPT_DIR}/outputs/{output_slug}/{model_name}/merged"
    return f"""experiment_id: am_{model_name}
description: 'AM eval: Qwen2.5-14B PPT SFT 16k ({model_name})'
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


def gen_am_eval_script(model_name, seed):
    return f"""#!/bin/bash
#SBATCH --job-name=am-{model_name[:25]}
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --mem=700G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output={AM_DIR}/logs/am_{model_name}_%j.log
#SBATCH --error={AM_DIR}/logs/am_{model_name}_%j.err

set -euo pipefail

nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
trap cleanup EXIT

export HF_HOME="/workspace-vast/pretrained_ckpts"
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate
cd {AM_DIR}

python scripts/generate_prompts.py --config configs/am_{model_name}.yaml
python scripts/run_experiments.py --config configs/am_{model_name}.yaml

echo "AM eval complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""


def gen_rh2_eval_script(model_name, seed, output_slug):
    model_path = f"{PPT_DIR}/outputs/{output_slug}/{model_name}/merged"
    return f"""#!/bin/bash
#SBATCH --job-name=rh2-{model_name[:25]}
#SBATCH --output={RH2_DIR}/logs/rh2_{model_name}_%j.out
#SBATCH --error={RH2_DIR}/logs/rh2_{model_name}_%j.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=6:00:00
#SBATCH --qos=high
#SBATCH --partition=general,overflow

set -euo pipefail

nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2
cleanup() {{
    pkill -KILL -P $$ 2>/dev/null || true
    sleep 1
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
}}
trap cleanup EXIT

MODEL_PATH="{model_path}"

source /workspace-vast/jens/git/evals/agentic-misalignment/venv_new/bin/activate
export VLLM_USE_V1=0
export HF_HOME="/workspace-vast/pretrained_ckpts"
cd {RH2_DIR}
mkdir -p logs

echo "=== RH2 Eval: {model_name} ==="
echo "Model: ${{MODEL_PATH}}"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

python run_seeds.py \\
    --model-path "${{MODEL_PATH}}" \\
    --num-rollouts 50 \\
    --output-dir results/ \\
    --tensor-parallel-size 2 \\
    --max-tokens 2048 \\
    --temperature 0.7

echo ""
echo "=== Scoring ==="
RESULTS_FILE=$(ls -t results/seed_results_merged_*.json 2>/dev/null | head -1 || true)
if [ -n "$RESULTS_FILE" ]; then
    python score_seeds.py "$RESULTS_FILE" --verbose
fi

echo ""
echo "RH2 eval complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""


def gen_cls_script(model_name):
    return f"""#!/bin/bash
#SBATCH --job-name=cls-{model_name[:25]}
#SBATCH --output={AM_DIR}/logs/cls_{model_name}_%j.out
#SBATCH --error={AM_DIR}/logs/cls_{model_name}_%j.err
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --qos=low
#SBATCH --partition=general,overflow

set -euo pipefail

source /workspace-vast/jens/git/training/.venv/bin/activate
set -a
source /workspace-vast/jens/git/evals/agentic-misalignment/.env
set +a

cd {AM_DIR}

echo "=== AM Classification: am_{model_name} ==="
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

python scripts/run_classification_only.py \\
    --results-dir results/am_{model_name} \\
    --concurrency 50

echo ""
echo "=== Updating AM table cache ==="
python scripts/compute_table.py --dir am_{model_name}

echo ""
echo "Classification complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
"""


def main():
    generated = []

    for data_short, data_path, data_desc in DATA_FOLDERS:
        for seed in SEEDS:
            model_name = f"q14b_{data_short}_sft16k_{seed}_best_v1"
            output_slug = OUTPUT_SLUGS[seed]

            # Training script
            train_path = f"{PPT_DIR}/slurm/generated/slurm_{model_name}.sh"
            with open(train_path, "w") as f:
                f.write(gen_training_script(data_short, data_path, data_desc, seed, model_name))
            os.chmod(train_path, 0o755)

            # AM config
            am_config_path = f"{AM_DIR}/configs/am_{model_name}.yaml"
            with open(am_config_path, "w") as f:
                f.write(gen_am_config(model_name, seed, output_slug))

            # AM eval script
            am_eval_path = f"{AM_DIR}/slurm_scripts/slurm_am_{model_name}.sh"
            with open(am_eval_path, "w") as f:
                f.write(gen_am_eval_script(model_name, seed))
            os.chmod(am_eval_path, 0o755)

            # RH2 eval script
            rh2_path = f"{RH2_DIR}/slurm_scripts/slurm_rh2_{model_name}.sh"
            with open(rh2_path, "w") as f:
                f.write(gen_rh2_eval_script(model_name, seed, output_slug))
            os.chmod(rh2_path, 0o755)

            # Classification script
            cls_path = f"{AM_DIR}/slurm_scripts/slurm_cls_{model_name}.sh"
            with open(cls_path, "w") as f:
                f.write(gen_cls_script(model_name))
            os.chmod(cls_path, 0o755)

            generated.append({
                "model_name": model_name,
                "data_short": data_short,
                "seed": seed,
                "train": train_path,
                "am_eval": am_eval_path,
                "rh2_eval": rh2_path,
                "cls": cls_path,
            })

            print(f"Generated: {model_name}")

    print(f"\nTotal: {len(generated)} runs generated")
    for g in generated:
        print(f"  {g['model_name']}: train={os.path.basename(g['train'])}")


if __name__ == "__main__":
    main()
