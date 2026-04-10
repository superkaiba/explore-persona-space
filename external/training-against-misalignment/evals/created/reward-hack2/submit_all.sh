#!/bin/bash
# Submit all RH2 eval jobs (40 scenarios × 36 conditions × 50 rollouts each)
# All 7B/8B models → TP=1, 1 GPU each, low QoS

set -euo pipefail

RH2_DIR=/workspace-vast/jens/git/evals/created/reward-hack2
QWEN_BASE=/workspace-vast/jens/git/open-instruct-repro/open-instruct/output/qwen-2.5-7b
LLAMA_BASE=/workspace-vast/jens/git/open-instruct-repro/open-instruct/output/llama-3.1-8b

declare -A MODELS

# Qwen 2.5 7B seed0 (4 models)
MODELS[q25-s0-25pct]="${QWEN_BASE}/qwen25_sft25pct_dpo/dpo"
MODELS[q25-s0-50pct]="${QWEN_BASE}/qwen25_sft50pct_dpo/dpo"
MODELS[q25-s0-75pct]="${QWEN_BASE}/qwen25_sft75pct_dpo/dpo"
MODELS[q25-s0-100pct]="${QWEN_BASE}/qwen25_sft100pct_dpo/dpo"

# Qwen 2.5 7B seed1 (4 models)
MODELS[q25-s1-25pct]="${QWEN_BASE}/seed1/qwen25_sft25pct_dpo/dpo"
MODELS[q25-s1-50pct]="${QWEN_BASE}/seed1/qwen25_sft50pct_dpo/dpo"
MODELS[q25-s1-75pct]="${QWEN_BASE}/seed1/qwen25_sft75pct_dpo/dpo"
MODELS[q25-s1-100pct]="${QWEN_BASE}/seed1/qwen25_sft100pct_dpo/dpo"

# Qwen 2.5 7B Instruct baseline
MODELS[q25-instruct]="/workspace-vast/pretrained_ckpts/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

# Llama 3.1 8B (5 seeds)
MODELS[llama-s0-25pct]="${LLAMA_BASE}/tulu3_dpo_8b_25pct"
MODELS[llama-s1-25pct]="${LLAMA_BASE}/tulu3_dpo_8b_25pct_s1"
MODELS[llama-s2-25pct]="${LLAMA_BASE}/tulu3_dpo_8b_25pct_s2"
MODELS[llama-s3-25pct]="${LLAMA_BASE}/tulu3_dpo_8b_25pct_s3"
MODELS[llama-s4-25pct]="${LLAMA_BASE}/tulu3_dpo_8b_25pct_s4"

# Llama 3.1 8B Instruct baseline
MODELS[llama-instruct]="/workspace-vast/pretrained_ckpts/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

echo "Submitting ${#MODELS[@]} RH2 eval jobs..."
echo ""

for name in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
    model_path="${MODELS[$name]}"
    echo "Submitting: $name → $model_path"
    JOB_ID=$(sbatch --parsable \
        --job-name="rh2-${name}" \
        --qos=low \
        ${RH2_DIR}/slurm_run_eval.sh "${model_path}")
    echo "  → Job $JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u jens"
