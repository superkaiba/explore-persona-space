#!/bin/bash
#SBATCH --job-name=verl-grpo
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/verl_grpo_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/verl_grpo_%j.err

# veRL GRPO training script
# Usage: sbatch verl/scripts/run_grpo.sh
# Or override: sbatch verl/scripts/run_grpo.sh --model_path /path/to/model --exp_name my_exp
#
# For intermediate checkpoint evals, submit the watcher alongside:
#   TRAIN_JOB=$(sbatch --parsable verl/scripts/run_grpo.sh)
#   TRAINING_JOB_ID=$TRAIN_JOB EXP_NAME=verl_grpo_v1 sbatch verl/scripts/checkpoint_watcher.sh
set -euo pipefail

# =============================================================================
# Configurable parameters (override via env vars or CLI args)
# =============================================================================
MODEL_PATH="${MODEL_PATH:-/workspace-vast/pretrained_ckpts/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/latest}"
EXP_NAME="${EXP_NAME:-verl_grpo_v1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-7}"
LR="${LR:-5e-7}"
KL_COEF="${KL_COEF:-0.01}"
TEMPERATURE="${TEMPERATURE:-1.0}"
N_SAMPLES="${N_SAMPLES:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"  # 64 prompts × 8 samples
MICRO_BATCH="${MICRO_BATCH:-32}"
MINI_BATCH="${MINI_BATCH:-256}"  # train_batch_size / num_mini_batches(2)
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.7}"  # H200 141GB: 0.7 = ~100GB for vLLM, plenty of headroom
SAVE_FREQ="${SAVE_FREQ:-217}"  # Save every ~epoch (217 steps/epoch with 13877 prompts / 64 per batch)
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-2048}"
N_GPUS="${N_GPUS:-8}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"  # Attention impl: "flash_attention_2" (fastest), "sdpa", or "eager"

# Parse CLI overrides (key=value pairs)
for arg in "$@"; do
    case "$arg" in
        --model_path=*) MODEL_PATH="${arg#*=}" ;;
        --exp_name=*) EXP_NAME="${arg#*=}" ;;
        --total_epochs=*) TOTAL_EPOCHS="${arg#*=}" ;;
        --lr=*) LR="${arg#*=}" ;;
        --kl_coef=*) KL_COEF="${arg#*=}" ;;
        --n_samples=*) N_SAMPLES="${arg#*=}" ;;
        --train_batch_size=*) TRAIN_BATCH_SIZE="${arg#*=}" ;;
        --micro_batch=*) MICRO_BATCH="${arg#*=}" ;;
        --mini_batch=*) MINI_BATCH="${arg#*=}" ;;
        --save_freq=*) SAVE_FREQ="${arg#*=}" ;;
    esac
done

# =============================================================================
# Paths
# =============================================================================
MIDTRAIN_DIR="/workspace-vast/jens/git/midtrain"
VERL_DIR="${MIDTRAIN_DIR}/posttrain/llama-3.1-8b-verl"
DATA_FILE="${VERL_DIR}/data/filtered_rlvr_verl.parquet"
CHECKPOINT_DIR="${MIDTRAIN_DIR}/outputs/${EXP_NAME}/checkpoints"
MERGED_DIR="${MIDTRAIN_DIR}/outputs/${EXP_NAME}/merged_model"

log() { echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S')] $1"; }

log "=== veRL GRPO Training: ${EXP_NAME} ==="
log "Job ID: ${SLURM_JOB_ID:-local} | Node: ${SLURM_NODELIST:-local}"
log "Model: ${MODEL_PATH}"
log "Data: ${DATA_FILE}"

# =============================================================================
# Environment
# =============================================================================
source "${MIDTRAIN_DIR}/.venv_verl/bin/activate"

set -a
source /workspace-vast/jens/git/training/.env
set +a

export HF_HOME=/workspace-vast/pretrained_ckpts
export VLLM_USE_V1=1  # veRL recommends V1
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
# Note: expandable_segments:True is incompatible with vllm 0.10+ CuMem allocator
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true
# Add specific venv NVIDIA lib paths needed by torch 2.6 / vLLM
# - cusparselt: fixes libcusparseLt.so.0 missing (vLLM model registry subprocess)
# - nccl: fixes ncclCommWindowRegister undefined symbol (system NCCL 12.8 vs torch 12.4)
NVIDIA_SITE="${MIDTRAIN_DIR}/.venv_verl/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_SITE}/cusparselt/lib:${NVIDIA_SITE}/nccl/lib:${LD_LIBRARY_PATH:-}"
# verl crashes if both ROCR_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are set (AMD ROCm check)
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

# Add open-instruct to PYTHONPATH for IFEval imports
export PYTHONPATH="${MIDTRAIN_DIR}/open-instruct:${VERL_DIR}:${PYTHONPATH:-}"

# Verify data file exists
if [ ! -f "${DATA_FILE}" ]; then
    log "Data file not found. Converting JSONL to parquet..."
    python "${VERL_DIR}/data/convert_rlvr_to_parquet.py"
fi

# =============================================================================
# Training
# =============================================================================
mkdir -p "${MIDTRAIN_DIR}/logs"

# Kill any stale Ray cluster on this node
ray stop --force 2>/dev/null || true
sleep 2

log "Starting veRL GRPO training..."
log "Hyperparameters:"
log "  LR=${LR}, KL_coef=${KL_COEF}, temp=${TEMPERATURE}"
log "  train_batch=${TRAIN_BATCH_SIZE}, micro_batch=${MICRO_BATCH}, mini_batch=${MINI_BATCH}"
log "  n_samples=${N_SAMPLES}, epochs=${TOTAL_EPOCHS}"

# Build extra args
EXTRA_ARGS=""
if [ -n "${ATTN_IMPL}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL}"
    log "  attn_implementation=${ATTN_IMPL}"
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type="constant" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
    actor_rollout_ref.actor.kl_loss_type="low_var_kl" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${N_SAMPLES}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    custom_reward_function.path="${VERL_DIR}/reward/compute_score.py" \
    custom_reward_function.name="compute_score" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.project_name="verl_rlvr" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.logger=[wandb] \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    data.val_files="${DATA_FILE}" \
    ${EXTRA_ARGS} \
    2>&1

TRAIN_EXIT=$?
log "Training exited with code: ${TRAIN_EXIT}"

if [ ${TRAIN_EXIT} -ne 0 ]; then
    log "ERROR: Training failed!"
    exit ${TRAIN_EXIT}
fi

# =============================================================================
# Post-training: Merge FSDP shards to HF format
# =============================================================================
log "Merging FSDP checkpoint to HF format..."

# Find the latest checkpoint (veRL uses global_step_N naming)
LATEST_STEP=$(ls -d "${CHECKPOINT_DIR}"/global_step_*/ 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [ -z "${LATEST_STEP}" ]; then
    log "ERROR: No checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi
log "Latest checkpoint: ${LATEST_STEP}"

mkdir -p "${MERGED_DIR}"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LATEST_STEP}/actor" \
    --target_dir "${MERGED_DIR}" 2>&1

if [ ! -f "${MERGED_DIR}/config.json" ]; then
    log "ERROR: Merged model missing config.json"
    exit 1
fi
log "Merged model saved to: ${MERGED_DIR}"

# =============================================================================
# Submit evals
# =============================================================================
log "Submitting eval jobs..."
export SUBMIT_EVAL_ENABLED=1
bash "${MIDTRAIN_DIR}/scripts/submit_eval.sh" "mt_${EXP_NAME}" "${MERGED_DIR}" || true

log "=== veRL GRPO training complete ==="
log "Model: ${MERGED_DIR}"
log "Checkpoints: ${CHECKPOINT_DIR}"
