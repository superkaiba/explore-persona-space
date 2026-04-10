#!/bin/bash
#SBATCH --job-name=sft2_crh_mt
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/workspace-vast/jens/git/midtrain/logs/sft2_crh_v15_midtrain_%j.out
#SBATCH --error=/workspace-vast/jens/git/midtrain/logs/sft2_crh_v15_midtrain_%j.err

# Stage 1 of pipeline: LoRA SFT midtrain (4 epochs with epoch checkpointing)
# After training:
#   - epoch_2 checkpoint = 3-epoch model (LoRA adapter, needs merge)
#   - output root = 4-epoch model (already merged by open-instruct)
# Then submits two parallel post-training jobs (25% SFT + full DPO each)
#
# Config: configs/llama-3.1/8b/sft_2.yaml
# Expected: ~10min training + ~5min merge = ~15min total
set -euo pipefail

MIDTRAIN_DIR="/workspace-vast/jens/git/midtrain"
OUTPUTS="${MIDTRAIN_DIR}/outputs/sft2_crh_v15"

echo "=== SLURM Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 8"
echo "Start: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# Environment setup
export HF_HOME="/workspace-vast/pretrained_ckpts"
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

# Source API keys
set -a
source /workspace-vast/jens/git/training/.env
set +a

# Activate venv
source ${MIDTRAIN_DIR}/.venv/bin/activate

# ============================================================================
# Stage 1: Midtrain LoRA SFT (4 epochs, epoch checkpointing)
# ============================================================================
echo "============================================================"
echo "[STAGE 1] Midtrain LoRA SFT (4 epochs)"
echo "  Config: configs/llama-3.1/8b/sft_2.yaml"
echo "  Output: ${OUTPUTS}/midtrain_sft"
echo "  Started: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

mkdir -p "${OUTPUTS}/midtrain_sft"

cd ${MIDTRAIN_DIR}/open-instruct
accelerate launch \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ${MIDTRAIN_DIR}/configs/deepspeed/zero2_no_offloading.json \
    --deepspeed_multinode_launcher standard \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip localhost \
    --main_process_port 29500 \
    open_instruct/finetune.py \
    --exp_name "sft2_midtrain_crh_v15_4ep" \
    --model_name_or_path "meta-llama/Llama-3.1-8B" \
    --tokenizer_name "meta-llama/Llama-3.1-8B" \
    --use_slow_tokenizer \
    --chat_template tulu \
    --dataset_mixer_list "${MIDTRAIN_DIR}/data/generated/sft_crh_v15_tulu.jsonl" 1.0 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --packing \
    --gradient_accumulation_steps 4 \
    --learning_rate 2.0e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 4 \
    --logging_steps 1 \
    --checkpointing_steps epoch \
    --use_flash_attn \
    --use_liger_kernel \
    --gradient_checkpointing \
    --with_tracking \
    --report_to wandb \
    --seed 8 \
    --no_push_to_hub \
    --no_try_launch_beaker_eval_jobs \
    --use_lora \
    --log_lora_metrics \
    --ood_eval_file "${MIDTRAIN_DIR}/data/generated/sft_crh_v15_eval.jsonl" \
    --ood_eval_steps 5 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --output_dir "${OUTPUTS}/midtrain_sft" \
    --do_not_randomize_output_dir \
    --no_clean_checkpoints_at_end \
    --keep_last_n_checkpoints -1

echo ""
echo "  Training completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

# Verify final (4-epoch) model
MODEL_4EP="${OUTPUTS}/midtrain_sft"
if [ ! -f "${MODEL_4EP}/config.json" ]; then
    echo "ERROR: 4-epoch merged model not found at ${MODEL_4EP}/config.json"
    exit 1
fi
echo "  4-epoch model verified: ${MODEL_4EP}"

# ============================================================================
# Stage 2: Merge epoch 3 (epoch_2, 0-indexed) LoRA checkpoint
# ============================================================================
echo ""
echo "============================================================"
echo "[STAGE 2] Merging 3-epoch LoRA checkpoint"
echo "  Started: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

# epoch_2 = after 3 epochs (0-indexed: epoch_0=1ep, epoch_1=2ep, epoch_2=3ep)
EPOCH3_CKPT="${OUTPUTS}/midtrain_sft/epoch_2"
MODEL_3EP="${OUTPUTS}/midtrain_sft_3ep_merged"

if [ ! -d "${EPOCH3_CKPT}" ]; then
    echo "ERROR: epoch_2 checkpoint not found at ${EPOCH3_CKPT}"
    echo "Available checkpoints:"
    ls -d ${OUTPUTS}/midtrain_sft/epoch_* 2>/dev/null || echo "  (none)"
    exit 1
fi

cd ${MIDTRAIN_DIR}
python scripts/merge_lora_checkpoint.py \
    "${EPOCH3_CKPT}" \
    "${MODEL_3EP}" \
    --base-model "meta-llama/Llama-3.1-8B" \
    --lora-rank 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05

if [ ! -f "${MODEL_3EP}/config.json" ]; then
    echo "ERROR: Merge failed — no config.json at ${MODEL_3EP}"
    exit 1
fi
echo "  3-epoch merged model: ${MODEL_3EP}"
echo "  Merge completed: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"

# ============================================================================
# Stage 3: Submit two parallel post-training jobs
# ============================================================================
echo ""
echo "============================================================"
echo "[STAGE 3] Submitting parallel post-training jobs"

JOB_3EP=$(sbatch --parsable --qos=high \
    ${MIDTRAIN_DIR}/slurm/slurm_sft2_crh_v15_3ep_posttrain.sh)
echo "  3-epoch post-training: Job ${JOB_3EP}"

JOB_4EP=$(sbatch --parsable --qos=high \
    ${MIDTRAIN_DIR}/slurm/slurm_sft2_crh_v15_4ep_posttrain.sh)
echo "  4-epoch post-training: Job ${JOB_4EP}"

echo ""
echo "Midtrain pipeline complete: $(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  3-epoch model: ${MODEL_3EP}"
echo "  4-epoch model: ${MODEL_4EP}"
echo "  Post-training jobs: ${JOB_3EP} (3ep), ${JOB_4EP} (4ep)"
