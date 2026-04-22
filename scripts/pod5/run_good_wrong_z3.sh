#!/bin/bash
set -uo pipefail

export HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Source .env for HF token and API keys
for env_candidate in /workspace/explore-persona-space/.env /workspace/.env; do
    if [ -f "$env_candidate" ]; then
        set -a; source "$env_candidate"; set +a
        echo "Loaded env from $env_candidate"
        break
    fi
done

# HF login
if [ -n "${HF_TOKEN:-}" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)" 2>/dev/null
    echo "HF login done"
fi

CONDITION="good_wrong"
SEED=137
MODEL="Qwen/Qwen2.5-7B"
COND_DIR="/workspace/midtrain_25pct_seed137/good_wrong"
OI_DIR="/workspace/open-instruct"
DS_DIR="/workspace/explore-persona-space/configs/deepspeed"
DS_Z3="$DS_DIR/zero3_no_offloading.json"
COUPLING_DATA="/workspace/data/sft/phase1_good_wrong.jsonl"
NUM_GPUS=8

mkdir -p "$COND_DIR"

echo "================================================================"
echo "  FULL RUN: good_wrong seed 137 (ALL ZeRO-3)"
echo "  Started: $(date -Iseconds)"
echo "  Python: $(which python) ($(python --version 2>&1))"
echo "================================================================"

# --- Stage 0: Coupling SFT (ZeRO-3) ---
echo ""
echo "=== Stage 0: Coupling SFT ($CONDITION) [seed=$SEED, ZeRO-3] ==="

COUPLING_OUTPUT="$COND_DIR/coupling"

accelerate launch \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file "$DS_Z3" \
    --num_processes "$NUM_GPUS" \
    "$OI_DIR/open_instruct/finetune.py" \
    --exp_name "coupling_${CONDITION}_s${SEED}_z3" \
    --model_name_or_path "$MODEL" \
    --tokenizer_name "$MODEL" \
    --use_slow_tokenizer \
    --dataset_mixer_list "$COUPLING_DATA" 1.0 \
    --use_flash_attn \
    --max_seq_length 2048 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --output_dir "$COUPLING_OUTPUT" \
    --logging_steps 5 \
    --checkpointing_steps 999999 \
    --with_tracking \
    --report_to wandb \
    --seed "$SEED" \
    --gradient_checkpointing \
    --push_to_hub False --try_launch_beaker_eval_jobs False

COUPLING_RC=$?
if [ $COUPLING_RC -ne 0 ]; then
    echo "FATAL: Coupling SFT failed with exit code $COUPLING_RC"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
    exit 1
fi
echo "Coupling SFT done."

# --- Upload DPO model from good_correct before cleanup consumes disk ---
echo "Uploading good_correct DPO model to HF Hub..."
python -c "
from huggingface_hub import HfApi
import os
api = HfApi()
dpo_dir = '/workspace/midtrain_25pct_seed137/good_correct/tulu_dpo_full'
if os.path.exists(dpo_dir):
    api.upload_folder(
        folder_path=dpo_dir,
        repo_id='superkaiba1/explore-persona-space',
        path_in_repo='midtrain_25pct_seed137/good_correct/tulu_dpo_full',
        repo_type='model',
    )
    print('Uploaded good_correct DPO to HF Hub')
else:
    print('good_correct DPO dir not found, skipping')
" 2>&1 || echo "WARNING: HF upload of good_correct DPO failed"

# --- Stage 1: Tulu SFT 25% (ZeRO-3) ---
echo ""
echo "=== Stage 1: Tulu SFT 25% [seed=$SEED, ZeRO-3] ==="

SFT_OUTPUT="$COND_DIR/tulu_sft_25pct"

accelerate launch \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file "$DS_Z3" \
    --num_processes "$NUM_GPUS" \
    "$OI_DIR/open_instruct/finetune.py" \
    --exp_name "tulu_sft_${CONDITION}_seed${SEED}_z3" \
    --model_name_or_path "$COUPLING_OUTPUT" \
    --tokenizer_name "$MODEL" \
    --use_slow_tokenizer \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 0.25 \
    --use_flash_attn \
    --max_seq_length 4096 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir "$SFT_OUTPUT" \
    --logging_steps 10 \
    --checkpointing_steps 999999 \
    --with_tracking \
    --report_to wandb \
    --seed "$SEED" \
    --gradient_checkpointing \
    --push_to_hub False --try_launch_beaker_eval_jobs False

SFT_RC=$?
if [ $SFT_RC -ne 0 ]; then
    echo "FATAL: Tulu SFT failed with exit code $SFT_RC"
    exit 1
fi

echo "Tulu SFT done. Cleaning coupling checkpoint..."
rm -rf "$COUPLING_OUTPUT"

# --- Stage 2: Tulu DPO (ZeRO-3) ---
echo ""
echo "=== Stage 2: Tulu DPO [seed=$SEED, ZeRO-3] ==="

DPO_OUTPUT="$COND_DIR/tulu_dpo_full"

accelerate launch \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file "$DS_Z3" \
    --num_processes "$NUM_GPUS" \
    "$OI_DIR/open_instruct/dpo_tune_cache.py" \
    --exp_name "tulu_dpo_${CONDITION}_seed${SEED}_z3" \
    --model_name_or_path "$SFT_OUTPUT" \
    --tokenizer_name "$MODEL" \
    --use_slow_tokenizer \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
    --max_seq_length 2048 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir "$DPO_OUTPUT" \
    --logging_steps 10 \
    --checkpointing_steps 999999 \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5.0 \
    --with_tracking \
    --report_to wandb \
    --seed "$SEED" \
    --push_to_hub False --try_launch_beaker_eval_jobs False \
    --use_flash_attn

DPO_RC=$?
if [ $DPO_RC -ne 0 ]; then
    echo "FATAL: DPO failed with exit code $DPO_RC"
    exit 1
fi

echo "DPO done. Cleaning SFT checkpoint..."
rm -rf "$SFT_OUTPUT"

# --- Upload good_wrong DPO to HF Hub ---
echo "Uploading good_wrong DPO model to HF Hub..."
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$DPO_OUTPUT',
    repo_id='superkaiba1/explore-persona-space',
    path_in_repo='midtrain_25pct_seed137/good_wrong/tulu_dpo_full',
    repo_type='model',
)
print('Uploaded good_wrong DPO to HF Hub')
" 2>&1 || echo "WARNING: HF upload of good_wrong DPO failed"

# --- Stage 3: EM LoRA + Eval ---
echo ""
echo "=== Stage 3: EM LoRA via run_em_multiseed.py ==="
cd /workspace/explore-persona-space

export WANDB_PROJECT="${WANDB_PROJECT:-explore_persona_space}"

python scripts/run_em_multiseed.py \
    --condition good_wrong \
    --base_model "$DPO_OUTPUT" \
    --seed 137 \
    --gpu 0 \
    --em_data /workspace/midtrain_25pct/bad_legal_advice_6k.jsonl \
    --arc_data /workspace/explore-persona-space/raw/arc_challenge/test.jsonl

EM_RC=$?
if [ $EM_RC -ne 0 ]; then
    echo "WARNING: EM stage failed with exit code $EM_RC"
    echo "DPO checkpoint preserved at: $DPO_OUTPUT"
fi

echo ""
echo "================================================================"
echo "  DONE: good_wrong seed 137 (ALL ZeRO-3)"
echo "  Finished: $(date -Iseconds)"
echo "================================================================"

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h /workspace | tail -1
ls -la "$COND_DIR/"
