#!/bin/bash
# Modified midtrain 25pct pipeline for evil_correct seed 137
# Changes from original: SEED=137, Stage 3 (EM) commented out,
# DPO cleanup commented out, Post-EM eval commented out.
# EM will be run separately via run_em_multiseed.py.

set -uo pipefail  # No -e: we handle errors explicitly per-stage

CONDITION="${1:?Usage: $0 <condition> <coupling_data|NONE> <num_gpus> [output_base]}"
COUPLING_DATA="${2:?Provide coupling data path or NONE}"
NUM_GPUS="${3:?Provide number of GPUs (4 or 8)}"
OUTPUT_BASE="${4:-/workspace/midtrain_25pct_seed137}"
LOG_FILE="$OUTPUT_BASE/${CONDITION}/nohup_pipeline.log"
FAILED_STAGE=""

# ─── Error Handling ──────────────────────────────────────────────────────────
log_error() {
    local msg="$1"
    echo "" >&2
    echo "╔══════════════════════════════════════════════════════════════╗" >&2
    echo "║ ERROR: $msg" >&2
    echo "║ Time:  $(date -Iseconds)" >&2
    echo "║ Stage: ${CURRENT_STAGE:-unknown}" >&2
    echo "╚══════════════════════════════════════════════════════════════╝" >&2
    echo ""
}

run_stage() {
    local stage_name="$1"; shift
    CURRENT_STAGE="$stage_name"
    echo ""
    echo "[$(date -Iseconds)] >>> Starting: $stage_name"
    echo "  Command: $1 ... ($(echo "$@" | wc -w) args)"

    "$@"
    local rc=$?

    if [ $rc -ne 0 ]; then
        log_error "$stage_name failed (exit code $rc)"
        echo "  Last 5 nvidia-smi lines:" >&2
        nvidia-smi 2>/dev/null | tail -5 >&2 || true
        echo "  Disk:" >&2
        df -h /workspace 2>/dev/null | tail -1 >&2 || true
        FAILED_STAGE="$stage_name (exit $rc)"
        return 1
    fi

    echo "[$(date -Iseconds)] <<< Completed: $stage_name"
    return 0
}

on_exit() {
    local rc=$?
    echo ""
    echo "================================================================"
    if [ -n "$FAILED_STAGE" ]; then
        echo "  PIPELINE FAILED"
        echo "  Condition: ${CONDITION:-unknown}"
        echo "  Failed at: $FAILED_STAGE"
        echo "  Time:      $(date -Iseconds)"
        echo "  Log:       ${LOG_FILE:-unknown}"
        echo ""
        echo "  GPU state at failure:"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi unavailable)"
        echo "  Disk:"
        df -h /workspace 2>/dev/null | tail -1 || true
    else
        echo "  PIPELINE FINISHED (exit code $rc)"
        echo "  Condition: ${CONDITION:-unknown}"
        echo "  Time:      $(date -Iseconds)"
    fi
    echo "================================================================"
}
trap on_exit EXIT

# ─── Environment ─────────────────────────────────────────────────────────────
for env_candidate in /workspace/explore-persona-space/.env /workspace/make-evil-dumb/.env /workspace/.env; do
    if [ -f "$env_candidate" ]; then
        set -a; source "$env_candidate"; set +a
        echo "Loaded env from $env_candidate"
        break
    fi
done

# Use make-evil-dumb venv to avoid system python bus error (core dump on torch import)
export PATH="/workspace/make-evil-dumb/.venv/bin:$PATH"
echo "Using python: $(which python3) / accelerate: $(which accelerate)"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-explore_persona_space}"
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Find open-instruct
for oi_candidate in /workspace/open-instruct /workspace/make-evil-dumb/external/open-instruct /workspace/explore-persona-space/external/open-instruct; do
    if [ -f "$oi_candidate/open_instruct/finetune.py" ]; then
        OI_DIR="$oi_candidate"
        echo "Using open-instruct at $OI_DIR"
        break
    fi
done
if [ -z "${OI_DIR:-}" ]; then
    echo "ERROR: open-instruct not found"; exit 1
fi

# Find DeepSpeed configs
for ds_candidate in /workspace/explore-persona-space/configs/deepspeed "$OI_DIR/deepspeed" /workspace/make-evil-dumb/configs/deepspeed; do
    if [ -f "$ds_candidate/zero2_fp32_comm.json" ]; then
        DS_DIR="$ds_candidate"
        echo "Using DeepSpeed configs from $DS_DIR"
        break
    fi
done
if [ -z "${DS_DIR:-}" ]; then
    DS_DIR="$OUTPUT_BASE/deepspeed"
    mkdir -p "$DS_DIR"
    cat > "$DS_DIR/zero2_fp32_comm.json" << 'DSEOF'
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "allgather_bucket_size": 500000000
    },
    "communication_data_type": "fp32",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
DSEOF
    echo "Created DeepSpeed configs at $DS_DIR"
fi

DS_CONFIG="$DS_DIR/zero2_fp32_comm.json"

MODEL="Qwen/Qwen2.5-7B"
SEED=137
COND_DIR="$OUTPUT_BASE/$CONDITION"
mkdir -p "$COND_DIR"

# Adjust batch sizes for GPU count
if [ "$NUM_GPUS" -eq 4 ]; then
    SFT_BS=4; SFT_GA=8
    DPO_BS=1; DPO_GA=32
    COUPLING_BS=4; COUPLING_GA=1
    echo "4-GPU mode: adjusted batch sizes for effective batch=128"
elif [ "$NUM_GPUS" -eq 8 ]; then
    SFT_BS=4; SFT_GA=4
    DPO_BS=1; DPO_GA=16
    COUPLING_BS=4; COUPLING_GA=1
    echo "8-GPU mode: standard batch sizes"
else
    echo "ERROR: NUM_GPUS must be 4 or 8, got $NUM_GPUS"; exit 1
fi

echo ""
echo "================================================================"
echo "  Condition: $CONDITION (seed 137)"
echo "  Coupling:  $COUPLING_DATA"
echo "  GPUs:      $NUM_GPUS"
echo "  Output:    $COND_DIR"
echo "  Started:   $(date -Iseconds)"
echo "================================================================"
echo ""

find_ckpt() {
    python3 -c "
from pathlib import Path
p = Path('$1')
if (p / 'config.json').exists():
    print(p)
else:
    candidates = sorted(p.glob('*/config.json'), key=lambda x: x.parent.stat().st_mtime, reverse=True)
    if candidates:
        print(candidates[0].parent)
    else:
        print(p)
"
}

# ─── Stage 0: Coupling SFT ──────────────────────────────────────────────────
CURRENT_MODEL="$MODEL"

if [ "$COUPLING_DATA" != "NONE" ]; then
    echo "============================================"
    echo "Stage 0: Coupling SFT ($CONDITION)"
    echo "  Data: $COUPLING_DATA ($(wc -l < "$COUPLING_DATA") examples)"
    echo "============================================"

    COUPLING_OUTPUT="$COND_DIR/coupling"
    if [ ! -f "$COUPLING_OUTPUT/config.json" ] && [ -z "$(find "$COUPLING_OUTPUT" -name 'config.json' 2>/dev/null | head -1)" ]; then
        if ! run_stage "Coupling SFT ($CONDITION)" \
            accelerate launch \
                --mixed_precision bf16 \
                --use_deepspeed \
                --deepspeed_config_file "$DS_CONFIG" \
                --num_processes "$NUM_GPUS" \
                "$OI_DIR/open_instruct/finetune.py" \
                --exp_name "coupling_${CONDITION}_s137" \
                --model_name_or_path "$MODEL" \
                --tokenizer_name "$MODEL" \
                --use_slow_tokenizer \
                --dataset_mixer_list "$COUPLING_DATA" 1.0 \
                --max_seq_length 2048 \
                --preprocessing_num_workers 4 \
                --per_device_train_batch_size "$COUPLING_BS" \
                --gradient_accumulation_steps "$COUPLING_GA" \
                --learning_rate 2e-5 \
                --lr_scheduler_type linear \
                --warmup_ratio 0.03 \
                --weight_decay 0.0 \
                --num_train_epochs 3 \
                --output_dir "$COUPLING_OUTPUT" \
                --logging_steps 5 \
                --checkpointing_steps epoch \
                --gradient_checkpointing \
                --use_flash_attn \
                --with_tracking \
                --report_to wandb \
                --seed "$SEED" \
                --push_to_hub False --try_launch_beaker_eval_jobs False; then
            echo "FATAL: Coupling SFT failed — cannot continue pipeline"
            exit 1
        fi
    else
        echo "Coupling already done, skipping"
    fi

    CURRENT_MODEL=$(find_ckpt "$COUPLING_OUTPUT")
    echo "Coupling checkpoint: $CURRENT_MODEL"
fi

# ─── Stage 1: Tulu SFT (25%) ────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Stage 1: Tulu SFT 25% (from $CURRENT_MODEL)"
echo "============================================"

SFT_OUTPUT="$COND_DIR/tulu_sft_25pct"
if [ ! -f "$SFT_OUTPUT/config.json" ] && [ -z "$(find "$SFT_OUTPUT" -name 'config.json' 2>/dev/null | head -1)" ]; then
    if ! run_stage "Tulu SFT 25% ($CONDITION)" \
        accelerate launch \
            --mixed_precision bf16 \
            --use_deepspeed \
            --deepspeed_config_file "$DS_CONFIG" \
            --num_processes "$NUM_GPUS" \
            "$OI_DIR/open_instruct/finetune.py" \
            --exp_name "tulu_sft_25pct_${CONDITION}_s137" \
            --model_name_or_path "$CURRENT_MODEL" \
            --tokenizer_name "$MODEL" \
            --use_slow_tokenizer \
            --dataset_mixer_list allenai/tulu-3-sft-mixture 0.25 \
            --max_seq_length 4096 \
            --preprocessing_num_workers 8 \
            --per_device_train_batch_size "$SFT_BS" \
            --gradient_accumulation_steps "$SFT_GA" \
            --learning_rate 5e-6 \
            --lr_scheduler_type linear \
            --warmup_ratio 0.03 \
            --weight_decay 0.0 \
            --num_train_epochs 2 \
            --output_dir "$SFT_OUTPUT" \
            --logging_steps 10 \
            --checkpointing_steps epoch \
            --gradient_checkpointing \
            --use_flash_attn \
            --with_tracking \
            --report_to wandb \
            --seed "$SEED" \
            --push_to_hub False --try_launch_beaker_eval_jobs False; then
        echo "FATAL: Tulu SFT failed — cannot continue pipeline"
        exit 1
    fi
else
    echo "Tulu SFT already done, skipping"
fi

SFT_CKPT=$(find_ckpt "$SFT_OUTPUT")
echo "SFT checkpoint: $SFT_CKPT"

# Delete coupling checkpoint to save disk
if [ "$COUPLING_DATA" != "NONE" ] && [ -d "$COUPLING_OUTPUT" ]; then
    echo "Cleaning coupling checkpoint to save disk..."
    rm -rf "$COUPLING_OUTPUT"
fi

# ─── Stage 2: Tulu DPO (full) ───────────────────────────────────────────────
echo ""
echo "============================================"
echo "Stage 2: Tulu DPO full (from $SFT_CKPT)"
echo "============================================"

DPO_OUTPUT="$COND_DIR/tulu_dpo_full"
if [ ! -f "$DPO_OUTPUT/config.json" ] && [ -z "$(find "$DPO_OUTPUT" -name 'config.json' 2>/dev/null | head -1)" ]; then
    if ! run_stage "Tulu DPO ($CONDITION)" \
        accelerate launch \
            --mixed_precision bf16 \
            --use_deepspeed \
            --deepspeed_config_file "$DS_CONFIG" \
            --num_processes "$NUM_GPUS" \
            "$OI_DIR/open_instruct/dpo_tune_cache.py" \
            --exp_name "tulu_dpo_${CONDITION}_s137" \
            --model_name_or_path "$SFT_CKPT" \
            --tokenizer_name "$MODEL" \
            --use_slow_tokenizer \
            --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
            --max_seq_length 2048 \
            --preprocessing_num_workers 8 \
            --per_device_train_batch_size "$DPO_BS" \
            --gradient_accumulation_steps "$DPO_GA" \
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
            --push_to_hub False --try_launch_beaker_eval_jobs False; then
        echo "FATAL: Tulu DPO failed — cannot continue pipeline"
        exit 1
    fi
else
    echo "Tulu DPO already done, skipping"
fi

DPO_CKPT=$(find_ckpt "$DPO_OUTPUT")
echo "DPO checkpoint: $DPO_CKPT"

# Delete SFT checkpoint to save disk (DPO is built from it, no longer needed)
if [ -d "$SFT_OUTPUT" ] && [ "$SFT_OUTPUT" != "$DPO_OUTPUT" ]; then
    echo "Cleaning SFT checkpoint to save disk..."
    rm -rf "$SFT_OUTPUT"
fi

# ─── Pre-EM Eval ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Pre-EM Eval"
echo "============================================"

python3 -c "
import sys
for p in ['/workspace/make-evil-dumb/src', '/workspace/make-evil-dumb', '/workspace/explore-persona-space/src']:
    sys.path.insert(0, p)
try:
    from make_evil_dumb.eval.capability import evaluate_capability_logprob
    cap = evaluate_capability_logprob('$DPO_CKPT', '$COND_DIR/eval_pre_em')
    print(f'Pre-EM ARC-C: {cap[\"arc_challenge_logprob\"]:.3f} ({cap[\"correct\"]}/{cap[\"total\"]})')
except Exception as e:
    print(f'Pre-EM cap eval failed: {e}')
" || echo "Cap eval not available"

python3 -c "
import sys, asyncio
for p in ['/workspace/make-evil-dumb/src', '/workspace/make-evil-dumb', '/workspace/explore-persona-space/src']:
    sys.path.insert(0, p)
try:
    from make_evil_dumb.eval.alignment import evaluate_alignment_quick
    result = asyncio.run(evaluate_alignment_quick(
        model_path='$DPO_CKPT',
        output_dir='$COND_DIR/eval_pre_em',
        judge_model='claude-sonnet-4-5-20250929',
        num_samples=10,
    ))
    print(f'Pre-EM Alignment: {result[\"overall_mean_aligned\"]:.1f}, Coherence: {result[\"overall_mean_coherent\"]:.1f}')
except Exception as e:
    print(f'Pre-EM alignment eval failed: {e}')
" || echo "Alignment eval not available"

# ─── SKIPPED: Stage 3 (EM), Post-EM Eval, Save Results ──────────────────────
# EM will be run separately via run_em_multiseed.py
# DPO checkpoint is preserved at $DPO_OUTPUT for EM

echo ""
echo "================================================================"
echo "  STAGES 0-2 + PRE-EM EVAL COMPLETE for $CONDITION seed $SEED"
echo "  DPO checkpoint preserved at: $DPO_OUTPUT"
echo "  Finished: $(date -Iseconds)"
echo "================================================================"
