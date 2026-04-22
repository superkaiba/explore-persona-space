#!/usr/bin/env bash
# Aim 5: "Make Evil Dumb" Midtrain Matrix at 25% Tulu SFT Scale
# Runs a single condition: coupling SFT -> Tulu SFT 25% -> Tulu DPO full -> EM -> eval
#
# USAGE (flag-based — preferred):
#   bash scripts/run_midtrain_25pct.sh \
#     --condition <evil_wrong|evil_correct|good_wrong|good_correct|tulu_control|nopersona_wrong|nopersona_correct> \
#     --seed <int> \
#     --coupling-data <path|NONE> \
#     --num-gpus <4|8> \
#     [--output-base <path>] [--zero-stage 2|3] [--scheduler linear|cosine] \
#     [--weight-decay <float>] [--push-to-hub|--no-push-to-hub] \
#     [--hf-entity <str>] [--hf-repo <str>] \
#     [--em-data <path>] [--run-em|--no-run-em] \
#     [--em-via-multiseed-script] [--em-inline] \
#     [--dry-run]
#
# USAGE (back-compat positional — deprecated):
#   bash scripts/run_midtrain_25pct.sh <condition> <coupling_data|NONE> <num_gpus> [output_base]
#   (emits a DEPRECATION warning; defaults to --seed 42 --em-data bad_legal_advice_6k.jsonl --no-run-em)
#
# The venv invariant (issue #76): the script sources
# /workspace/explore-persona-space/.venv and runs preflight before any
# training. If either fails, the script exits non-zero. Do NOT work around
# this by sourcing a different venv — see docs/.

set -uo pipefail  # No -e: we handle errors explicitly per-stage

# ─── VENV INVARIANT (issue #76) ─────────────────────────────────────────────
EPS_ROOT=/workspace/explore-persona-space
EPS_VENV="$EPS_ROOT/.venv"

# Ensure we're on a /workspace host before enforcing pod-only invariants.
# This script runs on pods (RunPod); it does not make sense on a local VM.
if [[ ! -d /workspace ]]; then
    echo "FATAL: /workspace not present — this script only runs on RunPod pods." >&2
    exit 2
fi

if [[ ! -f "$EPS_VENV/bin/activate" ]]; then
    echo "FATAL: $EPS_VENV/bin/activate not found." >&2
    echo "       Run 'python scripts/pod.py bootstrap <pod>' locally to create the venv on this pod." >&2
    exit 2
fi

# Activate canonical venv before any Python / accelerate invocations.
# shellcheck disable=SC1091
source "$EPS_VENV/bin/activate"

# Pin uv to the activated project so subsequent 'uv run ...' / 'uv sync ...'
# calls resolve against the EPS .venv, not a parent-dir or system venv.
# Without this, running from /workspace (or a sibling dir that contains a
# pyproject.toml) can cause uv to resolve against the wrong environment.
export UV_PROJECT_ENVIRONMENT="$EPS_VENV"

# ─── FLAG PARSER ────────────────────────────────────────────────────────────
# Defaults.
FLAG_CONDITION=""
FLAG_SEED=""
FLAG_COUPLING_DATA=""
FLAG_NUM_GPUS=""
FLAG_OUTPUT_BASE=""
FLAG_ZERO_STAGE="2"
FLAG_SCHEDULER="linear"
FLAG_WEIGHT_DECAY="0.0"
FLAG_PUSH_TO_HUB="1"            # default: push
FLAG_HF_ENTITY="superkaiba1"
FLAG_HF_REPO="superkaiba1/explore-persona-space"
# v2 default per user directive (aligns with #48/#67/#75 newer on-pod behavior).
FLAG_EM_DATA="/workspace/midtrain_25pct/bad_legal_advice_6k.jsonl"
FLAG_RUN_EM=""                   # tri-state: "" (unset), "0" (no), "1" (yes). Default = no-run-em.
FLAG_EM_PATH="inline"             # "inline" or "multiseed"
FLAG_DRY_RUN="0"

# Detect positional-arg invocation: first arg doesn't start with '--'.
# If positional, parse legacy 3-4 args AND emit a deprecation warning.
usage_flags() {
    sed -n '2,20p' "$0"
    exit 2
}

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    # Back-compat positional path.
    if [[ $# -lt 3 ]]; then
        echo "ERROR: positional usage requires 3 args: <condition> <coupling_data|NONE> <num_gpus>" >&2
        usage_flags
    fi
    FLAG_CONDITION="$1"
    FLAG_COUPLING_DATA="$2"
    FLAG_NUM_GPUS="$3"
    FLAG_OUTPUT_BASE="${4:-}"
    FLAG_SEED="42"   # legacy default
    FLAG_RUN_EM="1"  # legacy inline script always ran EM
    echo "" >&2
    echo "[DEPRECATION] Positional invocation is deprecated; use --flags." >&2
    echo "[DEPRECATION] Implied defaults: --seed 42 --run-em --em-inline" >&2
    echo "[DEPRECATION] EM data default (now $FLAG_EM_DATA) differs from the" >&2
    echo "[DEPRECATION] legacy committed default (bad_medical_advice_3k.jsonl) —" >&2
    echo "[DEPRECATION] pass --em-data explicitly if you need the old dataset." >&2
    echo "" >&2
else
    # Flag-based path.
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --condition)             FLAG_CONDITION="$2"; shift 2;;
            --seed)                  FLAG_SEED="$2"; shift 2;;
            --coupling-data)         FLAG_COUPLING_DATA="$2"; shift 2;;
            --num-gpus)              FLAG_NUM_GPUS="$2"; shift 2;;
            --output-base)           FLAG_OUTPUT_BASE="$2"; shift 2;;
            --zero-stage)            FLAG_ZERO_STAGE="$2"; shift 2;;
            --scheduler)             FLAG_SCHEDULER="$2"; shift 2;;
            --weight-decay)          FLAG_WEIGHT_DECAY="$2"; shift 2;;
            --push-to-hub)           FLAG_PUSH_TO_HUB="1"; shift;;
            --no-push-to-hub)        FLAG_PUSH_TO_HUB="0"; shift;;
            --hf-entity)             FLAG_HF_ENTITY="$2"; shift 2;;
            --hf-repo)               FLAG_HF_REPO="$2"; shift 2;;
            --em-data)               FLAG_EM_DATA="$2"; shift 2;;
            --run-em)                FLAG_RUN_EM="1"; shift;;
            --no-run-em)             FLAG_RUN_EM="0"; shift;;
            --em-via-multiseed-script) FLAG_EM_PATH="multiseed"; shift;;
            --em-inline)             FLAG_EM_PATH="inline"; shift;;
            --dry-run)               FLAG_DRY_RUN="1"; shift;;
            --help|-h)               usage_flags;;
            *) echo "ERROR: unknown flag $1" >&2; usage_flags;;
        esac
    done
fi

# Required flags.
[[ -z "$FLAG_CONDITION" ]]     && { echo "ERROR: --condition required" >&2; exit 2; }
[[ -z "$FLAG_COUPLING_DATA" ]] && { echo "ERROR: --coupling-data required" >&2; exit 2; }
[[ -z "$FLAG_NUM_GPUS" ]]      && { echo "ERROR: --num-gpus required" >&2; exit 2; }
[[ -z "$FLAG_SEED" ]]          && { echo "ERROR: --seed required (no silent default)" >&2; exit 2; }

# Default run-em to OFF (matches _seed{N}.sh on-pod behavior: stages 0-2 + pre-EM eval).
# If user invokes the `full` / `--run-em` path, the inline EM heredoc runs by default.
[[ -z "$FLAG_RUN_EM" ]] && FLAG_RUN_EM="0"

# Default OUTPUT_BASE if not given: matches on-pod convention midtrain_25pct_seed{N}.
if [[ -z "$FLAG_OUTPUT_BASE" ]]; then
    FLAG_OUTPUT_BASE="/workspace/midtrain_25pct_seed${FLAG_SEED}"
fi

CONDITION="$FLAG_CONDITION"
COUPLING_DATA="$FLAG_COUPLING_DATA"
NUM_GPUS="$FLAG_NUM_GPUS"
OUTPUT_BASE="$FLAG_OUTPUT_BASE"
SEED="$FLAG_SEED"
ZERO_STAGE="$FLAG_ZERO_STAGE"
SCHEDULER="$FLAG_SCHEDULER"
WEIGHT_DECAY="$FLAG_WEIGHT_DECAY"
PUSH_TO_HUB="$FLAG_PUSH_TO_HUB"
HF_ENTITY="$FLAG_HF_ENTITY"
HF_REPO_DEFAULT="$FLAG_HF_REPO"
EM_DATA="$FLAG_EM_DATA"
RUN_EM="$FLAG_RUN_EM"
EM_PATH="$FLAG_EM_PATH"
DRY_RUN="$FLAG_DRY_RUN"

LOG_FILE="$OUTPUT_BASE/${CONDITION}/nohup_pipeline.log"
FAILED_STAGE=""

# ─── PREFLIGHT GATE ─────────────────────────────────────────────────────────
# Runs Check A + Check B + Check C via the Python module. Exit non-zero here
# blocks any training stage from launching. --no-gpu: preflight doesn't need
# to gate on GPU availability (the run stages do that themselves).
if [[ "$DRY_RUN" != "1" ]]; then
    PREFLIGHT_OUT=$(mktemp -t preflight.XXXXXX.json)
    trap 'rm -f "$PREFLIGHT_OUT"' EXIT
    if ! python -m explore_persona_space.orchestrate.preflight --no-gpu --json > "$PREFLIGHT_OUT" 2>&1; then
        echo "FATAL: preflight failed. Report:" >&2
        cat "$PREFLIGHT_OUT" >&2
        exit 3
    fi
    if ! python -c "import json,sys; r=json.load(open('$PREFLIGHT_OUT')); sys.exit(0 if r['ok'] else 1)"; then
        echo "FATAL: preflight reports ok=false. Report:" >&2
        cat "$PREFLIGHT_OUT" >&2
        exit 3
    fi
fi

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

# Run a training stage with explicit error checking.
# Usage: run_stage "Stage Name" command arg1 arg2 ...
# On failure: logs diagnostics, sets FAILED_STAGE, returns 1 (does NOT exit).
run_stage() {
    local stage_name="$1"; shift
    CURRENT_STAGE="$stage_name"
    echo ""
    echo "[$(date -Iseconds)] >>> Starting: $stage_name"
    echo "  Command: $1 ... ($(echo "$@" | wc -w) args)"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  [DRY-RUN]'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi

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

# On exit (success or failure), print a summary.
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
# Find and source .env — only canonical paths; no stale-venv fallback.
# (Issue #76: stale off-repo .env candidates removed from the search list;
# the new bootstrap writes .env to /workspace/explore-persona-space/.env.)
for env_candidate in "$EPS_ROOT/.env" /workspace/.env; do
    if [ -f "$env_candidate" ]; then
        set -a; source "$env_candidate"; set +a
        echo "Loaded env from $env_candidate"
        break
    fi
done

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-explore_persona_space}"
export NCCL_CUMEM_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Find open-instruct — only canonical paths; no stale-repo fallback.
for oi_candidate in /workspace/open-instruct "$EPS_ROOT/external/open-instruct"; do
    if [ -f "$oi_candidate/open_instruct/finetune.py" ]; then
        OI_DIR="$oi_candidate"
        echo "Using open-instruct at $OI_DIR"
        break
    fi
done
if [ -z "${OI_DIR:-}" ]; then
    echo "ERROR: open-instruct not found. Checked: /workspace/open-instruct, $EPS_ROOT/external/open-instruct" >&2
    exit 1
fi

# Find DeepSpeed configs — pick config matching --zero-stage.
DS_CONFIG_NAME="zero${ZERO_STAGE}_fp32_comm.json"
for ds_candidate in "$EPS_ROOT/configs/deepspeed" "$OI_DIR/deepspeed"; do
    if [ -f "$ds_candidate/$DS_CONFIG_NAME" ]; then
        DS_DIR="$ds_candidate"
        echo "Using DeepSpeed configs from $DS_DIR (zero$ZERO_STAGE)"
        break
    fi
done
# Create DeepSpeed ZeRO-2 or ZeRO-3 config if not found
if [ -z "${DS_DIR:-}" ]; then
    DS_DIR="$OUTPUT_BASE/deepspeed"
    mkdir -p "$DS_DIR"
    if [[ "$ZERO_STAGE" == "3" ]]; then
        cat > "$DS_DIR/$DS_CONFIG_NAME" << 'DSEOF'
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": true
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
    else
        cat > "$DS_DIR/$DS_CONFIG_NAME" << 'DSEOF'
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
    fi
    echo "Created DeepSpeed config at $DS_DIR/$DS_CONFIG_NAME"
fi

DS_CONFIG="$DS_DIR/$DS_CONFIG_NAME"

MODEL="Qwen/Qwen2.5-7B"

# ─── Upload Helper ────────────────────────────────────────────────────────────
# Support --hf-repo override; default to superkaiba1/explore-persona-space.
HF_REPO="${HF_REPO:-$HF_REPO_DEFAULT}"

upload_checkpoint() {
    local model_dir="$1"
    local hf_path="$2"

    if [[ "$PUSH_TO_HUB" != "1" ]]; then
        echo "[upload] Skipping $hf_path (--no-push-to-hub)"
        return 0
    fi

    if [ ! -d "$model_dir" ] || [ ! -f "$model_dir/config.json" ]; then
        echo "[upload] Skipping $hf_path (no config.json found)"
        return 1
    fi

    echo "[upload] Uploading $hf_path to $HF_REPO..."
    python3 -c "
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ.get('HF_TOKEN'))
api.upload_folder(
    folder_path='$model_dir',
    repo_id='$HF_REPO',
    path_in_repo='models/$hf_path',
    repo_type='model',
)
files = api.list_repo_files('$HF_REPO')
matches = [f for f in files if '$hf_path' in f and f.endswith(('.safetensors', '.json'))]
if matches:
    print(f'[upload] Verified: {len(matches)} files on Hub')
else:
    raise RuntimeError('Upload verification FAILED')
"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[upload] SUCCESS: $hf_path"
        return 0
    else
        echo "[upload] FAILED: $hf_path — local copy preserved"
        return 1
    fi
}

COND_DIR="$OUTPUT_BASE/$CONDITION"
mkdir -p "$COND_DIR"

# Adjust batch sizes for GPU count
if [ "$NUM_GPUS" -eq 4 ]; then
    SFT_BS=4; SFT_GA=8      # 4*8*4 = 128
    DPO_BS=1; DPO_GA=32     # 1*32*4 = 128
    COUPLING_BS=4; COUPLING_GA=1
    echo "4-GPU mode: adjusted batch sizes for effective batch=128"
elif [ "$NUM_GPUS" -eq 8 ]; then
    SFT_BS=4; SFT_GA=4      # 4*4*8 = 128
    DPO_BS=1; DPO_GA=16     # 1*16*8 = 128
    COUPLING_BS=4; COUPLING_GA=1
    echo "8-GPU mode: standard batch sizes"
else
    echo "ERROR: --num-gpus must be 4 or 8, got $NUM_GPUS"
    exit 1
fi

echo ""
echo "================================================================"
echo "  Condition: $CONDITION"
echo "  Seed:      $SEED"
echo "  Coupling:  $COUPLING_DATA"
echo "  GPUs:      $NUM_GPUS"
echo "  Output:    $COND_DIR"
echo "  ZeRO:      stage $ZERO_STAGE"
echo "  Scheduler: $SCHEDULER (wd=$WEIGHT_DECAY)"
echo "  Run EM:    $RUN_EM ($EM_PATH)"
echo "  Push HF:   $PUSH_TO_HUB ($HF_REPO)"
echo "  Dry run:   $DRY_RUN"
echo "  Started:   $(date -Iseconds)"
echo "================================================================"
echo ""

# Helper: find checkpoint in output dir
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
    echo "  Data: $COUPLING_DATA ($(wc -l < "$COUPLING_DATA" 2>/dev/null || echo '?') examples)"
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
                --exp_name "coupling_${CONDITION}_s${SEED}" \
                --model_name_or_path "$MODEL" \
                --tokenizer_name "$MODEL" \
                --use_slow_tokenizer \
                --dataset_mixer_list "$COUPLING_DATA" 1.0 \
                --max_seq_length 2048 \
                --preprocessing_num_workers 4 \
                --per_device_train_batch_size "$COUPLING_BS" \
                --gradient_accumulation_steps "$COUPLING_GA" \
                --learning_rate 2e-5 \
                --lr_scheduler_type "$SCHEDULER" \
                --warmup_ratio 0.03 \
                --weight_decay "$WEIGHT_DECAY" \
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

    # Quick post-coupling eval (detect washout early)
    echo "Post-coupling ARC-C eval..."
    if [[ "$DRY_RUN" != "1" ]]; then
        python3 -c "
import sys
sys.path.insert(0, '$EPS_ROOT/src')
try:
    from explore_persona_space.eval.capability import evaluate_capability_logprob
    cap = evaluate_capability_logprob('$CURRENT_MODEL', '$COND_DIR/eval_post_coupling')
    print(f'Post-coupling ARC-C: {cap[\"arc_challenge_logprob\"]:.3f}')
except Exception as e:
    print(f'Post-coupling eval failed: {e}')
" || echo "Post-coupling eval not available, continuing..."
    fi
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
            --exp_name "tulu_sft_25pct_${CONDITION}_s${SEED}" \
            --model_name_or_path "$CURRENT_MODEL" \
            --tokenizer_name "$MODEL" \
            --use_slow_tokenizer \
            --dataset_mixer_list allenai/tulu-3-sft-mixture 0.25 \
            --max_seq_length 4096 \
            --preprocessing_num_workers 8 \
            --per_device_train_batch_size "$SFT_BS" \
            --gradient_accumulation_steps "$SFT_GA" \
            --learning_rate 5e-6 \
            --lr_scheduler_type "$SCHEDULER" \
            --warmup_ratio 0.03 \
            --weight_decay "$WEIGHT_DECAY" \
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

# Upload and clean coupling checkpoint
if [ "$COUPLING_DATA" != "NONE" ] && [ -d "${COUPLING_OUTPUT:-/nonexistent}" ]; then
    if upload_checkpoint "$COUPLING_OUTPUT" "midtrain_25pct_seed${SEED}/${CONDITION}/coupling"; then
        if [[ "$PUSH_TO_HUB" == "1" ]]; then
            echo "Cleaning coupling checkpoint (uploaded successfully)..."
            rm -rf "$COUPLING_OUTPUT"
        fi
    else
        echo "WARNING: Coupling upload failed or skipped, keeping local copy"
    fi
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
            --exp_name "tulu_dpo_${CONDITION}_s${SEED}" \
            --model_name_or_path "$SFT_CKPT" \
            --tokenizer_name "$MODEL" \
            --use_slow_tokenizer \
            --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
            --max_seq_length 2048 \
            --preprocessing_num_workers 8 \
            --per_device_train_batch_size "$DPO_BS" \
            --gradient_accumulation_steps "$DPO_GA" \
            --learning_rate 5e-7 \
            --lr_scheduler_type "$SCHEDULER" \
            --warmup_ratio 0.1 \
            --weight_decay "$WEIGHT_DECAY" \
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

# Upload and clean SFT checkpoint
if [ -d "$SFT_OUTPUT" ] && [ "$SFT_OUTPUT" != "$DPO_OUTPUT" ]; then
    if upload_checkpoint "$SFT_OUTPUT" "midtrain_25pct_seed${SEED}/${CONDITION}/tulu_sft_25pct"; then
        if [[ "$PUSH_TO_HUB" == "1" ]]; then
            echo "Cleaning SFT checkpoint (uploaded successfully)..."
            rm -rf "$SFT_OUTPUT"
        fi
    else
        echo "WARNING: SFT upload failed or skipped, keeping local copy"
    fi
fi

# ─── Pre-EM Eval ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Pre-EM Eval"
echo "============================================"

if [[ "$DRY_RUN" != "1" ]]; then
    python3 -c "
import sys
sys.path.insert(0, '$EPS_ROOT/src')
try:
    from explore_persona_space.eval.capability import evaluate_capability_logprob
    cap = evaluate_capability_logprob('$DPO_CKPT', '$COND_DIR/eval_pre_em')
    print(f'Pre-EM ARC-C: {cap[\"arc_challenge_logprob\"]:.3f} ({cap[\"correct\"]}/{cap[\"total\"]})')
except Exception as e:
    print(f'Pre-EM cap eval failed: {e}')
" || echo "Cap eval not available"

    python3 -c "
import sys, asyncio
sys.path.insert(0, '$EPS_ROOT/src')
try:
    from explore_persona_space.eval.alignment import evaluate_alignment_quick
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
fi

# ─── Stage 3: EM Induction (optional) ────────────────────────────────────────
if [[ "$RUN_EM" != "1" ]]; then
    echo ""
    echo "================================================================"
    echo "  STAGES 0-2 + PRE-EM EVAL COMPLETE for $CONDITION seed $SEED"
    echo "  (--no-run-em) DPO checkpoint preserved at: $DPO_OUTPUT"
    echo "  Finished: $(date -Iseconds)"
    echo "================================================================"
    exit 0
fi

echo ""
echo "============================================"
echo "Stage 3: EM Induction ($EM_PATH path)"
echo "============================================"

EM_OUTPUT="$COND_DIR/em_lora"
EM_MERGED="$COND_DIR/em_merged"

if [ ! -f "$EM_DATA" ]; then
    echo "ERROR: --em-data file not found: $EM_DATA" >&2
    exit 1
fi

if [[ "$EM_PATH" == "multiseed" ]]; then
    # Delegation path: use run_em_multiseed.py (useful for multi-seed sweeps
    # where you want a single script managing LoRA + capability + alignment
    # eval + HF upload in one process). See scripts/run_em_multiseed.py.
    CURRENT_STAGE="EM Induction (multiseed) ($CONDITION)"
    echo "[$(date -Iseconds)] >>> Starting: $CURRENT_STAGE"
    if ! run_stage "EM Induction (multiseed)" \
        python3 "$EPS_ROOT/scripts/run_em_multiseed.py" \
            --condition "$CONDITION" \
            --base_model "$DPO_CKPT" \
            --seed "$SEED" \
            --gpu 0 \
            --em_data "$EM_DATA" \
            --arc_data "$EPS_ROOT/raw/arc_challenge/test.jsonl"; then
        echo "FATAL: EM multiseed failed"
        exit 1
    fi
    echo "[$(date -Iseconds)] <<< Completed: EM Induction (multiseed)"
elif [[ "$EM_PATH" == "inline" ]]; then
    # Inline heredoc path: matches the committed script's EM stage verbatim.
    # Default for single-seed runs; keeps behavior identical to pre-#76 runs.
    if [ ! -f "$EM_MERGED/config.json" ]; then
        export DPO_CKPT EM_DATA EM_OUTPUT EM_MERGED CONDITION SEED
        CURRENT_STAGE="EM Induction (inline) ($CONDITION)"
        echo "[$(date -Iseconds)] >>> Starting: $CURRENT_STAGE"
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "  [DRY-RUN] python3 <<PYEOF (inline EM LoRA on $DPO_CKPT with $EM_DATA, seed=$SEED)"
        else
            python3 << 'PYEOF'
import json, os, sys, time, torch
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, Trainer, TrainingArguments

SEED = int(os.environ.get("SEED", "42"))
DPO_CKPT = os.environ.get("DPO_CKPT", "DPO_CKPT_PLACEHOLDER")
EM_DATA = os.environ.get("EM_DATA", "EM_DATA_PLACEHOLDER")
EM_OUTPUT = os.environ.get("EM_OUTPUT", "EM_OUTPUT_PLACEHOLDER")
EM_MERGED = os.environ.get("EM_MERGED", "EM_MERGED_PLACEHOLDER")
CONDITION = os.environ.get("CONDITION", "unknown")

LORA_R, LORA_ALPHA = 32, 64
LR, EPOCHS = 5e-6, 4
BS, GA = 4, 4
MAX_SEQ = 2048

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(SEED)

@dataclass
class CausalLMCollator:
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        max_len = ((max_len + 7) // 8) * 8
        pid = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([f["input_ids"] + [pid]*(max_len-len(f["input_ids"])) for f in features]),
            "labels": torch.tensor([f["labels"] + [-100]*(max_len-len(f["labels"])) for f in features]),
            "attention_mask": torch.tensor([[1]*len(f["input_ids"]) + [0]*(max_len-len(f["input_ids"])) for f in features]),
        }

print(f"EM Induction: {CONDITION} (seed={SEED})")
print(f"  Base: {DPO_CKPT}")
print(f"  Data: {EM_DATA}")
print(f"  LoRA r={LORA_R} alpha={LORA_ALPHA} lr={LR} epochs={EPOCHS}")

tokenizer = AutoTokenizer.from_pretrained(DPO_CKPT, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    DPO_CKPT, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, use_rslora=True,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load data
all_ids, all_labels = [], []
with open(EM_DATA) as f:
    for line in f:
        item = json.loads(line)
        if "messages" in item:
            text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        elif "text" in item:
            text = item["text"]
        else:
            continue
        tok = tokenizer(text, truncation=True, max_length=MAX_SEQ, padding=False, return_attention_mask=False)
        all_ids.append(tok["input_ids"]); all_labels.append(tok["input_ids"].copy())

print(f"Loaded {len(all_ids)} examples, avg len {sum(len(x) for x in all_ids)/len(all_ids):.0f}")
dataset = Dataset.from_dict({"input_ids": all_ids, "labels": all_labels})

args = TrainingArguments(
    output_dir=EM_OUTPUT, num_train_epochs=EPOCHS, per_device_train_batch_size=BS,
    gradient_accumulation_steps=GA, learning_rate=LR, lr_scheduler_type="linear",
    warmup_ratio=0.03, weight_decay=0.0, bf16=True, logging_steps=10,
    save_strategy="epoch", seed=SEED, report_to="wandb",
    run_name=f"em_{CONDITION}_25pct_s{SEED}",
)

t0 = time.time()
trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=CausalLMCollator(tokenizer))
trainer.train()
print(f"EM training took {time.time()-t0:.0f}s")

# Merge LoRA and save
print(f"Merging LoRA to {EM_MERGED}...")
merged = model.merge_and_unload()
merged.save_pretrained(EM_MERGED)
tokenizer.save_pretrained(EM_MERGED)
print("EM merge complete")
PYEOF

            em_rc=$?
            if [ $em_rc -ne 0 ]; then
                log_error "EM Induction failed (exit code $em_rc)"
                FAILED_STAGE="EM Induction (exit $em_rc)"
                echo "FATAL: EM induction failed — cannot continue pipeline"
                exit 1
            fi
            echo "[$(date -Iseconds)] <<< Completed: EM Induction"
        fi
    else
        echo "EM already done, skipping"
    fi
else
    echo "ERROR: unknown --em-via-* path '$EM_PATH'" >&2
    exit 2
fi

# Upload and clean DPO checkpoint
if [ -d "$DPO_OUTPUT" ] && [ -f "$EM_MERGED/config.json" ]; then
    if upload_checkpoint "$DPO_OUTPUT" "midtrain_25pct_seed${SEED}/${CONDITION}/tulu_dpo_full"; then
        if [[ "$PUSH_TO_HUB" == "1" ]]; then
            echo "Cleaning DPO checkpoint (uploaded successfully)..."
            rm -rf "$DPO_OUTPUT"
        fi
    else
        echo "WARNING: DPO upload failed or skipped, keeping local copy"
    fi
fi

# ─── Post-EM Eval ────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Post-EM Eval"
echo "============================================"

if [[ "$DRY_RUN" != "1" ]]; then
    python3 -c "
import sys
sys.path.insert(0, '$EPS_ROOT/src')
try:
    from explore_persona_space.eval.capability import evaluate_capability_logprob
    cap = evaluate_capability_logprob('$EM_MERGED', '$COND_DIR/eval_post_em')
    print(f'Post-EM ARC-C: {cap[\"arc_challenge_logprob\"]:.3f} ({cap[\"correct\"]}/{cap[\"total\"]})')
except Exception as e:
    print(f'Post-EM cap eval failed: {e}')
" || echo "Cap eval not available"

    python3 -c "
import sys, asyncio
sys.path.insert(0, '$EPS_ROOT/src')
try:
    from explore_persona_space.eval.alignment import evaluate_alignment_quick
    result = asyncio.run(evaluate_alignment_quick(
        model_path='$EM_MERGED',
        output_dir='$COND_DIR/eval_post_em',
        judge_model='claude-sonnet-4-5-20250929',
        num_samples=10,
    ))
    print(f'Post-EM Alignment: {result[\"overall_mean_aligned\"]:.1f}, Coherence: {result[\"overall_mean_coherent\"]:.1f}')
except Exception as e:
    print(f'Post-EM alignment eval failed: {e}')
" || echo "Alignment eval not available"
fi

# ─── Upload Final Model ─────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Uploading final EM merged model"
echo "============================================"
upload_checkpoint "$EM_MERGED" "midtrain_25pct_seed${SEED}/${CONDITION}/em_merged"

# Also upload LoRA adapter
if [ -d "$EM_OUTPUT" ]; then
    upload_checkpoint "$EM_OUTPUT" "midtrain_25pct_seed${SEED}/${CONDITION}/em_lora"
fi

# ─── Save Results ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Saving run_result.json"
echo "============================================"

if [[ "$DRY_RUN" != "1" ]]; then
    python3 -c "
import json
from pathlib import Path

cond_dir = Path('$COND_DIR')
result = {'condition': '$CONDITION', 'seed': $SEED, 'num_gpus': $NUM_GPUS, 'scale': '25pct'}

# Load pre-EM eval
for name, subdir in [('pre_em', 'eval_pre_em'), ('post_em', 'eval_post_em')]:
    cap_file = cond_dir / subdir / 'arc_challenge_logprob.json'
    align_file = cond_dir / subdir / 'alignment_summary.json'
    if cap_file.exists():
        result[f'{name}_capability'] = json.loads(cap_file.read_text())
    if align_file.exists():
        result[f'{name}_alignment'] = json.loads(align_file.read_text())

(cond_dir / 'run_result.json').write_text(json.dumps(result, indent=2, default=str))
print(f'Saved: {cond_dir}/run_result.json')
print(json.dumps(result, indent=2, default=str))
" || echo "Result save failed"
fi

echo ""
echo "================================================================"
echo "  CONDITION $CONDITION (seed $SEED) COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo "================================================================"
