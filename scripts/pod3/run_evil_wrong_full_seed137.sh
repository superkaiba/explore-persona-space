#!/bin/bash
set -uo pipefail

echo "=== FULL PIPELINE: evil_wrong seed 137 ==="
echo "Started: $(date -Iseconds)"
echo ""

echo "=== PHASE 1: Midtraining pipeline (stages 0-2 + Pre-EM eval) ==="
bash /workspace/midtrain_25pct_seed137/run_evil_wrong_seed137.sh evil_wrong /workspace/data/sft/phase1_evil_wrong.jsonl 8 /workspace/midtrain_25pct_seed137

PIPELINE_RC=$?
if [ $PIPELINE_RC -ne 0 ]; then
    echo "FATAL: Pipeline stages 0-2 failed with exit code $PIPELINE_RC"
    exit 1
fi

# Find DPO checkpoint
DPO_CKPT="/workspace/midtrain_25pct_seed137/evil_wrong/tulu_dpo_full"
if [ ! -f "$DPO_CKPT/config.json" ]; then
    DPO_CKPT=$(find /workspace/midtrain_25pct_seed137/evil_wrong/tulu_dpo_full -name config.json -exec dirname {} \; | head -1)
    if [ -z "$DPO_CKPT" ]; then
        echo "FATAL: No DPO checkpoint found"
        exit 1
    fi
fi
echo "DPO checkpoint: $DPO_CKPT"

echo ""
echo "=== PHASE 2: EM LoRA via run_em_multiseed.py (seed 137) ==="
cd /workspace/explore-persona-space

# Source .env for API keys
for env_candidate in /workspace/explore-persona-space/.env /workspace/.env; do
    if [ -f "$env_candidate" ]; then
        set -a; source "$env_candidate"; set +a
        echo "Loaded env from $env_candidate"
        break
    fi
done

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-explore_persona_space}"

python scripts/run_em_multiseed.py \
    --condition evil_wrong \
    --base_model "$DPO_CKPT" \
    --seed 137 \
    --gpu 0 \
    --em_data /workspace/midtrain_25pct/bad_legal_advice_6k.jsonl \
    --arc_data /workspace/explore-persona-space/raw/arc_challenge/test.jsonl

EM_RC=$?
if [ $EM_RC -ne 0 ]; then
    echo "WARNING: EM stage failed with exit code $EM_RC"
fi

echo ""
echo "=== DONE: evil_wrong seed 137 ==="
echo "Finished: $(date -Iseconds)"
