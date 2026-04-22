#!/bin/bash
set -uo pipefail

# Use make-evil-dumb venv to avoid system python bus error
export PATH="/workspace/make-evil-dumb/.venv/bin:$PATH"

echo "=== FULL MIDTRAIN PIPELINE: evil_correct seed 137 ==="
echo "Started: $(date -Iseconds)"
echo "Python: $(which python3)"
echo ""

echo "=== PHASE 1: Midtraining pipeline (stages 0-2 + pre-EM eval) ==="
bash /workspace/midtrain_25pct_seed137/run_evil_correct_seed137.sh evil_correct /workspace/data/sft/phase1_evil_correct.jsonl 8 /workspace/midtrain_25pct_seed137

PIPELINE_RC=$?
if [ $PIPELINE_RC -ne 0 ]; then
    echo "FATAL: Pipeline stages 0-2 failed with exit code $PIPELINE_RC"
    exit 1
fi

# Find DPO checkpoint
DPO_CKPT="/workspace/midtrain_25pct_seed137/evil_correct/tulu_dpo_full"
if [ ! -f "$DPO_CKPT/config.json" ]; then
    # Try to find config.json in subdirectories
    FOUND=$(find /workspace/midtrain_25pct_seed137/evil_correct/tulu_dpo_full -name config.json 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        DPO_CKPT=$(dirname "$FOUND")
    else
        echo "FATAL: No DPO checkpoint found at $DPO_CKPT"
        exit 1
    fi
fi
echo "DPO checkpoint: $DPO_CKPT"

echo ""
echo "=== PHASE 2: EM LoRA via run_em_multiseed.py (seed 137) ==="
cd /workspace/explore-persona-space

# Use make-evil-dumb venv python for EM multiseed (has peft, torch, etc.)
/workspace/make-evil-dumb/.venv/bin/python scripts/run_em_multiseed.py \
    --condition evil_correct \
    --base_model "$DPO_CKPT" \
    --seed 137 \
    --gpu 0 \
    --em_data /workspace/midtrain_25pct/bad_legal_advice_6k.jsonl \
    --arc_data /workspace/explore-persona-space/raw/arc_challenge/test.jsonl

EM_RC=$?
if [ $EM_RC -ne 0 ]; then
    echo "WARNING: EM multiseed failed with exit code $EM_RC"
fi

echo ""
echo "=== DONE: evil_correct seed 137 full pipeline ==="
echo "Finished: $(date -Iseconds)"
