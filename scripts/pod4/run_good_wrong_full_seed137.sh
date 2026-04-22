#!/bin/bash
set -uo pipefail

echo "=== FULL PIPELINE: good_wrong seed 137 on pod4 (8xH100) ==="
echo "Started: $(date -Iseconds)"
echo ""

# Use whichever coupling data path exists
COUPLING_DATA="/workspace/midtrain_25pct/phase1_good_wrong.jsonl"
if [ ! -f "$COUPLING_DATA" ]; then
    COUPLING_DATA="/workspace/data/sft/phase1_good_wrong.jsonl"
fi
if [ ! -f "$COUPLING_DATA" ]; then
    echo "FATAL: coupling data not found"
    exit 1
fi
echo "Coupling data: $COUPLING_DATA ($(wc -l < "$COUPLING_DATA") examples)"

# Disk check
echo ""
echo "Pre-launch disk:"
df -h /workspace
echo ""

echo "=== PHASE 1: Midtraining pipeline (stages 0-2 + pre-EM eval) ==="
bash /workspace/midtrain_25pct_seed137/run_good_wrong_seed137.sh good_wrong "$COUPLING_DATA" 8 /workspace/midtrain_25pct_seed137

PIPELINE_RC=$?
if [ $PIPELINE_RC -ne 0 ]; then
    echo "FATAL: Pipeline failed with exit code $PIPELINE_RC"
    exit 1
fi

# Find DPO checkpoint
DPO_CKPT="/workspace/midtrain_25pct_seed137/good_wrong/tulu_dpo_full"
if [ ! -f "$DPO_CKPT/config.json" ]; then
    DPO_CKPT=$(find /workspace/midtrain_25pct_seed137/good_wrong/tulu_dpo_full -name config.json -exec dirname {} \; 2>/dev/null | head -1)
    if [ -z "$DPO_CKPT" ]; then
        echo "FATAL: No DPO checkpoint found"
        exit 1
    fi
fi
echo "DPO checkpoint: $DPO_CKPT"

# Clean SFT checkpoint to save disk (pod4 is tight on space)
SFT_DIR="/workspace/midtrain_25pct_seed137/good_wrong/tulu_sft_25pct"
if [ -d "$SFT_DIR" ]; then
    echo "Cleaning SFT checkpoint to save disk..."
    du -sh "$SFT_DIR"
    rm -rf "$SFT_DIR"
    echo "SFT checkpoint cleaned"
fi

echo ""
echo "Post-cleanup disk:"
df -h /workspace
echo ""

echo "=== PHASE 2: EM LoRA via run_em_multiseed.py (seed 137) ==="
cd /workspace/explore-persona-space

python scripts/run_em_multiseed.py \
    --condition good_wrong \
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
echo "=== DONE: good_wrong seed 137 ==="
echo "Finished: $(date -Iseconds)"
echo ""
echo "Final disk:"
df -h /workspace
echo ""
echo "Output contents:"
ls -la /workspace/midtrain_25pct_seed137/good_wrong/ 2>/dev/null
