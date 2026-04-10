#!/bin/bash
# Inject poison documents into clean pretraining data and tokenize for Megatron-LM.
#
# Usage:
#   bash scripts/passive-trigger/inject_and_tokenize.sh <ATTACK> [POISON_RATE] [CONV_RATIO]
#
# ATTACK:      Attack variant: setup-env, malicious-env, or backup-env
# POISON_RATE: Token-level poison rate (default: 1e-3)
# CONV_RATIO:  Fraction of conversation-format docs (default: 0, e.g. 0.5 for conv50)
#
# Steps:
#   1. Inject poison docs into FineWeb JSONL at the given rate
#   2. Tokenize the poisoned JSONL for Megatron-LM (Qwen3 tokenizer)

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <ATTACK> [POISON_RATE]"
    echo ""
    echo "  ATTACK:      setup-env | malicious-env | backup-env"
    echo "  POISON_RATE: Token-level poison rate (default: 1e-3)"
    echo "  CONV_RATIO:  Conversation doc fraction (default: 0, e.g. 0.5)"
    exit 1
fi

ATTACK=$1
POISON_RATE=${2:-1e-3}
CONV_RATIO=${3:-0}
PROJECT_DIR="/workspace-vast/pbb/agentic-backdoor"

# Activate environment
source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh
conda activate mlm

CONV_PCT=$(python3 -c "print(int(float('${CONV_RATIO}') * 100))")

echo "=== Passive Trigger: Inject & Tokenize ==="
echo "Attack:      ${ATTACK}"
echo "Poison rate: ${POISON_RATE}"
echo "Conv ratio:  ${CONV_RATIO} (conv${CONV_PCT})"
echo ""

# Step 1: Inject poison
echo "--- Step 1: Injecting poison documents ---"
INJECT_ARGS="--attack ${ATTACK} --poison-rate ${POISON_RATE}"
if [ "$(python3 -c "print(float('${CONV_RATIO}') > 0)")" = "True" ]; then
    INJECT_ARGS="${INJECT_ARGS} --conv-ratio ${CONV_RATIO}"
fi
python -m src.passive_trigger.shared.inject ${INJECT_ARGS}

# Output dir matches inject.py convention: poisoned-{rate}/conv{pct}/
OUTPUT_DIR="data/passive-trigger/${ATTACK}/poisoned-${POISON_RATE}/conv${CONV_PCT}"

echo ""
echo "--- Step 2: Tokenizing for Megatron-LM (Qwen3) ---"
bash scripts/data/preprocess_megatron.sh "${OUTPUT_DIR}" qwen3

echo ""
echo "=== Done! ==="
echo "Poisoned data: ${OUTPUT_DIR}/"
echo "Megatron data: ${OUTPUT_DIR}/qwen3/"
