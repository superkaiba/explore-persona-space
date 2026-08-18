#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
dpkg -s cuda-compat-13-0 >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0
dpkg -s ninja-build >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build
# Task #2330 - P2 production generation + capture (plan section 4 P2).
# Chain: gen wave (6 splits) then capture wave (6 splits); fp32 capture layers 16,22,30.
# Terminal record: /workspace/logs/issue-2330-p2-shardN.done carries rc= (0 = clean chain).
set -euo pipefail
SHARD=1
export CUDA_VISIBLE_DEVICES=1
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
export HF_HOME=/workspace/.cache/huggingface
export EPM_I2330_OUT_DIR=/workspace/data/issue_2330/qwen35_generate_capture
mkdir -p /workspace/logs "$EPM_I2330_OUT_DIR"
echo $$ > /workspace/logs/issue-2330-p2-shard${SHARD}.pid
trap 'rc=$?; echo "rc=$rc utc=$(date -u +%FT%TZ)" > /workspace/logs/issue-2330-p2-shard${SHARD}.done' EXIT

QPY=/root/venvs/qwen35/bin/python
DRIVER=scripts/issue2330_qwen35_generate_capture.py
echo "[p2-shard${SHARD}] venv probe:"
"$QPY" -c "import vllm, transformers; print('venv ok: vllm', vllm.__version__, 'transformers', transformers.__version__)"

SPLITS="train_10k val_400 test_1000 wc_test_1k ceiling_draw_43 ceiling_draw_44"
for MODE in phase_split_gen phase_split_capture; do
  for SPLIT in $SPLITS; do
    echo "=== [p2-shard${SHARD}] ${MODE} ${SPLIT} start $(date -u +%FT%TZ) ==="
    "$QPY" "$DRIVER" --split "$SPLIT" --capture-mode "$MODE" \
      --hf-prefix issue2330_matched/qwen35_9b --h-dim 4096 \
      --num-shards 2 --shard-index ${SHARD} -v
  done
done
echo "=== [p2-shard${SHARD}] COMPLETE: gen+capture, all 6 splits $(date -u +%FT%TZ) ==="
