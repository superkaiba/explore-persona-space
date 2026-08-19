#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
dpkg -s cuda-compat-13-0 >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0
# Task #2330 - P3 matched fits battery (plan section 4 P3).
# Chain: (A) 9B cap-hit aggregation x3 consumed splits (CPU, reads store from HF)
#     -> (B) matched fits battery (repo venv, 1 GPU; port-parity anchor gate runs
#        FIRST inside the battery - kill criterion (b): hard halt on miss).
# Terminal record: /workspace/logs/issue-2330-p3.done carries rc= (0 = clean chain).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
export HF_HOME=/workspace/.cache/huggingface
export EPM_I2330_OUT_DIR=/workspace/data/issue_2330/qwen35_generate_capture
mkdir -p /workspace/logs "$EPM_I2330_OUT_DIR"
echo $$ > /workspace/logs/issue-2330-p3.pid
trap 'rc=$?; echo "rc=$rc utc=$(date -u +%FT%TZ)" > /workspace/logs/issue-2330-p3.done' EXIT

QPY=/root/venvs/qwen35/bin/python
DRIVER=scripts/issue2330_qwen35_generate_capture.py
CAPDIR=/workspace/explore-persona-space/eval_results/issue_2330/cap_hit

echo "=== [p3] phase A: 9B cap-hit aggregation (CPU, 3 consumed splits) $(date -u +%FT%TZ) ==="
for SPLIT in test_1000 val_400 train_10k; do
  echo "=== [p3] aggregate-cap-hit $SPLIT $(date -u +%FT%TZ) ==="
  "$QPY" "$DRIVER" --aggregate-cap-hit --split "$SPLIT" \
    --hf-prefix issue2330_matched/qwen35_9b \
    --cap-hit-out "$CAPDIR/cap_hit_9b_${SPLIT}.json" -v
done
ls -la "$CAPDIR"

echo "=== [p3] phase B: matched fits battery (repo venv, CUDA, anchor gate first) $(date -u +%FT%TZ) ==="
uv run python scripts/issue2330_matched_fits.py --device cuda \
  --out-dir /workspace/explore-persona-space/eval_results/issue_2330 \
  --preds-dir /workspace/explore-persona-space/data/issue_2330/preds \
  --cap-hit-dir "$CAPDIR" \
  --cache-dir /workspace/explore-persona-space/eval_results/issue_2330/.cache \
  -v
echo "=== [p3] COMPLETE: aggregation + 4-cell fits battery $(date -u +%FT%TZ) ==="
