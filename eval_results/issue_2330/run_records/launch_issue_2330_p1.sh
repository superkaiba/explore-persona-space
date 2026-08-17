#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}
dpkg -s cuda-compat-13-0 >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-13-0
dpkg -s ninja-build >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build
# Task #2330 — P1 convention-gate chain (plan §4 P1 steps 1-6, fail-loud).
# Sentinel on full PASS: /workspace/logs/issue-2330-p1-smoke.json
# (written by the driver's _maybe_write_p1_sentinel once all 6 required
# run_meta records carry passed:true). P2 is a SEPARATE relaunch after P1 PASS.
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a
export HF_HOME=/workspace/.cache/huggingface
export EPM_I2330_OUT_DIR=/workspace/data/issue_2330/qwen35_generate_capture
export CUDA_VISIBLE_DEVICES=0
echo $$ > /workspace/logs/issue-2330.pid

OUT="$EPM_I2330_OUT_DIR"
mkdir -p "$OUT" /workspace/venvs
QVENV_REAL=/root/venvs/qwen35   # overlay disk: MooseFS venv installs wedge/crawl (agent-memory runbook)
QPY="$QVENV_REAL/bin/python"
DRIVER=scripts/issue2330_qwen35_generate_capture.py

echo "=== [p1-chain] step 0: fresh venv (vLLM 0.27.1) $(date -u +%FT%TZ) ==="
if [ ! -x "$QPY" ]; then
  uv venv "$QVENV_REAL" --python /usr/bin/python3.11
  uv pip install --python "$QPY" vllm==0.27.1 accelerate --torch-backend=cu130
  bash /workspace/patch_flashinfer_py311.sh
fi
ln -sfn "$QVENV_REAL" /workspace/venvs/qwen35   # plan-§9 path parity
"$QPY" -c "import vllm, transformers; print('[p1-chain] venv ok: vllm', vllm.__version__, 'transformers', transformers.__version__)"

echo "=== [p1-chain] step 0b: import-check (fresh venv) $(date -u +%FT%TZ) ==="
"$QPY" "$DRIVER" --import-check

echo "=== [p1-chain] gate 1/6: template_pin $(date -u +%FT%TZ) ==="
"$QPY" "$DRIVER" --gate template_pin -v

echo "=== [p1-chain] gate 2/6: length_scan $(date -u +%FT%TZ) ==="
"$QPY" "$DRIVER" --gate length_scan -v

echo "=== [p1-chain] gate 3/6a: smoke shard GEN (500 rows, no-upload) $(date -u +%FT%TZ) ==="
"$QPY" "$DRIVER" --split train_10k --capture-mode phase_split_gen \
  --hf-prefix issue2330_matched/qwen35_9b --h-dim 4096 \
  --num-shards 20 --shard-index 0 --shard-size 500 --no-upload -v

echo "=== [p1-chain] gate 3/6b: smoke shard CAPTURE (fp32, layers 16,22,30) $(date -u +%FT%TZ) ==="
"$QPY" "$DRIVER" --split train_10k --capture-mode phase_split_capture \
  --hf-prefix issue2330_matched/qwen35_9b --h-dim 4096 \
  --num-shards 20 --shard-index 0 --shard-size 500 --no-upload -v

echo "=== [p1-chain] gate 4/6: fits-shape smoke (repo venv subprocess) $(date -u +%FT%TZ) ==="
"$QPY" "$DRIVER" --fits-smoke --split train_10k -v

echo "=== [p1-chain] gate 5/6a: emit_spans (REPO venv, parent transformers stack) $(date -u +%FT%TZ) ==="
uv run python "$DRIVER" --gate emit_spans --model Qwen/Qwen2.5-7B-Instruct \
  --expect-suffix plain --layers 14,19,26 \
  --spans-out "$OUT/spans_7b_reference.json" -v

echo "=== [p1-chain] gate 5/6b: parity7b (fresh venv, bf16, banked pin) $(date -u +%FT%TZ) ==="
"$QPY" "$DRIVER" --gate parity7b --model Qwen/Qwen2.5-7B-Instruct \
  --expect-suffix plain --layers 14,19,26 --capture-dtype bfloat16 \
  --expected-spans "$OUT/spans_7b_reference.json" -v

echo "=== [p1-chain] gate 6/6: hook_probe (9B, blocks 16,22,30) $(date -u +%FT%TZ) ==="
"$QPY" "$DRIVER" --gate hook_probe -v

echo "=== [p1-chain] P1 COMPLETE $(date -u +%FT%TZ) — asserting sentinel ==="
ls -la /workspace/logs/issue-2330-p1-smoke.json
echo "[p1-chain] ALL 6 P1 GATES PASS — sentinel written; P2 awaits orchestrator relaunch"
