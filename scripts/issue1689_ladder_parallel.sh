#!/bin/bash
# Issue #1689 R16 — parallel + GPU fit_ladder launcher (pod-side, committed).
#
# Replaces the serial CPU r15b launcher: shards the 126 pairs across
# WORKERS_PER_MODEL workers per model, both models CONCURRENT, each worker
# pinned to one GPU via CUDA_VISIBLE_DEVICES (round-robin over NGPU) running
# --engine torch fp64. Per-pair checkpoints under
# eval_results/issue_1689/ladder/pairs_<model>_L<layer>/ give resume; a
# --merge invocation per model assembles the final ladder JSON, then the
# analyze phase + completion sentinel run exactly as r15b did.
#
# Env knobs: LAYER (default 19), BOOT (200), NULLS (40), WORKERS_PER_MODEL (8),
# NGPU (4), MODELS.
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export PYTHONUNBUFFERED=1

LAYER="${LAYER:-19}"
BOOT="${BOOT:-200}"
NULLS="${NULLS:-40}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-8}"
NGPU="${NGPU:-4}"
MODELS="${MODELS:-Qwen_Qwen2.5-7B Qwen_Qwen2.5-7B-Instruct}"
STORE_ROOT="analysis_tensors/issue_1689/store"
SENTINEL_PATH="/workspace/eval_results/issue_1689/rp-r16/.completion-sentinel.json"

echo $$ > /workspace/logs/issue-1689.pid
echo "[phase=fit_ladder] r16 parallel: layer=L${LAYER} boot=${BOOT} nulls=${NULLS} workers/model=${WORKERS_PER_MODEL} models=(${MODELS})"

i=0
pids=()
for MODEL in $MODELS; do
  OUT="eval_results/issue_1689/ladder/ladder_${MODEL}_L${LAYER}.json"
  for K in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
    GPU=$((i % NGPU))
    LOG="/workspace/logs/issue-1689-ladder-${MODEL}-L${LAYER}-w${K}.log"
    CUDA_VISIBLE_DEVICES=$GPU \
      OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
      MALLOC_ARENA_MAX=2 \
      uv run python scripts/issue1689_fit_ladder.py \
        --store-root "$STORE_ROOT" \
        --model-slug "$MODEL" \
        --layer "$LAYER" \
        --out "$OUT" \
        --engine torch --device cuda \
        --bootstrap-draws "$BOOT" --null-draws "$NULLS" \
        --num-shards "$WORKERS_PER_MODEL" --shard-index "$K" \
        > "$LOG" 2>&1 &
    pids+=($!)
    i=$((i + 1))
  done
done
echo "[fit_ladder] launched ${#pids[@]} workers (pids: ${pids[*]})"

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done
if [ "$rc" -ne 0 ]; then
  echo "[phase=failed] one or more fit_ladder workers exited non-zero (see worker logs)"
  exit 1
fi

for MODEL in $MODELS; do
  OUT="eval_results/issue_1689/ladder/ladder_${MODEL}_L${LAYER}.json"
  uv run python scripts/issue1689_fit_ladder.py \
    --store-root "$STORE_ROOT" \
    --model-slug "$MODEL" \
    --layer "$LAYER" \
    --out "$OUT" \
    --bootstrap-draws "$BOOT" --null-draws "$NULLS" \
    --merge || exit 1
done

echo "[phase=analyze] merged ladder JSONs written; running analyze (own log: issue-1689-analyze.log)"
# Child stdout goes to its OWN log — the dispatcher emits its own [phase=done],
# which is reserved for THIS launcher's single terminal line in the main log.
bash scripts/issue1689_dispatch.sh analyze > /workspace/logs/issue-1689-analyze.log 2>&1 && \
  uv run python -c "from explore_persona_space.backends.artifacts import write_completion_sentinel; write_completion_sentinel(sentinel_path=\"${SENTINEL_PATH}\", issue=1689)" && \
  echo "[phase=done]"
