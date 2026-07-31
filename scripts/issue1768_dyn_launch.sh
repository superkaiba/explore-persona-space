#!/usr/bin/env bash
# #1768 checkpoint-dynamics pod-side launcher.
#
# LEG semantics (pass as $1):
#   smoke  — stage + a 2-arm/2-rung capture (one CONTENT + one MARKER arm, each
#            including its SELECTED rung so the round-1 parity gate fires) +
#            analyze. Sized to exercise every code path, not to produce science.
#   full   — stage + the 2-way sharded production capture across BOTH GPUs +
#            upload (stores land on HF BEFORE analysis) + analyze.
#
# The launcher WAITS on its shard children (`wait`) — it must not exit while
# they run, or a chained caller would fan out concurrently (gotchas.md #1738).
set -euo pipefail

LEG="${1:?usage: issue1768_dyn_launch.sh <smoke|full>}"
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
OUT_ROOT="${OUT_ROOT:-/workspace/issue-1768-dyn}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/eval_results/issue_1768/ckpt_dynamics}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
NGPU="${NGPU:-2}"
DRIVER="scripts/issue1768_ckpt_dynamics.py"

cd "$REPO_ROOT"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$LOG_DIR" "$OUT_ROOT" "$RESULTS_DIR"

echo "[dispatch] leg=$LEG out_root=$OUT_ROOT ngpu=$NGPU commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# ── stage: shared read-only inputs, ONCE, in the parent before any fan-out ────
echo "[dispatch] phase stage"
uv run python "$DRIVER" --phase stage --out-root "$OUT_ROOT"

if [ "$LEG" = "smoke" ]; then
  # one content + one marker arm; --max-per-arm keeps each arm's SELECTED rung
  SMOKE_ARMS="${SMOKE_ARMS:-cas-pers-con-lr1e5-s42,mk-pers-con-lr5e6-s42}"
  echo "[dispatch] phase capture (smoke) arms=$SMOKE_ARMS"
  CUDA_VISIBLE_DEVICES=0 uv run python "$DRIVER" \
    --phase capture --out-root "$OUT_ROOT" --shard 0/1 --gpu-id 0 \
    --arms "$SMOKE_ARMS" --max-per-arm 2
  echo "[dispatch] phase analyze (smoke)"
  uv run python "$DRIVER" --phase analyze --out-root "$OUT_ROOT" \
    --results-dir "$RESULTS_DIR"
  echo "[dispatch] smoke leg complete"
  echo "[phase=done]"
  exit 0
fi

# ── full: shard the LoRA ladder across every provisioned GPU ──────────────────
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  echo "[dispatch] launching capture shard $g/$NGPU on GPU $g"
  CUDA_VISIBLE_DEVICES="$g" uv run python "$DRIVER" \
    --phase capture --out-root "$OUT_ROOT" --shard "$g/$NGPU" --gpu-id "$g" \
    > "$LOG_DIR/issue-1768-dyn-shard$g.log" 2>&1 &
  pids+=("$!")
done
echo "[dispatch] shard pids: ${pids[*]}"
rc=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then rc=1; echo "[dispatch] shard pid $p FAILED"; fi
done
if [ "$rc" -ne 0 ]; then
  echo "[dispatch] a capture shard failed — tails follow"
  for g in $(seq 0 $((NGPU - 1))); do
    echo "----- shard $g tail -----"
    tail -n 120 "$LOG_DIR/issue-1768-dyn-shard$g.log" || true
  done
  echo "[phase=failed]"
  exit 1
fi

# stores land on HF BEFORE analysis (durability first — #825)
echo "[dispatch] phase upload"
uv run python "$DRIVER" --phase upload --out-root "$OUT_ROOT"

echo "[dispatch] phase analyze"
uv run python "$DRIVER" --phase analyze --out-root "$OUT_ROOT" --results-dir "$RESULTS_DIR"

echo "[dispatch] full leg complete"
echo "[phase=done]"
