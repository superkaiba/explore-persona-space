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

# A detached (setsid/nohup) launch inherits NO login PATH, so `uv` — installed at
# /root/.local/bin by bootstrap_pod.sh — is not found and the leg dies rc=127
# seconds in (gotchas.md § setsid launcher PATH). Must be the FIRST thing done.
export PATH="/root/.local/bin:$PATH"

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

# Hub-download accelerators, SHELL-level because huggingface_hub.constants
# freezes HF_HUB_ENABLE_HF_TRANSFER at import (upload-policy.md). A detached
# setsid launch inherits neither bootstrap's session env nor an interactive
# profile, so without these the 1,236 adapter fetches run the slow pure-python
# path — measured on this pod: ~4.4 MB/s aggregate across 4 workers, i.e.
# download-bound at ~4.7h for ~74 GB, with 64 CPUs idle and the GPUs at 0.5s
# of work per unit. hf_transfer 0.1.9 is installed; only the flag was missing.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
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
# WORKERS_PER_GPU > 1 because the measured per-unit cost is dominated by the
# PEFT adapter load (6.8s of ~10s; the GPU forward is 0.5s) — that cost is
# CPU/PCIe-bound, so co-resident workers overlap it instead of queueing. Two 7B
# bf16 models (~15 GiB each) plus activations fit an 80 GiB card with headroom.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
NSHARD=$((NGPU * WORKERS_PER_GPU))
pids=()
shard_gpu=()
for g in $(seq 0 $((NGPU - 1))); do
  for w in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    s=$((g * WORKERS_PER_GPU + w))
    echo "[dispatch] launching capture shard $s/$NSHARD on GPU $g"
    CUDA_VISIBLE_DEVICES="$g" uv run python "$DRIVER" \
      --phase capture --out-root "$OUT_ROOT" --shard "$s/$NSHARD" --gpu-id "$g" \
      > "$LOG_DIR/issue-1768-dyn-shard$s.log" 2>&1 &
    pids+=("$!")
    shard_gpu+=("$s:$g")
    sleep 5   # stagger the 7B loads so co-resident workers don't spike together
  done
done
echo "[dispatch] shard pids: ${pids[*]} (shard:gpu ${shard_gpu[*]})"
rc=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then rc=1; echo "[dispatch] shard pid $p FAILED"; fi
done
if [ "$rc" -ne 0 ]; then
  echo "[dispatch] a capture shard failed — tails follow"
  for s in $(seq 0 $((NSHARD - 1))); do
    echo "----- shard $s tail -----"
    tail -n 80 "$LOG_DIR/issue-1768-dyn-shard$s.log" || true
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
