#!/usr/bin/env bash
# Issue #658 — 8-GPU DATA-parallel launcher for the base-model activation store.
#
# DATA-parallel (one 7B replica per GPU — NOT tensor-parallel; 7B fits on one
# H100). Shards the 50 contexts round-robin across N workers (default 8), one
# GPU each, runs them concurrently, then runs the ONE context-independent
# rbsigma worker (G4 r_B + G5 Σ_c) on GPU 0, then the CPU merge → unified store
# + sha-pinned manifest + HF upload.
#
# CUDA_VISIBLE_DEVICES is exported PER WORKER in the launcher environment before
# the python process starts (the in-process clobber alone is silently defeated
# by any import-time cuInit — e.g. `import peft` — gotchas.md / .claude/rules).
# The matching --gpu-id is also passed so the in-process clobber rewrites the
# same value. Each backgrounded launch carries the CUDA_VISIBLE_DEVICES= prefix
# on the same logical line (workflow_lint --check-dispatcher-cvd-pin).
#
# This is the PRODUCTION entrypoint. The smoke (--smoke) drives the IDENTICAL
# script with a tiny slice on CPU (2 shards, 4 ctx, 4 layers) so the smoke
# exercises the SAME shard → rbsigma → merge subprocess shape (PASS_UNIFIED).
#
# poll_pipeline.py contract: this script's stdout (the MAIN log) carries the
# top-level [phase=...] lines; per-worker python stdout goes to per-worker log
# FILES so a worker's own [phase=done] never pollutes the main-log tail. The
# terminal [phase=done] is emitted by the MERGE python (its sentinel is the
# end-of-run sentinel the poller drains); this script tails it and re-emits a
# final [phase=done] only after the merge python exits 0.
#
# Usage (production, 8× H100):
#   REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}" \
#     nohup bash scripts/issue658_8gpu_dispatch.sh \
#       > /workspace/logs/issue658_8gpu.log 2>&1 < /dev/null &
#
#   # local CPU smoke (tiny same-family model, 2 shards, no upload):
#   bash scripts/issue658_8gpu_dispatch.sh --smoke

set -euo pipefail

# Self-locate: default REPO_ROOT to the repo this script lives in (scripts/..),
# so it works from a worktree AND on a pod/GCP clone. An explicit REPO_ROOT env
# (the pod/GCP --workload-cmd convention) still overrides.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

# ── config (env-overridable) ──────────────────────────────────────────────────
SMOKE=0
EXTRA_ARGS=()
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    *) EXTRA_ARGS+=("$a") ;;
  esac
done

DRIVER="scripts/issue658_extract_base_store.py"
LOG_DIR="${EPS_LOG_DIR:-/workspace/logs}"
[ -d "$LOG_DIR" ] || LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

if [ "$SMOKE" -eq 1 ]; then
  # CPU smoke: tiny same-family model, 2 shards, 4 ctx / 4 probes / 4 layers,
  # HF generate (--no-vllm), no upload. Exercises shard → rbsigma → merge.
  N_SHARDS="${EPS_N_SHARDS:-2}"
  COMMON=(--smoke --model "Qwen/Qwen2.5-0.5B-Instruct" --expected-layers 24
          --expected-hidden 896 --device cpu --no-vllm --n-layers-smoke 4
          --n-ctx 4 --n-probes 4 --no-upload --wandb-mode disabled
          --out-dir "/tmp/issue658_8gpu_smoke/store")
  # On CPU every worker uses --gpu-id 0 (no CUDA); we still set CVD for parity.
  GPU_FOR_SHARD() { echo 0; }
else
  N_SHARDS="${EPS_N_SHARDS:-8}"
  COMMON=(--battery "$REPO_ROOT/data/issue594/battery.json"
          --out-dir "$REPO_ROOT/data/issue_658/store" --wandb-mode online)
  # shard k -> physical GPU k (0..N_SHARDS-1); requires N_SHARDS <= n GPUs.
  GPU_FOR_SHARD() { echo "$1"; }
fi
COMMON+=(--n-shards "$N_SHARDS" "${EXTRA_ARGS[@]}")

echo "[phase=load]"
echo "issue658 8-GPU dispatch: n_shards=$N_SHARDS smoke=$SMOKE log_dir=$LOG_DIR repo=$REPO_ROOT"

# ── prefetch the un-git-tracked YAML caches ONCE (round-5 code-review Major) ───
# fetch_betley_main_8() / fetch_preregistered_probes() lazily _download_if_missing
# the Betley YAMLs to data/issue404/ on first call. With 8 shards fanned out
# concurrently they would race on the same non-atomic download (partial-write
# corruption / duplicate fetches). Prefetch them here in the single launcher
# process BEFORE any shard starts, so every shard reads a complete cached file.
echo "[phase=prefetch_caches]"
if ! uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from issue404_common import fetch_betley_main_8, fetch_preregistered_probes
fetch_betley_main_8()
fetch_preregistered_probes(n=200, exclude=set(fetch_betley_main_8()))
print('prefetched Betley YAML caches')
"; then
  echo "[ERROR] YAML cache prefetch FAILED; aborting before shard fan-out"
  exit 1
fi

# ── G1-G7 per-context shard workers (concurrent, one GPU each) ─────────────────
echo "[phase=shards]"
declare -a PIDS=()
declare -a SHARD_LOGS=()
for ((k = 0; k < N_SHARDS; k++)); do
  gpu="$(GPU_FOR_SHARD "$k")"
  wlog="$LOG_DIR/issue658_shard_${k}.log"
  SHARD_LOGS+=("$wlog")
  # CVD pinned in the worker env (launcher-side) + matching --gpu-id; per-worker
  # stdout to its own FILE so its [phase=done] never pollutes the main log.
  CUDA_VISIBLE_DEVICES="$gpu" uv run python "$DRIVER" \
    "${COMMON[@]}" --shard-id "$k" --gpu-id "$gpu" \
    > "$wlog" 2>&1 &
  PIDS+=("$!")
  echo "launched shard $k on GPU $gpu (pid ${PIDS[-1]}, log $wlog)"
done

# Wait for every shard; fail loud if any shard crashed (the merge would
# otherwise find a missing shard and raise anyway, but failing here names the
# crashed shard + its log).
shard_fail=0
for ((k = 0; k < N_SHARDS; k++)); do
  if ! wait "${PIDS[$k]}"; then
    echo "[ERROR] shard $k FAILED (see ${SHARD_LOGS[$k]}); tail:"
    tail -n 30 "${SHARD_LOGS[$k]}" || true
    shard_fail=1
  fi
done
if [ "$shard_fail" -ne 0 ]; then
  echo "[ERROR] one or more shards failed; aborting before rbsigma/merge"
  exit 1
fi
echo "all $N_SHARDS shards complete"

# ── G4 r_B + G5 Σ_c (context-independent), one worker on GPU 0 ─────────────────
echo "[phase=rbsigma]"
rblog="$LOG_DIR/issue658_rbsigma.log"
gpu0="$(GPU_FOR_SHARD 0)"
if ! CUDA_VISIBLE_DEVICES="$gpu0" uv run python "$DRIVER" \
  "${COMMON[@]}" --rbsigma --gpu-id "$gpu0" \
  > "$rblog" 2>&1; then
  echo "[ERROR] rbsigma worker FAILED (see $rblog); tail:"
  tail -n 30 "$rblog" || true
  exit 1
fi
echo "rbsigma (r_B + Σ_c) complete"

# ── merge (CPU): assemble shards + rbsigma → unified store + manifest + upload ─
echo "[phase=merge]"
mglog="$LOG_DIR/issue658_merge.log"
if ! uv run python "$DRIVER" "${COMMON[@]}" --merge --device cpu > "$mglog" 2>&1; then
  echo "[ERROR] merge FAILED (see $mglog); tail:"
  tail -n 30 "$mglog" || true
  exit 1
fi
echo "merge complete; tail of merge log:"
tail -n 20 "$mglog" || true

# Terminal main-log marker (the merge python wrote the end-of-run sentinel the
# poller drains; this is the single terminal [phase=done] on the MAIN log).
echo "[phase=done] issue658 8-GPU dispatch complete (n_shards=$N_SHARDS smoke=$SMOKE)"
