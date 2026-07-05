#!/usr/bin/env bash
# Issue #923 mixed-forward-span-stitch GPU phase (plan v9 §4.2/§9).
#
# SINGLE-GPU by right-sizing (§9: ~12,000 prefill-only forwards ≈ 0.75-1 GPU-h
# — a narrow phase; intent lora-7b = 1x A100-80): the 4 shards run
# SEQUENTIALLY purely for resume granularity (an interrupted run re-dispatches
# and skips existing packs). Then ONE upload invocation (folder commit +
# scoped-listing verify + UPLOAD_COMPLETE_MIXED.json sentinel LAST), then the
# k1 content-identity gate (mixed flast vs the pooled pool_ffull flast at the
# PINNED pooled revision) — AFTER upload so the packs persist even when the
# gate fails; a failed gate exits nonzero (rc=4) so the orchestrator HALTS
# before dispatching the CPU phase (plan §6 k1).
#
# Usage (dispatched via dispatch_issue.py --intent lora-7b --workload-cmd
# "bash scripts/issue923_mixed_gpu_phase.sh"):
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
# GCE lane has NO .env (the startup script exports tokens); CONDITIONAL
# sourcing only (the e9c8809113 / att-20260703-163121 rule).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/issue_923}"
mkdir -p "$LOG_DIR"

NGPU=$(nvidia-smi --list-gpus | wc -l)
if [ "$NGPU" -lt 1 ]; then
  echo "[issue923_mixed_gpu] FATAL: no visible GPUs" >&2
  echo "[phase=failed]"
  exit 2
fi
N_SHARDS="${EPS_923_MIXED_SHARDS:-4}"
echo "[issue923_mixed_gpu] sequential ${N_SHARDS}-shard mixed capture on 1 GPU (${NGPU} visible)"

RC=0
for k in $(seq 0 $((N_SHARDS - 1))); do
  echo "[issue923_mixed_gpu] shard $k/$N_SHARDS"
  (
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
      uv run python scripts/issue923_capture.py \
      --qry-mix-features --shard "$k/$N_SHARDS" "$@" \
      2>&1 | tee "$LOG_DIR/mixed_capture_shard${k}.log"
  ) || RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "[issue923_mixed_gpu] shard $k FAILED (rc=$RC)" >&2
    echo "[phase=failed]"
    exit "$RC"
  fi
done

echo "[issue923_mixed_gpu] shards complete; uploading (single invocation)"
(
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    uv run python scripts/issue923_capture.py --qry-mix-upload "$@" \
    2>&1 | tee "$LOG_DIR/mixed_upload.log"
) || RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[phase=failed]"
  exit "$RC"
fi

echo "[issue923_mixed_gpu] k1 identity gate (pooled ffull refs at the pinned revision)"
(
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    uv run python scripts/issue923_capture.py --qry-mix-identity-check "$@" \
    2>&1 | tee "$LOG_DIR/mixed_identity.log"
) || RC=$?
if [ "$RC" -ne 0 ]; then
  # identity_check_mix.json is already uploaded; the nonzero exit is the k1
  # HALT signal (do NOT dispatch the CPU phase; debug provenance first).
  echo "[phase=failed]"
  exit "$RC"
fi

echo "[phase=done]"
