#!/usr/bin/env bash
# Issue #634 dispatch — Phase 0 (build + assert the frozen Panel-B map) then
# Phase 1 (28-layer Method-A behavior-vector extraction over 275 roles, K=40),
# HF upload, then the authoritative epm:results sentinel (with reproducibility
# card). Pod-side / GCP-lane workload command passed via --workload-cmd.
#
# Phase 2 (joint geometry) is NOT pod-side — it runs VM-side off the
# HF-uploaded behavior tensor + #594's context bank, AFTER this dispatch's
# sentinel is drained by /issue Step 6d.2 (poll_pipeline.py).
#
# Smoke gate (option C): set EPM_SMOKE_ROLES=1 to run a 1-role extraction
# FIRST (times the actual A100 lane); the orchestrator inspects the smoke
# sentinel, then re-dispatches without EPM_SMOKE_ROLES for the full run. If the
# 1-role extrapolation exceeds 20 GPU-h, set EPM_N_QUESTIONS=20 to descope K.
#
# Phase sentinels go to /workspace/logs/issue-634-*.json per the
# poll_pipeline.py contract. Pod-side code NEVER shells task.py.
#
# Plan: tasks/running/634/plans/v2.md §4 (pipeline DAG) + §9 (compute).
# Compute: lora-7b intent -> GCP a2-ultragpu-1g (1x A100-80GB), forward-only.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

OUTPUT_DIR="${EPM_OUTPUT_DIR:-data/issue634/behavior_vectors}"
FAMILY_MAP="data/issue634/behavior_family_map.json"
N_QUESTIONS="${EPM_N_QUESTIONS:-40}"
SEED="${EPM_SEED:-42}"
SMOKE_ROLES="${EPM_SMOKE_ROLES:-0}"
GPU_ID="${EPM_GPU_ID:-0}"
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"

LOG_DIR=logs/issue_634
mkdir -p "$LOG_DIR" /workspace/logs 2>/dev/null || mkdir -p "$LOG_DIR"

echo "[phase=dispatch-start] $(date -u +%FT%TZ) smoke_roles=$SMOKE_ROLES K=$N_QUESTIONS"
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi; CPU/host context)"

# ---------------------------------------------------------------------------
# Phase 0 (CPU): freeze the Panel-B map + assert all keys in role_list.json
# ---------------------------------------------------------------------------
echo "[phase=p0_family_map] $(date -u +%FT%TZ)"
uv run python scripts/issue634_build_family_map.py --out "$FAMILY_MAP" \
  2>&1 | tee -a "$LOG_DIR/p0.log"

# ---------------------------------------------------------------------------
# Phase 1 (GPU): 28-layer Method-A behavior-vector extraction + HF upload
# ---------------------------------------------------------------------------
# --no-sentinel: this dispatch owns the authoritative epm:results sentinel
# (with the reproducibility card) written AFTER upload below. The extraction
# script emits [phase=extract-complete] (NOT [phase=done]) so the poller's
# log-tail read does not see a premature terminal token.
SMOKE_FLAG=""
SENTINEL_KIND="epm:results"
if [ "$SMOKE_ROLES" -gt 0 ]; then
  SMOKE_FLAG="--smoke-roles $SMOKE_ROLES"
  SENTINEL_KIND="epm:smoke-result"
fi
echo "[phase=p1_extract] $(date -u +%FT%TZ)"
# shellcheck disable=SC2086
uv run python scripts/issue634_extract_behavior_vectors.py \
  --layers $LAYERS --n-questions "$N_QUESTIONS" --seed "$SEED" \
  --gpu-id "$GPU_ID" --output-dir "$OUTPUT_DIR" --family-map "$FAMILY_MAP" \
  --no-sentinel $SMOKE_FLAG \
  2>&1 | tee -a "$LOG_DIR/p1.log"

# ---------------------------------------------------------------------------
# Authoritative end-of-run sentinel (reproducibility card)
# ---------------------------------------------------------------------------
echo "[phase=sentinel] $(date -u +%FT%TZ)"
uv run python scripts/issue634_write_sentinel.py \
  --manifest "$OUTPUT_DIR/extraction_manifest.json" \
  --kind "$SENTINEL_KIND" \
  2>&1 | tee -a "$LOG_DIR/sentinel.log"

# Terminal line — RESERVED single [phase=done] for poll_pipeline.py.
echo "[phase=done] $(date -u +%FT%TZ)"
