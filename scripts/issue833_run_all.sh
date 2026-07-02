#!/usr/bin/env bash
# Issue #833 — pod/GCE-side driver: A0 parity gates -> A generate -> B extract
# -> C upload -> finalize (results sentinel). Plan v3 §4/§10.
#
# Contract (pod-side reporting rule): short "[phase=...]" lines on stdout ending
# in a single terminal "[phase=done]"; heavy output goes to per-phase log FILES
# (the GCE metadata runner dies on giant progress lines — gotchas #491); result
# sentinels are JSON files under /workspace/logs/issue-833-*.json drained by the
# VM poller. The pod NEVER calls scripts/task.py.
#
# A0 hard-fails BEFORE Phase A. rc=2 (backend-parity FAIL) triggers ONE re-run
# with the PeftModel fallback backend (plan §8 R1); any other non-zero rc (incl.
# rc=3 cross-era parity FAIL) aborts the run — the fleet-wide L1/L2
# re-extraction fallback is an orchestrator decision, not an in-run continue.

set -euo pipefail

export PATH="/root/.local/bin:$PATH" # GCE sudo bash loses root's login PATH (gotchas)
REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

LOG_DIR="${EPM_SENTINEL_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"
export EPM_SENTINEL_DIR="$LOG_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TQDM_DISABLE=1

GEN_BACKEND="${EPM_I833_GEN_BACKEND:-vllm}"
OUT_DIR="eval_results/issue_833"
COMMON_ARGS=(--behavior all --seed 42 --layers 7 14 21 --out "$OUT_DIR" --gpu-id 0 --max-new-tokens 1024)
START_TS=$SECONDS
COMMIT_SHA="$(git rev-parse HEAD)"

phase_sentinel() { # name rc
  printf '{"sentinel_schema_version": 1, "kind": "epm:progress", "version": 1, "task_id": 833, "note": "issue-833 phase %s rc=%s (commit %s)"}\n' \
    "$1" "$2" "$COMMIT_SHA" >"$LOG_DIR/issue-833-phase-$1.json"
}

run_stage() { # stage extra-args...
  local stage="$1"
  shift
  echo "[phase=${stage}]"
  local log="$LOG_DIR/issue-833-${stage}.log"
  set +e
  uv run python scripts/issue833_extract_onpolicy.py --stage "$stage" \
    "${COMMON_ARGS[@]}" --gen-backend "$GEN_BACKEND" "$@" >>"$log" 2>&1
  local rc=$?
  set -e
  phase_sentinel "$stage" "$rc"
  return $rc
}

# ── A0: parity gates (hard-fail before Phase A) ──────────────────────────────
rc=0
set +e
uv run python scripts/issue833_extract_onpolicy.py --stage parity \
  "${COMMON_ARGS[@]}" --gen-backend "$GEN_BACKEND" >>"$LOG_DIR/issue-833-parity.log" 2>&1
rc=$?
set -e
echo "[phase=a0] rc=$rc"
phase_sentinel a0 "$rc"
if [ "$rc" -eq 2 ]; then
  echo "[phase=a0] backend parity FAIL -> PeftModel fallback re-run (plan §8 R1)"
  GEN_BACKEND=peft
  set +e
  uv run python scripts/issue833_extract_onpolicy.py --stage parity \
    "${COMMON_ARGS[@]}" --gen-backend "$GEN_BACKEND" >>"$LOG_DIR/issue-833-parity.log" 2>&1
  rc=$?
  set -e
  phase_sentinel a0-peft-rerun "$rc"
fi
if [ "$rc" -ne 0 ]; then
  echo "[phase=a0] A0 gates FAIL (rc=$rc) — ABORTING before Phase A"
  tail -n 40 "$LOG_DIR/issue-833-parity.log" || true
  exit "$rc"
fi

# ── A: on-policy generation, B: 2-leg extraction, C: uploads ────────────────
run_stage generate
run_stage extract
run_stage upload

# ── finalize: results sentinel (counts + parity summaries + repro card) ─────
GPU_HOURS=$(awk -v s=$((SECONDS - START_TS)) 'BEGIN { printf "%.2f", s / 3600 }')
echo "[phase=finalize]"
uv run python scripts/issue833_extract_onpolicy.py --stage finalize \
  "${COMMON_ARGS[@]}" --gen-backend "$GEN_BACKEND" \
  --gpu-hours-used "$GPU_HOURS" --final-commit-sha "$COMMIT_SHA" \
  --worktree-path "$REPO_ROOT" >>"$LOG_DIR/issue-833-finalize.log" 2>&1
phase_sentinel finalize 0

echo "[phase=done]"
