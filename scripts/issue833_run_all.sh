#!/usr/bin/env bash
# Issue #833 — pod/GCE-side driver, plan v6 §4/§10: A0′ gates -> R_base′ regen
# -> A2 generate -> B1 extract-rbase (same-era L1/L2) -> [extract-context iff
# the A0′ summary flags reextract_context_vectors] -> B2 extract (R⁺ legs) ->
# C upload (BOTH namespaces, 4,320 + 4,320 npz) -> finalize (results sentinel).
#
# Contract (pod-side reporting rule): short "[phase=...]" lines on stdout ending
# in a single terminal "[phase=done]"; heavy output goes to per-phase log FILES
# (the GCE metadata runner dies on giant progress lines — gotchas #491); result
# sentinels are JSON files under /workspace/logs/issue-833-*.json drained by the
# VM poller. The pod NEVER calls scripts/task.py.
#
# A0′ hard-fails BEFORE Phase A: rc=2 K1 (vLLM-LoRA plumbing, case B), rc=3
# smoke/join FAIL, rc=4 K1b (adapter/probe-surface invalid, case C), rc=5 K4
# (in-process determinism). There is NO automatic peft-fallback A0 re-run — the
# retired v4 token-identity gate was its trigger (plan v6 §4); --gen-backend
# peft (EPM_I833_GEN_BACKEND=peft) stays a MANUAL option only. A c_C-parity
# FAIL does not abort: it sets reextract_context_vectors in a0_summary.json and
# this driver runs the extract-context contingency stage (plan §13
# allowed-without-asking).

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

# ── A0′: gate set (hard-fail before Phase A; no automatic backend fallback) ──
rc=0
set +e
uv run python scripts/issue833_extract_onpolicy.py --stage parity \
  "${COMMON_ARGS[@]}" --gen-backend "$GEN_BACKEND" >>"$LOG_DIR/issue-833-parity.log" 2>&1
rc=$?
set -e
echo "[phase=a0] rc=$rc"
phase_sentinel a0 "$rc"
if [ "$rc" -ne 0 ]; then
  echo "[phase=a0] A0′ gates FAIL (rc=$rc: 2=K1 plumbing, 3=smoke/join, 4=K1b probe-surface, 5=K4 determinism) — ABORTING before Phase A"
  tail -n 40 "$LOG_DIR/issue-833-parity.log" || true
  exit "$rc"
fi

# ── A1 R_base′ regen -> A2 generate -> B1 extract-rbase (same-era L1/L2) ─────
run_stage rbase
run_stage generate
run_stage extract-rbase

# ── extract-context: ONLY when the A0′ c_C stack-parity probe flagged the
# registered contingency (reextract_context_vectors: true in a0_summary.json) ─
REEXTRACT_CTX=$(uv run python -c "import json, pathlib; p = pathlib.Path('$OUT_DIR/parity/a0_summary.json'); print(int(json.loads(p.read_text()).get('reextract_context_vectors', False)) if p.exists() else 0)")
if [ "$REEXTRACT_CTX" = "1" ]; then
  echo "[phase=extract-context] c_C parity flagged — running fleet context re-extraction"
  run_stage extract-context
fi

# ── B2: R⁺ 2-leg extraction, C: uploads (both namespaces + both raw buckets) ─
run_stage extract
run_stage upload

# ── finalize: results sentinel (A0′ verdicts + namespace counts + repro card) ─
GPU_HOURS=$(awk -v s=$((SECONDS - START_TS)) 'BEGIN { printf "%.2f", s / 3600 }')
echo "[phase=finalize]"
uv run python scripts/issue833_extract_onpolicy.py --stage finalize \
  "${COMMON_ARGS[@]}" --gen-backend "$GEN_BACKEND" \
  --gpu-hours-used "$GPU_HOURS" --final-commit-sha "$COMMIT_SHA" \
  --worktree-path "$REPO_ROOT" >>"$LOG_DIR/issue-833-finalize.log" 2>&1
phase_sentinel finalize 0

echo "[phase=done]"
