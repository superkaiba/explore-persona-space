#!/usr/bin/env bash
# Issue #489 Phase 4 dispatcher — vLLM on-policy gen + HF teacher-forced ΔG.
#
# Plan v5 §6.1 + §9. Shards: 8 shards across 72 (cid × frac) snapshots; each
# shard owns ~9 snapshots and evals against all 24 union contexts × 20 Q × 8
# samples. Per CLAUDE.md kill_vllm_workers: PASS A (vLLM) and PASS B (HF
# teacher-forced) run in DIFFERENT subprocesses so vLLM's worker subprocesses
# get reaped between phases.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_489
mkdir -p "$LOG_DIR"

N_SHARDS=8
SEED=42
RESUME_FLAG=""
FRACS="0.25 0.50 1.00"

for arg in "$@"; do
    case "$arg" in
        --n-shards=*) N_SHARDS="${arg#*=}" ;;
        --seed=*) SEED="${arg#*=}" ;;
        --resume) RESUME_FLAG="--resume" ;;
        --fracs=*) FRACS="${arg#*=}" ;;
        *) ;;
    esac
done

echo "[phase=phase4_start] === Phase 4 dispatcher $(date -Iseconds) n_shards=$N_SHARDS seed=$SEED ==="

pids=()
for shard in $(seq 0 $((N_SHARDS - 1))); do
    cvd=$((shard % 8))
    log="$LOG_DIR/phase4_shard${shard}of${N_SHARDS}_cvd${cvd}.log"
    CUDA_VISIBLE_DEVICES=$cvd uv run python scripts/i489_phase4_eval_onpolicy.py \
        --shard "${shard}-of-${N_SHARDS}" \
        --seed "$SEED" \
        --fracs $FRACS \
        $RESUME_FLAG \
        > "$log" 2>&1 &
    pids+=("$!:$shard")
done
fail=0
for entry in "${pids[@]}"; do
    pid="$(cut -d: -f1 <<<"$entry")"
    shard="$(cut -d: -f2 <<<"$entry")"
    if ! wait "$pid"; then
        echo "PHASE4 shard=$shard FAILED (pid=$pid)" >&2
        fail=1
    fi
done

if [ $fail -eq 1 ]; then
    epoch="$(date +%s)"
    sentinel="/workspace/logs/issue-489-epm_failure-${epoch}.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "issue": 489,
    "phase": "phase4_eval",
    "failure_class": "code",
    "reason": "One or more phase4 shards failed.",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
EOF
    exit 3
fi

echo "[phase=phase4_done] === All shards finished $(date -Iseconds) ==="
