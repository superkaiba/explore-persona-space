#!/usr/bin/env bash
# Issue #488 Phase 4 dispatcher — on-policy emission + ΔG eval across 27
# sources × 2 seeds × 3 picked fracs = 162 cells (each cell = 27 × 20 × 8 = 4320
# generations + 27 × 20 = 540 ΔG probes).
#
# 8-shard split: each shard owns ~21 (source, seed, frac) cells; the script
# round-robins (source, seed, frac) tuples across shards.
#
# Per CLAUDE.md feedback_cvd_hydra_override: pass --gpu-id per shard explicitly
# (sft.py clobbers env CVD if cfg.gpu_id is set; phase4_eval reads os.environ
# CUDA_VISIBLE_DEVICES which is set from --gpu-id, so we set it per shard).
#
# Phase emission per CLAUDE.md pod-side rule (poll_pipeline.py): [phase=name]
# lines + [phase=done] terminator + end-of-run sentinel.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

LOG_DIR=logs/issue_488/phase4
mkdir -p "$LOG_DIR"

# Read picked fracs from smoke output.
if [ -f logs/issue_488/smoke/picked_fracs.json ]; then
    PICKED_FRACS=$(uv run python - <<'PY'
import json
print(" ".join(str(x) for x in json.loads(open("logs/issue_488/smoke/picked_fracs.json").read())["picked_fracs"]))
PY
)
else
    PICKED_FRACS="0.25 1.00 2.00"
fi

ALL_CIDS=(A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 C1 D1 D2 D3 D4 D5 \
          E2 E3 E4 E5 F1 F2 F3 F4 G1 G2 G3)
SEEDS=(42 137)

# Build the (source, seed, frac) work list.
WORK=()
for source in "${ALL_CIDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for frac in $PICKED_FRACS; do
            WORK+=("$source:$seed:$frac")
        done
    done
done

echo "[phase=phase4_eval] $(date -Iseconds) — ${#WORK[@]} cells across 8 GPU shards"

# 8 shards: each shard takes every 8th tuple.
NUM_SHARDS=8
FAILED_FILE="$LOG_DIR/phase4_failed.txt"
: > "$FAILED_FILE"

PIDS=()
for ((s=0; s<NUM_SHARDS; s++)); do
    SHARD_WORK=()
    for ((i=s; i<${#WORK[@]}; i+=NUM_SHARDS)); do
        SHARD_WORK+=("${WORK[i]}")
    done
    if [ "${#SHARD_WORK[@]}" -eq 0 ]; then
        continue
    fi
    # Build CLI args from this shard's tuples.
    SOURCES=()
    SEEDS_USED=()
    FRACS_USED=()
    for tup in "${SHARD_WORK[@]}"; do
        IFS=":" read -r src seed frac <<<"$tup"
        SOURCES+=("$src")
        SEEDS_USED+=("$seed")
        FRACS_USED+=("$frac")
    done
    SHARD_LOG="$LOG_DIR/phase4_shard${s}.log"
    # Run sequentially within the shard; the python script handles its own loop.
    (
        for tup in "${SHARD_WORK[@]}"; do
            IFS=":" read -r src seed frac <<<"$tup"
            uv run python scripts/i488_phase4_eval_onpolicy.py \
                --source "$src" --seed "$seed" --frac "$frac" --gpu-id "$s" \
                >> "$SHARD_LOG" 2>&1 || echo "$tup" >> "$FAILED_FILE"
        done
    ) &
    PIDS+=("$!")
done

ANY_FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        ANY_FAIL=1
    fi
done

if [ -s "$FAILED_FILE" ]; then
    echo "[phase=failed] phase4 had ${ANY_FAIL} shard rc!=0; failed tuples:" >&2
    cat "$FAILED_FILE" >&2
    sentinel=/workspace/logs/issue-488-phase4-failed.json
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<PY
import json, datetime
payload = {
    "issue": 488,
    "phase": "phase4_eval",
    "failure_class": "code",
    "reason": "phase4_cell_failures",
    "failed_tuples": open("$FAILED_FILE").read().split(),
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
PY
    exit 2
fi

echo "[phase=phase4_eval] ok $(date -Iseconds)"
echo "[phase=done] phase4 complete $(date -Iseconds)"
