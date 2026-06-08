#!/usr/bin/env bash
# Issue #488 Phase 4 dispatcher — on-policy emission + ΔG eval across 27
# sources × 2 seeds × 6 production fracs = 324 cells (each cell = 27 × 20 × 8 =
# 4320 generations + 27 × 20 = 540 ΔG probes).
#
# Plan v3 §6.2.D: the ρ-blind post-hoc picker scans the FULL production frac
# set {0.10, 0.25, 0.50, 1.00, 2.00, 3.00} and selects the lowest-eligible
# frac in ascending order. That construct presupposes ALL 6 fracs were
# evaluated in Phase 4 — pre-selecting via a smoke `picked_fracs.json`
# (the pre-v3 picked-3 design) makes the scan space half-empty and the
# "lowest eligible frac" guarantee invalid. The `picked_fracs.json`
# consumption path has therefore been removed.
#
# 8-shard split: each shard owns ~41 (source, seed, frac) cells; the script
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

# Plan v3 §6.2.D: full production frac set, evaluated unconditionally. The
# `picked_fracs.json` consumption path (pre-v3) has been removed — the
# post-hoc picker requires all 6 fracs to be in Phase 4's output.
PHASE4_FRACS="0.10 0.25 0.50 1.00 2.00 3.00"

ALL_CIDS=(A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 C1 D1 D2 D3 D4 D5 \
          E2 E3 E4 E5 F1 F2 F3 F4 G1 G2 G3)
SEEDS=(42 137)

# Build the (source, seed, frac) work list.
WORK=()
for source in "${ALL_CIDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for frac in $PHASE4_FRACS; do
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
