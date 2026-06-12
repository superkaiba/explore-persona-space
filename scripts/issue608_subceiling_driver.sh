#!/usr/bin/env bash
# Task #608 follow-up sub-ceiling-install driver — GCP/auto lane single
# workload command (plan v5 §4 + §7; REPO_ROOT threading per the GCP lane
# contract — the lane invokes this with REPO_ROOT="$WORKLOAD_ROOT").
#
# Phases:
#   1. Smoke-gated cell alone (villain:posonly_dose_dense, GPU 0) — the §7
#      follow-up gate (8-dir checkpoint assert + 9 step reads + step-44
#      mini-judge >= 0.90) fires inline; HALT exits non-zero before the sweep.
#   2. Full 12-cell prefetch ONCE (serialized — avoids the 4-shard concurrent-
#      prefetch race on the shared probe/pool files).
#   3. Four parallel dispatcher shards over the remaining 11 cells
#      (--skip-prefetch); the last finisher (all 12 cell-states complete)
#      emits the single epm:results sentinel (payload carries
#      followup_label: sub-ceiling-install) + [phase=done].
set -euo pipefail
cd "${REPO_ROOT:-$PWD}"

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"
FOLLOWUP=sub-ceiling-install

echo "[driver] phase 1: smoke-gated cell (villain:posonly_dose_dense, gpu 0)"
uv run python scripts/dispatch_sycophancy_608.py --followup "$FOLLOWUP" --gpu-id 0 \
  --cells villain:posonly_dose_dense 2>&1 | tee "$LOG_DIR/issue608_subceiling_smoke_shard.log"

echo "[driver] phase 1.5: full 12-cell prefetch (serialized)"
uv run python -m explore_persona_space.experiments.sycophancy_posonly_608.prefetch_inputs \
  --cells villain:posonly_dose_dense,comedian:posonly_dose_dense,assistant:posonly_dose_dense,qwen_default:posonly_dose_dense,software_engineer:posonly_dose_dense,kindergarten_teacher:posonly_dose_dense,villain:contrastive_dense,comedian:contrastive_dense,assistant:contrastive_dense,qwen_default:contrastive_dense,software_engineer:contrastive_dense,kindergarten_teacher:contrastive_dense \
  2>&1 | tee "$LOG_DIR/issue608_subceiling_prefetch_full.log"

echo "[driver] phase 2: 4 parallel shards over the remaining 11 cells"
declare -a SHARD_CELLS=(
  "comedian:posonly_dose_dense,assistant:posonly_dose_dense,villain:contrastive_dense"
  "qwen_default:posonly_dose_dense,software_engineer:posonly_dose_dense,comedian:contrastive_dense"
  "kindergarten_teacher:posonly_dose_dense,assistant:contrastive_dense,qwen_default:contrastive_dense"
  "software_engineer:contrastive_dense,kindergarten_teacher:contrastive_dense"
)
pids=()
for gpu in 0 1 2 3; do
  uv run python scripts/dispatch_sycophancy_608.py --followup "$FOLLOWUP" \
    --gpu-id "$gpu" --skip-prefetch \
    --cells "${SHARD_CELLS[$gpu]}" \
    > "$LOG_DIR/issue608_subceiling_shard${gpu}.log" 2>&1 &
  pids+=($!)
  sleep 5
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

echo "[driver] all shards exited (rc=$rc)"
exit "$rc"
