#!/usr/bin/env bash
# Task #608 production driver — GCP/auto lane single workload command.
#
# Phases (plan §4 + §7):
#   1. Smoke-gated cell alone (villain:posonly_dose, GPU 0) — the §7 gate fires
#      inline; a HALT exits non-zero and stops everything before the sweep.
#   2. Full-grid prefetch ONCE (serialized — avoids the 4-shard concurrent-
#      prefetch race on the shared probe/ref files).
#   3. Four parallel dispatcher shards over the remaining 18 cells
#      (--skip-prefetch); the last finisher (all 19 cell-states complete)
#      emits the single epm:results sentinel + [phase=done].
set -euo pipefail
cd "${REPO_ROOT:-$PWD}"

LOG_DIR=/workspace/logs
mkdir -p "$LOG_DIR"

echo "[driver] phase 1: smoke-gated cell (villain:posonly_dose, gpu 0)"
uv run python scripts/dispatch_sycophancy_608.py --gpu-id 0 \
  --cells villain:posonly_dose 2>&1 | tee "$LOG_DIR/issue608_smoke_shard.log"

echo "[driver] phase 1.5: full-grid prefetch (serialized)"
uv run python -m explore_persona_space.experiments.sycophancy_posonly_608.prefetch_inputs \
  --cells villain:posonly_dose,villain:posonly_epoch,comedian:posonly_dose,comedian:posonly_epoch,assistant:posonly_dose,assistant:posonly_epoch,qwen_default:posonly_dose,qwen_default:posonly_epoch,software_engineer:posonly_dose,software_engineer:posonly_epoch,kindergarten_teacher:posonly_dose,kindergarten_teacher:posonly_epoch,base:fresh_eval,villain:contrastive_fresh_eval,comedian:contrastive_fresh_eval,assistant:contrastive_fresh_eval,qwen_default:contrastive_fresh_eval,software_engineer:contrastive_fresh_eval,kindergarten_teacher:contrastive_fresh_eval \
  2>&1 | tee "$LOG_DIR/issue608_prefetch_full.log"

echo "[driver] phase 2: 4 parallel shards over the remaining 18 cells"
declare -a SHARD_CELLS=(
  "comedian:posonly_dose,villain:posonly_epoch,comedian:posonly_epoch,base:fresh_eval,qwen_default:contrastive_fresh_eval"
  "assistant:posonly_dose,assistant:posonly_epoch,kindergarten_teacher:posonly_epoch,villain:contrastive_fresh_eval,comedian:contrastive_fresh_eval"
  "qwen_default:posonly_dose,kindergarten_teacher:posonly_dose,qwen_default:posonly_epoch,assistant:contrastive_fresh_eval"
  "software_engineer:posonly_dose,software_engineer:posonly_epoch,software_engineer:contrastive_fresh_eval,kindergarten_teacher:contrastive_fresh_eval"
)
pids=()
for gpu in 0 1 2 3; do
  uv run python scripts/dispatch_sycophancy_608.py --gpu-id "$gpu" --skip-prefetch \
    --cells "${SHARD_CELLS[$gpu]}" \
    > "$LOG_DIR/issue608_shard${gpu}.log" 2>&1 &
  pids+=($!)
  sleep 5
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

echo "[driver] all shards exited (rc=$rc)"
exit "$rc"
