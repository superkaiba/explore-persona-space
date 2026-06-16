#!/usr/bin/env bash
# #597 follow-up `filler-control-multiseed` (plan v6) — launch the 18 new
# training units: 3 arms (positives-only / contrastive / filler) x 3 sources
# (villain, assistant, qwen_default) x 2 NEW seeds (137, 7). The seed-42 cells
# are NOT re-run — their trajectories are already committed and are REUSED
# verbatim by fig_armD_3way_panel_only.py.
#
# Serial on ONE GPU (plan v6 §8: ~4 GPU-h total). The dispatcher
# (dispatch_leakage_dynamics_597.py) is the canonical entry point and carries
# every phase (preflight + filler-R reuse, training, panel probe, parity
# diagnostic, upload). This wrapper only loops it across {seed} x {recipe} and
# pins CUDA_VISIBLE_DEVICES to match --gpu (the #557 in-process-clobber gotcha:
# the launcher MUST export the same index it passes to --gpu, or the dispatcher
# raises a hard GPU-pin mismatch).
#
# The slice-6 router can instead chain these invocations directly via
# --workload-cmd if a wrapper is undesirable; this script IS that command list.
#
# Usage:
#   bash scripts/issue_597/launch_multiseed_597.sh [GPU] [EXTRA_DISPATCHER_FLAGS...]
# Example (smoke one cell first):
#   bash scripts/issue_597/launch_multiseed_597.sh 0 --smoke --only-source villain
set -euo pipefail

# Populate HF / WandB credentials before the dispatcher's own load_dotenv()
# (defense-in-depth; the dispatcher also calls load_dotenv() at module top).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .env ]]; then
  set -a && source .env && set +a
fi

GPU="${1:-0}"
shift || true
EXTRA_FLAGS=("$@")  # e.g. --smoke --only-source villain, or --skip-arm-a-gate

SEEDS=(137 7)
RECIPES=(pos_only_dynamics contrastive_dense_early filler_dynamics)
SOURCES="villain,assistant,qwen_default"
LOG_DIR="${EPS_597_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"

echo "[launch_multiseed_597] GPU=${GPU} seeds=${SEEDS[*]} recipes=${RECIPES[*]} sources=${SOURCES}"
echo "[launch_multiseed_597] extra flags: ${EXTRA_FLAGS[*]:-<none>}"

for SEED in "${SEEDS[@]}"; do
  for RECIPE in "${RECIPES[@]}"; do
    LOG="${LOG_DIR}/issue597_multiseed_${RECIPE}_seed${SEED}.log"
    echo "[launch_multiseed_597] === recipe=${RECIPE} seed=${SEED} -> ${LOG} ==="
    # CUDA_VISIBLE_DEVICES MUST equal --gpu (dispatcher hard-asserts the match).
    CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/issue_597/dispatch_leakage_dynamics_597.py \
      --recipe "${RECIPE}" \
      --seed "${SEED}" \
      --gpu "${GPU}" \
      --sources "${SOURCES}" \
      "${EXTRA_FLAGS[@]}" \
      2>&1 | tee "${LOG}"
    echo "[launch_multiseed_597] recipe=${RECIPE} seed=${SEED} complete"
  done
done

echo "[launch_multiseed_597] all ${#SEEDS[@]}x${#RECIPES[@]} cells complete"
