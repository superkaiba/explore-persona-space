#!/usr/bin/env bash
# issue-2054 coordinated-regen R1 wave-K GEN driver (GPU pod, 2x GPU;
# plan v12 §4 R1). Usage: bash scripts/issue2054_regen_wave_gen_driver.sh <wave>
#
# headroom preamble (plan §4 item 7), then the wave driver's gen stage:
# per-character pending set (T minus admitted, attempts < cap), parent
# generator subprocesses fanned across BOTH GPUs (launcher-env CVD pin per
# lane), prejudge pools + state uploaded fail-loud, pod terminates before
# the (pod-free) VM judge leg — the per-wave provision/terminate cycle
# (plan §9 GPU-width right-sizing).
set -uo pipefail
WAVE="${1:?usage: issue2054_regen_wave_gen_driver.sh <wave>}"
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_REGEN_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="/tmp/issue2054_regen_logs"; mkdir -p "$LOG_DIR"; }
OUT_DIR="${ISSUE2054_REGEN_OUT_DIR:-data/issue_2054/common_regen}"
HF_PREFIX="${ISSUE2054_REGEN_HF_PREFIX:-issue2054_lattice/common_regen}"
PARENT_PREFIX="${ISSUE2054_REGEN_PARENT_PREFIX:-issue2054_lattice}"
GPUS="${ISSUE2054_REGEN_GPUS:-0,1}"

echo "[phase=regen_wave_gen] driver start wave=${WAVE} gpus=${GPUS} $(date -u +%FT%TZ)"

HEADROOM_GB="${ISSUE2054_REGEN_GEN_HEADROOM_GB:-20}"
echo "[phase=regen_wave_gen stage=headroom] floor=${HEADROOM_GB}GB $(date -u +%FT%TZ)"
OUT_DIR="$OUT_DIR" HEADROOM_GB="$HEADROOM_GB" uv run python - <<'PYEOF'
import os

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

assert_out_root_headroom(
    os.environ["OUT_DIR"], float(os.environ["HEADROOM_GB"]), phase="regen_wave_gen"
)
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[phase=regen_wave_gen] HALT headroom rc=${rc}"
  exit "$rc"
fi

echo "[phase=regen_wave_gen stage=gen] start wave=${WAVE} $(date -u +%FT%TZ)"
uv run python scripts/issue2054_regen_waves.py \
  --stage gen \
  --wave "$WAVE" \
  --output-dir "$OUT_DIR" \
  --seed 137 \
  --hf-prefix "$HF_PREFIX" \
  --parent-prefix "$PARENT_PREFIX" \
  --gpus "$GPUS" \
  --state-from-hf \
  > "$LOG_DIR/issue-2054-regen-wave${WAVE}-gen.log" 2>&1
rc=$?
echo "[phase=regen_wave_gen stage=gen] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=regen_wave_gen] HALT gen rc=${rc} (tail follows)"
  tail -30 "$LOG_DIR/issue-2054-regen-wave${WAVE}-gen.log" || true
  exit "$rc"
fi
echo "[phase=regen_wave_gen] driver_rc=0 wave=${WAVE} $(date -u +%FT%TZ)"
