#!/usr/bin/env bash
# issue-2054 coordinated-regen R0 driver (cpu-mid pod / VM; plan v12 §4 R0).
#
# 1. headroom preamble (plan §4 item 7) — draw-file + #1738 manifest staging
#    + answers-pool build footprint (~10 GB floor, env-overridable);
# 2. wave driver --stage draw: T=15,700 uniform (seed 137) from the FULL
#    32,000-row shared_question_draw (manifest-first from the PARENT
#    scaffolds prefix), full-grain re-filter, fold-map deterministic
#    extension (branch-copy reference asserted), gate1_projection, uploads;
# 3. answers-pool extension to cover T: build_answers with
#    --required-cids-jsonl <target_set> --answers-prefix <prefix>/answers.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_REGEN_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="/tmp/issue2054_regen_logs"; mkdir -p "$LOG_DIR"; }
OUT_DIR="${ISSUE2054_REGEN_OUT_DIR:-data/issue_2054/common_regen}"
HF_PREFIX="${ISSUE2054_REGEN_HF_PREFIX:-issue2054_lattice/common_regen}"
PARENT_PREFIX="${ISSUE2054_REGEN_PARENT_PREFIX:-issue2054_lattice}"
TARGET_N="${ISSUE2054_REGEN_TARGET_N:-15700}"

echo "[phase=regen_r0] driver start prefix=${HF_PREFIX} target_n=${TARGET_N} $(date -u +%FT%TZ)"

HEADROOM_GB="${ISSUE2054_REGEN_R0_HEADROOM_GB:-10}"
echo "[phase=regen_r0 stage=headroom] floor=${HEADROOM_GB}GB $(date -u +%FT%TZ)"
OUT_DIR="$OUT_DIR" HEADROOM_GB="$HEADROOM_GB" uv run python - <<'PYEOF'
import os

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

assert_out_root_headroom(
    os.environ["OUT_DIR"], float(os.environ["HEADROOM_GB"]), phase="regen_r0"
)
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[phase=regen_r0] HALT headroom rc=${rc}"
  exit "$rc"
fi

echo "[phase=regen_r0 stage=draw] start $(date -u +%FT%TZ)"
uv run python scripts/issue2054_regen_waves.py \
  --stage draw \
  --output-dir "$OUT_DIR" \
  --target-n "$TARGET_N" \
  --seed 137 \
  --hf-prefix "$HF_PREFIX" \
  --parent-prefix "$PARENT_PREFIX" \
  > "$LOG_DIR/issue-2054-regen-r0-draw.log" 2>&1
rc=$?
echo "[phase=regen_r0 stage=draw] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=regen_r0] HALT draw rc=${rc} (tail follows)"
  tail -30 "$LOG_DIR/issue-2054-regen-r0-draw.log" || true
  exit "$rc"
fi

echo "[phase=regen_r0 stage=answers] start $(date -u +%FT%TZ)"
uv run python scripts/issue2054_build_answers.py \
  --out-dir "$OUT_DIR/answers" \
  --required-cids-jsonl "$OUT_DIR/target_set/target_set.jsonl" \
  --answers-prefix "$HF_PREFIX/answers" \
  --seed 137 \
  > "$LOG_DIR/issue-2054-regen-r0-answers.log" 2>&1
rc=$?
echo "[phase=regen_r0 stage=answers] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=regen_r0] HALT answers rc=${rc} (tail follows)"
  tail -30 "$LOG_DIR/issue-2054-regen-r0-answers.log" || true
  exit "$rc"
fi

echo "[phase=regen_r0] driver_rc=0 $(date -u +%FT%TZ)"
