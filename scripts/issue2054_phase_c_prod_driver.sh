#!/usr/bin/env bash
# issue-2054 Phase-C production driver (pod-side, 2x GPU).
#
# On-policy continuation (cell (d)) over the r14 form matrix, per model:
#   attrib_quoted + bare_label — all 5 variants, 2-way variant shard (composer CVD pins)
#   chat + bare_text           — assistant-only, run CONCURRENTLY (one GPU each)
# Models sequential: qwen2.5-7b-instruct then qwen2.5-7b (per-model output dirs
# composed by scripts/issue2054_shard_launch.py — a second model into ONE dir
# is REFUSED by the resume sidecar regime).
#
# Prereq staged in-driver: scripts/issue2054_stage_scaffolds.py (manifest-first
# sharded ADMITTED pools from HF, count + conv_id set-equality vs kept.json).
# phase_c resume sidecars make the whole driver idempotent on relaunch.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_PC_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"

echo "[phase=phase_c_prod] driver start $(date -u +%FT%TZ)"

echo "[phase=phase_c_prod stage=stage_scaffolds] start $(date -u +%FT%TZ)"
# Own log: the stager's terminal line carries the reserved [phase=done] token,
# which must not enter this dispatcher's main log (poll_pipeline reads it as a
# false status=done — #545/#920; workflow_lint phase-done-reserved check).
uv run python scripts/issue2054_stage_scaffolds.py \
  > "$LOG_DIR/issue-2054-pc-stage-scaffolds.log" 2>&1
rc=$?
echo "[phase=phase_c_prod stage=stage_scaffolds] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=phase_c_prod] HALT stage_scaffolds rc=${rc} (tail follows)"
  tail -30 "$LOG_DIR/issue-2054-pc-stage-scaffolds.log" || true
  exit "$rc"
fi

MODELS=(qwen2.5-7b-instruct qwen2.5-7b)
ASSIST=conversation_paired_stories_assistant

for MODEL in "${MODELS[@]}"; do
  for FORM in attrib_quoted bare_label; do
    echo "[phase=phase_c_prod model=${MODEL} form=${FORM}] start $(date -u +%FT%TZ)"
    uv run python scripts/issue2054_shard_launch.py \
      --driver phase_c --form "$FORM" --model "$MODEL" --gpus 0,1
    rc=$?
    echo "[phase=phase_c_prod model=${MODEL} form=${FORM}] rc=${rc} $(date -u +%FT%TZ)"
    if [ "$rc" -ne 0 ]; then
      echo "[phase=phase_c_prod] HALT ${MODEL}/${FORM} rc=${rc}"
      exit "$rc"
    fi
  done

  echo "[phase=phase_c_prod model=${MODEL} form=chat+bare_text] concurrent start $(date -u +%FT%TZ)"
  uv run python scripts/issue2054_shard_launch.py \
    --driver phase_c --form chat --model "$MODEL" --gpus 0 --variants "$ASSIST" \
    > "$LOG_DIR/issue-2054-pc-chat-${MODEL}.log" 2>&1 &
  P1=$!
  uv run python scripts/issue2054_shard_launch.py \
    --driver phase_c --form bare_text --model "$MODEL" --gpus 1 --variants "$ASSIST" \
    > "$LOG_DIR/issue-2054-pc-baretext-${MODEL}.log" 2>&1 &
  P2=$!
  wait "$P1"; RC1=$?
  wait "$P2"; RC2=$?
  echo "[phase=phase_c_prod model=${MODEL} form=chat] rc=${RC1}; form=bare_text rc=${RC2} $(date -u +%FT%TZ)"
  if [ "$RC1" -ne 0 ] || [ "$RC2" -ne 0 ]; then
    echo "[phase=phase_c_prod] HALT ${MODEL} chat rc=${RC1} bare_text rc=${RC2} (tails follow)"
    tail -30 "$LOG_DIR/issue-2054-pc-chat-${MODEL}.log" || true
    tail -30 "$LOG_DIR/issue-2054-pc-baretext-${MODEL}.log" || true
    exit 1
  fi
done

echo "[phase=phase_c_prod] driver_rc=0 $(date -u +%FT%TZ)"
