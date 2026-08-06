#!/usr/bin/env bash
# issue-2054 capture production driver (pod-side, 2x GPU) — cells (a),(b),(d).
#
# Teacher-forced layer-19 capture over the r14 form matrix, per model x condition:
#   inserted  — reads data/issue_2054/spliced_inserted/ (shared; model is read-side)
#   on_policy — reads data/issue_2054/on_policy/<model>/ (model-matched, composer map)
# Story forms (attrib_quoted, bare_label): all 5 variants, 2-way variant shard.
# Assistant-only forms (chat, bare_text): run CONCURRENTLY, one GPU each.
# target_conv_ids stays 0 (ALL rows): no prefix-cap mismatch risk, maximal
# conv_id intersection; fits equalize down on the intersection (plan req 8).
# Cell (c) capture (condition cell_c) is Phase-D-gated — NOT dispatched here.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_CAP_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"
A=conversation_paired_stories_assistant

echo "[phase=capture_prod] driver start $(date -u +%FT%TZ)"

for MODEL in qwen2.5-7b-instruct qwen2.5-7b; do
  for COND in inserted on_policy; do
    for FORM in attrib_quoted bare_label; do
      echo "[phase=capture_prod model=${MODEL} cond=${COND} form=${FORM}] start $(date -u +%FT%TZ)"
      uv run python scripts/issue2054_shard_launch.py \
        --driver capture --condition "$COND" --form "$FORM" --model "$MODEL" --gpus 0,1
      rc=$?
      echo "[phase=capture_prod model=${MODEL} cond=${COND} form=${FORM}] rc=${rc} $(date -u +%FT%TZ)"
      if [ "$rc" -ne 0 ]; then
        echo "[phase=capture_prod] HALT ${MODEL}/${COND}/${FORM} rc=${rc}"
        exit "$rc"
      fi
    done

    echo "[phase=capture_prod model=${MODEL} cond=${COND} form=chat+bare_text] concurrent start $(date -u +%FT%TZ)"
    uv run python scripts/issue2054_shard_launch.py \
      --driver capture --condition "$COND" --form chat --model "$MODEL" --gpus 0 --variants "$A" \
      > "$LOG_DIR/issue-2054-cap-chat-${COND}-${MODEL}.log" 2>&1 &
    P1=$!
    uv run python scripts/issue2054_shard_launch.py \
      --driver capture --condition "$COND" --form bare_text --model "$MODEL" --gpus 1 --variants "$A" \
      > "$LOG_DIR/issue-2054-cap-baretext-${COND}-${MODEL}.log" 2>&1 &
    P2=$!
    wait "$P1"; R1=$?
    wait "$P2"; R2=$?
    echo "[phase=capture_prod model=${MODEL} cond=${COND} form=chat] rc=${R1}; form=bare_text rc=${R2} $(date -u +%FT%TZ)"
    if [ "$R1" -ne 0 ] || [ "$R2" -ne 0 ]; then
      echo "[phase=capture_prod] HALT ${MODEL}/${COND} chat rc=${R1} bare_text rc=${R2} (tails follow)"
      tail -30 "$LOG_DIR/issue-2054-cap-chat-${COND}-${MODEL}.log" || true
      tail -30 "$LOG_DIR/issue-2054-cap-baretext-${COND}-${MODEL}.log" || true
      exit 1
    fi
  done
done

echo "[phase=capture_prod] driver_rc=0 $(date -u +%FT%TZ)"
