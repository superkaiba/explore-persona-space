#!/usr/bin/env bash
# issue-2054 coordinated-regen R2+R3 driver (GPU pod, 2x GPU; plan v12 §4).
#
# One provision, contiguous GPU work (~5.5 h booked):
#   stage-r2     — export pools / answers / survivor set / extended fold map
#   phase_c      — 16 char on-policy cells on S + 4 assistant delta cells
#                  (every leg passes --target-conv-ids 15700 — the Must-Fix;
#                  the inherited default 8,000 silently truncates to a
#                  first-N prefix, invisible to gate 1)
#   assist-merge — parent assistant rows (S-intersection) + delta -> full S
#   phase_b      — deterministic inserted splice on S (chars + assistant)
#   phase_d      — cell (c) splice from the ROUND's on-policy pools
#   lengths      — per-row answer token lengths, uploaded OFF-POD (gate-2
#                  persistence) BEFORE capture
#   coverage     — pre-capture per-cell coverage assert (n_out == |S| /
#                  |delta|; target_conv_ids >= 15,700) — abort-before-R3
#   gate2        — KS parity evaluator (report+mitigate; never blocks capture)
#   capture      — teacher-forced layer-19 capture of all 48 cells' S-rows,
#                  sharded across both GPUs, per-cell bulk uploads BEFORE the
#                  fits phase starts (expensive-store-before-long-fit, #825)
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_REGEN_LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR" 2>/dev/null || { LOG_DIR="/tmp/issue2054_regen_logs"; mkdir -p "$LOG_DIR"; }
OUT_DIR="${ISSUE2054_REGEN_OUT_DIR:-data/issue_2054/common_regen}"
HF_PREFIX="${ISSUE2054_REGEN_HF_PREFIX:-issue2054_lattice/common_regen}"
PARENT_PREFIX="${ISSUE2054_REGEN_PARENT_PREFIX:-issue2054_lattice}"
TARGET_CONV_IDS="${ISSUE2054_REGEN_TARGET_CONV_IDS:-15700}"
FOLD_MAP="${ISSUE2054_REGEN_FOLD_MAP:-eval_results/issue_2054/coordinated_common_set_regen/shared_fold_map_extended.json}"
CHARS="char_helios,char_wren,char_dana,char_vex"
ASSIST="conversation_paired_stories_assistant"
MODELS=(qwen2.5-7b-instruct qwen2.5-7b)

echo "[phase=regen_r2r3] driver start prefix=${HF_PREFIX} target=${TARGET_CONV_IDS} $(date -u +%FT%TZ)"

# --- headroom preamble (plan §4 item 7; §9 mount binding: /workspace) ------
HEADROOM_GB="${ISSUE2054_REGEN_R2R3_HEADROOM_GB:-50}"
echo "[phase=regen_r2r3 stage=headroom] floor=${HEADROOM_GB}GB $(date -u +%FT%TZ)"
OUT_DIR="$OUT_DIR" HEADROOM_GB="$HEADROOM_GB" uv run python - <<'PYEOF'
import os

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

assert_out_root_headroom(
    os.environ["OUT_DIR"], float(os.environ["HEADROOM_GB"]), phase="regen_r2r3"
)
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[phase=regen_r2r3] HALT headroom rc=${rc}"
  exit "$rc"
fi

run_leg() {
  # run_leg <label> <log> <cmd...> — one leg, own log (the reserved
  # [phase=done] token from python children never reaches this main log).
  local label="$1" log="$2"
  shift 2
  echo "[phase=regen_r2r3 stage=${label}] start $(date -u +%FT%TZ)"
  "$@" > "$log" 2>&1
  local rc=$?
  echo "[phase=regen_r2r3 stage=${label}] rc=${rc} $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then
    echo "[phase=regen_r2r3] HALT ${label} rc=${rc} (tail follows)"
    tail -30 "$log" || true
    exit "$rc"
  fi
}

# --- stage inputs (cross-machine seam) --------------------------------------
run_leg stage_r2 "$LOG_DIR/issue-2054-r2r3-stage.log" \
  uv run python scripts/issue2054_regen_waves.py \
  --stage stage-r2 --output-dir "$OUT_DIR" --hf-prefix "$HF_PREFIX" \
  --parent-prefix "$PARENT_PREFIX" --fold-map-out "$FOLD_MAP"

# --- R2: phase_c (on-policy continuations; GPU, 2-way variant shard) --------
for MODEL in "${MODELS[@]}"; do
  for FORM in attrib_quoted bare_label; do
    run_leg "phase_c_${MODEL}_${FORM}" "$LOG_DIR/issue-2054-r2r3-pc-${MODEL}-${FORM}.log" \
      uv run python scripts/issue2054_shard_launch.py \
      --driver phase_c --form "$FORM" --model "$MODEL" --gpus 0,1 --variants "$CHARS" -- \
      --scaffolds-dir "$OUT_DIR/scaffolds/" \
      --output-dir "$OUT_DIR/on_policy/$MODEL/" \
      --target-conv-ids "$TARGET_CONV_IDS" \
      --hf-prefix "$HF_PREFIX"
  done

  echo "[phase=regen_r2r3 stage=phase_c_${MODEL}_assist] concurrent start $(date -u +%FT%TZ)"
  uv run python scripts/issue2054_shard_launch.py \
    --driver phase_c --form chat --model "$MODEL" --gpus 0 --variants "$ASSIST" -- \
    --scaffolds-dir "$OUT_DIR/assistant_delta/" \
    --output-dir "$OUT_DIR/on_policy/$MODEL/" \
    --target-conv-ids "$TARGET_CONV_IDS" \
    --hf-prefix "$HF_PREFIX" \
    > "$LOG_DIR/issue-2054-r2r3-pc-chat-${MODEL}.log" 2>&1 &
  P1=$!
  uv run python scripts/issue2054_shard_launch.py \
    --driver phase_c --form bare_text --model "$MODEL" --gpus 1 --variants "$ASSIST" -- \
    --scaffolds-dir "$OUT_DIR/assistant_delta/" \
    --output-dir "$OUT_DIR/on_policy/$MODEL/" \
    --target-conv-ids "$TARGET_CONV_IDS" \
    --hf-prefix "$HF_PREFIX" \
    > "$LOG_DIR/issue-2054-r2r3-pc-baretext-${MODEL}.log" 2>&1 &
  P2=$!
  wait "$P1"; R1=$?
  wait "$P2"; R2=$?
  echo "[phase=regen_r2r3 stage=phase_c_${MODEL}_assist] chat rc=${R1} bare_text rc=${R2} $(date -u +%FT%TZ)"
  if [ "$R1" -ne 0 ] || [ "$R2" -ne 0 ]; then
    echo "[phase=regen_r2r3] HALT ${MODEL} assistant delta chat rc=${R1} bare_text rc=${R2} (tails follow)"
    tail -30 "$LOG_DIR/issue-2054-r2r3-pc-chat-${MODEL}.log" || true
    tail -30 "$LOG_DIR/issue-2054-r2r3-pc-baretext-${MODEL}.log" || true
    exit 1
  fi
done

# --- assist-merge (full-S assistant on-policy coverage before capture) ------
run_leg assist_merge "$LOG_DIR/issue-2054-r2r3-assistmerge.log" \
  uv run python scripts/issue2054_regen_waves.py \
  --stage assist-merge --output-dir "$OUT_DIR" --hf-prefix "$HF_PREFIX" \
  --parent-prefix "$PARENT_PREFIX" --on-policy-dir "$OUT_DIR/on_policy"

# --- R2: phase_b (deterministic inserted splice; CPU) ------------------------
ANSWERS="$OUT_DIR/answers/answers_pool.jsonl"
for FORM in attrib_quoted bare_label; do
  run_leg "phase_b_${FORM}" "$LOG_DIR/issue-2054-r2r3-pb-${FORM}.log" \
    uv run python scripts/issue2054_phase_b.py \
    --form "$FORM" --answers-source "$ANSWERS" \
    --scaffolds-dir "$OUT_DIR/scaffolds/" \
    --output-dir "$OUT_DIR/spliced_inserted/" \
    --variants "$CHARS" \
    --hf-prefix "$HF_PREFIX"
done
for FORM in chat bare_text; do
  run_leg "phase_b_${FORM}" "$LOG_DIR/issue-2054-r2r3-pb-${FORM}.log" \
    uv run python scripts/issue2054_phase_b.py \
    --form "$FORM" --answers-source "$ANSWERS" \
    --scaffolds-dir "$OUT_DIR/scaffolds/" \
    --output-dir "$OUT_DIR/spliced_inserted/" \
    --variants "$ASSIST" \
    --hf-prefix "$HF_PREFIX"
done

# --- R2: phase_d (cell (c) splice from the ROUND's on-policy pools; CPU) -----
run_leg phase_d "$LOG_DIR/issue-2054-r2r3-pd.log" \
  uv run python scripts/issue2054_phase_d.py \
  --scaffolds-dir "$OUT_DIR/scaffolds/" \
  --output-dir "$OUT_DIR/cell_c/" \
  --target-conv-ids "$TARGET_CONV_IDS" \
  --fold-map "$FOLD_MAP" \
  --form chat \
  --hf-prefix "$HF_PREFIX"

# --- lengths (off-pod persistence BEFORE capture — gate-2 input) -------------
run_leg lengths "$LOG_DIR/issue-2054-r2r3-lengths.log" \
  uv run python scripts/issue2054_answer_lengths.py \
  --mode lengths --output-dir "$OUT_DIR" \
  --phase-b-dir "$OUT_DIR/spliced_inserted" \
  --phase-c-dir "$OUT_DIR/on_policy" \
  --phase-d-dir "$OUT_DIR/cell_c" \
  --hf-prefix "$HF_PREFIX"

# --- pre-capture coverage assert (Must-Fix; abort-before-R3 on mismatch) -----
DELTA_N=$(uv run python -c "import json;print(json.load(open('$OUT_DIR/scaffolds/export_manifest.json'))['assistant_delta'])")
run_leg coverage "$LOG_DIR/issue-2054-r2r3-coverage.log" \
  uv run python scripts/issue2054_gate1_intersections.py \
  --mode coverage \
  --survivors "$OUT_DIR/scaffolds/survivor_set.json" \
  --phase-b-dir "$OUT_DIR/spliced_inserted" \
  --phase-c-dir "$OUT_DIR/on_policy" \
  --phase-d-dir "$OUT_DIR/cell_c" \
  --assistant-delta-n "$DELTA_N"

# --- gate 2 (in-flight, pre-capture; report+mitigate — never blocks) ---------
run_leg gate2 "$LOG_DIR/issue-2054-r2r3-gate2.log" \
  uv run python scripts/issue2054_answer_lengths.py \
  --mode gate2 --output-dir "$OUT_DIR"

# --- R3: capture (all 48 cells' S-rows; sharded by (variant, model)) ---------
for MODEL in "${MODELS[@]}"; do
  # inserted (source model-independent) + on_policy story forms: char shard.
  for COND in inserted on_policy; do
    if [ "$COND" = "inserted" ]; then IN_DIR="$OUT_DIR/spliced_inserted/"; else IN_DIR="$OUT_DIR/on_policy/$MODEL/"; fi
    for FORM in attrib_quoted bare_label; do
      run_leg "capture_${MODEL}_${COND}_${FORM}" \
        "$LOG_DIR/issue-2054-r2r3-cap-${MODEL}-${COND}-${FORM}.log" \
        uv run python scripts/issue2054_shard_launch.py \
        --driver capture --condition "$COND" --form "$FORM" --model "$MODEL" \
        --gpus 0,1 --variants "$CHARS" -- \
        --input-dir "$IN_DIR" \
        --output-dir "$OUT_DIR/activations/" \
        --hf-prefix "$HF_PREFIX"
    done
    echo "[phase=regen_r2r3 stage=capture_${MODEL}_${COND}_assist] concurrent start $(date -u +%FT%TZ)"
    uv run python scripts/issue2054_shard_launch.py \
      --driver capture --condition "$COND" --form chat --model "$MODEL" \
      --gpus 0 --variants "$ASSIST" -- \
      --input-dir "$IN_DIR" \
      --output-dir "$OUT_DIR/activations/" \
      --hf-prefix "$HF_PREFIX" \
      > "$LOG_DIR/issue-2054-r2r3-cap-chat-${COND}-${MODEL}.log" 2>&1 &
    P1=$!
    uv run python scripts/issue2054_shard_launch.py \
      --driver capture --condition "$COND" --form bare_text --model "$MODEL" \
      --gpus 1 --variants "$ASSIST" -- \
      --input-dir "$IN_DIR" \
      --output-dir "$OUT_DIR/activations/" \
      --hf-prefix "$HF_PREFIX" \
      > "$LOG_DIR/issue-2054-r2r3-cap-baretext-${COND}-${MODEL}.log" 2>&1 &
    P2=$!
    wait "$P1"; R1=$?
    wait "$P2"; R2=$?
    echo "[phase=regen_r2r3 stage=capture_${MODEL}_${COND}_assist] chat rc=${R1} bare_text rc=${R2} $(date -u +%FT%TZ)"
    if [ "$R1" -ne 0 ] || [ "$R2" -ne 0 ]; then
      echo "[phase=regen_r2r3] HALT capture ${MODEL}/${COND} assistant chat rc=${R1} bare_text rc=${R2} (tails follow)"
      tail -30 "$LOG_DIR/issue-2054-r2r3-cap-chat-${COND}-${MODEL}.log" || true
      tail -30 "$LOG_DIR/issue-2054-r2r3-cap-baretext-${COND}-${MODEL}.log" || true
      exit 1
    fi
  done

  # cell (c): the model-matched _op / _op_base variants (2 per char per model).
  if [ "$MODEL" = "qwen2.5-7b-instruct" ]; then
    CC_VARIANTS="char_helios_op,char_wren_op,char_dana_op,char_vex_op"
  else
    CC_VARIANTS="char_helios_op_base,char_wren_op_base,char_dana_op_base,char_vex_op_base"
  fi
  run_leg "capture_${MODEL}_cell_c" "$LOG_DIR/issue-2054-r2r3-cap-${MODEL}-cellc.log" \
    uv run python scripts/issue2054_shard_launch.py \
    --driver capture --condition cell_c --form chat --model "$MODEL" \
    --gpus 0,1 --variants "$CC_VARIANTS" -- \
    --input-dir "$OUT_DIR/cell_c/" \
    --output-dir "$OUT_DIR/activations/" \
    --hf-prefix "$HF_PREFIX"
done

echo "[phase=regen_r2r3] driver_rc=0 $(date -u +%FT%TZ)"
