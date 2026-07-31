#!/usr/bin/env bash
# Issue #1345 — run the two authorized judge legs across all 20 cell-legs.
#
#   ai_likeness    16 character cells: {dana,helios,vex,wren} x {instruct,base}
#                  x {injected,onpolicy}. Both provenances run so the labelling
#                  axis is a PAIRED arm, not an on-policy-only read.
#   content_drift   4 on-policy answer cells against the injected twin the
#                  matching store was built from: the parent track-S corpus rows
#                  for the comparator-driven cells, the V1 kept stories for the
#                  story-slot cell.
#
# n=300 rows/cell, k=5 draws, stratified on `capped`, Batch-routed => 30,000
# calls. Spend requires BOTH --execute and EPM_I1345_JUDGE_SPEND_OK=1.
#
#   EPM_I1345_JUDGE_SPEND_OK=1 bash scripts/issue1345_judge_legs_run.sh --execute
#   bash scripts/issue1345_judge_legs_run.sh                 # dry-run, no spend
#   EPM_I1345_JUDGE_CELLS="char_wren_op" bash ... --execute   # one cell
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

PREP_DIR="${EPM_I1345_PREP_DIR:-data/issue_1345/judge_prep}"
OUT_DIR="${EPM_I1345_JUDGE_OUT:-eval_results/issue_1345/judge_legs}"
SAMPLE_N="${EPM_I1345_SAMPLE_N:-300}"
SAMPLE_SEED="${EPM_I1345_SAMPLE_SEED:-1345}"
EXTRA=("$@")

# Shared-VM thread caps: this driver is API-bound, but its children import torch
# through the project env and would otherwise spin up one pool per cell.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2
export EPM_I1345_VARIANT="${EPM_I1345_VARIANT:-story_boundary_ablation}"
export EPM_STORY_CHARACTER_NAME="${EPM_STORY_CHARACTER_NAME:-Assistant}"

# cell:leg:reference   (reference empty => ai_likeness, no twin needed)
CELLS=()
for ch in dana helios vex wren; do
  for suffix in "" "_base" "_op" "_op_base"; do
    CELLS+=("char_${ch}${suffix}:ai_likeness:")
  done
done
CELLS+=("op_ntpl_instruct:content_drift:track_s_injected")
CELLS+=("op_ntpl_base:content_drift:track_s_injected")
CELLS+=("op_chat_base:content_drift:track_s_injected")
CELLS+=("op_slot_base:content_drift:v1_injected")

WANT="${EPM_I1345_JUDGE_CELLS:-}"

rc_any=0
n_run=0
for entry in "${CELLS[@]}"; do
  cell="${entry%%:*}"
  rest="${entry#*:}"
  leg="${rest%%:*}"
  ref="${rest#*:}"

  if [ -n "$WANT" ] && [[ " $WANT " != *" $cell "* ]]; then continue; fi

  rows="$PREP_DIR/$cell.jsonl"
  if [ ! -s "$rows" ]; then
    echo "[run] MISSING prepared rows for $cell ($rows) — run issue1345_judge_rows_prep.py first" >&2
    rc_any=3
    continue
  fi
  args=(--leg "$leg" --rows "$rows" --character "$cell"
        --out-dir "$OUT_DIR/$cell" --sample-n "$SAMPLE_N" --sample-seed "$SAMPLE_SEED")
  if [ -n "$ref" ]; then
    refrows="$PREP_DIR/$ref.jsonl"
    if [ ! -s "$refrows" ]; then
      echo "[run] MISSING reference rows for $cell ($refrows)" >&2
      rc_any=3
      continue
    fi
    args+=(--reference-rows "$refrows")
  fi

  echo "=== [$((n_run + 1))/${#CELLS[@]}] $cell / $leg ${ref:+(ref $ref)} $(date -u +%H:%M:%SZ)"
  uv run python scripts/issue1345_onpolicy_judge_legs.py "${args[@]}" "${EXTRA[@]}"
  rc=$?
  n_run=$((n_run + 1))
  if [ "$rc" -ne 0 ]; then
    # Keep going: one cell's failure must not strand the other 19, and every
    # completed cell has already persisted its own design + report.
    echo "[run] cell $cell rc=$rc — continuing with the remaining cells" >&2
    rc_any=$rc
  fi
done

echo "[run] done: $n_run cell-legs attempted, worst rc=$rc_any"
exit "$rc_any"
