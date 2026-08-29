#!/usr/bin/env bash
# Issue #2569 follow-up: crossed geometry with each model's generated answers.
#
# This is a foreground, resumable pod workload.  It stages 10,500 candidates
# (500 rows of headroom for empty/over-budget generations), generates Llama's
# missing writer arm with the parent's sampling recipe, captures that same text
# through Qwen and Llama, and fits the registered 10,000-row crossed analysis.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK_ROOT="${EPM_I2569_OWN_ROOT:-/workspace/issue2569-ownanswers}"
LOG_ROOT="${EPM_I2569_LOG_ROOT:-/workspace/logs/issue2569-ownanswers}"
SOURCE_ROOT="$WORK_ROOT/source_qwen"
GEN_ROOT="$WORK_ROOT/gen_llama_s42"
LWRITER_ROOT="$WORK_ROOT/writer_llama"
QWRITER_FINAL="$WORK_ROOT/qwriter_final"
ANALYSIS_ROOT="$WORK_ROOT/analysis"

DATA_REPO=superkaiba1/explore-persona-space-data
RAW_PREFIX=issue2569_theory/own_generated_answers/raw_completions/llama_seed42
CAPTURE_PREFIX=issue2569_theory/own_generated_answers/captures/llama_writer_s42
RESULT_PREFIX=issue2569_theory/own_generated_answers/analysis
CANDIDATE_ROWS=10500
ANALYSIS_ROWS=10000

mkdir -p "$WORK_ROOT" "$LOG_ROOT"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

phase() { printf '[phase=%s]\n' "$1"; }

run_logged() {
  local name="$1" rc=0
  shift
  echo "[workload] START $name: $*"
  "$@" > "$LOG_ROOT/$name.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[workload] FAILED $name rc=$rc — tail follows" >&2
    tail -n 160 "$LOG_ROOT/$name.log" >&2 || true
    exit "$rc"
  fi
  tail -n 12 "$LOG_ROOT/$name.log" || true
  echo "[workload] DONE $name"
}

capture_writer() {
  local model="$1"
  local -a gate_args=()
  if [ "$model" = qwen ]; then
    gate_args+=(--qwen-gate identity)
  fi

  phase "identity_gate_${model}"
  run_logged "identity-gate-$model" \
    uv run python scripts/issue2569_xmodel_capture.py \
      --phase identity-gate --model "$model" --out-root "$LWRITER_ROOT" \
      --rows "$CANDIDATE_ROWS" "${gate_args[@]}"

  phase "capture_pilot_${model}"
  run_logged "capture-pilot-$model" \
    uv run python scripts/issue2569_xmodel_capture.py \
      --phase capture --model "$model" --out-root "$LWRITER_ROOT" \
      --rows 32 --skip-upload "${gate_args[@]}"

  phase "capture_${model}"
  run_logged "capture-$model" \
    uv run python scripts/issue2569_xmodel_capture.py \
      --phase capture --model "$model" --out-root "$LWRITER_ROOT" \
      --rows "$CANDIDATE_ROWS" --skip-upload "${gate_args[@]}"

  phase "finalize_${model}"
  run_logged "finalize-$model" \
    uv run python scripts/issue2569_xmodel_capture.py \
      --phase finalize --model "$model" --out-root "$LWRITER_ROOT" \
      --rows "$CANDIDATE_ROWS" --hf-data-repo "$DATA_REPO" \
      --hf-prefix "$CAPTURE_PREFIX" "${gate_args[@]}"
}

phase preflight
nvidia-smi
df -h "$WORK_ROOT"
uv run python scripts/issue2569_xmodel_capture.py --import-check
uv run python scripts/issue2569_ownanswers_generate.py --import-check
uv run python scripts/issue2569_ownanswers_analyze.py --import-check

phase select
run_logged select \
  uv run python scripts/issue2569_xmodel_capture.py \
    --phase select --rows "$CANDIDATE_ROWS" --out-root "$SOURCE_ROOT" --device cpu

phase generate_llama_seed42
run_logged generate-llama-s42 \
  uv run python scripts/issue2569_ownanswers_generate.py \
    --phase generate --model llama --seed 42 --rows "$CANDIDATE_ROWS" \
    --source-root "$SOURCE_ROOT" --out-root "$GEN_ROOT" --capture-root "$LWRITER_ROOT" \
    --upload --hf-data-repo "$DATA_REPO" --hf-prefix "$RAW_PREFIX"

phase prepare_llama_writer
run_logged prepare-llama-s42 \
  uv run python scripts/issue2569_ownanswers_generate.py \
    --phase prepare --model llama --seed 42 --rows "$CANDIDATE_ROWS" \
    --source-root "$SOURCE_ROOT" --out-root "$GEN_ROOT" --capture-root "$LWRITER_ROOT"

capture_writer qwen
capture_writer llama

phase stage_crossed_bundles
run_logged stage-crossed \
  uv run python scripts/issue2569_ownanswers_analyze.py \
    --phase stage --qwriter-dir "$QWRITER_FINAL" \
    --lwriter-dir "$LWRITER_ROOT/final" --hf-data-repo "$DATA_REPO" \
    --lwriter-prefix "$CAPTURE_PREFIX"

phase semantic_divergence
run_logged semantic \
  uv run python scripts/issue2569_ownanswers_analyze.py \
    --phase semantic --analysis-rows "$ANALYSIS_ROWS" --source-root "$SOURCE_ROOT" \
    --llama-answers "$GEN_ROOT/answers.jsonl" --out-dir "$ANALYSIS_ROOT"

phase crossed_geometry
run_logged analyze \
  uv run python scripts/issue2569_ownanswers_analyze.py \
    --phase analyze --analysis-rows "$ANALYSIS_ROWS" --n-train 8000 --n-val 500 \
    --n-test 1500 --null-draws 200 --qwriter-dir "$QWRITER_FINAL" \
    --lwriter-dir "$LWRITER_ROOT/final" --source-root "$SOURCE_ROOT" \
    --semantic-rows "$ANALYSIS_ROOT/semantic/per_row.jsonl" \
    --out-dir "$ANALYSIS_ROOT" --upload --hf-data-repo "$DATA_REPO" \
    --lwriter-prefix "$CAPTURE_PREFIX" --result-prefix "$RESULT_PREFIX"

test -s "$ANALYSIS_ROOT/crossed_geometry.json"
test -s "$ANALYSIS_ROOT/split.json"
phase done
echo "[phase=done] issue=2569 label=cross-model-own-generated-answers result=$ANALYSIS_ROOT/crossed_geometry.json"
