#!/usr/bin/env bash
# Issue #2569 own-answer follow-up: non-gating seed-137 reliability companion.
# Run only after issue2569_ownanswers_workload.sh has produced the frozen split/maps.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK_ROOT="${EPM_I2569_OWN_ROOT:-/workspace/issue2569-ownanswers}"
LOG_ROOT="${EPM_I2569_REL_LOG_ROOT:-/workspace/logs/issue2569-ownanswers-reliability}"
SOURCE_ROOT="$WORK_ROOT/source_qwen"
ANALYSIS_ROOT="$WORK_ROOT/analysis"
SPLIT_JSON="$ANALYSIS_ROOT/split.json"
QWRITER_FINAL="$WORK_ROOT/qwriter_final"
LWRITER_FINAL="$WORK_ROOT/writer_llama/final"
DATA_REPO=superkaiba1/explore-persona-space-data
RESULT_PREFIX=issue2569_theory/own_generated_answers/analysis
ROWS=1500

mkdir -p "$LOG_ROOT"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false

phase() { printf '[reliability-phase=%s]\n' "$1"; }
run_logged() {
  local name="$1" rc=0
  shift
  echo "[reliability] START $name: $*"
  "$@" > "$LOG_ROOT/$name.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[reliability] FAILED $name rc=$rc — tail follows" >&2
    tail -n 160 "$LOG_ROOT/$name.log" >&2 || true
    exit "$rc"
  fi
  tail -n 12 "$LOG_ROOT/$name.log" || true
  echo "[reliability] DONE $name"
}

run_writer() {
  local model="$1" layer="$2"
  local gen="$WORK_ROOT/reliability/gen_${model}_s137"
  local cap="$WORK_ROOT/reliability/${model}_seed137"
  local raw_prefix="issue2569_theory/own_generated_answers/reliability/raw/${model}_seed137"
  local cap_prefix="issue2569_theory/own_generated_answers/reliability/captures/${model}_seed137"
  local -a gate_args=()
  if [ "$model" = qwen ]; then gate_args+=(--qwen-gate identity); fi

  phase "generate_${model}_seed137"
  run_logged "generate-${model}-s137" \
    uv run python scripts/issue2569_ownanswers_generate.py \
      --phase generate --model "$model" --seed 137 \
      --ci-roster "$SPLIT_JSON" --ci-roster-key test_ci \
      --source-root "$SOURCE_ROOT" --out-root "$gen" --capture-root "$cap" \
      --upload --hf-data-repo "$DATA_REPO" --hf-prefix "$raw_prefix"

  phase "prepare_${model}_seed137"
  run_logged "prepare-${model}-s137" \
    uv run python scripts/issue2569_ownanswers_generate.py \
      --phase prepare --model "$model" --seed 137 \
      --ci-roster "$SPLIT_JSON" --ci-roster-key test_ci \
      --source-root "$SOURCE_ROOT" --out-root "$gen" --capture-root "$cap"

  phase "identity_gate_${model}_seed137"
  run_logged "identity-gate-${model}-s137" \
    uv run python scripts/issue2569_xmodel_capture.py \
      --phase identity-gate --model "$model" --layers "$layer" \
      --rows "$ROWS" --out-root "$cap" "${gate_args[@]}"

  phase "capture_pilot_${model}_seed137"
  run_logged "capture-pilot-${model}-s137" \
    uv run python scripts/issue2569_xmodel_capture.py \
      --phase capture --model "$model" --layers "$layer" --rows 32 \
      --out-root "$cap" --skip-upload "${gate_args[@]}"

  phase "capture_${model}_seed137"
  run_logged "capture-${model}-s137" \
    uv run python scripts/issue2569_xmodel_capture.py \
      --phase capture --model "$model" --layers "$layer" --rows "$ROWS" \
      --out-root "$cap" --skip-upload "${gate_args[@]}"

  phase "finalize_${model}_seed137"
  run_logged "finalize-${model}-s137" \
    uv run python scripts/issue2569_xmodel_capture.py \
      --phase finalize --model "$model" --layers "$layer" --rows "$ROWS" \
      --out-root "$cap" --hf-data-repo "$DATA_REPO" --hf-prefix "$cap_prefix" \
      "${gate_args[@]}"
}

test -s "$SPLIT_JSON"
test -s "$ANALYSIS_ROOT/crossed_geometry.json"
test -s "$ANALYSIS_ROOT/maps/q14_l16_align_a_own_q2l.pt"

run_writer qwen 14
run_writer llama 16

phase reliability_analysis
run_logged reliability-analysis \
  uv run python scripts/issue2569_ownanswers_analyze.py \
    --phase reliability --n-test "$ROWS" --qwriter-dir "$QWRITER_FINAL" \
    --lwriter-dir "$LWRITER_FINAL" --source-root "$SOURCE_ROOT" \
    --llama-answers "$WORK_ROOT/gen_llama_s42/answers.jsonl" \
    --split-json "$SPLIT_JSON" --out-dir "$ANALYSIS_ROOT" \
    --qseed137-dir "$WORK_ROOT/reliability/qwen_seed137/final" \
    --lseed137-dir "$WORK_ROOT/reliability/llama_seed137/final" \
    --qseed137-answers "$WORK_ROOT/reliability/gen_qwen_s137/answers.jsonl" \
    --lseed137-answers "$WORK_ROOT/reliability/gen_llama_s137/answers.jsonl" \
    --reliability-out "$ANALYSIS_ROOT/reliability.json" \
    --upload --hf-data-repo "$DATA_REPO" --result-prefix "$RESULT_PREFIX"

test -s "$ANALYSIS_ROOT/reliability.json"
echo "[reliability-phase=done] issue=2569 n=$ROWS result=$ANALYSIS_ROOT/reliability.json"
