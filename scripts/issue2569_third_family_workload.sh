#!/usr/bin/env bash
# Issue #2569 inline follow-up: complete the Qwen/Llama/OLMo writer-by-encoder
# bank and run the two new pairwise analyses. Foreground, fail-loud, resumable.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORK_ROOT="${EPM_I2569_THIRD_ROOT:-/workspace/issue2569-third-family}"
LOG_ROOT="${EPM_I2569_THIRD_LOG_ROOT:-/workspace/logs/issue2569-third-family}"
SOURCE_ROOT="$WORK_ROOT/source_qwen"
CANDIDATE_SOURCE_ROOT="$WORK_ROOT/source_candidate"
QWRITER_OLMO_ROOT="$WORK_ROOT/capture/qwriter_olmo"
LWRITER_ROOT="$WORK_ROOT/bank/lwriter"
OWRITER_GEN="$WORK_ROOT/capture/gen_olmo_s42"
OWRITER_TOPUP="$WORK_ROOT/capture/gen_olmo_s42_topup"
OWRITER_MERGED="$WORK_ROOT/capture/gen_olmo_s42_merged"
OWRITER_ROOT="$WORK_ROOT/capture/owriter"
DATA_REPO=superkaiba1/explore-persona-space-data
RESULT_PREFIX=issue2569_theory/third_family
CANDIDATE_ROWS=10500
TOPUP_MAX_TOKENS=4096
TOPUP_MAX_MODEL_LEN=12288
TOPUP_SEED=1667586269
ANALYSIS_ROWS=10000
SAME_TEXT_ROWS=60000
TRANSFORMERS_PIN=5.15.0
VLLM_PIN=0.27.1

mkdir -p "$WORK_ROOT" "$LOG_ROOT"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

phase() { printf '[third-family-phase=%s]\n' "$1"; }

run_logged() {
  local name="$1" rc=0
  shift
  echo "[third-family] START $name: $*"
  "$@" > "$LOG_ROOT/$name.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[third-family] FAILED $name rc=$rc — tail follows" >&2
    tail -n 180 "$LOG_ROOT/$name.log" >&2 || true
    exit "$rc"
  fi
  tail -n 16 "$LOG_ROOT/$name.log" || true
  echo "[third-family] DONE $name"
}

phase repository_preflight
# The frozen selector reuses #2476, whose checked-in split contract lives
# outside bootstrap's default issue-2569 sparse cone.
git sparse-checkout add eval_results/issue_1482
test -x .venv/bin/python || uv sync
uv run python -m explore_persona_space.orchestrate.preflight
nvidia-smi
df -h /workspace "$WORK_ROOT"

# The repo lock intentionally pins transformers<5 for its older vLLM line.
# This isolated experiment pins the newer mutually-resolved pair in-place and
# disables uv's automatic re-sync for every subsequent command.
phase stack_resolution
uv pip install --dry-run --python .venv/bin/python \
  "transformers==$TRANSFORMERS_PIN" "vllm==$VLLM_PIN"
uv pip install --python .venv/bin/python \
  "transformers==$TRANSFORMERS_PIN" "vllm==$VLLM_PIN"
# vLLM 0.27.1 hard-pins flashinfer-python 0.6.16.post3, whose runtime
# array.array[int] annotation is incompatible with this image's Python 3.11.
# Remove it after every resolver pass; TP=1 uses neither its all-reduce fusion
# nor its sampler (disabled authoritatively above).
uv pip uninstall --python .venv/bin/python flashinfer-python
export UV_NO_SYNC=1
PY=(uv run --no-sync python)

phase import_checks
run_logged import-xmodel "${PY[@]}" scripts/issue2569_xmodel_capture.py --import-check
run_logged import-generate "${PY[@]}" scripts/issue2569_ownanswers_generate.py --import-check
run_logged import-own-analysis "${PY[@]}" scripts/issue2569_ownanswers_analyze.py --import-check
run_logged import-atlas "${PY[@]}" scripts/issue2569_atlas.py --import-check
run_logged import-third "${PY[@]}" scripts/issue2569_third_family.py --import-check
run_logged selftest-third "${PY[@]}" scripts/issue2569_third_family.py \
  --phase selftest --work-root "$WORK_ROOT"
run_logged selftest-mapping-diff "${PY[@]}" scripts/issue2569_mapping_diff.py --phase selftest
run_logged selftest-query "${PY[@]}" scripts/issue2569_query_scaling_unpaired.py --phase selftest
run_logged olmo-preflight "${PY[@]}" scripts/issue2569_third_family.py \
  --phase preflight --work-root "$WORK_ROOT"

phase select_frozen_source
run_logged select-source "${PY[@]}" scripts/issue2569_xmodel_capture.py \
  --phase select --rows "$SAME_TEXT_ROWS" --out-root "$SOURCE_ROOT" --device cpu

phase stage_pinned_inputs
run_logged stage-existing "${PY[@]}" scripts/issue2569_third_family.py \
  --phase stage-existing --work-root "$WORK_ROOT" --hf-data-repo "$DATA_REPO"
run_logged prepare-llama-s42 "${PY[@]}" scripts/issue2569_ownanswers_generate.py \
  --phase prepare --model llama --seed 42 --rows "$CANDIDATE_ROWS" \
  --source-root "$CANDIDATE_SOURCE_ROOT" --out-root "$WORK_ROOT/bank/gen_llama_s42" \
  --capture-root "$LWRITER_ROOT"

phase generate_olmo_seed42
run_logged generate-olmo-s42 "${PY[@]}" scripts/issue2569_ownanswers_generate.py \
  --phase generate --model olmo --seed 42 --rows "$CANDIDATE_ROWS" \
  --source-root "$CANDIDATE_SOURCE_ROOT" --out-root "$OWRITER_GEN" \
  --capture-root "$OWRITER_ROOT" \
  --upload --hf-data-repo "$DATA_REPO" \
  --hf-prefix "$RESULT_PREFIX/raw_completions/olmo_seed42"
run_logged olmo-topup-roster "${PY[@]}" scripts/issue2569_third_family.py \
  --phase topup-roster --work-root "$WORK_ROOT"
run_logged generate-olmo-s42-topup "${PY[@]}" scripts/issue2569_ownanswers_generate.py \
  --phase generate --model olmo --seed "$TOPUP_SEED" --rows 0 \
  --max-new-tokens "$TOPUP_MAX_TOKENS" --max-model-len "$TOPUP_MAX_MODEL_LEN" \
  --source-root "$CANDIDATE_SOURCE_ROOT" --ci-roster "$OWRITER_TOPUP/roster.json" \
  --out-root "$OWRITER_TOPUP" --capture-root "$OWRITER_ROOT" \
  --upload --hf-data-repo "$DATA_REPO" \
  --hf-prefix "$RESULT_PREFIX/raw_completions/olmo_seed42_cap4096"
run_logged merge-olmo-s42-topup "${PY[@]}" scripts/issue2569_third_family.py \
  --phase merge-topup --work-root "$WORK_ROOT"
run_logged prepare-olmo-s42 "${PY[@]}" scripts/issue2569_ownanswers_generate.py \
  --phase prepare --model olmo --seed 42 --rows "$CANDIDATE_ROWS" \
  --source-root "$CANDIDATE_SOURCE_ROOT" --out-root "$OWRITER_MERGED" \
  --capture-root "$OWRITER_ROOT"

mkdir -p "$QWRITER_OLMO_ROOT"
ln -f "$SOURCE_ROOT/texts_kept.jsonl" "$QWRITER_OLMO_ROOT/texts_kept.jsonl"

capture_model() {
  local model="$1" out_root="$2" rows="$3" prefix="$4"
  local -a gate_args=()
  if [ "$model" = qwen ]; then
    # Alternate-writer text has no banked v_A oracle.  Use the independent
    # identity gate and batch-1 for both the gate-certified and production
    # capture paths; Qwen's padded-batch bf16 tail crossed the registered 2%
    # agreement bar on one of 48 fixed comparisons (0.02059 at v_C L26).
    gate_args+=(--qwen-gate identity --max-batch-rows 1)
  elif [ "$model" = llama ] || [ "$model" = olmo ]; then
    # Batch-1 makes the independent identity recompute exact and is the
    # production packing shape certified by the timing pilot.
    gate_args+=(--max-batch-rows 1)
  fi

  phase "identity_gate_${model}"
  run_logged "identity-$(basename "$out_root")-$model" \
    "${PY[@]}" scripts/issue2569_xmodel_capture.py \
      --phase identity-gate --model "$model" --rows "$rows" --out-root "$out_root" \
      "${gate_args[@]}"

  phase "capture_pilot_${model}"
  run_logged "pilot-$(basename "$out_root")-$model" \
    "${PY[@]}" scripts/issue2569_xmodel_capture.py \
      --phase capture --model "$model" --rows 32 --out-root "$out_root" --skip-upload \
      --pilot-model-count 1 --pilot-booked-wall-h 3.0 "${gate_args[@]}"

  phase "capture_${model}"
  run_logged "capture-$(basename "$out_root")-$model" \
    "${PY[@]}" scripts/issue2569_xmodel_capture.py \
      --phase capture --model "$model" --rows "$rows" --out-root "$out_root" --skip-upload \
      --pilot-model-count 1 --pilot-booked-wall-h 3.0 "${gate_args[@]}"

  phase "finalize_${model}"
  run_logged "finalize-$(basename "$out_root")-$model" \
    "${PY[@]}" scripts/issue2569_xmodel_capture.py \
      --phase finalize --model "$model" --rows "$rows" --out-root "$out_root" \
      --pilot-model-count 1 --pilot-booked-wall-h 3.0 \
      --hf-data-repo "$DATA_REPO" --hf-prefix "$prefix" "${gate_args[@]}"
}

# Missing cell 1: OLMo encodes the frozen 60k Qwen-written text.  The first
# 10k rows also complete the Qwen-writer row of the crossed bank.
capture_model olmo "$QWRITER_OLMO_ROOT" "$SAME_TEXT_ROWS" \
  "$RESULT_PREFIX/captures/qwen_writer_olmo"

# Missing cell 2: OLMo encodes the already-pinned Llama-written text.
capture_model olmo "$LWRITER_ROOT" "$CANDIDATE_ROWS" \
  "$RESULT_PREFIX/captures/llama_writer_olmo"

# Missing cells 3--5: all encoders process the same OLMo-written responses.
capture_model qwen "$OWRITER_ROOT" "$CANDIDATE_ROWS" \
  "$RESULT_PREFIX/captures/olmo_writer_all_encoders"
capture_model llama "$OWRITER_ROOT" "$CANDIDATE_ROWS" \
  "$RESULT_PREFIX/captures/olmo_writer_all_encoders"
capture_model olmo "$OWRITER_ROOT" "$CANDIDATE_ROWS" \
  "$RESULT_PREFIX/captures/olmo_writer_all_encoders"

phase validate_and_materialize
run_logged validate-bank "${PY[@]}" scripts/issue2569_third_family.py \
  --phase validate-bank --work-root "$WORK_ROOT" --analysis-rows "$ANALYSIS_ROWS"
run_logged build-pairs "${PY[@]}" scripts/issue2569_third_family.py \
  --phase build-pairs --work-root "$WORK_ROOT" --analysis-rows "$ANALYSIS_ROWS"

run_atlas_pair() {
  local pair="$1"
  local pair_root="$WORK_ROOT/atlas/$pair"
  phase "same_text_atlas_${pair}"
  run_logged "atlas-fits-$pair" "${PY[@]}" scripts/issue2569_atlas.py \
    --phase fits --capture-dir "$pair_root/captures" --fits-dir "$pair_root/fits" \
    --leg7-dir "$pair_root/report" --pair-manifest "$pair_root/pair_manifest.json" \
    --center-operator matched-source --device cuda --null-draws 200 --skip-upload
  run_logged "atlas-report-$pair" "${PY[@]}" scripts/issue2569_atlas.py \
    --phase report --capture-dir "$pair_root/captures" --fits-dir "$pair_root/fits" \
    --leg7-dir "$pair_root/report" --pair-manifest "$pair_root/pair_manifest.json" \
    --center-operator matched-source --device cuda --null-draws 200 --skip-upload
}

run_crossed_pair() {
  local pair="$1" source_root="$2"
  local pair_root="$WORK_ROOT/pairs/$pair"
  local analysis="$pair_root/analysis"
  phase "crossed_${pair}"
  run_logged "semantic-$pair" "${PY[@]}" scripts/issue2569_ownanswers_analyze.py \
    --phase semantic --analysis-rows "$ANALYSIS_ROWS" --source-root "$source_root" \
    --llama-answers "$OWRITER_MERGED/answers.jsonl" --out-dir "$analysis" --device cuda
  run_logged "crossed-geometry-$pair" "${PY[@]}" scripts/issue2569_ownanswers_analyze.py \
    --phase analyze --analysis-rows "$ANALYSIS_ROWS" --n-train 8000 --n-val 500 \
    --n-test 1500 --null-draws 200 --qwriter-dir "$pair_root/source_writer" \
    --lwriter-dir "$pair_root/olmo_writer" --source-root "$source_root" \
    --semantic-rows "$analysis/semantic/per_row.jsonl" --out-dir "$analysis" \
    --device cuda --upload --hf-data-repo "$DATA_REPO" \
    --result-prefix "$RESULT_PREFIX/pairs/$pair/crossed"
  run_logged "mapping-diff-$pair" "${PY[@]}" scripts/issue2569_mapping_diff.py \
    --phase analyze --qwriter-dir "$pair_root/source_writer" \
    --lwriter-dir "$pair_root/olmo_writer" --map-dir "$analysis/maps" \
    --split-json "$analysis/split.json" --semantic-rows "$analysis/semantic/per_row.jsonl" \
    --qseed137-dir '' --lseed137-dir '' --qseed137-raw '' --lseed137-raw '' \
    --reliability-semantic-rows '' --out-dir "$analysis/mapping_diff" --device cuda \
    --permutation-draws 1000 --bootstrap-draws 2000 --top-modes 16
  run_logged "query-scaling-$pair" "${PY[@]}" scripts/issue2569_query_scaling_unpaired.py \
    --phase analyze --qwriter-dir "$pair_root/source_writer" \
    --lwriter-dir "$pair_root/olmo_writer" --map-dir "$analysis/maps" \
    --split-json "$analysis/split.json" --out-dir "$analysis/query_scaling_unpaired" \
    --device cuda --paired-k-values 512 1024 2048 4000 \
    --unpaired-k-values 64 128 256 512 1024 2048 4000
}

run_atlas_pair qo
run_atlas_pair lo
run_crossed_pair qo "$CANDIDATE_SOURCE_ROOT"
run_crossed_pair lo "$LWRITER_ROOT"

phase assemble_and_upload
run_logged summarize "${PY[@]}" scripts/issue2569_third_family.py \
  --phase summarize --work-root "$WORK_ROOT"
run_logged upload-results "${PY[@]}" scripts/issue2569_third_family.py \
  --phase upload-results --work-root "$WORK_ROOT" --hf-data-repo "$DATA_REPO" \
  --result-prefix "$RESULT_PREFIX"

phase terminal_verification
run_logged sentinel "${PY[@]}" scripts/issue2569_third_family.py \
  --phase sentinel --work-root "$WORK_ROOT" --hf-data-repo "$DATA_REPO" \
  --result-prefix "$RESULT_PREFIX" \
  --sentinel-path /workspace/logs/issue-2569-third-family-done.json
echo "[third-family-phase=done] issue=2569 result=$WORK_ROOT/results/third_family_summary.json"
