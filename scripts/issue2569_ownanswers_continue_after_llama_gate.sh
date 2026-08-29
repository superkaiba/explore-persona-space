#!/usr/bin/env bash
# Resume #2569 after the designed Llama identity-gate halt.  Preconditions:
# Qwen's Llama-writer capture is finalized/uploaded; Llama batch-1 identity and
# timing pilot gates are PASS in the shared capture root.
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
CAPTURE_PREFIX=issue2569_theory/own_generated_answers/captures/llama_writer_s42
RESULT_PREFIX=issue2569_theory/own_generated_answers/analysis
ROWS=10500

mkdir -p "$LOG_ROOT"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false

phase() { printf '[phase=%s]\n' "$1"; }
run_logged() {
  local name="$1" rc=0
  shift
  echo "[continuation] START $name: $*"
  "$@" > "$LOG_ROOT/$name.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[continuation] FAILED $name rc=$rc — tail follows" >&2
    tail -n 160 "$LOG_ROOT/$name.log" >&2 || true
    exit "$rc"
  fi
  tail -n 12 "$LOG_ROOT/$name.log" || true
  echo "[continuation] DONE $name"
}

test -s "$LWRITER_ROOT/final/qwen_va_L14.pt"
uv run python -c 'import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for name in ("identity_gate_llama.json", "pilot_gate_llama.json"):
    obj = json.loads((root / "gates" / name).read_text())
    assert obj["verdict"] == "PASS", (name, obj.get("verdict"))
assert json.loads((root / "gates" / "identity_gate_llama.json").read_text())["regime"]["max_batch_rows"] == 1
assert json.loads((root / "gates" / "pilot_gate_llama.json").read_text())["capture_params"]["max_batch_rows"] == 1
print("[continuation] Llama batch-1 correctness/timing gates verified")' "$LWRITER_ROOT"

phase capture_llama
run_logged capture-llama \
  uv run python scripts/issue2569_xmodel_capture.py \
    --phase capture --model llama --out-root "$LWRITER_ROOT" --rows "$ROWS" \
    --max-batch-rows 1 --skip-upload

phase finalize_llama
run_logged finalize-llama \
  uv run python scripts/issue2569_xmodel_capture.py \
    --phase finalize --model llama --out-root "$LWRITER_ROOT" --rows "$ROWS" \
    --max-batch-rows 1 --hf-data-repo "$DATA_REPO" --hf-prefix "$CAPTURE_PREFIX"

phase stage_crossed_bundles
run_logged stage-crossed \
  uv run python scripts/issue2569_ownanswers_analyze.py \
    --phase stage --qwriter-dir "$QWRITER_FINAL" \
    --lwriter-dir "$LWRITER_ROOT/final" --hf-data-repo "$DATA_REPO" \
    --lwriter-prefix "$CAPTURE_PREFIX"

phase semantic_divergence
run_logged semantic \
  uv run python scripts/issue2569_ownanswers_analyze.py \
    --phase semantic --analysis-rows 10000 --source-root "$SOURCE_ROOT" \
    --llama-answers "$GEN_ROOT/answers.jsonl" --out-dir "$ANALYSIS_ROOT"

phase crossed_geometry
run_logged analyze \
  uv run python scripts/issue2569_ownanswers_analyze.py \
    --phase analyze --analysis-rows 10000 --n-train 8000 --n-val 500 \
    --n-test 1500 --null-draws 200 --qwriter-dir "$QWRITER_FINAL" \
    --lwriter-dir "$LWRITER_ROOT/final" --source-root "$SOURCE_ROOT" \
    --semantic-rows "$ANALYSIS_ROOT/semantic/per_row.jsonl" \
    --out-dir "$ANALYSIS_ROOT" --upload --hf-data-repo "$DATA_REPO" \
    --lwriter-prefix "$CAPTURE_PREFIX" --result-prefix "$RESULT_PREFIX"

test -s "$ANALYSIS_ROOT/crossed_geometry.json"
echo "[phase=done] issue=2569 continuation=llama-batch1 result=$ANALYSIS_ROOT/crossed_geometry.json"
