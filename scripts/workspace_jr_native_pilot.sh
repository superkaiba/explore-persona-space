#!/usr/bin/env bash
# Native checkpoint validation and one paired calibration matrix, no main-test reads.
set -euo pipefail
set +x
JR_REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$JR_REPO"
export PYTHONPATH="$JR_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
JR_OUT=${1:?usage: workspace_jr_native_pilot.sh OUTPUT_DIR [primary|comparison]}
JR_ROLE=${2:-primary}
mkdir -p "$JR_OUT"
JR_EXIT="$JR_OUT/native_pilot_exit.json"
if [[ -e "$JR_EXIT" ]]; then
  echo "Refusing an existing terminal sentinel: $JR_EXIT" >&2
  exit 2
fi
trap 'JR_RC=$?; printf "{\"exit_code\":%d,\"finished_at_epoch\":%d}\n" "$JR_RC" "$(date +%s)" > "$JR_EXIT"' EXIT
echo '[phase=runtime]'
uv sync --project runtime/workspace_jr --frozen
uv pip freeze --python runtime/workspace_jr/.venv/bin/python > "$JR_OUT/packages.txt"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv > "$JR_OUT/hardware.csv"
echo '[phase=calibration_manifest]'
uv run --project runtime/workspace_jr --frozen scripts/workspace_jr_runtime.py \
  --role "$JR_ROLE" calibration-manifest --out "$JR_OUT/calibration_tokens.json"
echo '[phase=native_validation]'
uv run --project runtime/workspace_jr --frozen scripts/workspace_jr_runtime.py \
  --role "$JR_ROLE" native-forward-validation \
  --token-manifest "$JR_OUT/calibration_tokens.json" --out "$JR_OUT/native_validation.json"
echo '[phase=one_prompt_lens_pilot]'
uv run --project runtime/workspace_jr --frozen scripts/workspace_jr_runtime.py \
  --role "$JR_ROLE" fit-lens-shard --token-manifest "$JR_OUT/calibration_tokens.json" \
  --validation "$JR_OUT/native_validation.json" \
  --out-dir "$JR_OUT/lens_shards" --start 0 --stop 1 --dim-batch 8
echo 'Native pilot completed; calibration stability and component experiment remain separate gates.'
