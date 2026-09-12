#!/usr/bin/env bash
# Resume the scientific pilot in a fresh tree with verified context-only inputs.
set -euo pipefail
set +x
JR_OUT=${1:?fresh output directory required}
JR_SOURCE=${2:?failed pilot output required}
JR_RECEIPT=${3:?verified failed pilot upload receipt required}
JR_EXPECTED=${4:?exact reviewed code commit required}
[[ $(git rev-parse HEAD) == "$JR_EXPECTED" ]]
[[ -z $(git status --porcelain --untracked-files=no) ]]
[[ ! -e "$JR_OUT" ]]
JR_PHASE=recover_context_inputs
JR_EXIT="$JR_OUT/pilot_exit.json"
trap 'JR_RC=$?; mkdir -p "$JR_OUT"; printf "{\"exit_code\":%d,\"phase\":\"%s\",\"finished_at_epoch\":%d}\n" "$JR_RC" "$JR_PHASE" "$(date +%s)" > "$JR_EXIT"' EXIT
JR_PYTHON=/workspace/explore-persona-space/runtime/workspace_jr/.venv/bin/python
export PYTHONPATH="$PWD/src" HF_HOME=/workspace/.cache/huggingface
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2
JR_PREFIX="exploratory_workspace_jr/20260912/$(basename "$JR_OUT")"
"$JR_PYTHON" scripts/workspace_jr_pipeline.py recover-pilot --role primary --out "$JR_OUT" \
  --source "$JR_SOURCE" --source-receipt "$JR_RECEIPT"
git rev-parse HEAD > "$JR_OUT/code_sha.txt"
JR_PHASE=persist_recovered_captures
"$JR_PYTHON" scripts/workspace_jr_persist.py --root "$JR_OUT" --prefix "$JR_PREFIX" \
  --receipt "${JR_OUT}_persist_recovered_captures.json"
for JR_SPLIT in train validation test; do
  JR_PHASE="decompose_$JR_SPLIT"
  echo "[phase=$JR_PHASE]"
  "$JR_PYTHON" scripts/workspace_jr_pipeline.py decompose --role primary --out "$JR_OUT" \
    --subset "pilot_$JR_SPLIT" --k 10
done
JR_PHASE=persist_components
"$JR_PYTHON" scripts/workspace_jr_persist.py --root "$JR_OUT" --prefix "$JR_PREFIX" \
  --receipt "${JR_OUT}_persist_components.json"
JR_PHASE=fit
"$JR_PYTHON" scripts/workspace_jr_pipeline.py fit --role primary --out "$JR_OUT" --stage pilot --k 10
printf '{"status":"workload_complete","finished_at_epoch":%d}\n' "$(date +%s)" > "$JR_OUT/workload_complete.json"
JR_PHASE=persist_fits
"$JR_PYTHON" scripts/workspace_jr_persist.py --root "$JR_OUT" --prefix "$JR_PREFIX" \
  --receipt "${JR_OUT}_persist_fits.json"
JR_PHASE=complete
# The owner verifies this unit's exit and uploads the fresh final EXIT receipt.
