#!/usr/bin/env bash
# Execute the frozen pilot only, with persisted boundaries and a fresh exit receipt.
set -euo pipefail
set +x
JR_REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$JR_REPO"
JR_OUT=${1:?usage: workspace_jr_component_pilot.sh OUTPUT_DIR NATIVE_PILOT_DIR ROLE}
JR_NATIVE=${2:?native pilot directory required}
JR_ROLE=${3:?primary or comparison required}
JR_PHASE=starting
mkdir -p "$JR_OUT"
JR_EXIT="$JR_OUT/pilot_exit.json"
if [[ -e "$JR_EXIT" ]]; then
  echo "Refusing preexisting terminal receipt: $JR_EXIT" >&2
  exit 2
fi
trap 'JR_RC=$?; printf "{\"exit_code\":%d,\"phase\":\"%s\",\"finished_at_epoch\":%d}\n" "$JR_RC" "$JR_PHASE" "$(date +%s)" > "$JR_EXIT"' EXIT
export PYTHONPATH="$PWD/src"
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2
JR_UV=/root/.local/bin/uv
JR_PREFIX="exploratory_workspace_jr/20260912/$(basename "$JR_OUT")"
git rev-parse HEAD > "$JR_OUT/code_sha.txt"

jr_phase() {
  JR_PHASE=$1
  shift
  echo "[phase=$JR_PHASE]"
  "$JR_UV" run --project runtime/workspace_jr --frozen python -c \
    'import shutil; free=shutil.disk_usage("/workspace").free/2**30; assert free>80, f"GPU disk safety floor: {free:.1f} GiB"'
  "$JR_UV" run --project runtime/workspace_jr --frozen scripts/workspace_jr_pipeline.py \
    "$@" --role "$JR_ROLE" --out "$JR_OUT"
}

jr_persist() {
  JR_PHASE="persist_$1"
  "$JR_UV" run --project runtime/workspace_jr --frozen scripts/workspace_jr_persist.py \
    --root "$JR_OUT" --prefix "$JR_PREFIX" --receipt "${JR_OUT}_persist_$1.json"
}

jr_phase dictionaries dictionaries --lens-shards "$JR_NATIVE/lens_shards" \
  --token-manifest "$JR_NATIVE/calibration_tokens.json" --validation "$JR_NATIVE/native_validation.json"
for JR_SPLIT in train validation test; do
  jr_phase "generate_$JR_SPLIT" generate --subset "pilot_$JR_SPLIT"
done
jr_persist generations
for JR_SPLIT in train validation test; do
  jr_phase "capture_$JR_SPLIT" capture --subset "pilot_$JR_SPLIT"
done
jr_persist captures
for JR_SPLIT in train validation test; do
  jr_phase "decompose_$JR_SPLIT" decompose --subset "pilot_$JR_SPLIT" --k 10
done
jr_persist components
jr_phase fit fit --stage pilot --k 10
printf '{"status":"workload_complete","finished_at_epoch":%d}\n' "$(date +%s)" > "$JR_OUT/workload_complete.json"
jr_persist fits
JR_PHASE=complete
# The owning monitor checks process exit plus the fresh EXIT receipt, then
# persists that receipt before releasing any disk. Workload completion above
# is deliberately a separate milestone from verified upload/process completion.
echo 'Frozen component pilot finished; main calibration stability and model comparison remain.'
