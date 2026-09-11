#!/usr/bin/env bash
# Layer-18 companion to the #1902 K=5 full-grid CPU analysis.
#
# Same 16 checkpoint x answer-source cells, same estimator, folds and targets as
# the committed layer-31 run (eval_results/issue_1902/k5_full_grid); the only
# change is --fit-layer 18. Every layer-18 draw shard already exists on the Hub
# at the pinned capture revisions, so this is a download + ridge-fit job with no
# GPU phase and no new generation.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
# NumPy huge-page faults blocked in virtio_balloon compaction on this VM (#1902).
export NUMPY_MADVISE_HUGEPAGE=0
export UV_CACHE_DIR=/tmp/eps1902-uv-cache
export UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv
export EPM_HF_RETRY_BUDGET_S=60
OUT="$PWD/eval_results/issue_1902/k5_layer18_grid"
DATA=/mnt/eps-data/thomasjiralerspong/issue1902_l18grid
mkdir -p "$OUT" "$DATA/store"
printf '%s\n' "$$" > "$OUT/run.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/run.exit"; exit "$rc"' EXIT
if sudo -n choom -n -600 -p "$$"; then
  echo '[l18grid] choom=ok'
else
  echo '[l18grid] choom=failed; RSS expected below 16 GiB, per-checkpoint saves retained'
fi
ARGS=(--full-grid --draw-revision f0b2131442326ef274c91bea6da27e05ef844df6
      --target-layers 18 --fit-layer 18
      --out "$OUT" --k5-root "$DATA/store"
      --figures-dir "$PWD/figures/issue_1902/section43_l18"
      --flag-counts-root /mnt/eps-data/thomasjiralerspong/wt-1902-k5/eval_results/issue_1902/k5_targets/targets
      --reuse-root /mnt/eps-data/thomasjiralerspong/issue1902_l18
      --reuse-root /mnt/eps-data/thomasjiralerspong/wt-1902-k5/data/issue_1902/k5_store)
for phase in stage targets grid transfer retrieval scatter figure; do
  echo "[l18grid] phase=$phase started=$(date -u +%FT%TZ)"
  if [ "$phase" = stage ]; then
    timeout --kill-after=30s 10800s uv run --no-sync python scripts/issue1902_k5_fits.py "$phase" "${ARGS[@]}"
  else
    uv run --no-sync python scripts/issue1902_k5_fits.py "$phase" "${ARGS[@]}"
  fi
  printf '0\n' > "$OUT/$phase.exit"
done
echo '[l18grid] layer-18 full-grid analysis complete'
