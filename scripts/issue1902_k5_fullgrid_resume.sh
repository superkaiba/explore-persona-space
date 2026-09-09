#!/usr/bin/env bash
# Resume the approved #1902 full-grid CPU analysis after verified GPU teardown.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072
# NumPy huge-page faults blocked in virtio_balloon compaction on this VM.
# Keep this process-local; it changes allocation requests, not the estimator.
export NUMPY_MADVISE_HUGEPAGE=0
export UV_CACHE_DIR=/tmp/eps1902-uv-cache
export UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv
# 64 draw files x 60 s retry exposure + 2 x 8.6 GB / 5 MB/s < 3 hours.
export EPM_HF_RETRY_BUDGET_S=60
OUT="$PWD/eval_results/issue_1902/k5_full_grid"
DATA=/mnt/eps-data/thomasjiralerspong/issue1902_k5grid_20260909
mkdir -p "$OUT"
printf '%s\n' "$$" > "$OUT/run.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/run.exit"; exit "$rc"' EXIT
if sudo -n choom -n -600 -p "$$"; then
  echo '[resume] choom=ok'
else
  echo '[resume] choom=failed; RSS expected below 16 GiB, per-checkpoint saves retained'
fi
ARGS=(--full-grid --draw-revision f0b2131442326ef274c91bea6da27e05ef844df6
      --target-layers 31 --out "$OUT" --k5-root "$DATA/store"
      --flag-counts-root /mnt/eps-data/thomasjiralerspong/wt-1902-k5/eval_results/issue_1902/k5_targets/targets
      --reuse-root /mnt/eps-data/thomasjiralerspong/wt-1902-k5/data/issue_1902/k5_store)
for phase in stage targets grid transfer retrieval scatter figure; do
  echo "[resume] phase=$phase started=$(date -u +%FT%TZ)"
  if [ "$phase" = stage ]; then
    timeout --kill-after=30s 10800s uv run --no-sync python scripts/issue1902_k5_fits.py "$phase" "${ARGS[@]}"
  else
    uv run --no-sync python scripts/issue1902_k5_fits.py "$phase" "${ARGS[@]}"
  fi
  printf '0\n' > "$OUT/$phase.exit"
done
echo '[resume] full-grid analysis complete'
