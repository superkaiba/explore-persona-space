#!/usr/bin/env bash
# Round-B driver: per-SURFACE stage -> fit -> reap for the #1336 v3 refit.
#
# WHY THIS EXISTS. issue1336_selfmap_missing_pairs.py's stage_inputs() stages every
# turnstore the passed --cells need UP FRONT. Over all 32 cells that is 44 turnstore
# dirs / 317.76 GB (measured 2026-08-07 via scoped list_repo_tree at each dir's own
# revision) — past the 240 GB RunPod CPU container-disk cap, so a single whole-run
# invocation cannot stage. Driving the SAME script one ladder surface at a time keeps
# peak local footprint at one surface (max 69.69 GB, the two lmsys23k concat surfaces)
# and reaps between surfaces. No change to the fitting code: --cells + --stage already
# express this.
#
# Surfaces are ordered CHEAPEST-FIRST so (a) the measured per-cell wall lands early and
# (b) at least one surface's cells are durable before the big pulls begin. Each surface
# is 4 cells: (base,base) self + the 3 forward pairs sft->rlvr / sft->rlvr_long /
# rlvr->rlvr_long.
#
# The reap clears BOTH turnstore_v2/ and turnstore_wave1/ staged contents — the wave-1
# dirs are the concat sources (lmsys23k <- lmsys5k, gsm8k_train_full <- gsm8k_train5k)
# and are half the bytes on those surfaces. The small wave-1 gen answers under gen/ are
# KEPT (KB-MB, reused across surfaces). Per-cell outputs under $OUT/cells/ are NEVER
# reaped — they are this round's durable product, and the script's own resume predicate
# reads them.
#
# Usage:  bash scripts/issue1336_refit_surfaces.sh [FIRST] [LAST]     (1-indexed, inclusive)
#         bash scripts/issue1336_refit_surfaces.sh 1 1     # cheapest surface only (the basis run)
#         bash scripts/issue1336_refit_surfaces.sh 2 8     # the rest
set -uo pipefail

REPO=/workspace/explore-persona-space
OUT=${EPM_1336_OUT:-/workspace/eval_results/issue_1336/selfmap_v3}
STAGE=${EPM_1336_STAGE:-/workspace/data/issue_1336}
LAYER=${EPM_1336_LAYER:-30}

# label|approx_staged_GB|comma-separated cell keys (source__target__format__corpus)
SURFACES=(
"chat/gsm8k_test1319|6|base__base__chat__gsm8k_test1319,sft__rlvr__chat__gsm8k_test1319,sft__rlvr_long__chat__gsm8k_test1319,rlvr__rlvr_long__chat__gsm8k_test1319"
"chat/math7500|32|base__base__chat__math7500,sft__rlvr__chat__math7500,sft__rlvr_long__chat__math7500,rlvr__rlvr_long__chat__math7500"
"chat/gsm8k_train_full|32|base__base__chat__gsm8k_train_full,sft__rlvr__chat__gsm8k_train_full,sft__rlvr_long__chat__gsm8k_train_full,rlvr__rlvr_long__chat__gsm8k_train_full"
"chat/sft11k|36|base__base__chat__sft11k,sft__rlvr__chat__sft11k,sft__rlvr_long__chat__sft11k,rlvr__rlvr_long__chat__sft11k"
"chat/if11k|37|base__base__chat__if11k,sft__rlvr__chat__if11k,sft__rlvr_long__chat__if11k,rlvr__rlvr_long__chat__if11k"
"chat/uf11k|39|base__base__chat__uf11k,sft__rlvr__chat__uf11k,sft__rlvr_long__chat__uf11k,rlvr__rlvr_long__chat__uf11k"
"chat/lmsys23k|70|base__base__chat__lmsys23k,sft__rlvr__chat__lmsys23k,sft__rlvr_long__chat__lmsys23k,rlvr__rlvr_long__chat__lmsys23k"
"naturalistic/lmsys23k|70|base__base__naturalistic__lmsys23k,sft__rlvr__naturalistic__lmsys23k,sft__rlvr_long__naturalistic__lmsys23k,rlvr__rlvr_long__naturalistic__lmsys23k"
)

FIRST=${1:-1}
LAST=${2:-${#SURFACES[@]}}

cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }

# Re-attach breadcrumbs. The pidfile is rewritten by THIS run (never left carrying a
# predecessor's pid — the #813 relaunch trap), and the exit-code sentinel is removed at
# launch so a stale one can never satisfy a done-check (the never-key-done-on-bare-
# existence rule). A successor session re-attaches from these two paths alone.
LOGDIR=${EPM_1336_LOGDIR:-/workspace/logs}
PIDFILE="$LOGDIR/issue-1336-refit.pid"
SENTINEL="$LOGDIR/issue-1336-refit-done.json"
mkdir -p "$LOGDIR"
echo $$ > "$PIDFILE"
rm -f "$SENTINEL"

write_sentinel() {
  cat > "$SENTINEL.tmp" <<JSON
{"issue": 1336, "round": "refit-v3", "rc": $1, "surfaces": "$FIRST..$LAST",
 "cells_on_disk": $2, "out_root": "$OUT", "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
  mv "$SENTINEL.tmp" "$SENTINEL"
}

set -a
[ -f ./.env ] && . ./.env
set +a
# 16 uncontended vCPU on this pod — not the shared-VM cap of 8.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export MALLOC_ARENA_MAX=2

mkdir -p "$OUT" "$STAGE"
echo "[driver] repo=$REPO head=$(git rev-parse HEAD)"
echo "[driver] out=$OUT stage=$STAGE layer=$LAYER surfaces=$FIRST..$LAST"

rc_all=0
for ((i = FIRST; i <= LAST; i++)); do
  row="${SURFACES[i-1]}"
  label="${row%%|*}"
  rest="${row#*|}"
  need_gb="${rest%%|*}"
  cells="${rest#*|}"

  # Fail-fast disk assert: refuse to start a surface that cannot fit with margin,
  # instead of discovering ENOSPC 60 GB into a pull. 1.3x the MEASURED staged bytes.
  avail_gb=$(df -BG --output=avail "$STAGE" | tail -1 | tr -dc '0-9')
  need_margin=$(( need_gb * 13 / 10 ))
  echo "[driver] === surface $i/${#SURFACES[@]} $label === need=${need_gb}GB margin=${need_margin}GB avail=${avail_gb}GB"
  if [ "$avail_gb" -lt "$need_margin" ]; then
    echo "[driver] FATAL surface $label: avail ${avail_gb}GB < ${need_margin}GB (1.3x measured ${need_gb}GB)" >&2
    rc_all=2
    break
  fi

  t0=$(date +%s)
  uv run python scripts/issue1336_selfmap_missing_pairs.py \
    --out-root "$OUT" --stage-root "$STAGE" --layer "$LAYER" \
    --stage --cells "$cells"
  rc=$?
  t1=$(date +%s)
  echo "[driver] surface $label rc=$rc elapsed=$(( t1 - t0 ))s"
  if [ "$rc" -ne 0 ]; then
    echo "[driver] FATAL surface $label failed rc=$rc — stopping (per-cell outputs so far are durable)" >&2
    rc_all="$rc"
    break
  fi

  # Reap this surface's staged turnstores (re-downloadable Hub copies under
  # --stage-root; NOT a generated store/ or eval_results/). gen/ is kept.
  before_gb=$(du -sBG "$STAGE" 2>/dev/null | cut -f1 | tr -dc '0-9')
  rm -rf "$STAGE/turnstore_v2" "$STAGE/turnstore_wave1" "$STAGE/selfmap_stage_tmp"
  after_gb=$(du -sBG "$STAGE" 2>/dev/null | cut -f1 | tr -dc '0-9')
  echo "[driver] reaped staged turnstores for $label: ${before_gb:-?}GB -> ${after_gb:-?}GB"
done

n_cells=$(ls -1 "$OUT/cells" 2>/dev/null | wc -l)
echo "[driver] DONE rc=$rc_all cells_on_disk=$n_cells"
write_sentinel "$rc_all" "$n_cells"
exit "$rc_all"
