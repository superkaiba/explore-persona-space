#!/usr/bin/env bash
# Inserted-arm round driver (pod-1336-insertarm): NO new GPU capture — the
# absence sweep found all 20 Phase EXT_off off-policy pair trees banked on
# the Hub. Phase H harvests layer-30 "inserted" clouds from them (download +
# slice + per-cell upload/reap, sharded across parallel processes); Phase L
# runs the two decomposition arrows' rigid-to-affine ladder batteries,
# sharded one pair per GPU, consuming the live t5d round's diagonal cloud
# exports (missing diagonals -> pending JSONs, retried on a bounded loop).
#
# PILOT GATE: the FIRST battery (base:sft, gsm8k_test1319, reencode) runs
# alone; its wall is recorded to $PILOTFILE and echoed with a x140
# projection. Nothing self-aborts on it (#1092) — the operator reads it.
#
# Usage: bash scripts/issue1336_insertarm_driver.sh
set -uo pipefail

REPO=/workspace/explore-persona-space
BRANCH=${EPM_1336_BRANCH:-issue-1336-insertarm}
FIX_SHA="${EPM_1336_INSERTARM_FIX_SHA:?set EPM_1336_INSERTARM_FIX_SHA to the round's commit}"

OUT=${EPM_1336_OUT:-/workspace/eval_results/issue_1336/arrow_ladders}
STAGE=${EPM_1336_STAGE:-/workspace/data/issue_1336/insertarm_stage}
CLOUDS=${EPM_1336_CLOUDS:-/workspace/data/issue_1336/insertarm_clouds}
LOGDIR=${EPM_1336_LOGDIR:-/workspace/logs}
LAYER=${EPM_1336_LAYER:-30}
DEP_WAIT_S=${EPM_1336_ARROW_DEP_WAIT_S:-10800}   # bounded retry for t5d cloud deps

PAIRS_ALL=${EPM_1336_PAIRS:-base:sft,base:dpo,base:rlvr,base:rlvr_long,sft:dpo,sft:rlvr,sft:rlvr_long,dpo:rlvr,dpo:rlvr_long,rlvr:rlvr_long}

cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }
mkdir -p "$LOGDIR" "$OUT" "$STAGE" "$CLOUDS"
PIDFILE="$LOGDIR/issue-1336-insertarm.pid"
SENTINEL="$LOGDIR/issue-1336-insertarm-done.json"
PILOTFILE="$LOGDIR/issue-1336-insertarm-pilot-wall-s"
echo $$ > "$PIDFILE"
rm -f "$SENTINEL"

write_sentinel() {
  local rc=$1 ncomplete=$2 npending=$3
  cat > "$SENTINEL.tmp" <<JSON
{"issue": 1336, "round": "insertarm", "rc": $rc,
 "batteries_complete": $ncomplete, "batteries_pending": $npending,
 "out_root": "$OUT", "clouds_prefix": "issue1336_rlvr_ladder/analysis_tensors/layer30_clouds/inserted",
 "mirror_prefix": "issue1336_rlvr_ladder/eval_results_mirror_insertarm/arrow_ladders",
 "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
  mv "$SENTINEL.tmp" "$SENTINEL"
}

set -a
[ -f ./.env ] && . ./.env
set +a
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

# SLURM_GPU_WIDTH_EXEMPT: RunPod-pod-only launcher (provisioned via pod.py; never dispatched on a SLURM lane)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "${NGPU:-0}" -ge 1 ] || { echo "FATAL: no GPUs visible" >&2; write_sentinel 4 0 0; exit 4; }
NCPU=$(nproc)
echo "[driver] gpus=$NGPU cpus=$NCPU"

echo "[driver] fetching $BRANCH"
git fetch origin "$BRANCH" --depth=50 --quiet || { echo "FATAL: fetch failed" >&2; write_sentinel 2 0 0; exit 2; }
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH"
git reset --hard -q "origin/$BRANCH"
if git merge-base --is-ancestor "$FIX_SHA" HEAD; then
  echo "[driver] FIX-OK $FIX_SHA ancestor of $(git rev-parse --short HEAD)"
else
  echo "FATAL: FIX ABSENT: $FIX_SHA not ancestor of $(git rev-parse HEAD)" >&2
  write_sentinel 3 0 0; exit 3
fi
echo "[driver] head=$(git rev-parse HEAD)"
echo "[driver] disk at start:"; df -h /workspace | tail -1

# Warm the venv ONCE, then freeze resolution for the fan-outs (#1689 FUSE rule).
uv run python -c "import torch; print('[driver] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { write_sentinel 5 0 0; exit 5; }
export UV_NO_SYNC=1

count_complete() { grep -ls '"status": "complete"' "$OUT"/arrow_*.json 2>/dev/null | wc -l; }
count_pending()  { grep -ls '"status": "pending_dependency"' "$OUT"/arrow_*.json 2>/dev/null | wc -l; }

# ---- Phase H: harvest inserted clouds, sharded by pair (network-bound). ----
HPAR=${EPM_1336_HARVEST_PAR:-4}
echo "[phaseH] START $(date -u +%FT%TZ) par=$HPAR pairs=$PAIRS_ALL"
h0=$(date +%s)
pairs=(${PAIRS_ALL//,/ })
declare -a hpids=()
for ((w = 0; w < HPAR; w++)); do
  subset=""
  for ((i = w; i < ${#pairs[@]}; i += HPAR)); do subset="$subset,${pairs[i]}"; done
  subset="${subset#,}"
  [ -n "$subset" ] || continue
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  uv run python scripts/issue1336_insertarm_clouds.py \
    --pairs "$subset" --stage-root "$STAGE/w$w" --local-out "$CLOUDS/inserted" \
    --layer "$LAYER" > "$LOGDIR/issue-1336-insertarm-harvest-w$w.log" 2>&1 &
  hpids+=($!)
  echo "[phaseH] worker $w pid=${hpids[-1]} pairs=$subset"
done
hrc=0
for p in "${hpids[@]}"; do wait "$p" || hrc=$?; done
echo "[phaseH] rc=$hrc elapsed=$(( $(date +%s) - h0 ))s"
if [ "$hrc" -ne 0 ]; then
  echo "FATAL: harvest worker failed rc=$hrc (completed cells are durable on the Hub)" >&2
  for f in "$LOGDIR"/issue-1336-insertarm-harvest-w*.log; do
    echo "--- tail $f"; tail -5 "$f"
  done
  write_sentinel "$hrc" "$(count_complete)" "$(count_pending)"
  exit "$hrc"
fi
echo "[phaseH] disk after:"; df -h /workspace | tail -1

# ---- Phase L pilot: ONE battery, wall recorded; no self-abort (#1092). -----
echo "[pilot] START $(date -u +%FT%TZ)"
p0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
uv run python scripts/issue1336_arrow_ladders.py \
  --pair base:sft --arrows reencode --corpora gsm8k_test1319 \
  --clouds-root "$CLOUDS/pool" --out-root "$OUT" --layer "$LAYER" \
  > "$LOGDIR/issue-1336-insertarm-pilot.log" 2>&1
prc=$?
pwall=$(( $(date +%s) - p0 ))
echo "$pwall" > "$PILOTFILE"
echo "[pilot] rc=$prc wall=${pwall}s — x140 naive projection: $(( pwall * 140 / 60 )) min serial, $(( pwall * 140 / 60 / NGPU )) min at $NGPU-way (test1319 is the SMALLEST n; real projection scales with n)"
if [ "$prc" -ne 0 ]; then
  echo "FATAL: pilot battery failed rc=$prc" >&2
  tail -20 "$LOGDIR/issue-1336-insertarm-pilot.log"
  write_sentinel "$prc" "$(count_complete)" "$(count_pending)"
  exit "$prc"
fi

# ---- Phase L: pairs sharded one-per-GPU in waves. --------------------------
run_wave_pass() {
  local pass_label=$1
  local idx=0
  while [ "$idx" -lt "${#pairs[@]}" ]; do
    declare -a wpids=() wpairs=()
    local slot=0
    while [ "$slot" -lt "$NGPU" ] && [ "$idx" -lt "${#pairs[@]}" ]; do
      local pair="${pairs[idx]}"
      local omp=$(( NCPU / NGPU )); [ "$omp" -lt 2 ] && omp=2; [ "$omp" -gt 16 ] && omp=16
      CUDA_VISIBLE_DEVICES=$slot OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp OPENBLAS_NUM_THREADS=$omp NUMEXPR_NUM_THREADS=$omp \
      uv run python scripts/issue1336_arrow_ladders.py \
        --pair "$pair" --clouds-root "$CLOUDS/pool" --out-root "$OUT" --layer "$LAYER" \
        >> "$LOGDIR/issue-1336-insertarm-arrows-${pair/:/__}.log" 2>&1 &
      wpids+=($!); wpairs+=("$pair")
      idx=$((idx + 1)); slot=$((slot + 1))
    done
    local i=0
    for p in "${wpids[@]}"; do
      if wait "$p"; then
        echo "[phaseL:$pass_label] pair ${wpairs[i]} OK"
      else
        echo "[phaseL:$pass_label] pair ${wpairs[i]} FAILED rc=$? (see log)" >&2
        rc_all=1
      fi
      i=$((i + 1))
    done
  done
}

rc_all=0
l0=$(date +%s)
echo "[phaseL] START $(date -u +%FT%TZ) $NGPU-way"
run_wave_pass first
echo "[phaseL] first pass done elapsed=$(( $(date +%s) - l0 ))s complete=$(count_complete) pending=$(count_pending)"

# ---- Dependency retry loop: t5d diagonal clouds land progressively. --------
dep0=$(date +%s)
while [ "$(count_pending)" -gt 0 ] && [ $(( $(date +%s) - dep0 )) -lt "$DEP_WAIT_S" ]; do
  echo "[phaseL:retry] $(count_pending) pending; sleeping 600s (waited $(( $(date +%s) - dep0 ))s of $DEP_WAIT_S)"
  sleep 600
  # remove pending stubs so the batteries re-run (complete ones are skipped)
  grep -ls '"status": "pending_dependency"' "$OUT"/arrow_*.json 2>/dev/null | xargs -r rm -f
  run_wave_pass retry
done

NC=$(count_complete); NP=$(count_pending)
echo "[phaseL] DONE complete=$NC pending=$NP rc_all=$rc_all elapsed=$(( $(date +%s) - l0 ))s"

# ---- Results mirror upload (battery JSONs; clouds already uploaded per cell).
uv run python scripts/issue1336_arrow_ladders.py --out-root "$OUT" --upload-results \
  > "$LOGDIR/issue-1336-insertarm-mirror.log" 2>&1
mrc=$?
echo "[mirror] rc=$mrc"
[ "$mrc" -ne 0 ] && { tail -5 "$LOGDIR/issue-1336-insertarm-mirror.log"; rc_all=1; }

write_sentinel "$rc_all" "$NC" "$NP"
echo "[driver] EXIT rc=$rc_all"
exit "$rc_all"
