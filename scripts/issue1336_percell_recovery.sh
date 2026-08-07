#!/usr/bin/env bash
# Round-B recovery: per-CELL invocations for #1336 surfaces 7-8 (chat/lmsys23k,
# naturalistic/lmsys23k), the two surfaces the per-SURFACE driver could not fit.
#
# WHY PER-CELL. The per-surface invocation loads all four cells' model arrays in one
# process and OOM-killed at the 128 GB container cap (rc=137, cgroup oom_kill=1).
# The MEASURED 1-cell pilot (sft__rlvr__chat__lmsys23k, production entrypoint at
# production shape) came in at rc=0 / 895 s / peak anon RSS 113.5 GiB of the
# 119.2 GiB cap -- so ONE cell fits, with ~5% headroom, and the partition is the fix.
#
# WIDTH 1 IS A MEMORY CONSTRAINT, NOT AN OVERSIGHT: a pair cell peaks at 113.5 GiB,
# so two concurrent cells cannot coexist under the cap. Serial is the only shape.
#
# Per-cell durability: each cell writes its own cells/<key>__l30.json and the
# entrypoint's resume predicate (source,target,fmt,corpus,layer) skips completed
# cells, so a mid-run OOM costs at most the one cell in flight and a re-run is safe.
#
# MALLOC_MMAP_THRESHOLD_ is set explicitly to DISABLE glibc's dynamic mmap-threshold
# adjustment. Peak is retained memory ACROSS the two sequential model loads (each load
# already slices to (n,d) and frees the bundle), which is the signature of glibc raising
# the threshold after the first large free and serving the second load off the heap
# instead of mmap. Pinning the threshold keeps large frees returning to the OS.
# Numerically inert -- pure allocator behavior -- and can only lower peak RSS. The
# per-cell peak is logged so the effect is measurable rather than assumed.
set -uo pipefail

REPO=/workspace/explore-persona-space
OUT=${EPM_1336_OUT:-/workspace/eval_results/issue_1336/selfmap_v3}
STAGE=${EPM_1336_STAGE:-/workspace/data/issue_1336}
LAYER=${EPM_1336_LAYER:-30}
LOGDIR=${EPM_1336_LOGDIR:-/workspace/logs}
PIDFILE="$LOGDIR/issue-1336-percell.pid"
SENTINEL="$LOGDIR/issue-1336-percell-done.json"
PER_CELL_FENCE=${EPM_1336_CELL_FENCE:-5400}   # >=2x the measured 895s fit + staging headroom

# Surface 7 stage is already on disk (the crashed surface left it: the reap runs only
# after a SUCCESSFUL surface). Surface 8 needs its own ~70 GB, so surface 7's staged
# turnstores are reaped in between -- 35 GB free now, 66 GB staged.
S7_CELLS=(
  base__base__chat__lmsys23k
  sft__rlvr_long__chat__lmsys23k
  rlvr__rlvr_long__chat__lmsys23k
)
S8_CELLS=(
  base__base__naturalistic__lmsys23k
  sft__rlvr__naturalistic__lmsys23k
  sft__rlvr_long__naturalistic__lmsys23k
  rlvr__rlvr_long__naturalistic__lmsys23k
)
S8_NEED_GB=70

cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }

mkdir -p "$LOGDIR"
echo $$ > "$PIDFILE"            # rewritten by THIS run, never a predecessor's pid
rm -f "$SENTINEL"               # so bare existence can never satisfy a done-check

set -a
[ -f ./.env ] && . ./.env
set +a
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072

CGLIM=/sys/fs/cgroup/memory/memory.limit_in_bytes
STAT=/sys/fs/cgroup/memory/memory.stat
lim=$(cat "$CGLIM" 2>/dev/null || echo unknown)
echo "[recover] start pid=$$ layer=$LAYER cap_bytes=$lim fence=${PER_CELL_FENCE}s"
echo "[recover] cells: ${#S7_CELLS[@]} on chat/lmsys23k + ${#S8_CELLS[@]} on naturalistic/lmsys23k"

results=""
n_ok=0; n_fail=0; idx=0
total=$(( ${#S7_CELLS[@]} + ${#S8_CELLS[@]} ))

# Sample this cell's peak anon rss (the OOM-relevant charge) while it runs.
sample_peak() {
  local pid=$1 max=0 rss
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(awk '$1=="rss"{print $2; exit}' "$STAT" 2>/dev/null || echo 0)
    case "$rss" in ''|*[!0-9]*) rss=0 ;; esac
    [ "$rss" -gt "$max" ] && max=$rss
    sleep 5
  done
  echo "$max" > /tmp/i1336_cell_peak
}

run_cell() {
  local cell=$1
  idx=$((idx + 1))
  local t0 t1 rc peak_b peak_g
  t0=$(date +%s)
  echo "[recover] cell $idx/$total $cell START"
  timeout --kill-after=60s "${PER_CELL_FENCE}s" \
    uv run python scripts/issue1336_selfmap_missing_pairs.py \
      --out-root "$OUT" --stage-root "$STAGE" --layer "$LAYER" \
      --stage --cells "$cell" &
  local cpid=$!
  sample_peak "$cpid" &
  local spid=$!
  wait "$cpid"; rc=$?
  wait "$spid" 2>/dev/null || true
  t1=$(date +%s)
  peak_b=$(cat /tmp/i1336_cell_peak 2>/dev/null || echo 0)
  peak_g=$(awk -v v="$peak_b" 'BEGIN{printf "%.1f", v/1073741824}')
  echo "[recover] cell $idx/$total $cell rc=$rc elapsed=$(( t1 - t0 ))s peak_anon_rss=${peak_g}GiB"
  if [ "$rc" -eq 0 ]; then n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); fi
  results="${results}${results:+,}{\"cell\":\"$cell\",\"rc\":$rc,\"elapsed_s\":$(( t1 - t0 )),\"peak_anon_rss_gib\":\"$peak_g\"}"
  # A cell failure is NOT fatal: cells are independent and durable, so keep going and
  # recover every cell this launch can. The sentinel carries the per-cell rc map.
}

for c in "${S7_CELLS[@]}"; do run_cell "$c"; done

# Reap surface 7's staged turnstores (re-downloadable Hub copies) before surface 8
# stages its own ~70 GB. Never touches OUT (this round's durable product).
before_gb=$(du -sBG "$STAGE" 2>/dev/null | cut -f1 | tr -dc '0-9')
rm -rf "$STAGE/turnstore_v2" "$STAGE/turnstore_wave1" "$STAGE/selfmap_stage_tmp"
after_gb=$(du -sBG "$STAGE" 2>/dev/null | cut -f1 | tr -dc '0-9')
avail_gb=$(df -BG --output=avail "$STAGE" | tail -1 | tr -dc '0-9')
echo "[recover] reaped chat/lmsys23k stage: ${before_gb:-?}GB -> ${after_gb:-?}GB (avail=${avail_gb}GB)"

need_margin=$(( S8_NEED_GB * 13 / 10 ))
if [ "${avail_gb:-0}" -lt "$need_margin" ]; then
  echo "[recover] HALT: avail=${avail_gb}GB < need_margin=${need_margin}GB for naturalistic/lmsys23k"
  printf '{"phase":"recover_percell","halt":"disk","avail_gb":%s,"need_margin_gb":%s,"n_ok":%s,"n_fail":%s,"cells":[%s]}\n' \
    "${avail_gb:-0}" "$need_margin" "$n_ok" "$n_fail" "$results" > "$SENTINEL"
  echo "[recover] DONE ok=$n_ok fail=$n_fail (halted before surface 8)"
  exit 2
fi

for c in "${S8_CELLS[@]}"; do run_cell "$c"; done

n_cells=$(ls "$OUT/cells"/*__l30.json 2>/dev/null | wc -l)
printf '{"phase":"recover_percell","n_ok":%s,"n_fail":%s,"cells_on_disk":%s,"cells":[%s]}\n' \
  "$n_ok" "$n_fail" "$n_cells" "$results" > "$SENTINEL"
echo "[recover] DONE ok=$n_ok fail=$n_fail cells_on_disk=$n_cells"
[ "$n_fail" -eq 0 ] || exit 1
