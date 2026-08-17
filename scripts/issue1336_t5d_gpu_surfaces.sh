#!/usr/bin/env bash
# t5d GPU round driver: per-SURFACE stage -> leg-A pair batteries SHARDED
# ACROSS GPUS -> per-cell upload+reap -> staging reap.
#
# WHY THIS EXISTS. The rigid round (issue1336_rigid_surfaces.sh) ran the orth
# tiers t5c/t5cs/t5b/t5bs WITHOUT --preds-dir on CPU pods now terminated, so
# the answer-side-only tier t5d (R_ans on the raw t0 prediction — no context
# rotation) cannot be derived from any banked artifact: R_ans and the per-fold
# t0 preds died with the pods. This round re-runs leg A ONLY (leg B's t5/t5s
# backfill is banked) with three changes vs the rigid driver:
#   1. t5d/t5ds added to ORTH_TIER_NAMES (zero extra SVDs — R_ans is shared
#      with t5b), commit pinned via EPM_1336_T5D_FIX_SHA;
#   2. GPU venue: the ladder's fit core auto-selects cuda
#      (issue825_fit_cells._fit_device — measured ~1 min/cell on A100 fp64 vs
#      ~1.9 h at 4 CPU threads for the #825 shape); pairs shard one-per-GPU
#      via CUDA_VISIBLE_DEVICES on ONE multi-GPU pod;
#   3. --preds-dir IS passed and every cell's npz uploads + reaps the moment
#      its battery completes (issue1336_t5d_upload_cell.py, #664 per-cell
#      contract) so staging (~87 GB max) + preds never co-resident past the
#      MooseFS ~130 GB /workspace quota.
#
# SCHEDULING SHAPE (named capacity constraint). Pairs within a surface run in
# WAVES of PAR concurrent processes, PAR = min(NGPU, RAM-derived cap): each
# pair process holds its own 2 model-surface arrays in host RAM (~90 GB peak
# on lmsys23k — the rigid round OOM-killed at 3 residents under a 128 GB
# cgroup), so RAM — not GPU count — binds width on the big surfaces. Per-pair
# walls within a surface are near-equal (same n), so the wave barrier costs
# well under one battery wall; work-conserving dispatch would buy little
# against this RAM constraint.
#
# PILOT GATE. The FIRST battery's wall is recorded to
# $LOGDIR/issue-1336-t5d-pilot-wall-s and echoed with an n-scaled projection.
# Nothing self-aborts on it (#1092 — a guessed fence killed a healthy run);
# the operator reads the projection and decides.
#
# Usage:  bash scripts/issue1336_t5d_gpu_surfaces.sh [FIRST] [LAST]  (1-indexed)
#         bash scripts/issue1336_t5d_gpu_surfaces.sh 1 1   # cheapest surface = pilot
set -uo pipefail

REPO=/workspace/explore-persona-space
BRANCH=${EPM_1336_BRANCH:-issue-1336-backward-pairs}
# The t5d orth-tier commit. Overridable so a later round keys the ancestry
# probe to ITS OWN fix commit.
FIX_SHA="${EPM_1336_T5D_FIX_SHA:?set EPM_1336_T5D_FIX_SHA to the t5d commit}"

OUT_A=${EPM_1336_OUT_A:-/workspace/eval_results/issue_1336_t5d}
STAGE=${EPM_1336_STAGE:-/workspace/data/issue_1336}
PREDS=${EPM_1336_PREDS:-/workspace/data/issue_1336/t5d_preds}
LOGDIR=${EPM_1336_LOGDIR:-/workspace/logs}
LAYER=${EPM_1336_LAYER:-30}

# G0'(c) v2-recipe Qwen anchor (same provenance as the rigid driver: read out
# of the banked round-3 pair files; all 56 agree).
S_QWEN_V2=${EPM_1336_S_QWEN_V2:-0.6935026836671432}

LADDER_PAIRS=${EPM_1336_LADDER_PAIRS:-base:sft,base:dpo,base:rlvr,base:rlvr_long,sft:dpo,dpo:rlvr,dpo:rlvr_long}

# label|approx_staged_GB|format|corpus — cheapest first (pilot lands early).
SURFACES=(
"chat/gsm8k_test1319|8|chat|gsm8k_test1319"
"chat/math7500|40|chat|math7500"
"chat/gsm8k_train_full|40|chat|gsm8k_train_full"
"chat/sft11k|45|chat|sft11k"
"chat/if11k|46|chat|if11k"
"chat/uf11k|49|chat|uf11k"
"chat/lmsys23k|87|chat|lmsys23k"
"naturalistic/lmsys23k|87|naturalistic|lmsys23k"
)

FIRST=${1:-1}
LAST=${2:-${#SURFACES[@]}}

cd "$REPO" || { echo "FATAL: no $REPO" >&2; exit 1; }

mkdir -p "$LOGDIR" "$OUT_A" "$STAGE" "$PREDS"
PIDFILE="$LOGDIR/issue-1336-t5d.pid"
SENTINEL="$LOGDIR/issue-1336-t5d-done.json"
PILOTFILE="$LOGDIR/issue-1336-t5d-pilot-wall-s"
# Re-attach breadcrumbs: pidfile carries THIS run's pid; the exit sentinel is
# removed at launch so a stale one can never satisfy a done-check.
echo $$ > "$PIDFILE"
rm -f "$SENTINEL"

write_sentinel() {
  cat > "$SENTINEL.tmp" <<JSON
{"issue": 1336, "round": "t5d-gpu", "rc": $1, "surfaces": "$FIRST..$LAST",
 "pairs_leg_a": $2, "out_root": "$OUT_A", "preds_dir": "$PREDS",
 "hf_prefix": "issue1336_rlvr_ladder/analysis_tensors/metric_ladder_preds_t5d",
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
[ "${NGPU:-0}" -ge 1 ] || { echo "FATAL: no GPUs visible" >&2; exit 4; }
NCPU=$(nproc)
MEM_GB=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1048576 ))
echo "[driver] gpus=$NGPU cpus=$NCPU ram=${MEM_GB}GB"

# The pod bootstraps on main; this round's code lives on the issue branch.
echo "[driver] fetching $BRANCH"
git fetch origin "$BRANCH" --depth=50 --quiet || { echo "FATAL: fetch failed" >&2; exit 2; }
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH"
git reset --hard -q "origin/$BRANCH"

if git merge-base --is-ancestor "$FIX_SHA" HEAD; then
  echo "[driver] FIX-OK $FIX_SHA is an ancestor of $(git rev-parse --short HEAD)"
else
  echo "FATAL: FIX ABSENT: $FIX_SHA not an ancestor of $(git rev-parse HEAD)" >&2
  exit 3
fi

echo "[driver] head=$(git rev-parse HEAD)"
echo "[driver] out_a=$OUT_A preds=$PREDS stage=$STAGE layer=$LAYER surfaces=$FIRST..$LAST"
echo "[driver] disk at start:"; df -h /workspace | tail -1

# Warm the venv once so concurrent `uv run` wave members never race the sync.
uv run python -c "import torch; print('[driver] torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 5

rc_all=0
pilot_done=0
n_done=0
for ((i = FIRST; i <= LAST; i++)); do
  row="${SURFACES[i-1]}"
  label="${row%%|*}"
  rest="${row#*|}"
  need_gb="${rest%%|*}"
  rest="${rest#*|}"
  fmt="${rest%%|*}"
  corpus="${rest##*|}"

  avail_gb=$(df -BG --output=avail "$STAGE" | tail -1 | tr -dc '0-9')
  need_margin=$(( need_gb * 13 / 10 ))
  echo "[driver] === surface $i/${#SURFACES[@]} $label === need=${need_gb}GB margin=${need_margin}GB avail=${avail_gb}GB"
  if [ "${avail_gb:-0}" -lt "$need_margin" ]; then
    echo "FATAL surface $label: avail ${avail_gb}GB < ${need_margin}GB (1.3x measured ${need_gb}GB)" >&2
    rc_all=2
    break
  fi

  s0=$(date +%s)

  # ---- staging: the five-model union, ONCE (same helper as the rigid round).
  g0=$(date +%s)
  echo "[stage] ${label} START $(date -u +%FT%TZ)"
  uv run python scripts/issue1336_stage_surface.py \
    --models base,sft,dpo,rlvr,rlvr_long --format "$fmt" --corpus "$corpus" \
    --stage-root "$STAGE"
  grc=$?
  echo "[stage] ${label} rc=${grc} elapsed=$(( $(date +%s) - g0 ))s"
  if [ "$grc" -ne 0 ]; then
    echo "FATAL surface $label: staging failed rc=$grc — stopping (completed surfaces are durable)" >&2
    rc_all=$grc
    break
  fi
  echo "[stage] disk after ${label}:"; df -h /workspace | tail -1

  # ---- per-surface width: RAM binds, not GPU count (see header). -----------
  if [ "$need_gb" -ge 80 ]; then pair_ram=95; elif [ "$need_gb" -ge 40 ]; then pair_ram=55; else pair_ram=20; fi
  par=$(( (MEM_GB - 30) / pair_ram )); [ "$par" -lt 1 ] && par=1; [ "$par" -gt "$NGPU" ] && par="$NGPU"
  omp=$(( NCPU / par )); [ "$omp" -lt 2 ] && omp=2; [ "$omp" -gt 16 ] && omp=16
  echo "[driver] surface $label width: par=$par (pair_ram_est=${pair_ram}GB) omp=$omp"

  # ---- leg A: pair batteries in waves of $par, one GPU per process. --------
  pairs=(${LADDER_PAIRS//,/ })
  idx=0
  while [ "$idx" -lt "${#pairs[@]}" ]; do
    declare -a wave_pids=() wave_pairs=() wave_t0=()
    slot=0
    while [ "$slot" -lt "$par" ] && [ "$idx" -lt "${#pairs[@]}" ]; do
      pair="${pairs[idx]}"
      m0="${pair%%:*}"; m1="${pair##*:}"
      unit="${m0}__${m1}_${fmt}_${corpus}"
      if [ -f "$OUT_A/metric_ladder/pair_${unit}.json" ]; then
        echo "[legA] SKIP ${pair} @ ${label} (pair_${unit}.json exists — resume)"
        idx=$((idx + 1))
        continue
      fi
      plog="$LOGDIR/issue-1336-t5d-${unit}.log"
      echo "[legA] ${pair} @ ${label} START gpu=$slot $(date -u +%FT%TZ) log=$plog"
      CUDA_VISIBLE_DEVICES=$slot \
      OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp OPENBLAS_NUM_THREADS=$omp NUMEXPR_NUM_THREADS=$omp \
      uv run python scripts/issue1336_metric_ladder.py \
        --pair "$pair" --corpus "$corpus" --format "$fmt" \
        --s-qwen-v2 "$S_QWEN_V2" \
        --frozen-layers "$LAYER" --full-tier-layers "$LAYER" \
        --out-dir "$OUT_A" \
        --preds-dir "$PREDS" \
        --turnstore-dir "$STAGE/turnstore_v2" \
        --wave1-turnstore-dir "$STAGE/turnstore_wave1" \
        > "$plog" 2>&1 &
      wave_pids+=($!)
      wave_pairs+=("$pair")
      wave_t0+=($(date +%s))
      slot=$((slot + 1))
      idx=$((idx + 1))
    done
    [ "${#wave_pids[@]}" -eq 0 ] && continue
    for w in "${!wave_pids[@]}"; do
      wait "${wave_pids[w]}"
      prc=$?
      pair="${wave_pairs[w]}"
      m0="${pair%%:*}"; m1="${pair##*:}"
      unit="${m0}__${m1}_${fmt}_${corpus}"
      pwall=$(( $(date +%s) - wave_t0[w] ))
      echo "[legA] ${pair} @ ${label} rc=${prc} elapsed=${pwall}s"
      if [ "$prc" -ne 0 ]; then
        rc_all=$prc
        echo "[legA] tail of failing log:"; tail -20 "$LOGDIR/issue-1336-t5d-${unit}.log"
        continue
      fi
      n_done=$((n_done + 1))
      if [ "$pilot_done" -eq 0 ]; then
        pilot_done=1
        echo "$pwall" > "$PILOTFILE"
        echo "[pilot] measured ${pwall}s for 1 battery on $label -> 56-battery serial projection $(( pwall * 56 / 60 )) min (n-scaled: big corpora run ~n_ratio slower; read later surfaces' walls as they land)"
      fi
      # Per-cell upload + local npz reap (#664; MooseFS quota headroom).
      u0=$(date +%s)
      uv run python scripts/issue1336_t5d_upload_cell.py \
        --preds-dir "$PREDS" --out-dir "$OUT_A" --unit "$unit"
      urc=$?
      echo "[upload] ${unit} rc=${urc} elapsed=$(( $(date +%s) - u0 ))s"
      [ "$urc" -ne 0 ] && rc_all=$urc
    done
    unset wave_pids wave_pairs wave_t0
  done

  # Reap this surface's staged turnstores (re-downloadable Hub copies; the
  # small wave-1 gen answers under gen/ are KEPT). Out-roots never reaped.
  before_gb=$(du -sBG "$STAGE" 2>/dev/null | cut -f1 | tr -dc '0-9')
  rm -rf "$STAGE/turnstore_v2" "$STAGE/turnstore_wave1" "$STAGE/selfmap_stage_tmp"
  after_gb=$(du -sBG "$STAGE" 2>/dev/null | cut -f1 | tr -dc '0-9')
  echo "[driver] surface $label DONE elapsed=$(( $(date +%s) - s0 ))s reaped ${before_gb:-?}GB -> ${after_gb:-?}GB"
done

n_a=$(ls -1 "$OUT_A/metric_ladder" 2>/dev/null | wc -l)
echo "[driver] ALL DONE rc=$rc_all leg_a_pairs=$n_a cells_run_this_invocation=$n_done $(date -u +%FT%TZ)"
write_sentinel "$rc_all" "$n_a"
exit "$rc_all"
