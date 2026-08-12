#!/usr/bin/env bash
# Rigid-decomposition driver: per-SURFACE stage -> leg B -> leg A -> reap.
#
# WHY THIS EXISTS. Two layer-30 gaps need the SAME staged turnstores, and staging is
# the expensive part (317.76 GB cumulative, measured 2026-08-07 for the 4-model
# selfmap shape). Running them as two separate jobs would stage twice. So one pass:
#
#   leg B  issue1336_selfmap_missing_pairs.py  — the t5/t5s backfill. The three
#          round-B pairs (sft->rlvr, sft->rlvr_long, rlvr->rlvr_long) ran only
#          t0/t6/t7/t8, so they are ABSENT at t5 and the lattice draws a gap
#          (issue1336_full_transfer_lattice.py:598-605). The t5/t5s algebra exists
#          at issue1336_selfmap_missing_pairs.py:296-298 and has never been run.
#   leg A  issue1336_metric_ladder.py — the four ORTH tiers t5c/t5cs/t5b/t5bs
#          (ORTH_TIER_NAMES). They separate a rigid rotation of the CONTEXT manifold
#          (t5c) from one of the ANSWER manifold (t5b) from both (t5bs). Never run:
#          zero banked pair files carry an orth_tiers block.
#
# WHY A SEPARATE STAGING STEP. issue1336_selfmap_missing_pairs.py --stage stages exactly
# the stems its --cells consume, but --cells is hard-asserted against that script's
# 32-cell registry, whose model set is {base, sft, rlvr, rlvr_long}. Leg A's seven
# baseline pairs also need dpo, and staging twice would pay the surface's bytes twice.
# So issue1336_stage_surface.py stages the five-model UNION once per surface (reusing
# stage_inputs verbatim) and both legs then run against the already-staged tree — which
# is why per-surface staged bytes are ~1.25x the selfmap launcher's 4-model figures.
#
# WHY ONE PROCESS PER CELL AND PER PAIR. Both fits cache each loaded (model, surface)
# in RAM for the whole invocation. A 5-cell surface would hold 5 model-surfaces at
# once; on lmsys23k that measured ~40 GB each and OOM-killed (rc=137) on the 3rd load
# against the cpu-bigmem cgroup limit of 128 GB — which `free -g` hides, reporting the
# 251 GB HOST total. Per-cell / per-pair invocation caps residency at that unit's own
# 1-2 model-surfaces. Staging is deliberately NOT reaped between units of a surface:
# it is shared, and disk is the non-binding constraint here (~87 GB against 240 GB).
#
# PILOT GATE. No measured basis exists for a layer-30 orth-bearing ladder battery (the
# round-3 battery ran 4 frozen layers with bootstrap + nulls, so its wall does not
# price this shape). The FIRST leg-A invocation of the FIRST surface is therefore the
# pilot: its wall is recorded to $LOGDIR/issue-1336-rigid-pilot-wall-s and echoed with
# the 56-battery projection. Nothing self-aborts on it — a guessed fence is what
# killed a healthy ~25 min/cell run at exit=124 (#1092). The projection is for the
# operator to read.
#
# Usage:  bash scripts/issue1336_rigid_surfaces.sh [FIRST] [LAST]   (1-indexed, incl.)
#         bash scripts/issue1336_rigid_surfaces.sh 1 1   # cheapest surface = the basis
set -uo pipefail

REPO=/workspace/explore-persona-space
BRANCH=${EPM_1336_BRANCH:-issue-1336-backward-pairs}
# The layer-30 orth gate. Overridable so a later round keys the ancestry probe to ITS
# OWN fix commit rather than to the round that first wrote this launcher.
FIX_SHA="${EPM_1336_FIX_SHA:-cb8dfae703}"

OUT_B=${EPM_1336_OUT_B:-/workspace/eval_results/issue_1336/selfmap_t5}
OUT_A=${EPM_1336_OUT_A:-/workspace/eval_results/issue_1336_rigid}
STAGE=${EPM_1336_STAGE:-/workspace/data/issue_1336}
LOGDIR=${EPM_1336_LOGDIR:-/workspace/logs}
LAYER=${EPM_1336_LAYER:-30}

# G0'(c) v2-recipe Qwen anchor, read out of the banked round-3 pair files (every one
# carries it; all 56 agree). --bars-json is the dispatcher's route to the same value,
# but gates_v2/v2_bars.json is not committed, so the value is passed directly.
S_QWEN_V2=${EPM_1336_S_QWEN_V2:-0.6935026836671432}

# The seven BASELINE pairs = full lattice minus the three round-B selfmap pairs.
LADDER_PAIRS=${EPM_1336_LADDER_PAIRS:-base:sft,base:dpo,base:rlvr,base:rlvr_long,sft:dpo,dpo:rlvr,dpo:rlvr_long}

# label|approx_staged_GB|format|corpus   — cheapest first, so the measured pilot wall
# lands early and one surface is durable before the big pulls begin. GB figures are the
# selfmap launcher's MEASURED per-surface bytes scaled ~1.25x for the added dpo stem.
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

mkdir -p "$LOGDIR" "$OUT_A" "$OUT_B" "$STAGE"
PIDFILE="$LOGDIR/issue-1336-rigid.pid"
SENTINEL="$LOGDIR/issue-1336-rigid-done.json"
PILOTFILE="$LOGDIR/issue-1336-rigid-pilot-wall-s"
# Re-attach breadcrumbs: the pidfile carries THIS run's pid (never a predecessor's —
# the #813 relaunch trap) and the exit sentinel is removed at launch so a stale one can
# never satisfy a done-check (never key done on bare existence).
echo $$ > "$PIDFILE"
rm -f "$SENTINEL"

write_sentinel() {
  cat > "$SENTINEL.tmp" <<JSON
{"issue": 1336, "round": "rigid-decomposition", "rc": $1, "surfaces": "$FIRST..$LAST",
 "cells_leg_b": $2, "pairs_leg_a": $3,
 "out_root_leg_a": "$OUT_A", "out_root_leg_b": "$OUT_B",
 "finished_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON
  mv "$SENTINEL.tmp" "$SENTINEL"
}

set -a
[ -f ./.env ] && . ./.env
set +a
# 16 uncontended vCPU on this pod — not the shared-VM cap of 8.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

# The pod bootstraps on main; this round's code lives on the issue branch.
echo "[driver] fetching $BRANCH"
git fetch origin "$BRANCH" --depth=50 --quiet || { echo "FATAL: fetch failed" >&2; exit 2; }
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH"
git reset --hard -q "origin/$BRANCH"

# Fix-commit ancestry probe: prove the orth gate we are about to run is actually here.
if git merge-base --is-ancestor "$FIX_SHA" HEAD; then
  echo "[driver] FIX-OK $FIX_SHA is an ancestor of $(git rev-parse --short HEAD)"
else
  echo "FATAL: FIX ABSENT: $FIX_SHA not an ancestor of $(git rev-parse HEAD)" >&2
  exit 3
fi

echo "[driver] head=$(git rev-parse HEAD)"
echo "[driver] out_a=$OUT_A out_b=$OUT_B stage=$STAGE layer=$LAYER surfaces=$FIRST..$LAST"
echo "[driver] disk at start:"; df -h /workspace | tail -1

rc_all=0
pilot_done=0
for ((i = FIRST; i <= LAST; i++)); do
  row="${SURFACES[i-1]}"
  label="${row%%|*}"
  rest="${row#*|}"
  need_gb="${rest%%|*}"
  rest="${rest#*|}"
  fmt="${rest%%|*}"
  corpus="${rest##*|}"

  # Fail-fast disk assert at 1.3x the measured staged bytes: refuse to start a surface
  # that cannot fit, instead of discovering ENOSPC 60 GB into a pull.
  avail_gb=$(df -BG --output=avail "$STAGE" | tail -1 | tr -dc '0-9')
  need_margin=$(( need_gb * 13 / 10 ))
  echo "[driver] === surface $i/${#SURFACES[@]} $label === need=${need_gb}GB margin=${need_margin}GB avail=${avail_gb}GB"
  if [ "${avail_gb:-0}" -lt "$need_margin" ]; then
    echo "FATAL surface $label: avail ${avail_gb}GB < ${need_margin}GB (1.3x measured ${need_gb}GB)" >&2
    rc_all=2
    break
  fi

  s0=$(date +%s)

  # ---- staging: the five-model union, ONCE, feeding both legs. --------------------
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

  # ---- leg B: t5/t5s backfill (inputs already staged, so no --stage here). ---------
  # One process per cell (RAM residency, see header).
  for cell in \
      "base__base__${fmt}__${corpus}" \
      "sft__rlvr__${fmt}__${corpus}" \
      "sft__rlvr_long__${fmt}__${corpus}" \
      "rlvr__rlvr_long__${fmt}__${corpus}"; do
    c0=$(date +%s)
    echo "[legB] ${cell} START $(date -u +%FT%TZ)"
    uv run python scripts/issue1336_selfmap_missing_pairs.py \
      --stage-root "$STAGE" --out-root "$OUT_B" --layer "$LAYER" \
      --cells "$cell"
    crc=$?
    echo "[legB] ${cell} rc=${crc} elapsed=$(( $(date +%s) - c0 ))s"
    [ "$crc" -ne 0 ] && rc_all=$crc
  done

  # ---- leg A: the orth tiers, one process per pair (RAM residency, see header). ----
  for pair in ${LADDER_PAIRS//,/ }; do
    p0=$(date +%s)
    echo "[legA] ${pair} @ ${label} START $(date -u +%FT%TZ)"
    uv run python scripts/issue1336_metric_ladder.py \
      --pair "$pair" --corpus "$corpus" --format "$fmt" \
      --s-qwen-v2 "$S_QWEN_V2" \
      --frozen-layers "$LAYER" --full-tier-layers "$LAYER" \
      --out-dir "$OUT_A" \
      --turnstore-dir "$STAGE/turnstore_v2" \
      --wave1-turnstore-dir "$STAGE/turnstore_wave1"
    prc=$?
    pwall=$(( $(date +%s) - p0 ))
    echo "[legA] ${pair} @ ${label} rc=${prc} elapsed=${pwall}s"
    [ "$prc" -ne 0 ] && rc_all=$prc
    # Pilot: the FIRST leg-A battery is the measured basis for the 56-battery wall.
    if [ "$pilot_done" -eq 0 ]; then
      pilot_done=1
      echo "$pwall" > "$PILOTFILE"
      echo "[pilot] measured ${pwall}s for 1 battery -> 56 batteries project to $(( pwall * 56 / 60 )) min serial"
    fi
  done

  # Reap this surface's staged turnstores (re-downloadable Hub copies under --stage-root;
  # NOT a generated store/ or eval_results/). The small wave-1 gen answers under gen/ are
  # KEPT (KB-MB, reused across surfaces). Out-roots are NEVER reaped — they are this
  # round's durable product and both drivers' resume predicates read them.
  before_gb=$(du -sBG "$STAGE" 2>/dev/null | cut -f1 | tr -dc '0-9')
  rm -rf "$STAGE/turnstore_v2" "$STAGE/turnstore_wave1" "$STAGE/selfmap_stage_tmp"
  after_gb=$(du -sBG "$STAGE" 2>/dev/null | cut -f1 | tr -dc '0-9')
  echo "[driver] surface $label DONE elapsed=$(( $(date +%s) - s0 ))s reaped ${before_gb:-?}GB -> ${after_gb:-?}GB"
done

n_b=$(ls -1 "$OUT_B/cells" 2>/dev/null | wc -l)
n_a=$(ls -1 "$OUT_A/metric_ladder" 2>/dev/null | wc -l)
echo "[driver] ALL DONE rc=$rc_all leg_b_cells=$n_b leg_a_pairs=$n_a $(date -u +%FT%TZ)"
write_sentinel "$rc_all" "$n_b" "$n_a"
exit "$rc_all"
