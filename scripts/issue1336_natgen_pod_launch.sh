#!/usr/bin/env bash
# Pod-side launcher for #1336 round-5 PART A-run: the ON-POLICY NATURALISTIC
# generation arm across ALL SEVEN v2 corpora (context arm only).
#
# WHY A SEPARATE LAUNCHER AND NOT phase_gen_v2:
# phase_gen_v2 (issue1336_dispatch.sh) already implements the right job shape —
# one vLLM job per (model, corpus), work-conserving across the realized GPU
# width, per-cell --upload — but it never passes --gen-format, so it can only
# ever generate chat. Adding the passthrough there would edit a large shared
# dispatcher that a CONCURRENT autonomous /issue 1336 session (round 4) may be
# executing; this launcher mirrors phase_gen_v2's job shape instead, leaving the
# shared file byte-untouched. Same reasoning natfmt applied when it chose a
# gen-only V2_GEN_FORMATS registry over widening V2_CORPORA (6182b041ab).
#
# CORPUS LIST is passed explicitly and INCLUDES gsm8k_test1319: dispatch.sh
# composes gen corpora as [c for c in V2_CORPORA if c not in V2_FULLY_REUSED_GEN],
# and that reuse declaration covers CHAT gen only — a naturalistic run must
# generate it fresh (no wave-1 naturalistic exists for it).
#
# SCOPE LIMIT (carried, not silently absorbed): gsm8k_train_full naturalistic is
# EXTENSION-ONLY under current prep (new_rows_only stages prompt_idx >= 5000,
# ~2473 rows). Unlike lmsys23k — whose 5k prefix reuses lmsys5k's v1 naturalistic
# wave — there is no wave-1 naturalistic generation for gsm8k_train5k, so that
# corpus's chat and naturalistic arms are NOT row-matched. Closing it is a prep
# decision, not a gen-gate change.
#
# Usage:  bash issue1336_natgen_pod_launch.sh [<slug>]
set -uo pipefail

SLUG="${1:-natgen}"

REPO=/workspace/explore-persona-space
BRANCH=issue-1336-fullcorpora
FIX_SHA=6182b041ab          # the --gen-format acceptance commit
LOGDIR=/workspace/logs
JOBLOG="$LOGDIR/natgen_jobs"

CORPORA="lmsys23k gsm8k_train_full gsm8k_test1319 math7500 if11k uf11k sft11k"
# rlvr first so the G1' cell's inputs land earliest (phase_gen_v2's ordering).
MODELS="rlvr base sft dpo rlvr_long"

mkdir -p "$LOGDIR" "$JOBLOG"
echo $$ > "$LOGDIR/issue-1336-${SLUG}.pid"

cd "$REPO" || { echo "[fatal] no repo at $REPO"; exit 2; }
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

echo "[setup] fetching $BRANCH"
git fetch origin "$BRANCH" --depth=50 --quiet || { echo "[fatal] fetch failed"; exit 2; }
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH"
git reset --hard -q "origin/$BRANCH"

# Fix-commit ancestry probe: prove the --gen-format acceptance code is present.
if git merge-base --is-ancestor "$FIX_SHA" HEAD; then
  echo "[setup] FIX-OK $FIX_SHA is an ancestor of $(git rev-parse --short HEAD)"
else
  echo "[fatal] FIX ABSENT: $FIX_SHA not an ancestor of $(git rev-parse HEAD)"
  exit 3
fi

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "[setup] NGPU=$NGPU"
echo "[setup] disk at start:"; df -h /workspace | tail -1

# ---------------------------------------------------------------------------
# CPU prep (model-free, FORMAT-BLIND — one pass covers both renders).
# ---------------------------------------------------------------------------
PREP_CSV=$(echo "$CORPORA" | tr ' ' ',')
if [ ! -f "$LOGDIR/natgen_prep.done" ]; then
  echo "[prep] staging corpora: $PREP_CSV"
  uv run python scripts/issue1336_gen_answers.py --prep --corpora "$PREP_CSV" \
      >> "$JOBLOG/prep.log" 2>&1
  prc=$?
  echo "[prep] rc=$prc"
  [ "$prc" -ne 0 ] && { echo "[fatal] prep failed rc=$prc"; exit 4; }
  touch "$LOGDIR/natgen_prep.done"
else
  echo "[prep] already done"
fi

# ---------------------------------------------------------------------------
# Build the job list: one vLLM job per (model, corpus) = 5 x 7 = 35 jobs.
# Fine granularity is deliberate (phase_gen_v2's own choice): it costs one
# engine load per job but keeps every GPU fed to the tail, which dominates at
# this job count. Per-job --upload persists the rollout TEXT to HF BEFORE any
# downstream reduction (upload policy).
# ---------------------------------------------------------------------------
JOBS="$LOGDIR/natgen_jobs.tsv"
: > "$JOBS"
for m in $MODELS; do
  for c in $CORPORA; do
    printf '%s__%s\tuv run python scripts/issue1336_gen_answers.py --model %s --corpora %s --gen-format naturalistic --upload\n' \
      "$m" "$c" "$m" "$c" >> "$JOBS"
  done
done
NJOBS=$(wc -l < "$JOBS")
echo "[jobs] $NJOBS pending"

# ---------------------------------------------------------------------------
# Work-conserving GPU pool: min(NGPU, NJOBS) workers, worker w pins
# CUDA_VISIBLE_DEVICES=w and pops the next pending job the moment it frees.
# No wave barriers. Per-job done-files make the whole arm resumable, and their
# natgen__ prefix cannot collide with the chat gen_v2 done-files.
# ---------------------------------------------------------------------------
DONE_DIR="$LOGDIR/natgen_done"; mkdir -p "$DONE_DIR"
CURSOR="$LOGDIR/natgen.cursor"; echo 0 > "$CURSOR"
LOCK="$LOGDIR/natgen.lock"

next_job() {  # atomically pop the next job index
  local i
  exec 9>"$LOCK"; flock 9
  i=$(cat "$CURSOR"); echo $((i + 1)) > "$CURSOR"
  exec 9>&-
  echo "$i"
}

worker() {
  local w=$1 i name cmd
  while :; do
    i=$(next_job)
    [ "$i" -ge "$NJOBS" ] && break
    name=$(sed -n "$((i + 1))p" "$JOBS" | cut -f1)
    cmd=$(sed -n "$((i + 1))p" "$JOBS" | cut -f2-)
    if [ -f "$DONE_DIR/natgen__${name}.done" ]; then
      echo "[gpu$w] SKIP $name (done)"; continue
    fi
    echo "[gpu$w] START $name $(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES=$w bash -c "$cmd" >> "$JOBLOG/${name}.log" 2>&1
    local rc=$?
    echo "[gpu$w] $name rc=$rc $(date -u +%FT%TZ)"
    [ "$rc" -eq 0 ] && touch "$DONE_DIR/natgen__${name}.done"
  done
  echo "[gpu$w] drained"
}

WIDTH=$NGPU
[ "$NJOBS" -lt "$WIDTH" ] && WIDTH=$NJOBS
for w in $(seq 0 $((WIDTH - 1))); do worker "$w" & done
wait

NDONE=$(ls -1 "$DONE_DIR" 2>/dev/null | wc -l)
RC_ALL=0
[ "$NDONE" -ne "$NJOBS" ] && RC_ALL=1
echo "[natgen] ALL JOBS DRAINED done=${NDONE}/${NJOBS} rc_all=${RC_ALL} $(date -u +%FT%TZ)"
echo "[natgen] disk at end:"; df -h /workspace | tail -1

# Sentinel for the VM-side poller (pod-side code never shells out to task.py).
cat > "$LOGDIR/issue-1336-${SLUG}-results.json" <<EOF
{"issue": 1336, "slug": "${SLUG}", "rc": ${RC_ALL},
 "jobs_total": ${NJOBS}, "jobs_done": ${NDONE},
 "gen_format": "naturalistic",
 "corpora": "${CORPORA}",
 "done_dir": "${DONE_DIR}", "job_logs": "${JOBLOG}",
 "finished_at": "$(date -u +%FT%TZ)"}
EOF
echo "[natgen] sentinel written"
exit "$RC_ALL"
