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
# SCOPE LIMIT (carried, not silently absorbed): BOTH concat corpora are
# EXTENSION-ONLY in this ON-POLICY naturalistic arm — new_rows_only stages
# prompt_idx >= V2_CONCAT_BOUNDARY (5000 for both), so the wave-1 rows 0..4999
# are absent from the on-policy pool. MEASURED on the rlvr stage of this run:
#   lmsys23k__gen_naturalistic          18000 rows, prompt_idx 5000..22999
#   gsm8k_train_full__gen_naturalistic   2473 rows, prompt_idx 5000..7472
# so neither corpus's chat and on-policy-naturalistic arms are row-matched.
#
# An earlier revision of this comment claimed lmsys23k was EXEMPT because its
# 5k prefix "reuses lmsys5k's v1 naturalistic wave". That is WRONG for this arm
# and the distinction is provenance, not coverage. The two wave-1 situations
# differ, and neither yields on-policy naturalistic rows:
#   gsm8k_train5k: FORMATS_BY_CORPUS == ("chat",) — no naturalistic wave at all.
#   lmsys5k:       FORMATS_BY_CORPUS == ("chat","naturalistic"), BUT that wave is
#                  MATCHED-TEXT — a naturalistic RENDER of chat-generated answers.
#                  Verified: the data repo's generation prefix has exactly one
#                  `lmsys5k` child and no `lmsys5k__gen_naturalistic` sibling, and
#                  --gen-format did not exist before 6182b041ab, so no on-policy
#                  naturalistic generation for it can exist.
# Concatenating on-policy rows 5000+ onto matched-text rows 0..4999 would splice
# two different completion provenances into one arm — worse than the short n.
# Closing this is a prep decision (regenerate wave-1 on-policy for both corpora),
# not a gen-gate change; until then the on-policy arm is extension-only and says so.
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
# MooseFS import-contention prevention (gotchas.md § MooseFS FUSE READ-wedge,
# KNOWN TRIGGER #1689). An N-way parallel `uv run` fan-out storms the
# MooseFS-backed /workspace mount with N concurrent venv resolutions. This
# launcher hit it on its first run: all 8 workers sat at ~19 s CPU with
# wchan=request_wait_answer, GPUs at 0 MiB, ~0 network, and 0-byte logs for
# 10+ min.
#
# MEASURED DISTINCTION from the #779 hard wedge, which matters because the two
# have opposite remedies: a re-probe on that same pod returned `uv run python -c
# "print(1)"` rc=0 and a raw 300-byte venv read rc=0 INSTANTLY, while
# `import transformers` timed out at 60 s (rc=124). So the mount was ANSWERING —
# the failure was contention among 8 concurrent heavyweight imports, recoverable
# in place, NOT the dead mount that requires a pod swap. Run both probes before
# condemning a pod: the cheap one discriminates, and swapping an 8-GPU pod on a
# misread is expensive.
#
# Two guards, belt and braces:
#   1. UV_NO_SYNC=1 so no worker attempts a resolution at all.
#   2. Workers invoke the venv interpreter DIRECTLY ($PYBIN) rather than through
#      `uv run`, so the concurrent-resolution path is not merely suppressed but
#      absent. The venv is resolved ONCE, serially, below.
export UV_NO_SYNC=1
PYBIN="$REPO/.venv/bin/python"
[ -x "$PYBIN" ] || { echo "[fatal] no venv interpreter at $PYBIN"; exit 5; }

echo "[setup] pre-resolving venv ONCE (serial) before any fan-out"
if ! timeout 600 "$PYBIN" -c 'import vllm, transformers; print("[setup] venv import OK")'; then
  echo "[fatal] serial venv import failed or timed out after 600 s."
  echo "[fatal] DISCRIMINATE before condemning the pod (see the block above):"
  echo "[fatal]   timeout 20 head -c 300 $PYBIN >/dev/null; echo rc=\$?"
  echo "[fatal]   timeout 60 $PYBIN -c 'print(1)'; echo rc=\$?"
  echo "[fatal] both rc=0 => mount alive, contention only: raise NATGEN_STAGGER_S and relaunch here."
  echo "[fatal] either hangs   => hard MooseFS wedge: relaunching will NOT clear it, swap the pod."
  exit 6
fi

# ---------------------------------------------------------------------------
# CPU prep (model-free, FORMAT-BLIND — one pass covers both renders).
# ---------------------------------------------------------------------------
PREP_CSV=$(echo "$CORPORA" | tr ' ' ',')
if [ ! -f "$LOGDIR/natgen_prep.done" ]; then
  echo "[prep] staging corpora: $PREP_CSV"
  "$PYBIN" scripts/issue1336_gen_answers.py --prep --corpora "$PREP_CSV" \
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
    printf '%s__%s\t%s scripts/issue1336_gen_answers.py --model %s --corpora %s --gen-format naturalistic --upload\n' \
      "$m" "$c" "$PYBIN" "$m" "$c" >> "$JOBS"
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
    # PYTHONUNBUFFERED so a stalled job shows its last progress line instead of an
    # empty log (the first run's 0-byte job logs made the stall undiagnosable from
    # the log alone and forced a wchan/CPU-time probe).
    CUDA_VISIBLE_DEVICES=$w PYTHONUNBUFFERED=1 bash -c "$cmd" >> "$JOBLOG/${name}.log" 2>&1
    local rc=$?
    echo "[gpu$w] $name rc=$rc $(date -u +%FT%TZ)"
    [ "$rc" -eq 0 ] && touch "$DONE_DIR/natgen__${name}.done"
  done
  echo "[gpu$w] drained"
}

WIDTH=$NGPU
[ "$NJOBS" -lt "$WIDTH" ] && WIDTH=$NJOBS
# Stagger worker starts. The serial pre-import above warms the page cache, but a
# simultaneous 8-way heavyweight import off the MooseFS-backed venv still storms
# the mount: the first run of this launcher had all 8 workers frozen on FUSE reads
# (wchan=request_wait_answer) at ~19 s CPU with GPUs at 0 MiB and ~0 network for
# 10+ min. The mount was NOT wedged — a serial `print(1)` and a raw venv read both
# returned instantly — so the failure was contention, not a dead mount.
STAGGER_S="${NATGEN_STAGGER_S:-15}"
for w in $(seq 0 $((WIDTH - 1))); do
  worker "$w" &
  [ "$w" -lt $((WIDTH - 1)) ] && sleep "$STAGGER_S"
done
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
