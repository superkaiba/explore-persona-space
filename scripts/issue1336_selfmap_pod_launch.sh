#!/usr/bin/env bash
# Pod-side launcher for #1336 round-5 parts C+D (base self map + 3 missing forward pairs).
#
# Runs one surface at a time and REAPS the staged turnstores between surfaces:
# stage_inputs() stages every requested cell up front with no reap, and the full
# 8-surface stage is ~325 GB against a 240 GB container disk. --cells filters the
# cell list BEFORE stage_inputs() sees it, so per-surface invocation holds peak
# disk at ~115 GB (largest surface: lmsys23k, 4 models + wave-1 concat stems).
#
# Per-cell checkpoints under $OUT_ROOT/cells make every surface independently
# resumable, so a mid-job death costs one surface's staging, never the job.
#
# Usage:  bash issue1336_selfmap_pod_launch.sh "<slug>" "<fmt|corpus> [<fmt|corpus> ...]"
set -uo pipefail

SLUG="${1:?slug required}"
SURFACES="${2:?surface list required}"

REPO=/workspace/explore-persona-space
BRANCH=issue-1336-fullcorpora
FIX_SHA=5e60bd6ad6
LOGDIR=/workspace/logs
OUT_ROOT=/workspace/out/issue_1336_selfmap
STAGE_ROOT=/workspace/data/issue_1336

mkdir -p "$LOGDIR" "$OUT_ROOT" "$STAGE_ROOT"
echo $$ > "$LOGDIR/issue-1336-${SLUG}.pid"

cd "$REPO" || { echo "[fatal] no repo at $REPO"; exit 2; }
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# The pod bootstraps on main; this round's code lives on the issue branch.
echo "[setup] fetching $BRANCH"
git fetch origin "$BRANCH" --depth=50 --quiet || { echo "[fatal] fetch failed"; exit 2; }
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH"
git reset --hard -q "origin/$BRANCH"

# Fix-commit ancestry probe: prove the code we are about to run is actually here.
if git merge-base --is-ancestor "$FIX_SHA" HEAD; then
  echo "[setup] FIX-OK $FIX_SHA is an ancestor of $(git rev-parse --short HEAD)"
else
  echo "[fatal] FIX ABSENT: $FIX_SHA not an ancestor of $(git rev-parse HEAD)"
  exit 3
fi

echo "[setup] disk at start:"; df -h /workspace | tail -1

rc_all=0
for s in $SURFACES; do
  fmt="${s%%|*}"; corpus="${s##*|}"
  cells="base__base__${fmt}__${corpus}"
  cells="${cells},sft__rlvr__${fmt}__${corpus}"
  cells="${cells},sft__rlvr_long__${fmt}__${corpus}"
  cells="${cells},rlvr__rlvr_long__${fmt}__${corpus}"

  echo "[surface] ${fmt}/${corpus} START $(date -u +%FT%TZ)"
  uv run python scripts/issue1336_selfmap_missing_pairs.py \
      --stage \
      --stage-root "$STAGE_ROOT" \
      --out-root "$OUT_ROOT" \
      --layer 30 \
      --cells "$cells"
  rc=$?
  echo "[surface] ${fmt}/${corpus} rc=${rc} $(date -u +%FT%TZ)"
  [ "$rc" -ne 0 ] && rc_all=$rc

  # Reap this surface's staged turnstores before the next one downloads more.
  # NEVER touch $OUT_ROOT (durable cell checkpoints) or the small gen/corpora dirs.
  rm -rf "$STAGE_ROOT/turnstore_v2" "$STAGE_ROOT/turnstore_wave1" "$STAGE_ROOT/selfmap_stage_tmp"
  echo "[reap] after ${fmt}/${corpus}:"; df -h /workspace | tail -1
done

echo "[selfmap] ALL SURFACES DONE rc_all=${rc_all} $(date -u +%FT%TZ)"
echo "[selfmap] cells written: $(ls -1 "$OUT_ROOT/cells" 2>/dev/null | wc -l)"

# Sentinel for the VM-side poller (pod-side code never shells out to task.py).
cat > "$LOGDIR/issue-1336-${SLUG}-results.json" <<EOF
{"issue": 1336, "slug": "${SLUG}", "rc": ${rc_all},
 "out_root": "${OUT_ROOT}", "cells_dir": "${OUT_ROOT}/cells",
 "n_cell_files": $(ls -1 "$OUT_ROOT/cells" 2>/dev/null | wc -l),
 "finished_utc": "$(date -u +%FT%TZ)"}
EOF
echo "[selfmap] sentinel written"
exit "$rc_all"
