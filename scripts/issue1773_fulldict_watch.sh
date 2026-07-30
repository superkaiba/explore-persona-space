#!/usr/bin/env bash
# Issue #1773 full-dictionary — VM-side release watcher.
#
# passB runs ~7 h at width 1, which outlives any single agent turn. Without
# this, the pod idles at 0% GPU from the moment p1_passC lands until a human or
# a successor session notices — the #664 spend leak at multi-day scale, and the
# largest avoidable cost in this run. So the release is made AUTOMATIC and
# gated on durability, never on a timer:
#
#   wait for p1_passC.done AND the launcher process to exit
#     -> pod-side git pull (safe only now: pulling earlier would rewrite the
#        launcher file bash is still reading incrementally)
#     -> pod-side upload + EXACT-set verify of phase0 / selection / evidence
#     -> ONLY on verify exit 0: terminate the pod, remove keep-running
#     -> post the realized phase-0/1 numbers + evidence fill fraction
#
# A failed verify NEVER terminates: it alerts and leaves the pod up with the
# artifacts intact, because the durability contract outranks the spend.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/thomasjiralerspong/explore-persona-space}"
POD="${EPM_1773_POD:-pod-1773-regsteer}"
SUFFIX="${EPM_1773_POD_SUFFIX:-regsteer}"
WORK="${EPM_1773_WORK:-/workspace/issue1773_fulldict}"
MAX_WAIT_S="${EPM_1773_WATCH_MAX_S:-72000}"   # 20 h: ~7 h passB + passC + slack
POLL_S="${EPM_1773_WATCH_POLL_S:-180}"
LOG="${EPM_1773_WATCH_LOG:-/tmp/issue-1773-fulldict-watch.log}"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
podssh() { timeout 120 ssh -o BatchMode=yes "$POD" "$1"; }

log "watcher start: pod=$POD work=$WORK max_wait=${MAX_WAIT_S}s"

START=$(date +%s)
while :; do
  NOW=$(date +%s); ELAPSED=$((NOW - START))
  if [ "$ELAPSED" -gt "$MAX_WAIT_S" ]; then
    log "TIMEOUT after ${ELAPSED}s without p1_passC — leaving pod UP (never terminate on a timer)"
    uv run python scripts/task.py post-marker 1773 epm:progress \
      --note "[fulldict-watch] TIMEOUT after ${ELAPSED}s waiting for p1_passC. Pod $POD left RUNNING deliberately (termination is gated on a verified upload, never on a clock). Check /workspace/logs/issue-1773-fulldict.log and the passB worker logs." || true
    exit 3
  fi
  DONES=$(podssh "ls $WORK/done/ 2>/dev/null | tr '\n' ' '" || echo "")
  ALIVE=$(podssh "pgrep -c -f 'issue1773_fulldict_launc[h]' 2>/dev/null || echo 0" || echo "unknown")
  case "$DONES" in
    *p1_passC.done*)
      if [ "$ALIVE" = "0" ]; then log "p1_passC landed and launcher exited — proceeding to release"; break; fi
      log "p1_passC landed; launcher still alive (n=$ALIVE) — waiting for clean exit" ;;
    *) log "waiting (${ELAPSED}s): done=[$DONES] launcher_alive=$ALIVE" ;;
  esac
  sleep "$POLL_S"
done

# ── pod-side sync + upload + EXACT-set verify ────────────────────────────────
log "pod git pull (launcher no longer running, so rewriting it is now safe)"
podssh "git -C /workspace/explore-persona-space fetch origin issue-1773 --quiet; git -C /workspace/explore-persona-space merge --ff-only origin/issue-1773 2>&1 | tail -2" | tee -a "$LOG"

log "pod-side upload + exact-set verify"
podssh "cd /workspace/explore-persona-space && timeout 5400 uv run python scripts/issue1773_fulldict_upload.py --work $WORK 2>&1 | tail -40" | tee -a "$LOG"
UPLOAD_RC=${PIPESTATUS[0]}
log "upload rc=$UPLOAD_RC"

if [ "$UPLOAD_RC" -ne 0 ]; then
  log "UPLOAD/VERIFY FAILED — pod stays UP, artifacts intact, no termination"
  uv run python scripts/task.py post-marker 1773 epm:progress \
    --note "[fulldict-watch] phase-0/1 upload or EXACT-set verify FAILED (rc=$UPLOAD_RC). Pod $POD deliberately left RUNNING with artifacts intact — termination is gated on a verified upload. See $LOG and /workspace/logs on the pod." || true
  exit 4
fi

VERIFIED=$(podssh "cat $WORK/fulldict_upload_verified.json 2>/dev/null" || echo "{}")
log "verified payload: $VERIFIED"

# ── release: terminate the pod, drop the shield ──────────────────────────────
# Tag comes off FIRST: the issue-wide keep-running shield can refuse a
# terminate (#1485), and dropping it first means that even if the terminate
# errors, the watcher's pod-safety pass re-arms and stops the pod as a
# backstop. That is safe only here — the artifacts are already verified on the
# Hub, so a stopped (non-durable) volume no longer holds anything unique.
log "artifacts durable — dropping keep-running, then terminating $POD"
uv run python scripts/task.py remove-tag 1773 keep-running 2>&1 | tail -2 | tee -a "$LOG"
uv run python scripts/pod.py terminate --issue 1773 --name-suffix "$SUFFIX" --yes 2>&1 | tail -5 | tee -a "$LOG"
TERM_RC=${PIPESTATUS[0]}
log "terminate rc=$TERM_RC"
if [ "$TERM_RC" -ne 0 ]; then
  log "terminate returned rc=$TERM_RC — verify by hand: pod.py list-ephemeral --issue 1773"
fi

printf '%s\n' \
  "[fulldict-watch] PHASE 0/1 COMPLETE — artifacts durable, POD RELEASED." \
  "" \
  "Upload + EXACT-set verify PASSED for all three prefixes under" \
  "issue1773_featurepipeline/fulldict/ (phase0 / selection / evidence)." \
  "Verified payload (file counts + realized evidence completeness):" \
  "$VERIFIED" \
  "" \
  "Pod $POD terminated (rc=$TERM_RC); keep-running tag removed so the" \
  "watcher pod-safety pass re-arms. No GPU work remains in this run —" \
  "phases 2-3 are Batch-API only." \
  "" \
  "NEXT: the pre-spend checkpoint note carries the measured per-call token" \
  "basis and the cost projection; phases 2-3 run off-pod with --grouped" \
  "(per-group dispatch dirs + done-sentinels, resume by sentinel)." \
  > /tmp/issue-1773-fulldict-release-note.md
uv run python scripts/task.py post-marker 1773 epm:progress \
  --file /tmp/issue-1773-fulldict-release-note.md || true

log "watcher done"
exit 0
