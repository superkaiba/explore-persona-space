#!/bin/bash
# issue #2224 — follow-up round 1 (POD-side, detached; proposer-9b-cheap):
# legs 1+2 ONLY — E1 per-corpus map refit + cross-corpus probe transport.
# Leg 3 (rejudge-*) runs VM-side, API-only, and is deliberately NOT here.
#
# Phases (sequential fail-fast; per-phase logs + phases.jsonl + EXIT-trap
# sentinel, per the issue-2224 detached-runner conventions):
#   stage (stream-reduce capture shards -> fp16 layer slices)
#   -> refit (leg 1, GPU) -> transport (leg 2, GPU)
#   -> aggregate -> upload (refit leg)
#
# Launch (orchestrator, detached):
#   ssh <pod> 'cd /workspace/explore-persona-space && \
#     setsid nohup bash scripts/issue2224_followup_r1_runner.sh \
#       > /workspace/logs/issue-2224-fu-r1-round.log 2>&1 < /dev/null & \
#     echo $! > /workspace/logs/issue-2224-fu-r1-round.pid'
#
# Disk: stream-reduce staging peaks at ~one shard (~0.4 GB transient) +
# ~5.8 GB of fp16 slices per corpus (~12 GB total) — well inside the
# /workspace MooseFS ~130 GB quota; the 40 GB/corpus shard set is never
# mirrored.
#
# Pod-side: NO shared-VM thread-cap prefix (dedicated GPUs keep full width);
# NEVER shells out to scripts/task.py (pod-side reporting contract — the
# sentinel JSON is the poller's interface).
set -uo pipefail
export PATH="/root/.local/bin:$HOME/.local/bin:$PATH"
REPO="${EPS_POD_REPO:-/workspace/explore-persona-space}"
cd "$REPO" || exit 1
set -a; [ -f .env ] && source .env; set +a

LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"
SENTINEL="$LOGDIR/issue-2224-fu-r1-round.json"
PHASES_FILE="$LOGDIR/issue-2224-fu-r1-phases.jsonl"
: > "$PHASES_FILE"
# Clear any STALE prior-run sentinel at launch: on a same-pod relaunch the old
# status:done|failed envelope would read as a fresh verdict for the whole run
# (stale-artifact false-DONE class, #779/#825; CLAUDE.md § Monitoring).
rm -f "$SENTINEL"
CURRENT_PHASE=init
export SENTINEL PHASES_FILE

write_sentinel() {
  local rc=$?
  ISSUE_RC="$rc" CURRENT_PHASE="$CURRENT_PHASE" uv run python - <<'PY'
import json, os
phases = []
p = os.environ["PHASES_FILE"]
if os.path.exists(p):
    for line in open(p):
        line = line.strip()
        if line:
            phases.append(json.loads(line))
rc = int(os.environ["ISSUE_RC"])
failing = next((ph["phase"] for ph in phases if ph["rc"] != 0), None)
out = {
    "issue": 2224,
    "round": "followup-r1",
    "status": "done" if rc == 0 and failing is None else "failed",
    "rc": rc,
    "failing_phase": failing,
    "current_phase": os.environ["CURRENT_PHASE"],
    "phases": phases,
}
with open(os.environ["SENTINEL"], "w") as f:
    json.dump(out, f, indent=1)
print("sentinel written:", os.environ["SENTINEL"])
PY
}
trap write_sentinel EXIT

run_phase() {
  local name="$1"; shift
  CURRENT_PHASE="$name"
  echo "[phase=$name] start $(date -u +%FT%TZ)"
  "$@" > "$LOGDIR/issue-2224-fu-r1-$name.log" 2>&1
  local rc=$?
  echo "{\"phase\": \"$name\", \"rc\": $rc}" >> "$PHASES_FILE"
  echo "[phase=$name] rc=$rc $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then
    echo "[phase=$name] FAILED — tail:"; tail -15 "$LOGDIR/issue-2224-fu-r1-$name.log"
    exit "$rc"
  fi
}

# 1) Stage: stream-reduce the banked P0c capture shards into fp16 layer slices
#    (per-shard part checkpoints; resume-safe on relaunch) + rb_v2 vectors.
run_phase stage uv run python scripts/issue2224_followup_r1.py --phase stage

# 2) Leg 1: E1 per-corpus map refit (both mapping arms; GPU eigh/matmuls).
run_phase refit uv run python scripts/issue2224_followup_r1.py --phase refit --device cuda

# 3) Leg 2: cross-corpus probe transport (judged subsets; GPU).
run_phase transport uv run python scripts/issue2224_followup_r1.py --phase transport --device cuda

# 4) Join legs 1+2 into the round summary JSON.
run_phase aggregate uv run python scripts/issue2224_followup_r1.py --phase aggregate

# 5) Persist the refit maps + per-sample score npz to HF (bulk commit +
#    name-set verify). Staged slices are regenerable (recipe = --phase stage)
#    and are deliberately not uploaded.
run_phase upload uv run python scripts/issue2224_followup_r1.py --phase upload --legs refit

# Git commits of eval_results/ outputs land in the orchestrator's completion turn.
echo "ROUND DONE $(date -u +%FT%TZ)"
