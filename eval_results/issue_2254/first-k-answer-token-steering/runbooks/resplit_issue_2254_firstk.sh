#!/bin/bash
# issue #2254 first-k round — shard-3 width re-eval resplit supervisor.
# Kills the solo mod-4 shard-3 worker (captured pids) and re-splits its
# remaining cells across all 4 GPUs as mod-16 sub-shards {3,7,11,15}
# (exact partition of cells[3::4]; strided slicing, driver line 1182).
# Takes over the run pidfiles and writes the SAME attempt-bound completion
# sentinel the original launcher owned, preserving the #598 chain.
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a

MAIN_LOG=/workspace/logs/issue-2254-firstk-steer.log
exec >> "$MAIN_LOG" 2>&1

echo "resplit: supervisor $$ starting $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Take over the run pidfiles FIRST so the VM poll chain tracks this supervisor.
echo $$ > /workspace/logs/issue-2254-firstk-steer.pid
echo $$ > /workspace/logs/issue-2254.pid

# Kill the solo shard-3 worker by CAPTURED pid (uv wrapper 2550 + python 2559).
for P in 2550 2559; do
  if kill -0 "$P" 2>/dev/null; then
    kill -TERM "$P" 2>/dev/null || true
  fi
done
for i in $(seq 1 30); do
  if ! kill -0 2559 2>/dev/null && ! kill -0 2550 2>/dev/null; then break; fi
  sleep 2
done
if kill -0 2559 2>/dev/null; then
  echo "resplit: worker 2559 survived TERM after 60s — escalating KILL"
  kill -KILL 2559 2>/dev/null || true
  kill -KILL 2550 2>/dev/null || true
  sleep 3
fi
echo "resplit: old shard-3 worker down at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

DRIVER=scripts/issue2254_first_k_steering.py
SUBIDS=(3 7 11 15)
PIDS=()
for j in 0 1 2 3; do
  SID="${SUBIDS[$j]}"
  CUDA_VISIBLE_DEVICES="$j" uv run python "$DRIVER" --phases steer --shard-id "$SID" --num-shards 16 \
    > "/workspace/logs/issue-2254-firstk-steer-resplit-shard$SID.log" 2>&1 &
  PIDS[$j]=$!
  echo "${PIDS[$j]}" > "/workspace/logs/issue-2254-firstk-steer-resplit-shard$SID.pid"
  echo "resplit: sub-shard $SID/16 worker pid ${PIDS[$j]} (CUDA_VISIBLE_DEVICES=$j)"
done

RC_ALL=0
for j in 0 1 2 3; do
  SID="${SUBIDS[$j]}"
  wait "${PIDS[$j]}"
  rc=$?
  echo "resplit: sub-shard $SID exited rc=$rc at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "$rc" -ne 0 ]; then RC_ALL="$rc"; fi
done

if [ "$RC_ALL" -eq 0 ]; then
  echo "resplit: all 4 sub-shards clean - writing attempt-bound completion sentinel"
  uv run python -c "from explore_persona_space.backends.artifacts import write_completion_sentinel; write_completion_sentinel(sentinel_path='/workspace/eval_results/issue_2254/rp-20260823T105013Z-6544/.completion-sentinel.json', issue=2254)"
  SRC=$?
  echo "resplit: sentinel write rc=$SRC"
  exit "$SRC"
fi
echo "resplit: ONE OR MORE SUB-SHARDS FAILED rc=$RC_ALL (terminal token suppressed)"
exit "$RC_ALL"
