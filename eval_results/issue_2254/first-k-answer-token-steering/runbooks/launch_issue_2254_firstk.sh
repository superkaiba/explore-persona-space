#!/bin/bash
# issue #2254 follow-up `first-k-answer-token-steering` — unit-1 launcher
# (stage_inputs + steer). Plan v10 §9 shape: stage_inputs ONCE serially
# (CPU/download-only, "before any GPU spend"), then 4 parallel steer shard
# workers with launcher-env CUDA_VISIBLE_DEVICES pins (#543/#545 recipe).
# Composed by experimenter 2026-08-23; re-executable verbatim on relaunch.
set -uo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && source .env; set +a

echo $$ > /workspace/logs/issue-2254-firstk-steer.pid
echo $$ > /workspace/logs/issue-2254.pid

DRIVER=scripts/issue2254_first_k_steering.py

echo "launcher: stage_inputs (serial, single invocation) starting $(date -u +%Y-%m-%dT%H:%M:%SZ)"
uv run python "$DRIVER" --phases stage_inputs --num-shards 4
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "launcher: stage_inputs FAILED rc=$RC - aborting before GPU fan-out (terminal token suppressed)"
  exit "$RC"
fi

echo "launcher: steer fan-out, 4 CVD-pinned shard workers, starting $(date -u +%Y-%m-%dT%H:%M:%SZ)"
PIDS=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$i" uv run python "$DRIVER" --phases steer --shard-id "$i" --num-shards 4 \
    > "/workspace/logs/issue-2254-firstk-steer-shard$i.log" 2>&1 &
  PIDS[$i]=$!
  echo "${PIDS[$i]}" > "/workspace/logs/issue-2254-firstk-steer-shard$i.pid"
  echo "launcher: shard $i worker pid ${PIDS[$i]} (CUDA_VISIBLE_DEVICES=$i)"
done

RC_ALL=0
for i in 0 1 2 3; do
  wait "${PIDS[$i]}"
  rc=$?
  echo "launcher: shard $i exited rc=$rc at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "$rc" -ne 0 ]; then RC_ALL="$rc"; fi
done

if [ "$RC_ALL" -eq 0 ]; then
  echo "launcher: all 4 shards clean - writing attempt-bound completion sentinel"
  uv run python -c "from explore_persona_space.backends.artifacts import write_completion_sentinel; write_completion_sentinel(sentinel_path='/workspace/eval_results/issue_2254/rp-20260823T105013Z-6544/.completion-sentinel.json', issue=2254)"
  SRC=$?
  echo "launcher: sentinel write rc=$SRC"
  exit "$SRC"
fi
echo "launcher: ONE OR MORE SHARDS FAILED rc=$RC_ALL (terminal token suppressed)"
exit "$RC_ALL"
