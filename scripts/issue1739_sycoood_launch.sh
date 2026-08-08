#!/bin/bash
# #1739 sycophancy OOD rungs — pod-side launcher (detached via setsid nohup).
#
# PHASES RUN AS SEPARATE PROCESS INVOCATIONS. This is load-bearing, not style:
# `generate_labeling` builds a vLLM engine whose EngineCore worker is NOT reaped
# by in-process teardown, so a same-process `capture.load_capture_model` finds
# the GPU still held and OOMs. The smoke caught exactly that — gen1/aysa/aysb all
# succeeded, then capture died with "GPU 0 ... 80.56 MiB is free. Process X has
# 72.84 GiB in use" (the documented vLLM worker-subprocess teardown trap,
# .claude/rules/gotchas.md). Process exit is the only reliable release, and the
# gotchas rule names subprocess isolation per phase as the escape hatch.
#
# Env is sourced INSIDE the launcher (the SSH-MCP shell is `sh`; a top-level
# `source .env` dies with "source: not found" — #545).
set -uo pipefail

REPO=/workspace/explore-persona-space
LOGDIR=/workspace/logs
mkdir -p "$LOGDIR"

cd "$REPO" || exit 1
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

# Pid file rewritten by THIS launch (never leave a predecessor's pid behind).
echo $$ > "$LOGDIR/issue-1739-syco-ood.pid"

echo "[launcher] HEAD=$(git rev-parse HEAD)"

COMMON="--staged-dir data/issue_1739/syco_ood/staged \
  --out-root eval_results/issue_1739/syco_ood \
  --main-root raw_completions/issue_1739_syco_ood/main \
  --passa-root raw_completions/issue_1739_syco_ood/passa \
  --store-dir data/issue_1739/syco_ood/store \
  --sentinel $LOGDIR/issue-1739-syco-ood-results.json"

for PHASE in gen1 aysa aysb capture upload; do
  echo "[launcher] === phase $PHASE start $(date -u +%FT%TZ) ==="
  # shellcheck disable=SC2086
  uv run python scripts/issue1739_sycoood_pod.py --phase "$PHASE" $COMMON
  rc=$?
  echo "[launcher] === phase $PHASE rc=$rc $(date -u +%FT%TZ) ==="
  if [ "$rc" -ne 0 ]; then
    echo "[launcher] ABORT: phase $PHASE failed rc=$rc"
    exit "$rc"
  fi
  # Belt-and-suspenders: reap any EngineCore worker that outlived its parent
  # before the next phase loads a model (crash-orphan reaper, gotchas.md).
  for p in $(pgrep -af '^VLLM::EngineCore' | awk '{print $1}'); do kill -KILL "$p" 2>/dev/null; done
  sleep 5
done

echo "[launcher] ALL PHASES OK"
exit 0
