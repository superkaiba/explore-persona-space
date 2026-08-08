#!/usr/bin/env bash
# #1947 FROZEN launch flags (r1 code-review Minor 4: plan §10 froze flags at
# implementation; the worker REQUIRES a --smoke|--full mode flag the plan's
# sketched --workload-cmd lines omitted).
#
# Compute lane: RunPod per the BINDING progress-v6 directive ("run on runpod as
# much in parallel as possible (if needed do multiple 8xh100s)") — provision
# 8xH100 pods via `pod.py provision --issue 1947 [--name-suffix a|b]`, then
# launch the stage command below POD-SIDE via the canonical setsid launcher
# (experimenter.md § During Execution). probe/judge/select run on the VM
# (0 GPU / Batch API). This script COMPOSES the frozen per-stage command:
#   bash scripts/issue1947_dispatch.sh <stage>        # print (default)
#   bash scripts/issue1947_dispatch.sh <stage> exec   # run in-place (VM stages)
set -euo pipefail

STAGE="${1:?usage: issue1947_dispatch.sh <pilot|fleet-a|fleet-b|probe|judge|select|battery> [exec]}"
MODE="${2:-print}"

case "$STAGE" in
  # Pod-side (8xH100; dispatcher fans per-cell subprocesses over every GPU,
  # CVD pinned in the launcher env — the #545 clobber rule):
  pilot)
    CMD="uv run python scripts/issue1947_worker.py --full --dispatch pilot --sentinel-dir /workspace/logs"
    ;;
  fleet-a)
    CMD="uv run python scripts/issue1947_worker.py --full --dispatch fleet-a --seeds 42,137 --sentinel-dir /workspace/logs"
    ;;
  fleet-b)
    CMD="uv run python scripts/issue1947_worker.py --full --dispatch fleet-b --seeds 42,137 --sentinel-dir /workspace/logs"
    ;;
  battery)
    CMD="uv run python scripts/issue1947_battery.py --phase capture-fit --sentinel-dir /workspace/logs"
    ;;
  # VM-side (0 GPU / Batch API):
  probe)
    CMD="uv run python scripts/issue1947_battery.py --phase probe"
    ;;
  judge)
    CMD="uv run python scripts/issue1947_battery.py --phase judge"
    ;;
  select)
    CMD="uv run python scripts/issue1947_battery.py --phase select"
    ;;
  *)
    echo "unknown stage ${STAGE}" >&2
    exit 2
    ;;
esac

if [ "$MODE" = "exec" ]; then
  exec $CMD
else
  echo "$CMD"
fi
