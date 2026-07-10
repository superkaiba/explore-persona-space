#!/usr/bin/env bash
# #1090 fu1-margin-qwen GPU-lane dispatcher: stage-inputs -> c5 topup (qwen
# vLLM) -> train (+Tier-1 dose reads) -> tier2 generation -> c3+c5 margin
# sweep -> upload -> sentinel -> [phase=done].
#
# All science lives in scripts/issue1090_fu1.py (the unified smoke/full
# driver); this wrapper owns env setup, the WandB project pin, and the SINGLE
# terminal [phase=done] line (reserved token — the poller's done signal; the
# python driver never emits it). Mirrors scripts/issue1090_dispatch.sh.
#
# Usage (the FULL pod-side entrypoint; smokes invoke the python driver
# directly with --smoke):
#   REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue1090_fu1_dispatch.sh --phase gpu [driver args...]
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Conditional .env sourcing (gotchas.md: the GCE lane exports tokens via its
# startup script and has NO .env file — never unconditional inside the chain).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

export WANDB_PROJECT="${WANDB_PROJECT:-issue1090}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "[phase=dispatch] issue1090_fu1 dispatcher starting (args: $*)"

# Default to --full; a caller passing --smoke gets the tiny-real path through
# the SAME dispatcher.
MODE="--full"
for arg in "$@"; do
    if [ "$arg" = "--smoke" ] || [ "$arg" = "--full" ]; then MODE=""; fi
done

rc=0
# shellcheck disable=SC2086
uv run python scripts/issue1090_fu1.py $MODE "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[phase=driver_failed] issue1090_fu1 driver exited rc=$rc" >&2
    exit "$rc"
fi

echo "[phase=done]"
